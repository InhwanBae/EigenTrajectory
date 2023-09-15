import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from . import *


class ETTrainer:
    r"""Base class for all Trainers"""

    def __init__(self, args, hyper_params):
        print("Trainer initiating...")

        # Reproducibility
        reproducibility_settings(seed=0)

        self.args, self.hyper_params = args, hyper_params
        self.model, self.optimizer, self.scheduler = None, None, None
        self.loader_train, self.loader_val, self.loader_test = None, None, None
        self.dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
        self.checkpoint_dir = hyper_params.checkpoint_dir + '/' + args.tag + '/' + hyper_params.dataset + '/'
        print("Checkpoint dir:", self.checkpoint_dir)
        self.log = {'train_loss': [], 'val_loss': []}
        self.stats_func, self.stats_meter = None, None
        self.reset_metric()

        if not args.test:
            # Save arguments and configs
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            with open(self.checkpoint_dir + 'args.pkl', 'wb') as fp:
                pickle.dump(args, fp)

            with open(self.checkpoint_dir + 'config.pkl', 'wb') as fp:
                pickle.dump(hyper_params, fp)

    def set_optimizer_scheduler(self):
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.hyper_params.lr,
                                           weight_decay=self.hyper_params.weight_decay)

        if self.hyper_params.lr_schd:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=self.hyper_params.lr_schd_step,
                                                             gamma=self.hyper_params.lr_schd_gamma)

    def init_descriptor(self):
        # Calculate ET descriptor parameters
        print("ET descriptor initialization...")
        obs_traj = torch.cat([self.loader_train.dataset.obs_traj, self.loader_val.dataset.obs_traj], dim=0)
        pred_traj = torch.cat([self.loader_train.dataset.pred_traj, self.loader_val.dataset.pred_traj], dim=0)
        obs_traj, pred_traj = augment_trajectory(obs_traj, pred_traj)
        self.model.calculate_parameters(obs_traj, pred_traj)
        print("Anchor generation...")

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def valid(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def fit(self):
        print("Training started...")
        for epoch in range(self.hyper_params.num_epochs):
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            # Save the best model
            if epoch == 0 or self.log['val_loss'][-1] < min(self.log['val_loss'][:-1]):
                self.save_model()

            print(" ")
            print("Dataset: {0}, Epoch: {1}".format(self.hyper_params.dataset, epoch))
            print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(self.log['train_loss'][-1], self.log['val_loss'][-1]))
            print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(np.array(self.log['val_loss']).argmin(),
                                                                     np.array(self.log['val_loss']).min()))
            print(" ")
        print("Done.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde,
                           'TCC': compute_batch_tcc, 'COL': compute_batch_col}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = self.checkpoint_dir + filename
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, filename='model_best.pth'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = self.checkpoint_dir + filename
        torch.save(self.model.state_dict(), model_path)


class ETSequencedMiniBatchTrainer(ETTrainer):
    r"""Base class using sequenced mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        obs_len, pred_len = hyper_params.obs_len, hyper_params.pred_len
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=1)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=1)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0
        is_first_loss = True

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_eigentraj"] + output["loss_euclidean_ade"] + output["loss_euclidean_fde"]
            loss[torch.isnan(loss)] = 0

            if (cnt + 1) % self.hyper_params.batch_size != 0 and (cnt + 1) != len(self.loader_train):
                if is_first_loss:
                    is_first_loss = False
                    loss_cum = loss
                else:
                    loss_cum += loss

            else:
                is_first_loss = True
                loss_cum += loss
                loss_cum /= self.hyper_params.batch_size
                loss_cum.backward()

                if self.hyper_params.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

                self.optimizer.step()
                loss_batch += loss_cum.item()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            # Trajectory noise perturbation
            # obs_traj = obs_traj + torch.randn_like(obs_traj) * 0.10

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)

                # Non-linear trajectory
                # nl_ped = batch[2].cuda().gt(0.5)
                # value = value[nl_ped]

                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class ETCollatedMiniBatchTrainer(ETTrainer):
    r"""Base class using collated mini-batch training strategy"""

    def __init__(self, args, hyper_params):
        super().__init__(args, hyper_params)

        # Dataset preprocessing
        obs_len, pred_len = hyper_params.obs_len, hyper_params.pred_len
        batch_size = hyper_params.batch_size
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_eigentraj"] + output["loss_euclidean_ade"] + output["loss_euclidean_fde"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class ETSTGCNNTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using Social-STGCNN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        predictor_model = base_model(n_stgcnn=1, n_txpcnn=5, input_feat=1, output_feat=hyper_params.num_samples,
                                     kernel_size=3, seq_len=hyper_params.k+2, pred_seq_len=hyper_params.k).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETSGCNTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using SGCN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        predictor_model = base_model(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1, dropout=0,
                                     obs_len=hyper_params.k+2, pred_len=hyper_params.k,
                                     n_tcn=5, in_dims=1, out_dims=hyper_params.num_samples).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETPECNetTrainer(ETCollatedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using PECNet baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        import yaml
        with open("./baseline/pecnet/optimal.yaml", 'r') as file:
            pecnet_params = yaml.load(file, Loader=yaml.FullLoader)

        predictor_model = base_model(pecnet_params["enc_past_size"], pecnet_params["enc_dest_size"],
                                     pecnet_params["enc_latent_size"], pecnet_params["dec_size"],
                                     pecnet_params["predictor_hidden_size"], pecnet_params['non_local_theta_size'],
                                     pecnet_params['non_local_phi_size'], pecnet_params['non_local_g_size'],
                                     pecnet_params["fdim"], pecnet_params["zdim"], pecnet_params["nonlocal_pools"],
                                     pecnet_params['non_local_dim'], pecnet_params["sigma"], hyper_params.k//2,
                                     (hyper_params.k)*hyper_params.num_samples//2+1, False).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]
            scene_mask, seq_start_end = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

            self.optimizer.zero_grad()

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, pred_traj, addl_info=additional_information)

            loss = output["loss_eigentraj"] + output["loss_euclidean_ade"] + output["loss_euclidean_fde"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]
            scene_mask, seq_start_end = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]
            scene_mask, seq_start_end = [tensor.cuda(non_blocking=True) for tensor in batch[-2:]]

            additional_information = {"scene_mask": scene_mask, "num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, addl_info=additional_information)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class ETAgentFormerTrainer(ETCollatedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using AgentFormer baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        from baseline.agentformer.utils.config import Config
        cfg = Config("./baseline/agentformer/agentformer_pre.yml", create_dirs=True)
        cfg.past_frames, cfg.future_frames = hyper_params.k + 2, hyper_params.k
        cfg.motion_dim, cfg.forecast_dim = 1, hyper_params.num_samples
        cfg.input_type, cfg.pred_type, cfg.sn_out_type, cfg.scene_orig_all_past = ['pos'], 'pos', None, False
        cfg.nz, cfg.ar_train, cfg.learn_prior = 0, False, False
        predictor_model = base_model(cfg).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETLBEBMTrainer(ETCollatedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using LB-EBM baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        lbebm_params = {'seed': 1, 'gpu_deterministic': False, "data_scale": 60, "dec_size": [1024, 512, 1024],
                        "enc_dest_size": [256, 128], "enc_latent_size": [256, 512], "enc_past_size": [512, 256],
                        "predictor_hidden_size": [1024, 512, 256], "non_local_theta_size": [256, 128, 64],
                        "non_local_phi_size": [256, 128, 64], "non_local_g_size": [256, 128, 64], "non_local_dim": 128,
                        "fdim": 16, "future_length": 12, "device": 1, "kld_coeff": 0.5, "future_loss_coeff": 1,
                        "dest_loss_coeff": 2, "learning_rate": 0.0001, "lr_decay_step_size": 30, "lr_decay_gamma": 0.5,
                        "mu": 0, "n_values": 20, "nonlocal_pools": 3, "num_epochs": 100, "num_workers": 0,
                        "past_length": 8, "sigma": 1.3, "zdim": 16, "print_log": 6, "sub_goal_indexes": [2, 5, 8, 11],
                        'e_prior_sig': 2, 'e_init_sig': 2, 'e_activation': 'lrelu', 'e_activation_leak': 0.2,
                        'e_energy_form': 'identity', 'e_l_steps': 20, 'e_l_steps_pcd': 20, 'e_l_step_size': 0.4,
                        'e_l_with_noise': True, 'e_sn': False, 'e_lr': 0.00003, 'e_is_grad_clamp': False,
                        'e_max_norm': 25, 'e_decay': 1e-4, 'e_gamma': 0.998, 'e_beta1': 0.9, 'e_beta2': 0.999,
                        'memory_size': 200000, 'dataset_name': 'univ', 'dataset_folder': 'dataset', 'obs': 8,
                        'preds': 12, 'delim': '\t', 'verbose': False, 'val_size': 0, 'batch_size': 70, 'ny': 1,
                        'model_path': 'saved_models/lbebm_univ.pt'}
        lbebm_params = DotDict(lbebm_params)
        lbebm_params.sub_goal_indexes = [11]
        predictor_model = base_model(lbebm_params.enc_past_size, lbebm_params.enc_dest_size,
                                     lbebm_params.enc_latent_size, lbebm_params.dec_size,
                                     lbebm_params.predictor_hidden_size, lbebm_params.fdim, lbebm_params.zdim,
                                     lbebm_params.sigma, hyper_params.k//2, hyper_params.k*hyper_params.num_samples//2,
                                     args=lbebm_params).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()

    def train(self, epoch):
        self.model.train()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            additional_information = {"num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, pred_traj, addl_info=additional_information)

            loss = output["loss_eigentraj"] + output["loss_euclidean_ade"] + output["loss_euclidean_fde"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            additional_information = {"num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.cuda(non_blocking=True) for tensor in batch[:2]]

            additional_information = {"num_samples": self.hyper_params.num_samples}
            output = self.model(obs_traj, addl_info=additional_information)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class ETDMRGCNTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using DMRGCN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        predictor_model = base_model(n_stgcn=1, n_tpcnn=4, input_feat=1, output_feat=hyper_params.num_samples,
                                     kernel_size=3, seq_len=hyper_params.k+2, pred_seq_len=hyper_params.k).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETGPGraphTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using GP-Graph baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        predictor_model = base_model(obs_len=hyper_params.k+2, pred_len=hyper_params.k,
                                     in_dims=1, out_dims=hyper_params.num_samples).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETGPGraphSGCNTrainer(ETGPGraphTrainer):
    r"""EigenTrajectory model trainer using GP-Graph-SGCN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(base_model, model, hook_func, args, hyper_params)


class ETGPGraphSTGCNNTrainer(ETGPGraphTrainer):
    r"""EigenTrajectory model trainer using GP-Graph-STGCNN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(base_model, model, hook_func, args, hyper_params)


class ETGraphTERNTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using Graph-TERN baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        predictor_model = base_model(n_epgcn=1, n_epcnn=6, input_feat=1, seq_len=hyper_params.k+2,
                                     pred_seq_len=hyper_params.k, n_smpl=hyper_params.num_samples).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()


class ETImplicitTrainer(ETSequencedMiniBatchTrainer):
    r"""EigenTrajectory model trainer using Social-Implicit baseline predictor"""

    def __init__(self, base_model, model, hook_func, args, hyper_params):
        super().__init__(args, hyper_params)

        # EigenTrajectory model preparation
        CFG = {"spatial_input": 2, "spatial_output": 2, "temporal_input": 8, "temporal_output": 12,
               "bins": [0, 0.01, 0.1, 1.2], "noise_weight": [0.05, 1, 4, 8], "noise_weight_eth": [0.175, 1.5, 4, 8]}
        predictor_model = base_model(spatial_input=1,
                                     spatial_output=hyper_params.num_samples,
                                     temporal_input=hyper_params.k+2,
                                     temporal_output=hyper_params.k,
                                     bins=CFG["bins"],
                                     noise_weight=CFG["noise_weight"]).cuda()
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params).cuda()
        self.model = eigentraj_model
        self.set_optimizer_scheduler()
