import torch
import torch.nn as nn
from torch.distributions import Categorical, Independent, Normal, MixtureSameFamily
from .stmrgcn import st_mrgcn, epcnn, trcnn


def generate_adjacency_matrix(V):
    # V[NATVC] -> temp[NATVVC]
    temp = V.unsqueeze(dim=3).repeat_interleave(repeats=V.size(3), dim=3)
    # temp[NATVVC] -> A[NATVV]
    A = (temp - temp.transpose(3, 4)).norm(p=2, dim=5)
    A_inv = 1. / A
    A_inv[A == 0] = 0
    # [A_dist, A_disp, A_dist_inv, A_disp_inv]
    return torch.cat([A, A_inv], dim=1)


class graph_tern(nn.Module):
    def __init__(self, n_epgcn=1, n_epcnn=6, n_trgcn=1, n_trcnn=4, seq_len=8, pred_seq_len=12, n_ways=3, n_smpl=20):
        super().__init__()
        # Control Point Prediction
        self.n_epgcn = n_epgcn
        self.n_epcnn = n_epcnn
        self.n_smpl = n_smpl

        # Trajectory Refinement
        self.n_trgcn = n_trgcn
        self.n_trcnn = n_trcnn

        # Observing & Predicting Sequence frames
        self.obs_seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        # parameters
        input_feat = 2
        hidden_feat = 16
        output_feat = 5
        kernel_size = 3
        total_seq_len = seq_len + pred_seq_len
        self.gamma = 8
        self.n_gmms = 8
        self.n_ways = n_ways

        # Control Point Prediction
        self.tp_mrgcns = nn.ModuleList()
        self.tp_mrgcns.append(st_mrgcn(in_channels=input_feat, out_channels=hidden_feat, kernel_size=(kernel_size, seq_len), relation=4))
        for j in range(1, self.n_epgcn):
            self.tp_mrgcns.append(st_mrgcn(in_channels=hidden_feat, out_channels=hidden_feat, kernel_size=(kernel_size, seq_len), relation=4))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(epcnn(obs_seq_len=seq_len, pred_seq_len=self.n_gmms, in_channels=hidden_feat, out_channels=hidden_feat))
        for j in range(1, self.n_epcnn - 1):
            self.tpcnns.append(epcnn(obs_seq_len=self.n_gmms, pred_seq_len=self.n_gmms, in_channels=hidden_feat, out_channels=hidden_feat))
        self.tpcnns.append(epcnn(obs_seq_len=self.n_gmms, pred_seq_len=self.n_gmms, in_channels=hidden_feat, out_channels=output_feat * self.n_ways))

        # Trajectory Refinement
        self.st_mrgcns = nn.ModuleList()
        self.st_mrgcns.append(st_mrgcn(in_channels=input_feat, out_channels=hidden_feat, kernel_size=(kernel_size, total_seq_len), relation=4))
        for j in range(1, self.n_trgcn):
            self.st_mrgcns.append(st_mrgcn(in_channels=hidden_feat, out_channels=hidden_feat, kernel_size=(kernel_size, total_seq_len), relation=4))

        self.trcnns = nn.ModuleList()
        for j in range(0, self.n_trcnn-1):
            self.trcnns.append(trcnn(total_seq_len=total_seq_len, pred_seq_len=total_seq_len, in_channels=hidden_feat, out_channels=hidden_feat, t_ksize=(n_trcnn-j)*2+1))
        self.trcnns.append(trcnn(total_seq_len=total_seq_len, pred_seq_len=pred_seq_len, in_channels=hidden_feat, out_channels=input_feat))

    def forward(self, S_obs, S_trgt=None, pruning=None):

        ##################################################
        # Control Point Conditioned Endpoint Prediction  #
        ##################################################

        # Generate multi-relational pedestrian graph
        # make adjacency matrix for observed 8 frames
        A_obs = generate_adjacency_matrix(S_obs).detach()

        # Graph Control Point Prediction
        V_obs_abs = S_obs[:, 0]
        V_obs_rel = S_obs[:, 1]

        # NTVC -> NCTV
        V_init = V_obs_rel.permute(0, 3, 1, 2).contiguous()

        for k in range(self.n_epgcn):
            V_init, A_obs = self.tp_mrgcns[k](V_init, A_obs)

        # NCTV -> NTCV
        V_init = V_init.permute(0, 2, 1, 3).contiguous()

        for k in range(self.n_epcnn):
            V_init = self.tpcnns[k](V_init)

        # NTCV -> NTVC
        V_init = V_init.transpose(2, 3).contiguous()

        ##################################################
        #             Trajectory Refinement              #
        ##################################################

        # Guided point sampling
        Gamma = V_obs_rel.mean(dim=1).norm(p=2, dim=-1).squeeze(dim=0) / self.gamma
        Gamma /= self.pred_seq_len  # code optimization for linear interpolation (pre-division)

        if S_trgt is not None:
            # Training phase
            # GT endpoint (NTVC ->  NVC)
            V_trgt_rel = S_trgt[:, 1]
            V_dest_rel = V_trgt_rel.mean(dim=1)

            # Endpoint sampling & classify positive / negative set
            # NMV(C*K) (C: [mu_x, mu_y, std_x, std_y, pi])
            V_init_list = V_init.chunk(chunks=self.n_ways, dim=-1)  # (NMVC)*K
            dest_s_list = []
            for i in range(self.n_ways):
                # NMVC -> NVMC
                temp = V_init_list[i].transpose(1, 2).contiguous()
                mix = Categorical(torch.nn.functional.softmax(temp[:, :, :, 4], dim=-1))
                comp = Independent(Normal(temp[:, :, :, 0:2], temp[:, :, :, 2:4].exp()), 1)
                gmm = MixtureSameFamily(mix, comp)
                dest_s_list.append(gmm.sample((self.n_smpl,)).squeeze(dim=1))  # NVC
            dest_s_list = torch.stack(dest_s_list, dim=3)
            dest_s = dest_s_list.mean(dim=3)
            valid_mask_s = (dest_s - V_dest_rel).norm(p=2, dim=-1).le(Gamma).type(torch.float)

            # Guided endpoint sampling
            eps_r = torch.rand(self.n_smpl, V_dest_rel.size(1), device='cuda') * Gamma  # NV
            eps_t = torch.rand(self.n_smpl, V_dest_rel.size(1), device='cuda')  # NV
            eps_x = eps_r * eps_t.cos()
            eps_y = eps_r * eps_t.sin()
            dest_g = V_dest_rel + torch.stack([eps_x, eps_y], dim=-1)
            valid_mask_g = torch.ones(self.n_smpl, V_dest_rel.size(1), device='cuda')

            # Concatenate all samples
            endpoint_set = torch.cat([dest_s, dest_g], dim=0)
            valid_mask = torch.cat([valid_mask_s, valid_mask_g], dim=0)
        elif pruning is None:
            # Validation phase
            # Endpoint sampling
            # NMV(C*K) (C: [mu_x, mu_y, std_x, std_y, pi])
            V_init_list = V_init.chunk(chunks=self.n_ways, dim=-1)  # (NMVC)*K
            dest_s_list = []
            for i in range(self.n_ways):
                # NMVC -> NVMC
                temp = V_init_list[i].transpose(1, 2).contiguous()
                mix = Categorical(torch.nn.functional.softmax(temp[:, :, :, 4], dim=-1))
                comp = Independent(Normal(temp[:, :, :, 0:2], temp[:, :, :, 2:4].exp()), 1)
                gmm = MixtureSameFamily(mix, comp)
                dest_s_list.append(gmm.sample((self.n_smpl,)).squeeze(dim=1))  # NVC

            dest_s_list = torch.stack(dest_s_list, dim=3)
            endpoint_set = dest_s_list.mean(dim=3)
            valid_mask = torch.ones(self.n_smpl, Gamma.size(0), device='cuda')
        else:
            # Test phase
            # Endpoint sampling with GMM pruning
            # NMV(C*K) (C: [mu_x, mu_y, std_x, std_y, pi])
            endpoint_set_prune = []
            for _ in range(self.n_smpl):
                V_init_list = V_init.chunk(chunks=self.n_ways, dim=-1)  # (NMVC)*K
                dest_s_list = []
                for i in range(self.n_ways):
                    # NMVC -> NVMC
                    temp = V_init_list[i].transpose(1, 2).contiguous()
                    mix_temp = temp[:, :, :, 4]
                    sort_index = torch.argsort(mix_temp.squeeze(dim=0), dim=-1).detach().cpu().numpy()
                    mix_temp[:, torch.arange(V_init.size(2)).unsqueeze(dim=1), sort_index[:, :pruning]] = -1e8
                    mix = Categorical(torch.nn.functional.softmax(mix_temp, dim=-1))
                    comp = Independent(Normal(temp[:, :, :, 0:2], temp[:, :, :, 2:4].exp()), 1)
                    gmm = MixtureSameFamily(mix, comp)
                    dest_s_list.append(gmm.sample((self.n_smpl,)).squeeze(dim=1))  # NVC

                dest_s_list = torch.stack(dest_s_list, dim=3)
                endpoint_set_prune.append(dest_s_list.mean(dim=3))

            endpoint_set_prune = torch.stack(endpoint_set_prune, dim=0)
            argmax_index = (endpoint_set_prune.unsqueeze(dim=2) - endpoint_set_prune.unsqueeze(dim=1))
            argmax_index = argmax_index.norm(p=2, dim=-1).kthvalue(k=2, dim=2)[0].sum(dim=1).argmax(dim=0)
            endpoint_set = endpoint_set_prune[argmax_index, :, torch.arange(V_init.size(2))].transpose(0, 1)
            valid_mask = torch.ones(self.n_smpl, Gamma.size(0), device='cuda')

        # Initial trajectory prediction
        # Linear interpolation NVC -> NTVC
        V_pred = endpoint_set.unsqueeze(dim=1).repeat_interleave(repeats=self.pred_seq_len, dim=1)
        V_pred_abs = (V_pred.cumsum(dim=1) + V_obs_abs.squeeze(dim=0)[-1, :, :]).detach().clone()

        # repeat to sampled times (batch size)
        V_obs_rept = V_obs_rel.repeat_interleave(V_pred.size(0), dim=0)
        A_obs = A_obs.repeat_interleave(V_pred.size(0), dim=0)

        # Graph Trajectory Refinement
        # make adjacency matrix for predicted 12 frames (will be iteratively change)
        A_pred = generate_adjacency_matrix(torch.stack([V_pred_abs, V_pred], dim=1))

        # concatenate to make full 20 frame sequences
        V = torch.cat([V_obs_rept, V_pred], dim=1).detach()
        A = torch.cat([A_obs, A_pred], dim=2).detach()

        # NTVC -> NCTV
        V_corr = V.permute(0, 3, 1, 2).contiguous()

        for k in range(self.n_trgcn):
            V_corr, A = self.st_mrgcns[k](V_corr, A)

        # NCTV -> NTCV
        V_corr = V_corr.permute(0, 2, 1, 3).contiguous()

        for k in range(self.n_trcnn):
            V_corr = self.trcnns[k](V_corr)

        # NTCV -> NTVC
        V_corr = V_corr.transpose(2, 3).contiguous()

        # Refine initial trajectory
        V_refi = V_pred_abs
        V_refi[:, :-1] += V_corr[:, :-1]

        return V_init, V_pred, V_refi, valid_mask


class graph_tern_light(nn.Module):
    def __init__(self, n_epgcn=1, n_epcnn=6, input_feat=2, seq_len=8, pred_seq_len=12, n_smpl=20):
        super().__init__()
        # Control Point Prediction
        self.n_epgcn = n_epgcn
        self.n_epcnn = n_epcnn
        self.n_smpl = n_smpl

        # parameters
        hidden_feat = 16
        kernel_size = 3

        self.tp_mrgcns = nn.ModuleList()
        self.tp_mrgcns.append(st_mrgcn(in_channels=input_feat, out_channels=hidden_feat, kernel_size=(kernel_size, seq_len), relation=4))
        for j in range(1, self.n_epgcn):
            self.tp_mrgcns.append(st_mrgcn(in_channels=hidden_feat, out_channels=hidden_feat, kernel_size=(kernel_size, seq_len), relation=4))

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(epcnn(obs_seq_len=seq_len, pred_seq_len=pred_seq_len, in_channels=hidden_feat, out_channels=hidden_feat))
        for j in range(1, self.n_epcnn - 1):
            self.tpcnns.append(epcnn(obs_seq_len=pred_seq_len, pred_seq_len=pred_seq_len, in_channels=hidden_feat, out_channels=hidden_feat))
        self.tpcnns.append(epcnn(obs_seq_len=pred_seq_len, pred_seq_len=pred_seq_len, in_channels=hidden_feat, out_channels=n_smpl))

    def forward(self, S_obs):
        A_obs = generate_adjacency_matrix(S_obs).detach()

        V_obs_abs = S_obs[:, 0]
        V_obs_rel = S_obs[:, 1]

        # NTVC -> NCTV
        V_init = V_obs_abs.permute(0, 3, 1, 2).contiguous()

        for k in range(self.n_epgcn):
            V_init, A_obs = self.tp_mrgcns[k](V_init, A_obs)

        # NCTV -> NTCV
        V_init = V_init.permute(0, 2, 1, 3).contiguous()

        for k in range(self.n_epcnn):
            V_init = self.tpcnns[k](V_init)

        # NTCV -> NTVC
        V_init = V_init.transpose(2, 3).contiguous()

        return V_init
