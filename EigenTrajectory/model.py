import torch
import torch.nn as nn
from .anchor import ETAnchor
from .descriptor import ETDescriptor


class EigenTrajectory(nn.Module):
    r"""The EigenTrajectory model

    Args:
        baseline_model (nn.Module): The baseline model
        hook_func (dict): The bridge functions for the baseline model
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, baseline_model, hook_func, hyper_params):
        super().__init__()

        self.baseline_model = baseline_model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.static_dist = hyper_params.static_dist

        self.ET_m_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=True)
        self.ET_s_descriptor = ETDescriptor(hyper_params=hyper_params, norm_sca=False)
        self.ET_m_anchor = ETAnchor(hyper_params=hyper_params)
        self.ET_s_anchor = ETAnchor(hyper_params=hyper_params)

    def calculate_parameters(self, obs_traj, pred_traj):
        r"""Calculate the ET descriptors of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        # Mask out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj, pred_m_traj = obs_traj[mask], pred_traj[mask]
        obs_s_traj, pred_s_traj = obs_traj[~mask], pred_traj[~mask]

        # Descriptor initialization
        data_m = self.ET_m_descriptor.parameter_initialization(obs_m_traj, pred_m_traj)
        data_s = self.ET_s_descriptor.parameter_initialization(obs_s_traj, pred_s_traj)

        # Anchor generation
        self.ET_m_anchor.anchor_generation(*data_m)
        self.ET_s_anchor.anchor_generation(*data_s)

    def forward(self, obs_traj, pred_traj=None, addl_info=None):
        r"""The forward function of the EigenTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)
            addl_info (dict): The additional information (optional, if baseline model requires)

        Returns:
            output (dict): The output of the model (recon_traj, loss, etc.)
        """

        n_ped = obs_traj.size(0)

        # Filter out static trajectory
        mask = (obs_traj[:, -1] - obs_traj[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        obs_m_traj = obs_traj[mask]
        obs_s_traj = obs_traj[~mask]
        pred_m_traj_gt = pred_traj[mask] if pred_traj is not None else None
        pred_s_traj_gt = pred_traj[~mask] if pred_traj is not None else None

        # Projection
        C_m_obs, C_m_pred_gt = self.ET_m_descriptor.projection(obs_m_traj, pred_m_traj_gt)
        C_s_obs, C_s_pred_gt = self.ET_s_descriptor.projection(obs_s_traj, pred_s_traj_gt)
        C_obs = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
        C_obs[:, mask], C_obs[:, ~mask] = C_m_obs, C_s_obs  # KN

        # Absolute coordinate
        obs_m_ori = self.ET_m_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_s_ori = self.ET_s_descriptor.traj_normalizer.traj_ori.squeeze(dim=1).T
        obs_ori = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj.device)
        obs_ori[:, mask], obs_ori[:, ~mask] = obs_m_ori, obs_s_ori
        obs_ori -= obs_ori.mean(dim=1, keepdim=True)  # move scene to origin

        # Trajectory prediction
        input_data = self.hook_func.model_forward_pre_hook(C_obs, obs_ori, addl_info)
        output_data = self.hook_func.model_forward(input_data, self.baseline_model)
        C_pred_refine = self.hook_func.model_forward_post_hook(output_data, addl_info)

        # Anchor refinement
        C_m_pred = self.ET_m_anchor(C_pred_refine[:, mask])
        C_s_pred = self.ET_s_anchor(C_pred_refine[:, ~mask])

        # Reconstruction
        pred_m_traj_recon = self.ET_m_descriptor.reconstruction(C_m_pred)
        pred_s_traj_recon = self.ET_s_descriptor.reconstruction(C_s_pred)
        pred_traj_recon = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj.device)
        pred_traj_recon[:, mask], pred_traj_recon[:, ~mask] = pred_m_traj_recon, pred_s_traj_recon

        output = {"recon_traj": pred_traj_recon}

        if pred_traj is not None:
            C_pred = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj.device)
            C_pred[:, mask], C_pred[:, ~mask] = C_m_pred, C_s_pred

            # Low-rank approximation for gt trajectory
            C_pred_gt = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj.device)
            C_pred_gt[:, mask], C_pred_gt[:, ~mask] = C_m_pred_gt, C_s_pred_gt
            C_pred_gt = C_pred_gt.detach()

            # Loss calculation
            error_coefficient = (C_pred - C_pred_gt.unsqueeze(dim=-1)).norm(p=2, dim=0)
            error_displacement = (pred_traj_recon - pred_traj.unsqueeze(dim=0)).norm(p=2, dim=-1)
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement[:, :, -1].min(dim=0)[0].mean()

        return output
