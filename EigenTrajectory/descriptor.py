import torch
import torch.nn as nn
from .normalizer import TrajNorm


class ETDescriptor(nn.Module):
    r"""EigenTrajectory descriptor model

    Args:
        hyper_params (DotDict): The hyper-parameters
        norm_ori (bool): Whether to normalize the trajectory with the origin
        norm_rot (bool): Whether to normalize the trajectory with the rotation
        norm_sca (bool): Whether to normalize the trajectory with the scale"""

    def __init__(self, hyper_params, norm_ori=True, norm_rot=True, norm_sca=True):
        super().__init__()

        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.traj_normalizer = TrajNorm(ori=norm_ori, rot=norm_rot, sca=norm_sca)

        self.U_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.k)))
        self.U_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))

    def normalize_trajectory(self, obs_traj, pred_traj=None):
        r"""Trajectory normalization

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (Optional, for training only)

        Returns:
            obs_traj_norm (torch.Tensor): The normalized observed trajectory
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
        """

        self.traj_normalizer.calculate_params(obs_traj)
        obs_traj_norm = self.traj_normalizer.normalize(obs_traj)
        pred_traj_norm = self.traj_normalizer.normalize(pred_traj) if pred_traj is not None else None
        return obs_traj_norm, pred_traj_norm

    def denormalize_trajectory(self, traj_norm):
        r"""Trajectory denormalization

        Args:
            traj_norm (torch.Tensor): The trajectory to be denormalized

        Returns:
            traj (torch.Tensor): The denormalized trajectory
        """

        traj = self.traj_normalizer.denormalize(traj_norm)
        return traj

    def to_ET_space(self, traj, evec):
        r"""Transform Euclidean trajectories to EigenTrajectory coefficients

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The ET descriptors (eigenvectors)

        Returns:
            C (torch.Tensor): The ET descriptor coefficients"""

        # Euclidean -> ET
        tdim = evec.size(0)
        M = traj.reshape(-1, tdim).T
        C = evec.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        r"""Transform EigenTrajectory coefficients to Euclidean trajectories

        Args:
            C (torch.Tensor): The ET descriptor coefficients
            evec (torch.Tensor): The ET descriptors (eigenvectors)

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        # ET -> Euclidean
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj

    def truncated_SVD(self, traj, k=None, full_matrices=False):
        r"""Truncated Singular Value Decomposition

        Args:
            traj (torch.Tensor): The trajectory to be decomposed
            k (int): The number of singular values and vectors to be computed
            full_matrices (bool): Whether to compute full-sized matrices

        Returns:
            U_trunc (torch.Tensor): The truncated left singular vectors
            S_trunc (torch.Tensor): The truncated singular values
            Vt_trunc (torch.Tensor): The truncated right singular vectors
        """

        assert traj.size(2) == self.dim  # NTC
        k = self.k if k is None else k

        # Singular Value Decomposition
        M = traj.reshape(-1, traj.size(1) * self.dim).T
        U, S, Vt = torch.linalg.svd(M, full_matrices=full_matrices)

        # Truncated SVD
        U_trunc, S_trunc, Vt_trunc = U[:, :k], S[:k], Vt[:k, :]
        return U_trunc, S_trunc, Vt_trunc.T

    def parameter_initialization(self, obs_traj, pred_traj):
        r"""Initialize the ET descriptor parameters (for training only)

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Returns:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            U_pred_trunc (torch.Tensor): The truncated eigenvectors of the predicted trajectory

        Note:
            This function should be called once before training the model."""

        # Normalize trajectory
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)

        # Singular Value Decomposition with truncation
        U_obs_trunc, _, _ = self.truncated_SVD(obs_traj_norm)
        U_pred_trunc, _, _ = self.truncated_SVD(pred_traj_norm)

        # Register eigenvectors as model parameters
        self.U_obs_trunc = nn.Parameter(U_obs_trunc.to(self.U_obs_trunc.device))
        self.U_pred_trunc = nn.Parameter(U_pred_trunc.to(self.U_pred_trunc.device))

        # Reuse values for anchor generation
        return pred_traj_norm, U_pred_trunc

    def projection(self, obs_traj, pred_traj=None):
        r"""Trajectory projection to the ET space

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)

        Returns:
            C_obs (torch.Tensor): The observed trajectory in the ET space
            C_pred (torch.Tensor): The predicted trajectory in the ET space (optional, for training only)
        """

        # Trajectory Projection
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        C_obs = self.to_ET_space(obs_traj_norm, evec=self.U_obs_trunc).detach()
        C_pred = self.to_ET_space(pred_traj_norm, evec=self.U_pred_trunc).detach() if pred_traj is not None else None
        return C_obs, C_pred

    def reconstruction(self, C_pred):
        r"""Trajectory reconstruction from the ET space

        Args:
            C_pred (torch.Tensor): The predicted trajectory in the ET space

        Returns:
            pred_traj (torch.Tensor): The predicted trajectory in the Euclidean space
        """

        # Trajectory Reconstruction
        pred_traj_norm = [self.to_Euclidean_space(C_pred[:, :, s], evec=self.U_pred_trunc) for s in range(self.s)]
        pred_traj = [self.denormalize_trajectory(pred_traj_norm[s]) for s in range(self.s)]
        pred_traj = torch.stack(pred_traj, dim=0)  # SNTC
        return pred_traj

    def forward(self, C_pred):
        r"""Alias for reconstruction"""

        return self.reconstruction(C_pred)
