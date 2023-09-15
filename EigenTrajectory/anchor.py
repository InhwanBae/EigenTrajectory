import torch
import torch.nn as nn


class ETAnchor(nn.Module):
    r"""EigenTrajectory anchor model

    Args:
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim

        self.C_anchor = nn.Parameter(torch.zeros((self.k, self.s)))

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

    def anchor_generation(self, pred_traj_norm, U_pred_trunc):
        r"""Anchor generation on EigenTrajectory space

        Args:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            U_pred_trunc (torch.Tensor): The truncated ET descriptors (eigenvectors) of the predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        from sklearn.cluster import KMeans
        # Trajectory projection
        C_pred = self.to_ET_space(pred_traj_norm, evec=U_pred_trunc).T.detach().numpy()

        # Anchor generation on EigenTrajectory space
        C_anchor = torch.FloatTensor(
            KMeans(n_clusters=self.s, random_state=0, init='k-means++', n_init=10).fit(C_pred).cluster_centers_.T)

        # Register anchors as model parameters
        self.C_anchor = nn.Parameter(C_anchor.to(self.C_anchor.device))

    def forward(self, C_pred):
        r"""Anchor refinement on EigenTrajectory space

        Args:
            C_pred (torch.Tensor): The predicted ET descriptor coefficients

        Returns:
            C_pred_refine (torch.Tensor): The refined ET descriptor coefficients
        """

        # Anchor Refinement
        C_pred_refine = self.C_anchor.unsqueeze(dim=1).detach() + C_pred
        return C_pred_refine
