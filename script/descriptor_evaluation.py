import sys
import argparse
import numpy as np
import torch
from EigenTrajectory import TrajNorm
from CurveModel import *
from utils import *


# Reproducibility
reproducibility_settings(seed=0)

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)


def eval_method(args):
    data_set = '../datasets/' + args.dataset + '/'
    print('Scene: {}'.format(args.dataset))

    args.batch_size = 1e8
    loader_test = get_dataloader(data_set, 'test', args.obs_len, args.pred_len, args.batch_size)

    # Preprocessing
    obs_traj = loader_test.dataset.obs_traj
    pred_traj = loader_test.dataset.pred_traj

    n_ped, t_obs, dim = obs_traj.shape
    n_ped, t_pred, dim = pred_traj.shape

    # Normalization
    traj_norm = TrajNorm(ori=True, rot=True, sca=False)
    traj_norm.calculate_params(obs_traj)

    obs_traj_norm = traj_norm.normalize(obs_traj)
    pred_traj_norm = traj_norm.normalize(pred_traj)

    # Linear
    print('===Linear===')
    obs_traj_linear_basis = torch.stack([torch.linspace(0, 1, t_obs), torch.linspace(1, 0, t_obs)], dim=1)
    obs_traj_linear_norm = curve_fitting(obs_traj_norm, obs_traj_linear_basis)
    obs_traj_linear_recon = traj_norm.denormalize(obs_traj_linear_norm)
    pred_traj_linear_basis = torch.stack([torch.linspace(0, 1, t_pred), torch.linspace(1, 0, t_pred)], dim=1)
    pred_traj_linear_norm = curve_fitting(pred_traj_norm, pred_traj_linear_basis)
    pred_traj_linear_recon = traj_norm.denormalize(pred_traj_linear_norm)

    print('num params: {}'.format(2 * dim), end='\t')
    print('obs error: {:.4f}'.format((obs_traj_linear_recon - obs_traj).norm(p=2, dim=-1).mean().item()), end='\t')
    print('pred error: {:.4f}'.format((pred_traj_linear_recon - pred_traj).norm(p=2, dim=-1).mean().item()))

    # Bezier Curve
    print('===Bezier Curve===')
    for n_degree in range(2, 6):
        # parameter settings for Bézier curve
        obs_traj_bezier_basis = bezier_basis(degree=n_degree, step=t_obs)
        obs_traj_bezier_norm = curve_fitting(obs_traj_norm, obs_traj_bezier_basis)
        obs_traj_bezier_recon = traj_norm.denormalize(obs_traj_bezier_norm)
        pred_traj_bezier_basis = bezier_basis(degree=n_degree, step=t_pred)
        pred_traj_bezier_norm = curve_fitting(pred_traj_norm, pred_traj_bezier_basis)
        pred_traj_bezier_recon = traj_norm.denormalize(pred_traj_bezier_norm)

        print('degree: {}'.format(n_degree), end='\t')
        print('num params: {}'.format((n_degree + 1) * dim), end='\t')
        print('obs error: {:.4f}'.format((obs_traj_bezier_recon - obs_traj).norm(p=2, dim=-1).mean().item()), end='\t')
        print('pred error: {:.4f}'.format((pred_traj_bezier_recon - pred_traj).norm(p=2, dim=-1).mean().item()))

    # B-Spline
    print('===B-Spline===')
    for n_degree in range(1, 4):
        for n_curve in range(2, 6):
            if n_curve <= n_degree:
                continue
            # parameter settings for B-spline
            obs_traj_bspline_basis = bspline_basis(cpoint=n_curve, degree=n_degree, step=t_obs)
            obs_traj_bspline_norm = curve_fitting(obs_traj_norm, obs_traj_bspline_basis)
            obs_traj_bspline_recon = traj_norm.denormalize(obs_traj_bspline_norm)
            pred_traj_bspline_basis = bspline_basis(cpoint=n_curve, degree=n_degree, step=t_pred)
            pred_traj_bspline_norm = curve_fitting(pred_traj_norm, pred_traj_bspline_basis)
            pred_traj_bspline_recon = traj_norm.denormalize(pred_traj_bspline_norm)

            print('n_curve: {}'.format(n_curve), end='\t')
            print('degree: {}'.format(n_degree), end='\t')
            print('num params: {}'.format((n_curve + 1) * dim), end='\t')
            print('obs error: {:.4f}'.format((obs_traj_bspline_recon - obs_traj).norm(p=2, dim=-1).mean().item()), end='\t')
            print('pred error: {:.4f}'.format((pred_traj_bspline_recon - pred_traj).norm(p=2, dim=-1).mean().item()))

    # Singular Value Decomposition
    print('===Singular Value Decomposition===')
    A = obs_traj_norm.reshape(n_ped, t_obs * dim).T
    B = pred_traj_norm.reshape(n_ped, t_pred * dim).T
    U_obs, S_obs, Vt_obs = torch.linalg.svd(A, full_matrices=False)
    U_pred, S_pred, Vt_pred = torch.linalg.svd(B, full_matrices=False)

    for k in range(1, 13):
        # Low-rank approximation
        U_obs_trunc = U_obs[:, :k]
        U_pred_trunc = U_pred[:, :k]
        C_obs = U_obs_trunc.T @ A
        C_pred = U_pred_trunc.T @ B
        A_recon = U_obs_trunc @ C_obs
        B_recon = U_pred_trunc @ C_pred

        # Denormalization
        obs_traj_norm_recon = A_recon.T.reshape(n_ped, t_obs, dim)
        pred_traj_norm_recon = B_recon.T.reshape(n_ped, t_pred, dim)
        obs_traj_recon = traj_norm.denormalize(obs_traj_norm_recon)
        pred_traj_recon = traj_norm.denormalize(pred_traj_norm_recon)

        print('k: {}'.format(k), end='\t')
        print('num params: {}'.format(k), end='\t')
        print('obs error: {:.4f}'.format((obs_traj_recon - obs_traj).norm(p=2, dim=-1).mean().item()), end='\t')
        print('pred error: {:.4f}'.format((pred_traj_recon - pred_traj).norm(p=2, dim=-1).mean().item()))


if __name__ == '__main__':
    args = parser.parse_args()
    dataset_all = ["eth", "hotel", "univ", "zara1", "zara2"]

    for scene in dataset_all:
        args.dataset = scene
        eval_method(args)
