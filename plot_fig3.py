import sys
import argparse
import numpy as np
import torch
from EigenTrajectory import TrajNorm
from CurveModel import *
from utils import *
import matplotlib.pyplot as plt
import os

# Reproducibility
reproducibility_settings(seed=0)

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=8)
parser.add_argument('--pred_len', type=int, default=12)


def eval_method(args):
    data_set = 'datasets/' + args.dataset + '/'
    print('Scene: {}'.format(args.dataset))

    args.batch_size = 1e8
    loader_test = get_dataloader(data_set, 'train', args.obs_len, args.pred_len, args.batch_size)
    # loader_test = get_dataloader(data_set, 'val', args.obs_len, args.pred_len, args.batch_size)
    # loader_test = get_dataloader(data_set, 'test', args.obs_len, args.pred_len, args.batch_size)

    # Preprocessing
    obs_traj = loader_test.dataset.obs_traj
    pred_traj = loader_test.dataset.pred_traj

    n_ped, t_obs, dim = obs_traj.shape
    n_ped, t_pred, dim = pred_traj.shape

    # Normalization
    traj_norm = TrajNorm(ori=True, rot=True, sca=False)
    traj_norm.calculate_params(obs_traj)##计算obs_traj最后一帧的位置和最后两帧所形成的方向

    obs_traj_norm = traj_norm.normalize(obs_traj)
    pred_traj_norm = traj_norm.normalize(pred_traj)

    # Singular Value Decomposition
    print('===Singular Value Decomposition===')
    A = obs_traj_norm.reshape(n_ped, t_obs * dim).T
    B = pred_traj_norm.reshape(n_ped, t_pred * dim).T



    U_obs, S_obs, Vt_obs = torch.linalg.svd(A, full_matrices=False)
    U_pred, S_pred, Vt_pred = torch.linalg.svd(B, full_matrices=False)


    for k in range(6, 7):
        # Low-rank approximation
        U_obs_trunc = U_obs[:, :k]
        U_pred_trunc = U_pred[:, :k]

        if os.path.exists('Eigen_fig3')==False:##创建文件夹
            os.makedirs('Eigen_fig3')
            
        # Define a list of colors to cycle through
        colors = ['blue', 'orange','green','red', 'purple', 'brown']
            
        for i in range(k):## code of Figure 3
            xy_u_pred =U_pred_trunc[:, i].numpy().reshape(-1,2)##
            
            # Choose a color for this iteration (cycle through the colors list)
            color = colors[i % len(colors)]
            
            # X-Y
            plt.figure()
            plt.plot(xy_u_pred[:,0], xy_u_pred[:,1], label=f'k={i}', color=color)
            plt.xlim(-0.5, 0.5)
            plt.ylim(-0.5, 0.5)
            plt.legend()
            plt.savefig(f'Eigen_fig3/xy_u_pred_{i}.png')
            plt.close()
            
            # T-X
            plt.figure()
            plt.plot(xy_u_pred[:,0], label=f'k={i},x', color=color)
            plt.xlim(0, 12)
            plt.ylim(-0.5, 0.5)
            plt.legend()
            plt.savefig(f'Eigen_fig3/x_u_pred_{i}.png')
            plt.close()
            
            # T-Y
            plt.figure()
            plt.plot(xy_u_pred[:,1], label=f'k={i},y', color=color)
            plt.xlim(0, 12)
            plt.ylim(-0.5, 0.5)
            plt.legend()
            plt.savefig(f'Eigen_fig3/y_u_pred_{i}.png')
            plt.close()
            

            
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
    # dataset_all = ["eth", "hotel", "univ", "zara1", "zara2"]
    dataset_all = ["eth"]

    for scene in dataset_all:
        args.dataset = scene
        eval_method(args)
