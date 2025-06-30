import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from EigenTrajectory import TrajNorm
from CurveModel import *
from utils import *


# Reproducibility
reproducibility_settings(seed=0)


def eval_method(args):
    data_set = './datasets/' + args.dataset + '/'
    print('Scene: {}'.format(args.dataset))

    args.batch_size = 1e8
    loader_train = get_dataloader(data_set, 'train', args.obs_len, args.pred_len, args.batch_size)

    # Preprocessing
    obs_traj = loader_train.dataset.obs_traj
    pred_traj = loader_train.dataset.pred_traj

    n_ped, t_obs, dim = obs_traj.shape
    n_ped, t_pred, dim = pred_traj.shape

    # Normalization
    traj_norm = TrajNorm(ori=True, rot=True, sca=False)
    traj_norm.calculate_params(obs_traj)

    obs_traj_norm = traj_norm.normalize(obs_traj)
    pred_traj_norm = traj_norm.normalize(pred_traj)

    # Singular Value Decomposition
    A = obs_traj_norm.reshape(n_ped, t_obs * dim).T
    B = pred_traj_norm.reshape(n_ped, t_pred * dim).T
    U_obs, S_obs, Vt_obs = torch.linalg.svd(A, full_matrices=False)
    U_pred, S_pred, Vt_pred = torch.linalg.svd(B, full_matrices=False)

    # Low-rank approximation
    k = 6  # You can change this value to test different ranks
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
    
    # Drawing the figures
    fig, axs = plt.subplots(3+1, k+1, figsize=((k+1)*2, (3+1)*2))
    for i in range(k):
        xy_u_pred = U_pred_trunc[:, i].numpy().reshape(-1, 2)

        # Choose a color for this iteration (cycle through the colors list)
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        rank_color = colors[i % len(colors)]

        # X-Y
        axs[0, i].plot(xy_u_pred[:, 0], xy_u_pred[:, 1], label=f'x,y', color=rank_color)
        axs[0, i].set_xlim(-0.5, 0.5)
        axs[0, i].set_ylim(-0.5, 0.5)
        axs[0, i].set_xticks([-0.5, 0, 0.5])
        axs[0, i].set_yticks([-0.5, 0, 0.5])
        axs[0, i].set_aspect('equal', adjustable='box')
        axs[0, i].set_box_aspect(1)

        # T-X
        axs[1, i].plot(xy_u_pred[:, 0], label=f'k={i},x', color=rank_color)
        axs[1, i].set_xlim(0, 12)
        axs[1, i].set_ylim(-0.5, 0.5)
        axs[1, i].set_xticks([0, 4, 8, 12])
        axs[1, i].set_yticks([-0.5, 0, 0.5])
        axs[1, i].set_box_aspect(1)

        # T-Y
        axs[2, i].plot(xy_u_pred[:, 1], label=f'k={i},y', color=rank_color)
        axs[2, i].set_xlim(0, 12)
        axs[2, i].set_ylim(-0.5, 0.5)
        axs[2, i].set_xticks([0, 4, 8, 12])
        axs[2, i].set_yticks([-0.5, 0, 0.5])
        axs[2, i].set_box_aspect(1)

        # Add k label
        axs[3, i].axis('off')
        axs[3, i].text(0.5, 1, rf"$u_{{{i+1}}}$", ha='center', va='center', fontsize=18)

    # Turn off corner subplot
    axs[3, k].axis('off')

    # 2) Rightmost column: row labels
    row_labels = ['X-Y', 'T-X', 'T-Y']
    for j, label in enumerate(row_labels):
        axs[j, k].axis('off')
        axs[j, k].text(0, 0.5, label, ha='center', va='center', rotation=90, fontsize=18)

    # Save the combined figure
    os.makedirs('output_vis', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'output_vis/EigenTrajectory_Figure_3_{args.dataset}.png')
    # plt.savefig(f'output_vis/EigenTrajectory_Figure_3_{args.dataset}.svg')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    args = parser.parse_args()

    dataset_all = ["eth", "hotel", "univ", "zara1", "zara2"]
    for scene in dataset_all:
        args.dataset = scene
        eval_method(args)
