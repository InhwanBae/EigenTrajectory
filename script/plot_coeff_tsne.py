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


def tsne_plot(S, args, labels, num_clusters):
    # Perform t-SNE
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import matplotlib.colors as mcolors
    
    tsne = TSNE(n_components=2, random_state=42)
    S_tsne = tsne.fit_transform(S.detach().numpy())
    
    # Create a scatter plot with a more distinct colormap
    plt.figure(figsize=(12, 10))
    
    # Use tab20 colormap for more distinct colors (up to 20 clusters)
    if num_clusters <= 20:
        cmap = plt.get_cmap('tab20', num_clusters)
    else:
        # For more than 20 clusters, we'll use hsv which can generate more distinct colors
        cmap = plt.get_cmap('hsv', num_clusters)
    
    scatter = plt.scatter(S_tsne[:, 0], S_tsne[:, 1], c=labels, cmap=cmap, 
                         marker='o', s=8, alpha=0.7, edgecolors='none')
    
    # Add a legend with more distinct appearance
    # If too many clusters, limit the legend or split it into columns
    if num_clusters > 10:
        # Create two columns of legend entries
        ncol = 2
        legend_elements = []
        for i in range(num_clusters):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=cmap(i/num_clusters), 
                                   markersize=10, markeredgecolor='black', 
                                   label=f'Cluster {i}'))
        
        plt.legend(handles=legend_elements, title="Clusters", 
                   loc='upper right', bbox_to_anchor=(1.15, 1), 
                   ncol=ncol, fontsize=9, framealpha=0.8)
    else:
        # For fewer clusters, use a simpler legend
        legend_elements = []
        for i in range(num_clusters):
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=cmap(i/num_clusters), 
                                   markersize=12, markeredgecolor='black',
                                   label=f'Cluster {i}'))
        
        plt.legend(handles=legend_elements, title="Clusters", 
                   loc='upper right', fontsize=10, framealpha=0.8)

    # Add a title and axis labels
    plt.title(f't-SNE Visualization of {num_clusters} Clusters for {args.dataset}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Save the figure
    os.makedirs('output_vis', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'output_vis/EigenTrajectory_k_clusters_tSNE_{args.dataset}.png', dpi=300)
    # plt.savefig(f'output_vis/EigenTrajectory_k_clusters_tSNE_{args.dataset}.svg')
    plt.close()


def eval_method(args):
    data_set = './datasets/' + args.dataset + '/'
    print('Scene: {}'.format(args.dataset))

    args.batch_size = 1e8
    loader_train = get_dataloader(data_set, 'train', args.obs_len, args.pred_len, args.batch_size)
    # loader_val = get_dataloader(data_set, 'val', args.obs_len, args.pred_len, args.batch_size)
    # loader_test = get_dataloader(data_set, 'test', args.obs_len, args.pred_len, args.batch_size)

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
    A = obs_traj_norm.reshape(n_ped, t_obs * dim).T  # shape: (16, 181)
    B = pred_traj_norm.reshape(n_ped, t_pred * dim).T  # shape: (24, 181)
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

    c_traj = C_obs.T

    os.makedirs('output_vis', exist_ok=True)
    plt.figure()
    for i in range(10):
        plt.plot(c_traj[i], label=f'Ped id {i}')
    plt.title(f'EigenTrajectory Coefficients for {args.dataset}')
    plt.xlabel('Coefficient Index')
    plt.ylabel('Coefficient Value')
    plt.legend()
    plt.savefig(f'output_vis/EigenTrajectory_c_traj_{args.dataset}.png')
    # plt.savefig(f'output_vis/EigenTrajectory_c_traj_{args.dataset}.svg')
    plt.close()

    from sklearn.cluster import KMeans
    # Anchor generation on EigenTrajectory space
    kmeans = KMeans(n_clusters=20, random_state=0, init='k-means++', n_init=10)
    kmeans.fit(c_traj)
    labels = kmeans.labels_
    num_clusters = len(set(labels))
    print(f'Number of clusters: {num_clusters}')
    print(f'Labels: {labels}')
    # Plot the clusters
    tsne_plot(c_traj, args, labels, num_clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=8)
    parser.add_argument('--pred_len', type=int, default=12)
    args = parser.parse_args()

    dataset_all = ["eth", "hotel", "univ", "zara1", "zara2"]
    for scene in dataset_all:
        args.dataset = scene
        eval_method(args)
