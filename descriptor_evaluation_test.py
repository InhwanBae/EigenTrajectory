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
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        # cmap = plt.colormaps['tab20']
    else:
        # For more than 20 clusters, we'll use hsv which can generate more distinct colors
        cmap = plt.cm.get_cmap('hsv', num_clusters)
        # cmap = plt.colormaps['hsv']
    
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
                  loc='best', fontsize=10, framealpha=0.8)

    # Add a title and axis labels
    plt.title(f't-SNE Visualization of {num_clusters} Clusters for {args.dataset}')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    
    # Add a colorbar for additional visual reference
    # cbar = plt.colorbar(scatter, ticks=range(num_clusters))
    # cbar.set_label('Cluster ID')
    
    # Save the figure
    save_dir = f'tSNE_ets'
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/k_clusters_{args.dataset}.png', dpi=300)
    plt.close()

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
    # print('A.shape: {}'.format(A.shape))## (16, 181)
    # print('B.shape: {}'.format(B.shape))## (24, 181)
    '''
    for eth: 
    test: A.shape: torch.Size([16, 181])
    val: A.shape: torch.Size([16, 5349])
    train: A.shape: torch.Size([16, 29809])
    cat(train,val,dim=0): A.shape: torch.Size([16, 35158])

    '''

    U_obs, S_obs, Vt_obs = torch.linalg.svd(A, full_matrices=False)
    U_pred, S_pred, Vt_pred = torch.linalg.svd(B, full_matrices=False)
    # print('U_obs.shape: {}'.format(U_obs.shape))## (16, 16)##(L, r)
    # print('U_pred.shape: {}'.format(U_pred.shape))## (24, 24)
    # print('S_obs.shape: {}'.format(S_obs.shape))## (16,)(r,)
    # print('S_pred.shape: {}'.format(S_pred.shape))## (24,)
    # print('Vt_obs.shape: {}'.format(Vt_obs.shape))## (16, 181)(r, N)
    # print('Vt_pred.shape: {}'.format(Vt_pred.shape))## (24, 181)

    # for k in range(1, 13):
    for k in range(6, 7):
        # Low-rank approximation
        U_obs_trunc = U_obs[:, :k]
        U_pred_trunc = U_pred[:, :k]
        # print('U_obs_trunc.shape: {}'.format(U_obs_trunc.shape))## (16, 6)
        # print('U_pred_trunc.shape: {}'.format(U_pred_trunc.shape))## (24, 6)
        # print('U_obs_trunc[:, 0]@U_obs_trunc[:, 1]',U_obs_trunc[:, 0]@U_obs_trunc[:, 1])## ==0
        print(f'A.shape: {A.shape}')## (16, 181)
        if os.path.exists('Eigen_figs')==False:##创建文件夹
            os.makedirs('Eigen_figs')
            
        # # Define a list of colors to cycle through
        # colors = ['blue', 'orange','green','red', 'purple', 'brown']
            
        # for i in range(k):## code of Figure 3
        #     '''
        #     # print('B.shape: {}'.format(B.shape))## (24, 181)
        #     U_pred, S_pred, Vt_pred = torch.linalg.svd(B, full_matrices=False)
        #     # print('U_pred.shape: {}'.format(U_pred.shape))## (24, 24)
        #     U_pred_trunc = U_pred[:, :k] ## [L=24, r=6]
        #     '''
        #     xy_u_pred =U_pred_trunc[:, i].numpy().reshape(-1,2)##
        #     # xy_u_pred = U_obs_trunc[:, i].numpy().reshape(-1,2)##(12,2)
        #     # print(f'xy_u_pred.shape: {xy_u_pred.shape}')## (12, 2)
            
        #     # Choose a color for this iteration (cycle through the colors list)
        #     color = colors[i % len(colors)]
            
        #     # X-Y
        #     plt.figure()
        #     plt.plot(xy_u_pred[:,0], xy_u_pred[:,1], label=f'k={i}', color=color)
        #     plt.xlim(-0.5, 0.5)
        #     plt.ylim(-0.5, 0.5)
        #     plt.legend()
        #     plt.show()
        #     plt.savefig(f'Eigen_figs/xy_u_pred_{i}.png')
        #     plt.close()
            
        #     # T-X
        #     plt.figure()
        #     plt.plot(xy_u_pred[:,0], label=f'k={i},x', color=color)
        #     plt.xlim(0, 12)
        #     plt.ylim(-0.5, 0.5)
        #     plt.legend()
        #     plt.show()
        #     plt.savefig(f'Eigen_figs/x_u_pred_{i}.png')
        #     plt.close()
            
        #     # T-Y
        #     plt.figure()
        #     plt.plot(xy_u_pred[:,1], label=f'k={i},y', color=color)
        #     plt.xlim(0, 12)
        #     plt.ylim(-0.5, 0.5)
        #     plt.legend()
        #     plt.show()
        #     plt.savefig(f'Eigen_figs/y_u_pred_{i}.png')
        #     plt.close()
            

        ## U_obs_trunc shape: (16, 6)
        ## U_pred_trunc shape: (24, 6)
        # A shape: (16, 181)
        # B shape: (24, 181)
        C_obs = U_obs_trunc.T @ A
        C_pred = U_pred_trunc.T @ B
        # print('C_obs.shape: {}'.format(C_obs.shape))## (6, 181)
        # print('C_pred.shape: {}'.format(C_pred.shape))## (6, 181)
        c_traj = C_obs.T ##(181, 6)
        # print('c_traj.shape: {}'.format(c_traj.shape))## (181, 6)
        #plot c_traj
        plt.figure()
        for i in range(10):
            plt.plot(c_traj[i], label=f'k={i}')
        # plt.xlim(0, 12)
        # plt.ylim(-0.5, 0.5)
        plt.legend()
        plt.show()
        plt.savefig(f'Eigen_figs/c_traj.png')
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
        




        A_recon = U_obs_trunc @ C_obs
        B_recon = U_pred_trunc @ C_pred
        # print('A_recon.shape: {}'.format(A_recon.shape))## (16, 181)
        # print('B_recon.shape: {}'.format(B_recon.shape))## (24, 181)

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
