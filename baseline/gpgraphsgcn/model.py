import torch.nn as nn
from .model_baseline import TrajectoryModel
from .model_groupwrapper import GPGraph


# class GPGraphSGCN(nn.Module):
#     def __init__(self, obs_len=8, pred_len=12, in_dims=2, out_dims=5):
#         super().__init__()
#         base_model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1,
#                                      dropout=0, obs_len=obs_len, pred_len=pred_len, n_tcn=5,
#                                      in_dims=in_dims, out_dims=out_dims)
#         self.model = GPGraph(baseline_model=base_model, in_channels=in_dims, out_channels=out_dims,
#                              obs_seq_len=obs_len, pred_seq_len=pred_len,
#                              d_type='learned_l2norm', d_th='learned', mix_type='mlp',
#                              group_type=(True, True, True), weight_share=True).cuda()
#
#     def forward(self, *args, **kwargs):
#         return self.model(*args, **kwargs)


def get_GPGraph_SGCN_model(obs_len=8, pred_len=12, in_dims=2, out_dims=5):
    base_model = TrajectoryModel(number_asymmetric_conv_layer=7, embedding_dims=64, number_gcn_layers=1,
                                 dropout=0, obs_len=obs_len, pred_len=pred_len, n_tcn=5,
                                 in_dims=in_dims, out_dims=out_dims)
    model = GPGraph(baseline_model=base_model, in_channels=in_dims, out_channels=out_dims,
                    obs_seq_len=obs_len, pred_seq_len=pred_len,
                    d_type='learned_l2norm', d_th='learned', mix_type='mlp',
                    group_type=(True, True, True), weight_share=True).cuda()
    return model
