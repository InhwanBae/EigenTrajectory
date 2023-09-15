from .model_baseline import social_stgcnn
from .model_groupwrapper import GPGraph


def get_GPGraph_STGCNN_model(obs_len=8, pred_len=12, in_dims=2, out_dims=5):
    base_model = social_stgcnn(n_stgcnn=1, n_txpcnn=5, input_feat=in_dims, output_feat=out_dims,
                               kernel_size=3, seq_len=obs_len, pred_seq_len=pred_len).cuda()
    model = GPGraph(baseline_model=base_model, in_channels=in_dims, out_channels=out_dims,
                    obs_seq_len=obs_len, pred_seq_len=pred_len,
                    d_type='learned_l2norm', d_th='learned', mix_type='mlp',
                    group_type=(True, True, True), weight_share=True).cuda()
    return model
