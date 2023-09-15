# Baseline model for "Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction"
# Source-code referred from GP-Graph at https://github.com/InhwanBae/GPGraph/blob/main/model_baseline.py

import torch
import torch.nn as nn


class GroupGenerator(nn.Module):
    def __init__(self, d_type='learned', th=1., in_channels=16, hid_channels=32, n_head=1, dropout=0):
        super().__init__()
        self.d_type = d_type
        if d_type == 'learned':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, 1),
                                           nn.ReLU(),
                                           nn.BatchNorm2d(hid_channels),
                                           nn.Dropout(dropout, inplace=True),
                                           nn.Conv2d(hid_channels, n_head, 1),)
        elif d_type == 'estimate_th':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, n_head, 1),)
        elif d_type == 'learned_l2norm':
            self.group_cnn = nn.Sequential(nn.Conv2d(in_channels, hid_channels, kernel_size=(3, 1), padding=(1, 0)))
        self.th = th if type(th) == float else nn.Parameter(torch.Tensor([1]))

    def find_group_indices(self, v, dist_mat):
        n_ped = v.size(-1)
        mask = torch.ones_like(dist_mat).mul(1e4).triu()
        top_row, top_column = torch.nonzero(dist_mat.tril(diagonal=-1).add(mask).le(self.th), as_tuple=True)
        indices_raw = torch.arange(n_ped, dtype=top_row.dtype, device=v.device)
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_row.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
        return indices

    def find_group_indices_ratio(self, v, dist_mat):
        n_ped = v.size(-1)
        group_num = n_ped - (n_ped + self.th - 1) // self.th
        top_list = (1. / dist_mat).tril(diagonal=-1).view(-1).topk(k=group_num)[1]
        top_row, top_column = top_list // n_ped, top_list % n_ped
        indices_raw = torch.arange(n_ped, dtype=top_list.dtype, device=v.device)
        for r, c in zip(top_row, top_column):
            mask = indices_raw == indices_raw[r]
            indices_raw[mask] = c
        indices_uniq = indices_raw.unique()
        indices_map = torch.arange(indices_uniq.size(0), dtype=top_list.dtype, device=v.device)
        indices = torch.zeros_like(indices_raw)
        for i, j in zip(indices_uniq, indices_map):
            indices[indices_raw == i] = j
        return indices

    def group_backprop_trick_threshold(self, v, dist_mat, tau=1, hard=False):
        """
        The main trick for hard is to do (v_hard - v_soft).detach() + v_soft
        Sample hard categorical using "Straight-through" trick
        It achieves two things: - makes the output value exactly one-hot (since we add then subtract v_soft value)
                                - makes the gradient equal to v_soft gradient (since we strip all other gradients)
        """
        sig = (-(dist_mat - self.th) / tau).sigmoid()
        sig_norm = sig / sig.sum(dim=0, keepdim=True)
        v_soft = v @ sig_norm
        return (v - v_soft).detach() + v_soft if hard else v_soft

    def forward(self, v, v_abs, tau=0.1, hard=True):
        assert v.size(0) == 1
        n_ped = v.size(-1)

        # Measure similarity between pedestrian pairs
        if self.d_type == 'euclidean':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned_l2norm':
            temp = self.group_cnn(v_abs).unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            dist_mat = (temp - temp.transpose(-2, -1)).norm(p=2, dim=1)
        elif self.d_type == 'learned':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-1, -2)).reshape(temp.size(0), -1, n_ped, n_ped)
            temp = self.group_cnn(temp).exp()
            dist_mat = torch.stack([temp, temp.transpose(-1, -2)], dim=-1).mean(dim=-1)  # symmetric
        elif self.d_type == 'estimate_th':
            temp = v_abs.unsqueeze(dim=-1).repeat_interleave(repeats=n_ped, dim=-1)
            temp = (temp - temp.transpose(-2, -1))
            dist_mat = temp.norm(p=2, dim=1)
            self.th = self.group_cnn(temp.reshape(temp.size(0), -1, n_ped, n_ped)).mean().exp()
        else:
            raise NotImplementedError

        dist_mat = dist_mat.squeeze(dim=0).mean(dim=0)
        indices = self.find_group_indices(v, dist_mat)
        v = self.group_backprop_trick_threshold(v, dist_mat, tau=tau, hard=hard)
        return v, indices

    @staticmethod
    def ped_group_pool(v, indices):
        assert v.size(-1) == indices.size(0)
        n_ped = v.size(-1)
        n_ped_pool = indices.unique().size(0)
        v_pool = torch.zeros(v.shape[:-1] + (n_ped_pool,), device=v.device)
        v_pool.index_add_(-1, indices, v)
        v_pool_num = torch.zeros((v.size(0), 1, 1, n_ped_pool), device=v.device)
        v_pool_num.index_add_(-1, indices, torch.ones((v.size(0), 1, 1, n_ped), device=v.device))
        v_pool /= v_pool_num
        return v_pool

    @staticmethod
    def ped_group_unpool(v, indices):
        assert v.size(-1) == indices.unique().size(0)
        return torch.index_select(input=v, dim=-1, index=indices)

    @staticmethod
    def ped_group_mask(indices):
        mask = torch.eye(indices.size(0), dtype=torch.bool, device=indices.device)
        for i in indices.unique():
            idx_list = torch.nonzero(indices.eq(i))
            for idx in idx_list:
                mask[idx, idx_list] = 1
        return mask


class GroupIntegrator(nn.Module):
    def __init__(self, mix_type='mean', n_mix=3, out_channels=5, pred_seq_len=12):
        super().__init__()
        self.mix_type = mix_type
        self.pred_seq_len = pred_seq_len
        if mix_type == 'mlp':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * pred_seq_len * n_mix, out_channels * pred_seq_len,
                                                       kernel_size=1), )
        elif mix_type == 'cnn':
            self.st_gcns_mix = nn.Sequential(nn.PReLU(),
                                             nn.Conv2d(out_channels * n_mix, out_channels,
                                                       kernel_size=(3, 1), padding=(1, 0)))

    def forward(self, v_stack):
        n_batch, n_ped = v_stack[0].shape[0], v_stack[0].shape[3]
        if self.mix_type == 'sum':
            v = torch.stack(v_stack, dim=0).sum(dim=0)
        elif self.mix_type == 'mean':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
        elif self.mix_type == 'mlp':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v_stack = torch.cat(v_stack, dim=1).reshape(n_batch, -1, 1, n_ped)
            v = v + self.st_gcns_mix(v_stack).view(n_batch, -1, self.pred_seq_len, n_ped)
        elif self.mix_type == 'cnn':
            v = torch.stack(v_stack, dim=0).mean(dim=0)
            v = v + self.st_gcns_mix(torch.cat(v_stack, dim=1))
        else:
            raise NotImplementedError
        return v


def generate_adjacency_matrix(v, mask=None):
    # return adjacency matrix for Social-STGCNN baseline
    n_ped = v.size(-1)
    temp = v.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    a = (temp - temp.transpose(2, 3)).norm(p=2, dim=-1)
    # inverse kernel
    a_inv = 1. / a
    a_inv[a == 0] = 0
    # masking
    a_inv = a_inv if mask is None else a_inv * mask
    # normalize
    a_hat = a_inv + torch.eye(n=n_ped, device=v.device)
    node_degrees = a_hat.sum(dim=-1).unsqueeze(dim=-1)
    degs_inv_sqrt = torch.pow(node_degrees, -0.5)
    degs_inv_sqrt[torch.isinf(degs_inv_sqrt)] = 0
    norm_degs_matrix = torch.eye(n=n_ped, device=v.device) * degs_inv_sqrt
    return torch.eye(n=n_ped, device=v.device) - norm_degs_matrix @ a_hat @ norm_degs_matrix


def generate_identity_matrix(v):
    # return spatial and temporal identity matrix for SGCN baseline
    i = [torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1),
         torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1)]
    return i


class GPGraph(nn.Module):
    def __init__(self, baseline_model, in_channels=2, out_channels=5, obs_seq_len=8, pred_seq_len=12,
                 d_type='learned_l2norm', d_th='learned', mix_type='mlp', group_type=None, weight_share=True):
        super().__init__()

        self.baseline_model = baseline_model
        self.obs_seq_len = obs_seq_len
        self.pred_seq_len = pred_seq_len
        self.mix_type = mix_type
        self.weight_share = weight_share

        group_type = (True,) * 3 if group_type is None else group_type
        self.include_original = group_type[0]
        self.include_inter_group = group_type[1]
        self.include_intra_group = group_type[2]

        self.group_gen = GroupGenerator(d_type=d_type, th=d_th, in_channels=in_channels, hid_channels=8)
        self.group_mix = GroupIntegrator(mix_type=mix_type, n_mix=sum(group_type),
                                         out_channels=out_channels, pred_seq_len=pred_seq_len)

    def forward(self, v_abs, v_rel):
        v_stack = []

        # Pedestrian graph
        if self.include_original:
            # Agent-agent interaction
            v = v_rel
            i = generate_identity_matrix(v)
            v = v.permute(0, 2, 3, 1)
            v = self.baseline_model(v, i) if self.weight_share else self.baseline_model[0](v, i)
            v = v.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v)

        # Intra-/Inter-group graph
        v_rel, indices = self.group_gen(v_rel, v_abs, hard=True)

        if self.include_inter_group:
            # Inter-group interaction
            v_e = self.group_gen.ped_group_pool(v_rel, indices)  # Pedestrian Group Pooling
            i_e = generate_identity_matrix(v_e)
            v_e = v_e.permute(0, 2, 3, 1)
            v_e = self.baseline_model(v_e, i_e) if self.weight_share else self.baseline_model[1](v_e, i_e)
            v_e = v_e.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_e = self.group_gen.ped_group_unpool(v_e, indices)  # Pedestrian Group Unpooling
            v_stack.append(v_e)

        if self.include_intra_group:
            # Intra-group interaction
            v_i = v_rel
            mask = self.group_gen.ped_group_mask(indices)
            i_i = generate_identity_matrix(v_i)
            v_i = v_i.permute(0, 2, 3, 1)
            v_i = self.baseline_model(v_i, i_i, mask) if self.weight_share else self.baseline_model[2](v_i, i_i, mask)
            v_i = v_i.unsqueeze(dim=0).permute(0, 3, 1, 2)
            v_stack.append(v_i)

        # Group Integration
        v = self.group_mix(v_stack)

        return v, indices
