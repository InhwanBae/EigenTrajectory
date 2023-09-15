import torch


def generate_adjacency_matrix(v, mask=None):
    # return adjacency matrix for DMRGCN baseline
    n_ped = v.size(-1)
    v_abs = v
    v_rel = torch.zeros_like(v)
    v_rel[..., 1:, :] = v[..., 1:, :] - v[..., :-1, :]
    temp_abs = v_abs.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    temp_rel = v_rel.permute(0, 2, 3, 1).unsqueeze(dim=-2).repeat_interleave(repeats=n_ped, dim=-2)
    A_dist = (temp_abs - temp_abs.transpose(2, 3)).norm(p=2, dim=-1)
    A_disp = (temp_rel - temp_rel.transpose(2, 3)).norm(p=2, dim=-1)

    # masking
    A_dist = A_dist if mask is None else A_dist * mask
    A_disp = A_disp if mask is None else A_disp * mask
    return torch.stack([A_disp, A_dist], dim=1)


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    v = obs_data[None, :, :, None].detach()
    v = v.permute(0, 3, 1, 2)
    a = generate_adjacency_matrix(v).detach()
    input_data = (v, a)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data[0].permute(0, 2, 3, 1).squeeze(dim=0)
    return pred_data
