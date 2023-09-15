import torch


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    S_obs_abs = obs_data[None, :, :, None].detach()
    S_obs_rel = torch.zeros_like(S_obs_abs)
    S_obs_rel[:, 1:] = S_obs_abs[:, 1:] - S_obs_abs[:, :-1]

    S_obs = torch.stack([S_obs_abs, S_obs_rel], dim=1)  # concat loc_pos
    input_data = (S_obs,)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data.squeeze(dim=0)
    return pred_data
