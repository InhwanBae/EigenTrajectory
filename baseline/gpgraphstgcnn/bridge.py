import torch


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    v_abs = obs_data[None, :, :, None].detach()
    v_abs = v_abs.permute(0, 3, 1, 2)
    # v_rel = torch.zeros_like(v_abs)
    # v_rel[..., 1:, :] = v[..., 1:, :] - v[..., :-1, :]
    v_rel = v_abs  # Use the actual coefficients rather than velocity for ET.
    input_data = (v_abs, v_rel)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data[0].permute(0, 2, 3, 1).squeeze(dim=0)
    return pred_data
