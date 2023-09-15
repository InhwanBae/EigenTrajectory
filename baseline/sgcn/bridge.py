import torch


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    v = obs_data[None, :, :, None].detach()
    i = [torch.eye(v.size(2), device=v.device).repeat(v.size(3), 1, 1),
         torch.eye(v.size(3), device=v.device).repeat(v.size(2), 1, 1)]
    input_data = (v, i)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data
    return pred_data
