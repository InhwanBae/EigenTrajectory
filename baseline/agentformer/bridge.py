from collections import defaultdict
import torch


def model_forward_pre_hook(obs_data, obs_ori, addl_info=None):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)

    input_data = defaultdict(lambda: None)
    input_data["pre_motion"] = obs_data.unsqueeze(dim=-1).contiguous()
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    baseline_model.set_data(input_data)
    baseline_model()
    output_data = baseline_model.data
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data['_dec_motion'].permute(1, 0, 2)
    return pred_data
