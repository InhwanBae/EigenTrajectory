def model_forward_pre_hook(obs_data, obs_ori, addl_info):
    # Pre-process input data for the baseline model
    input_data = (obs_data.T, obs_ori.T, addl_info["scene_mask"], obs_ori.T)
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model.predict(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info):
    # Post-process output data of the baseline model
    b, nk = output_data.shape
    pred_data = output_data.view(b, nk // addl_info["num_samples"], addl_info["num_samples"]).permute(1, 0, 2)
    return pred_data
