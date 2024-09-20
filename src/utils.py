import json

import torch


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def flatten_features(feature_dict):
    flatten_feature_dict = {}
    for feature_name, values in feature_dict.items():
        if isinstance(values, list):
            for i in range(len(values)):
                flatten_feature_dict[f'{feature_name}_{i}'] = values[i]
        else:
            flatten_feature_dict[feature_name] = values
    return flatten_feature_dict


def dict_to_device(data: dict, device: torch.device) -> dict:
    return {k: v.to(device) if not isinstance(v, list) else v for k, v in data.items()}
