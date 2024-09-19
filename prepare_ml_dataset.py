import os
from argparse import ArgumentParser, Namespace

import pandas as pd
from tqdm import tqdm

from src.utils import read_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Prepare ML dataset')
    parser.add_argument(
        '--data_path',
        type=str,
        default='nsynth-subtrain/examples.json'
    )
    parser.add_argument(
        '--feature_path',
        type=str,
        default='nsynth-subtrain/features.json'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='dataset/train.csv'
    )
    return parser.parse_args()


def flatten_features(feature_dict):
    flatten_feature_dict = {}
    for feature_name, values in feature_dict.items():
        if isinstance(values, list):
            for i in range(len(values)):
                flatten_feature_dict[f'{feature_name}_{i}'] = values[i]
        else:
            flatten_feature_dict[feature_name] = values
    return flatten_feature_dict


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    data_dict = read_json(args.data_path)
    feature_dict = read_json(args.feature_path)

    data_list = []
    for filename, info in tqdm(data_dict.items()):
        data_list.append(
            {
                'filename': filename,
                **flatten_features(feature_dict[filename]),
                'label': info['instrument_family'],
            }
        )
    pd.DataFrame(data_list).to_csv(args.save_path, index=False)
