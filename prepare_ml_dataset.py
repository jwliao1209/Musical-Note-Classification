import glob
import os
from argparse import ArgumentParser, Namespace

import pandas as pd
from tqdm import tqdm

from src.audio_extractor import AudioFeaturesExtractor
from src.utils import read_json, flatten_features


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Prepare ML dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='nsynth-subtrain'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='signal'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='dataset/train.csv'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    data_dict = read_json(os.path.join(args.data_dir, 'examples.json'))
    audio_paths = glob.glob(os.path.join(args.data_dir, 'audio', '*.wav'))

    extractor = AudioFeaturesExtractor(method=args.method)

    feature_dict = {}
    for audio_path in tqdm(audio_paths):
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        feature_dict[filename] = extractor(audio_path)

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
