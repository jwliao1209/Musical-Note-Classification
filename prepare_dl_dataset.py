import glob
import os
from argparse import ArgumentParser, Namespace

import numpy as np
from tqdm import tqdm

from src.constants import MEL_SPEC
from src.extract_mel_spectrogram import mel_spectrogram_extractor
from src.utils import read_json, save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Prepare DL dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='nsynth-subtrain'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='nsynth-subtrain/examples.json'
    )
    parser.add_argument(
        "--not_log_scale",
        action="store_true",
    )
    parser.add_argument(
        "--save_path",
        default='dataset/train.json',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    MEL_SPEC = f"{MEL_SPEC}_not_log" if args.not_log_scale else MEL_SPEC
    os.makedirs(os.path.join(args.data_dir, MEL_SPEC), exist_ok=True)
    audio_paths = glob.glob(os.path.join(args.data_dir, 'audio', '*.wav'))
    data_dict = read_json(args.data_path)

    data_list = []
    for audio_path in tqdm(audio_paths):
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        S, _ = mel_spectrogram_extractor(audio_path, log_scale=not args.not_log_scale)

        save_path = os.path.join(args.data_dir, MEL_SPEC, f"{filename}.npy")
        np.save(save_path, S)

        data_list.append(
            {
                "filename": filename,
                "mel_spec": save_path,
                "label": data_dict[filename]['instrument_family']
            }
        )

    save_json(data_list, args.save_path)
