import glob
import os
from argparse import ArgumentParser, Namespace

import librosa
import numpy as np
from tqdm import tqdm

from src.utils import save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Extract audio features')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='nsynth-subtrain/audio'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='nsynth-subtrain/features.json'
    )
    return parser.parse_args()


def extract_features(audio_path):
    y, sr = librosa.load(audio_path)

    # MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral Contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)
    spectral_contrast_std = np.std(spectral_contrast, axis=1)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)

    # Tempo
    tempo = librosa.beat.beat_track(y=y, sr=sr)[0] if np.any(y) else 0
    tempo = tempo.item() if isinstance(tempo, np.ndarray) else tempo

    return {
        'mfcc_mean': mfccs_mean.tolist(),
        'mfcc_std': mfccs_std.tolist(),
        'chroma_mean': chroma_mean.tolist(),
        'chroma_std': chroma_std.tolist(),
        'spectral_contrast_mean': spectral_contrast_mean.tolist(),
        'spectral_contrast_std': spectral_contrast_std.tolist(),
        'zcr_mean': float(zcr_mean),
        'zcr_std': float(zcr_std),
        'spectral_bandwidth_mean': float(spectral_bandwidth_mean),
        'spectral_bandwidth_std': float(spectral_bandwidth_std),
        'rms_mean': float(rms_mean),
        'rms_std': float(rms_std),
        'tempo': tempo
    }


if __name__ == '__main__':
    args = parse_arguments()
    audio_paths = glob.glob(os.path.join(args.data_dir, '*.wav'))

    feature_dict = {}
    for audio_path in tqdm(audio_paths):
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        feature_dict[filename] = extract_features(audio_path)
    save_json(feature_dict, args.output_path)
