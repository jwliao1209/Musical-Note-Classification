import os
from argparse import ArgumentParser, Namespace

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from src.constants import MEL_DIR, FFT_WINDOW_SIZE, HOP_LENGTH


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Visualize a Mel-Spectrogram')
    parser.add_argument(
        '--audio_path',
        type=str,
        default='nsynth-subtrain/audio/bass_acoustic_000-024-100.wav'
    )
    return parser.parse_args()


if __name__ == '__main__':
    os.makedirs(MEL_DIR, exist_ok=True)

    args = parse_arguments()
    filename = os.path.basename(args.audio_path)
    instrument, pitch = filename.split('_')[:2]

    # Load the audio file and compute the Mel-Spectrogram
    y, sr = librosa.load(args.audio_path)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000,
        n_fft=FFT_WINDOW_SIZE,
        hop_length=HOP_LENGTH
    )
    # Convert to decibels
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the Mel-Spectrogram
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'{instrument} - {pitch}')
    plt.tight_layout()
    plt.savefig(os.path.join(MEL_DIR, filename.replace('.wav', '.png')))

