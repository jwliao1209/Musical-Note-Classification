import os
from argparse import ArgumentParser, Namespace

import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.constants import MEL_DIR
from src.audio_extractor import mel_spec_extractor


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Visualize a Mel-Spectrogram')
    parser.add_argument(
        '--audio_paths',
        type=str,
        default=['nsynth-subtrain/audio/bass_acoustic_000-024-100.wav'],
        nargs='+',
        
    )
    return parser.parse_args()


if __name__ == '__main__':
    os.makedirs(MEL_DIR, exist_ok=True)

    args = parse_arguments()
    
    for path in tqdm(args.audio_paths):
        # Extract the instrument and pitch from the filename
        filename = os.path.basename(path)
        instrument = filename.split('_')[0]
        pitch = filename.split('_')[2].split('-')[1]

        # Load the audio file and compute the Mel-Spectrogram
        S_dB, sr = mel_spec_extractor(path)

        # Plot the Mel-Spectrogram
        plt.figure(figsize=(12, 6), dpi=600)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000, vmin=-80, vmax=0)
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'{instrument.title()} - Pitch {pitch}', fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Hz', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(MEL_DIR, filename.replace('.wav', '.png')))
