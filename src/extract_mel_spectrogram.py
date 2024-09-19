import librosa
import numpy as np

from src.constants import FFT_WINDOW_SIZE, HOP_LENGTH


def mel_spectrogram_extractor(audio_path, log_scale=True):
    y, sr = librosa.load(audio_path)
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000,
        n_fft=FFT_WINDOW_SIZE,
        hop_length=HOP_LENGTH
    )
    if log_scale:
        S = librosa.power_to_db(S, ref=np.max)
    return S, sr 
