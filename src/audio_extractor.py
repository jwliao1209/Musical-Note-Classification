import librosa
import numpy as np
import torch
from transformers import Wav2Vec2Processor, WavLMModel

from src.constants import FFT_WINDOW_SIZE, HOP_LENGTH
from src.utils import dict_to_device


def mel_spec_extractor(audio_path, log_scale=True):
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


class AudioSignalExtractor:
    def __init__(self):
        self.method = 'signal'

    def __call__(self, audio_path):
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


class BaseDLAudioExtractor:
    def __init__(self):
        self.method = None
        self.device = None
        self.processor = None
        self.model = None

    def __call__(self, audio_path):
        y, _ = librosa.load(audio_path)
        audio = self.processor(torch.as_tensor(y), sampling_rate=16000, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**dict_to_device(audio, self.device))
            hidden_states = outputs.last_hidden_state

        features = hidden_states.detach().squeeze().mean(dim=0).cpu().numpy()
        return {f'{self.method}_{i}': f for i, f in enumerate(features)}


class WavLMExtractor(BaseDLAudioExtractor):
    def __init__(self):
        super().__init__()
        self.method = 'wavlm'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained('patrickvonplaten/wavlm-libri-clean-100h-base-plus')
        self.model = WavLMModel.from_pretrained('microsoft/wavlm-base-plus').to(self.device)


class AudioFeaturesExtractor:
    def __init__(self, method):
        if method == 'signal':
            self.extractor = AudioSignalExtractor()
        elif method == 'wavlm':
            self.extractor = WavLMExtractor()

    def __call__(self, audio_path):
        return self.extractor(audio_path)
