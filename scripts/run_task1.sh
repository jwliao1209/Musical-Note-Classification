#!/bin/bash

python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/bass_acoustic_000-024-100.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/bass_electronic_000-022-127.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/bass_synthetic_000-024-100.wav

python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/flute_acoustic_000-055-100.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/flute_electronic_000-055-127.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/flute_synthetic_001-021-025.wav

python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/guitar_acoustic_000-021-025.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/guitar_electronic_000-024-025.wav
python visualize_mel_spectrogram.py --audio_path nsynth-subtrain/audio/guitar_synthetic_000-021-050.wav
