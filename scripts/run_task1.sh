#!/bin/bash

python visualize_mel_spec.py --audio_paths \
        nsynth-subtrain/audio/bass_synthetic_001-009-075.wav \
        nsynth-subtrain/audio/bass_synthetic_116-054-050.wav \
        nsynth-subtrain/audio/bass_synthetic_019-109-025.wav \
        nsynth-subtrain/audio/flute_acoustic_027-021-100.wav \
        nsynth-subtrain/audio/flute_acoustic_003-064-127.wav \
        nsynth-subtrain/audio/flute_synthetic_001-108-127.wav \
        nsynth-subtrain/audio/guitar_electronic_034-009-050.wav \
        nsynth-subtrain/audio/guitar_acoustic_031-064-127.wav \
        nsynth-subtrain/audio/guitar_synthetic_012-120-075.wav
