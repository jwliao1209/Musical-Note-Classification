#!/bin/bash

# Step 1: Extract audio features
if [ ! -f "nsynth-subtrain/features.json" ]; then
    python extract_audio_features.py --data_dir nsynth-subtrain --output_path nsynth-subtrain/features.json
fi

if [ ! -f "nsynth-valid/features.json" ]; then
    python extract_audio_features.py --data_dir nsynth-valid --output_path nsynth-valid/features.json
fi

if [ ! -f "nsynth-test/features.json" ]; then
    python extract_audio_features.py --data_dir nsynth-test --output_path nsynth-test/features.json
fi


# Step 2: Prepare dataset
if [ ! -f "dataset/train.csv" ]; then
    python prepare_ml_dataset.py --data_path nsynth-subtrain/examples.json --feature_path nsynth-subtrain/features.json --save_path dataset/train.csv
fi

if [ ! -f "dataset/valid.csv" ]; then
    python prepare_ml_dataset.py --data_path nsynth-valid/examples.json --feature_path nsynth-valid/features.json --save_path dataset/valid.csv
fi

if [ ! -f "dataset/test.csv" ]; then
    python prepare_ml_dataset.py --data_path nsynth-test/examples.json --feature_path nsynth-test/features.json --save_path dataset/test.csv
fi


# Step 3: Train the machine learning model
python train_ml.py
