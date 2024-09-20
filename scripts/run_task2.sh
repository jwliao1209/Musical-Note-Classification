#!/bin/bash

# Step 1: Prepare ML dataset
if [ ! -f "dataset/train.csv" ]; then
    python prepare_ml_dataset.py --data_dir nsynth-subtrain --save_path dataset/train.csv
fi

if [ ! -f "dataset/valid.csv" ]; then
    python prepare_ml_dataset.py --data_dir nsynth-valid --save_path dataset/valid.csv
fi

if [ ! -f "dataset/test.csv" ]; then
    python prepare_ml_dataset.py --data_dir nsynth-test --save_path dataset/test.csv
fi


# Step 2: Train ML model
python train_ml.py
