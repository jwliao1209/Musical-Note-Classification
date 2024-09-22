#!/bin/bash

# Step 1: Prepare DL dataset
if [ ! -f "dataset/train.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-subtrain --save_path dataset/train.json
fi

if [ ! -f "dataset/valid.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-valid --save_path dataset/valid.json
fi

if [ ! -f "dataset/test.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-test --save_path dataset/test.json
fi

if [ ! -f "dataset/train_not_log_scale.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-subtrain --save_path dataset/train_not_log_scale.json --not_log_scale
fi

if [ ! -f "dataset/valid_not_log_scale.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-valid --save_path dataset/valid_not_log_scale.json --not_log_scale
fi

if [ ! -f "dataset/test_not_log_scale.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-test --save_path dataset/test_not_log_scale.json --not_log_scale
fi


# Step 2: Train DL model

# w/o log-scale
# python train_dl.py --train_data_path dataset/train_not_log_scale.json --valid_data_path dataset/valid_not_log_scale.json --test_data_path dataset/test_not_log_scale.json

# w/ log-scale
# python train_dl.py --train_data_path dataset/train.json --valid_data_path dataset/valid.json --test_data_path dataset/test.json

python inference.py
