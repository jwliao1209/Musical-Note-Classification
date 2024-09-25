#!/bin/bash

# Step 1: Download test dataset
if [ ! -d "nsynth-test" ]; then
    wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
    tar -xzvf nsynth-test.jsonwav.tar.gz
    rm nsynth-test.jsonwav.tar.gz
fi

# Step 2: Download checkpoints
if [ ! -d "checkpoints" ]; then
    gdown 1cJOg4v2r5w640x2fMHiZCqLJU-VMfpa2 -O checkpoints.zip
    unzip -n checkpoints.zip
    rm checkpoints.zip
fi

# Step 3: Prepare DL dataset
if [ ! -f "dataset/test.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-test --save_path dataset/test.json
fi

if [ ! -f "dataset/test_not_log_scale.json" ]; then
    python prepare_dl_dataset.py --data_dir nsynth-test --save_path dataset/test_not_log_scale.json --not_log_scale
fi

# Step 4: Inference DL model
python infer_dl.py --test_data_path dataset/test.json --ckpt_dir checkpoints/09-25-15-46-12
python infer_dl.py --test_data_path dataset/test_not_log_scale.json --ckpt_dir checkpoints/09-25-15-50-18

python infer_dl.py --test_data_path dataset/test.json --ckpt_dir checkpoints/09-25-15-54-22
python infer_dl.py --test_data_path dataset/test_not_log_scale.json --ckpt_dir checkpoints/09-25-15-59-08
