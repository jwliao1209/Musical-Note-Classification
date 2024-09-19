#!/bin/bash

# Download training data
if [ ! -d "nsynth-subtrain" ]; then
    gdown 1wwNGbMD39_We9vqljmQa_fouK3GGU6Rk -O nsynth-subtrain.zip
    unzip -n nsynth-subtrain.zip
    rm nsynth-subtrain.zip
fi

# Download validation dataset
if [ ! -d "nsynth-valid" ]; then
    wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
    tar -xzvf nsynth-valid.jsonwav.tar.gz
    rm nsynth-valid.jsonwav.tar.gz
fi

# Download test dataset
if [ ! -d "nsynth-test" ]; then
    wget http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz
    tar -xzvf nsynth-test.jsonwav.tar.gz
    rm nsynth-test.jsonwav.tar.gz
fi
