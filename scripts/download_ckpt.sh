#!/bin/bash

# Download training data
if [ ! -d "checkpoints" ]; then
    gdown 1cJOg4v2r5w640x2fMHiZCqLJU-VMfpa2 -O checkpoints.zip
    unzip -n checkpoints.zip
    rm checkpoints.zip
fi
