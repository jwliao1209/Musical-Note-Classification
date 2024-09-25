# Musical-Note-Classification

This repository contains the implementation for Homework 0 of the CommE5070 Deep Learning for Music Analysis and Generation course, Fall 2024, at National Taiwan University. For a detailed report, please refer to this [slides](https://docs.google.com/presentation/d/1qzZkeOhSakKE9NnlswTEY2wXcdWsOeuAOOa0cAPT3Kg/edit?usp=sharing).


## Setup
To set up the virtual environment and install the required packages, use the following commands:
```
virtualenv --python=python3.10 deepmir_hw0
source deepmir_hw0/bin/activate
pip install -r requirements.txt
```

## Data and Checkpoint Download

### Dataset
To download the dataset, run the following script:
```
bash scripts/download_data.sh
```

### Checkpoint
To download the pre-trained model checkpoints, use the command:
```
bash scripts/download_ckpt.sh
```

## Tasks

### Task 1: Visualize a Mel-Spectrogram
To generate and visualize a Mel-Spectrogram:
```
bash scripts/run_task1.sh
```

### Task 2: Train a Traditional Machine Learning Model
To train a traditional machine learning model:
```
bash scripts/run_task2.sh
```

### Task 3: Train a Deep Learning Model
- To train a deep learning model:
```
bash scripts/run_task3_train.sh
```
- To run inference with the trained model:
```
bash scripts/run_task3_infer.sh
```

### Reproducing Task3 Result
- If you want to reproduce the inference result, you can run:
```
bash scripts/run_task3_reproduce.sh
```


## Environment
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
If you use this code, please cite the following:
```bibtex
@misc{musical_note_classification_2024,
    title  = {Music Note Classification},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Musical-Note-Classification},
    year   = {2024}
}
```
