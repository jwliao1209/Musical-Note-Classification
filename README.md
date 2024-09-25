# Musical-Note-Classification

This repository is implementation of Homework 0 for CommE5070 Deep Learning for Music Analysis and Generation course in 2024 Fall semester at National Taiwan University.



## Virtual Environment
```
virtualenv --python=python3.10 deepmir_hw0
source deepmir_hw0/bin/activate
pip install -r requirements.txt
```

## Download
- Dataset
```
bash scripts/download_data.sh
```
- Checkpoint
```
bash scripts/download_ckpt.sh
```


## Task 1: Visualize a Mel-Spectrogram
```
bash scripts/run_task1.sh
```

## Task 2: Train a Traditional Machine Learning Model
```
bash scripts/run_task2.sh
```

## Task 3: Train a Deep Learning Model
```
bash scripts/run_task3_train.sh
```
```
bash scripts/run_task3_infer.sh
```

## Operating System and Device
We implemented the code on an environment running Ubuntu 22.04.1, utilizing a 12th Generation Intel(R) Core(TM) i7-12700 CPU, along with a single NVIDIA GeForce RTX 4090 GPU equipped with 24 GB of dedicated memory.


## Citation
```bibtex
@misc{
    title  = {Music Note Classification},
    author = {Jia-Wei Liao},
    url    = {https://github.com/jwliao1209/Musical-Note-Classification},
    year   = {2023}
}
```
