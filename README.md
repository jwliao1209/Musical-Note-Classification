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
