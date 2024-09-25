import os
from argparse import ArgumentParser, Namespace

import torch
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch import nn
from tqdm import tqdm

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CKPT_FILE, KNOWN_LABELS, RESULT_DIR
from src.dataset import AudiosDataset
from src.evaluate import evaluator
from src.models import get_model
from src.trainer import Trainer
from src.transform import get_transforms
from src.utils import set_random_seeds, get_time, read_json, save_json, dict_to_device


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Inference DL model')
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='dataset/test.json',
    )
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='checkpoints/09-25-15-46-12',
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    test_data = read_json(args.test_data_path)

    config = read_json(os.path.join(args.ckpt_dir, 'config.json'))

    transforms = get_transforms()
    test_loader = AudiosDataset(test_data, transforms).get_loader()

    # Prepare inference
    checkpoint = torch.load(os.path.join(args.ckpt_dir, CKPT_FILE), weights_only=True)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = get_model(config['model'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()

    preds = []
    labels = []
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = dict_to_device(data, device)
            pred = model(data['mel_spec'])
            preds.append(pred)
            labels.append(data['label'])

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    results = evaluator(preds, labels)
    print(results)
    
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    log_scale = 'w/o log-scale' if 'not_log_scale' in args.test_data_path else 'w/ log-scale'
    

    # Plot the confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=KNOWN_LABELS, yticklabels=KNOWN_LABELS)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.title(f"{''.join([text.title() for text in config['model'].split('_')])} {log_scale} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR,
                f"{config['model']}_{'not_log_scale_' if 'not_log_scale' in args.test_data_path else ''}confusion_matrix.png"))
