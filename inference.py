import os
from argparse import ArgumentParser, Namespace

import torch
import wandb
from torch import nn
from tqdm import tqdm

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CKPT_FILE
from src.dataset import AudiosDataset
from src.evaluate import evaluator
from src.models import ShortChunkCNN, ShortChunkResCNN
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
        default='checkpoints/09-22-16-37-20',
    )
    return parser.parse_args()


if __name__ == "__main__":
    set_random_seeds()
    args = parse_arguments()
    test_data = read_json(args.test_data_path)

    transforms = get_transforms()
    test_loader = AudiosDataset(test_data, transforms).get_loader()

    # Prepare inference
    checkpoint = torch.load(os.path.join(args.ckpt_dir, CKPT_FILE), weights_only=True)
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = ShortChunkResCNN()
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
