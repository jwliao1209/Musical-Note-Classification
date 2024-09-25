import os
from argparse import ArgumentParser, Namespace

import torch
import wandb
from torch import nn

from src.constants import PROJECT_NAME, CHECKPOINT_DIR, CONFIG_FILE
from src.dataset import AudiosDataset
from src.models import get_model
from src.optimization import get_lr_scheduler
from src.trainer import Trainer
from src.transform import get_transforms
from src.utils import set_random_seeds, get_time, read_json, save_json


def parse_arguments() -> Namespace:
    parser = ArgumentParser(description='Train DL model')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='dataset/train.json',
    )
    parser.add_argument(
        '--valid_data_path',
        type=str,
        default='dataset/valid.json',
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='dataset/test.json',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='short_chunk_cnn',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='one_cycle',
    )
    return parser.parse_args()


if __name__ == '__main__':
    set_random_seeds()
    args = parse_arguments()
    checkpoint_dir = os.path.join(CHECKPOINT_DIR, get_time())
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_json(vars(args), os.path.join(checkpoint_dir, CONFIG_FILE))

    train_data = read_json(args.train_data_path)
    valid_data = read_json(args.valid_data_path)
    test_data = read_json(args.test_data_path)

    transforms = get_transforms()
    train_loader = AudiosDataset(train_data, transforms).get_loader(args.batch_size, True, 4)
    valid_loader = AudiosDataset(valid_data, transforms).get_loader(args.batch_size, False, 4)
    test_loader = AudiosDataset(test_data, transforms).get_loader(args.batch_size, False, 4)

    # Prepare training
    device = torch.device(f'cuda:0'if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    lr_scheduler = get_lr_scheduler(
        name=args.lr_scheduler,
        optimizer=optimizer,
        max_lr=args.lr,
        steps_for_one_epoch=len(train_loader),
        epochs=args.epochs,
    )

    # Prepare logger
    wandb.init(
        project=PROJECT_NAME,
        name=os.path.basename(checkpoint_dir),
        config=vars(args),
    )
    wandb.watch(model, log='all')

    # Start training
    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        valid_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accum_grad_step=1,
        clip_grad_norm=1.0,
        logger=wandb,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.fit(epochs=args.epochs)
