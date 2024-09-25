import os
from typing import Dict, Optional

import torch
from torch import nn
from torch.cuda.amp.grad_scaler import GradScaler
from tqdm import tqdm

from src.constants import CKPT_FILE
from src.evaluate import evaluator
from src.utils import dict_to_device


class Trainer:
    def __init__(
        self,
        model,
        device,
        train_loader,
        valid_loader,
        criterion=None,
        optimizer=None,
        lr_scheduler=None,
        logger=None,
        accum_grad_step: int = 1,
        clip_grad_norm: Optional[float] = 1.0,
        fp32: bool = False,
        disable_valid_on_start: bool = False,
        checkpoint_dir: str = None,
    ) -> None:

        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.accum_grad_step = accum_grad_step
        self.clip_grad_norm = clip_grad_norm
        self.lr_scheduler = lr_scheduler
        self.fp32 = fp32
        self.grad_scaler = GradScaler(enabled=not fp32)
        self.logger = logger
        self.disable_valid_on_start = disable_valid_on_start
        self.cur_ep = 0
        self.checkpoint_dir = checkpoint_dir
        self.best_score = 0.0
        print(self)

    def __repr__(self) -> str:
        tab = " " * 2
        return (
            f"{self.__class__.__name__}(\n"
            f" model={self.model},\n"
            f" device={self.device},\n"
            f" train_num={len(self.train_loader)},\n"
            f" valid_num={len(self.valid_loader)},\n"
            f" optimizer={self.optimizer},\n"
            f" train_batch_size={self.train_loader.batch_size},\n"
            f" accum_grad_step={self.accum_grad_step},\n"
            f" clip_grad_norm={self.clip_grad_norm},\n"
            f" lr_scheduler={self.lr_scheduler},\n"
        ).replace("\n", f"\n{tab}").replace(f'\n{tab}P', f'\n{tab}{tab}P') + "\n)"

    def train_step(self, batch_data) -> Dict[str, torch.Tensor]:
        outputs = self.model(batch_data['mel_spec'])
        loss = self.criterion(outputs, batch_data['label'])
        return {'loss': loss, 'lr': self.lr_scheduler.get_last_lr()[0]}

    def valid_step(self, batch_data) -> Dict[str, torch.Tensor]:
        pred = self.model(batch_data['mel_spec'])
        loss = self.criterion(pred, batch_data['label'])
        return {'loss': loss, 'pred': pred, 'label': batch_data['label']}

    def train_one_epoch(self) -> None:
        progress_bar = tqdm(self.train_loader, desc=f"Training {self.cur_ep}")
        self.model.train()

        for step, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)

            with torch.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" \
                    and not self.fp32 else torch.float32,
                device_type=self.device.type
            ):
                output = self.train_step(batch_data)

            self.grad_scaler.scale(output['loss'] / self.accum_grad_step).backward()

            if step % self.accum_grad_step == 0:
                self.grad_scaler.unscale_(self.optimizer)

                if self.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                self.grad_scaler.step(optimizer=self.optimizer)
                self.grad_scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()

            record = {
                f"train_{k}": v.item() if isinstance(v, torch.Tensor) else v
                for k, v in output.items()
            }
            progress_bar.set_postfix(record)
            self.log(record)
        progress_bar.close()

    @torch.no_grad()
    def valid_one_epoch(self)-> None:
        progress_bar = tqdm(self.valid_loader, desc=f"Validation {self.cur_ep}")
        self.model.eval()

        outputs = []
        for _, batch_data in enumerate(progress_bar, start=1):
            batch_data = dict_to_device(batch_data, self.device)
            with torch.amp.autocast(
                dtype=torch.bfloat16 if self.device.type == "cuda" \
                    and not self.fp32 else torch.float32,
                device_type=self.device.type
            ):
                output = self.valid_step(batch_data)
            outputs.append(output)
        progress_bar.close()

        preds = torch.cat([o['pred'] for o in outputs])
        labels = torch.cat([o['label'] for o in outputs])
        result = evaluator(preds, labels)
        record = {f"valid_{k}": round(v, 4) for k, v in result.items()}
        self.log({'epoch': self.cur_ep} | record | {'best_score': self.best_score})
        print(record)

        if record['valid_top1_acc'] > self.best_score:
            self.best_score = record['valid_top1_acc']
            self.save()
            print(f'Save best model: epoch={self.cur_ep}, score={self.best_score}')

    def log(self, record: Dict[str, float]) -> None:
        if self.logger is not None:
            self.logger.log(record)

    def fit(self, epochs: int) -> None:
        self.model.to(self.device)
        if not self.disable_valid_on_start:
            self.valid_one_epoch()
        for self.cur_ep in range(1, epochs + 1):
            self.train_one_epoch()
            self.valid_one_epoch()
    
    def save(self) -> None:
        checkpoint = {
            'epoch': self.cur_ep,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
        }
        save_path = os.path.join(self.checkpoint_dir, CKPT_FILE)
        torch.save(checkpoint, save_path)

    def load(self, path):
        checkpoint = torch.load(os.path.join(path, CKPT_FILE))
        self.model.load_state_dict(checkpoint['model'], weights_only=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
