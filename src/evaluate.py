from typing import Dict

import torch

from src.metrics import cal_top1_acc, cal_top3_acc


def evaluator(
        preds: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:

        eval_func = {
            "top1_acc": cal_top1_acc,
            "top3_acc": cal_top3_acc,
        }
        results = {}
        for metric, func in eval_func.items():
            results[metric] = func(preds, labels)
        return results
