from torchmetrics.functional import accuracy

from src.constants import N_CLASSES


def cal_top1_acc(preds, labels):
    return accuracy(preds, labels, task="multiclass", top_k=1, num_classes=N_CLASSES).item()

def cal_top3_acc(preds, labels):
    return accuracy(preds, labels, task="multiclass", top_k=3, num_classes=N_CLASSES).item()
