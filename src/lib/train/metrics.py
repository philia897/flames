import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
from bdd100k.eval.seg import fast_hist, per_class_acc, per_class_iou, whole_acc

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union


def fscore(pr, gt, beta=1, eps=1e-7, threshold=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta**2) * tp + eps) / (
        (1 + beta**2) * tp + beta**2 * fn + fp + eps
    )
    return score


class Fscore(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "Fscore"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            scores.append(fscore(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)


class IoU(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """

    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "IoU"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :]
            ygt = target[:, i, :, :]
            scores.append(iou(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)


class MetricMeter:

    def __init__(self):
        pass

    def calculate_and_log(self, gt:torch.Tensor, pred:torch.Tensor) -> None:
        pass

    def reset(self) -> None:
        pass

    def summary(self) -> dict:
        return {}

    def details(self) -> dict:
        return {}

class IoUMetricMeter(MetricMeter):
    '''IoU Metric Meter for Acc and IoU evaluation'''
    def __init__(self, num_classes:int):
        super().__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.hist = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

    def calculate_and_log(self, gt:Tensor, pred:Tensor):
        pred = torch.argmax(pred, dim=1).detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        # print(pred.shape, gt.shape)
        hist = fast_hist(gt, pred, self.num_classes)
        self.hist += hist

    def summary(self) -> dict:
        '''return a simple summary: Dict[pAcc:float]'''
        acc = whole_acc(self.hist)
        return {
            "pAcc": acc
        }
    
    def details(self) -> dict:
        '''return full metric details: Dict[pAcc:float, Acc:list, mIoU:float, IoU:list]'''
        acc = whole_acc(self.hist)
        ious = per_class_iou(self.hist)
        accs = per_class_acc(self.hist)
        return {
            "pAcc": acc,
            "mIoU": np.mean(ious),
            "Acc": accs.tolist(),
            "IoU": ious.tolist(),
        }
