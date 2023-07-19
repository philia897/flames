import numpy as np
import torch
from tqdm import tqdm
from . import metrics


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)

def pred_to_OneHot(pred):
    device = pred.device
    max_idx = torch.argmax(pred, dim=1)
    one_hot = torch.zeros(*(pred.shape)).to(device)
    # print(one_hot.device, max_idx.device)
    one_hot.scatter_(1, max_idx.unsqueeze(1), 1)
    return one_hot

def metric(input, target):
    """
    Args:
        input (tensor): prediction
        target (tensor): reference data

    Returns:
        float: harmonic fscore without including backgorund
    """
    # input = torch.softmax(input, dim=1)
    input = pred_to_OneHot(input)
    scores = []

    for i in range(input.shape[1]-1):  # background is not included
        ypr = input[:, i, :, :].view(input.shape[0], -1)
        ygt = target[:, i, :, :].view(target.shape[0], -1)
        scores.append(metrics.iou(ypr, ygt).item())

    return np.mean(scores), scores


def train_epoch(model, optimizer, criterion, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train", mininterval=1.)
    for x, y, *_ in iterator:
        x = x.to(device)
        y = y.to(device)
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            avg, scores = metric(outputs, y)
            score_meter.update(avg, n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
        
        # delete these variables to reliese gpu cache
        del x, y, outputs, loss 
    return logs


def valid_epoch(model, criterion, dataloader, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    iterator = tqdm(dataloader, desc="Valid", mininterval=1.)
    for x, y, *_ in iterator:
        x = x.to(device)
        y = y.to(device)
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)
            loss = criterion(outputs, y)

            loss_meter.update(loss.item(), n=n)
            avg, scores = metric(outputs, y)
            score_meter.update(avg, n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
        del x, y, outputs, loss
    return logs


def train_loop(model, train_data_loader, optimizer, criterion, epochs, device):

    model.to(device).train()
    # epoch_i = 1
    for epoch_i in range(epochs):
        # training
        print(f"\nEpoch: {epoch_i} / {epochs}\n-------------------------------")
        train_log = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
        )

def validate_loop(model, val_data_loader, criterion, device):
    model.eval()

    valid_logs = valid_epoch(
        model=model,
        criterion=criterion,
        dataloader=val_data_loader,
        device=device,
    )

    print(valid_logs)
    return valid_logs["Loss"], valid_logs["Score"]

