import json
import os
import torch
from lib.runners import AverageMeter, format_logs
import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import numpy as np

def get_img_list_by_condition(condition: dict, attr_file: str, prefix_dir: str):

    with open(attr_file) as f:
        data = json.load(f)

    # Extract the desired fields from the data
    results = {}
    for entry in data:
        if entry["attributes"]["weather"] not in condition["weather"]:
            continue
        if entry["attributes"]["timeofday"] not in condition["timeofday"]:
            continue
        if entry["attributes"]["scene"] not in condition["scene"]:
            continue
        results[os.path.join(prefix_dir, entry["name"])] = entry["attributes"]
    
    return results

def cmp_pred_and_lbl(pred_list, lbl_list):
    '''
    Output a same list of 1 and 0. 1 means same value.
    '''
    if len(pred_list) == len(lbl_list):
        output_list = [1 if torch.argmax(x) == y else 0 for x, y in zip(pred_list, lbl_list)]
        return output_list
    else:
        raise ValueError(f"Input lists should be the same length: {len(pred_list)} != {len(lbl_list)}")

def metric_fn(x, y):
    return sum(cmp_pred_and_lbl(x,y))/len(x)

class EvaluationCounter():
    def __init__(self, num_classes) -> None:
        self.total_num = np.zeros(num_classes)
        self.true_num = np.zeros(num_classes)

    def evaluate_and_add(self, pred_list, lbl_list):
        if len(pred_list) != len(lbl_list):
            raise ValueError(f"Input lists should be the same length: {len(pred_list)} != {len(lbl_list)}")
        for i in range(len(pred_list)):
            pred = torch.argmax(pred_list[i]).int()
            self.total_num[pred] += 1
            if pred == lbl_list[i]:
                self.true_num[pred] += 1

    def get_acc_by_classes(self):
        rst = np.nan_to_num(self.true_num / self.total_num, nan=-0.001)
        return rst.tolist()

    def get_total_acc(self):
        return sum(self.true_num) / sum(self.total_num)

def train_epoch(model, optimizer, criterion, dataloader, metric_fn: callable=metric_fn, device="cpu"):
    '''
        Returns:
        {'Loss': xxx, 'Score': yyy}
    '''
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
            avg = metric_fn(outputs, y)
            score_meter.update(avg, n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
        
        # delete these variables to reliese gpu cache
        del x, y, outputs, loss 
    return logs


def valid_epoch(model, criterion, dataloader, num_classes, metric_fn: callable=metric_fn, device="cpu"):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        {'Loss': xxx, 'Score': yyy}
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    detail_counter = EvaluationCounter(num_classes)
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
            avg = metric_fn(outputs, y)
            score_meter.update(avg, n=n)
            detail_counter.evaluate_and_add(outputs, y)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
        del x, y, outputs, loss
    logs.update({"Details": detail_counter.get_acc_by_classes()})
    return logs

def save_model_info(log_file, model_name, model_type, condition, condition_attr, acc, acc_details, idx_to_class_map):
    with open(log_file, "r+") as f:
        model_list = json.load(f)
        model_list[model_name] = {
            "model_type": model_type,
            "condition_attr": condition_attr,
            "acc": acc,
            "condition": condition,
            "mapping": idx_to_class_map,
            "acc_details": acc_details
        }
        f.seek(0)  # Move the file cursor to the beginning
        json.dump(model_list, f, indent=4)
        # Truncate the remaining contents (if any)
        f.truncate()

def load_model(num_classes, model_file=None):
    '''
    num_classes: int
    model_file: None|str, if None, it will return the pretrained default model
    ''' 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if isinstance(model_file, str):
        if os.path.exists(model_file):
            model.load_state_dict(torch.load(model_file))
        else:
            raise ValueError(f"File does not exist, please check: {model_file}")
    return model

