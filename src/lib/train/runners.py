import numpy as np
import torch
from tqdm import tqdm
from .metrics import MetricMeter
from typing import Callable, Dict, Any
from torch.utils.data import DataLoader
from lib.utils.logger import getLogger

LOGGER = getLogger()

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


def train_epoch(
        model:torch.nn.Module, 
        optimizer:torch.optim.Optimizer, 
        criterion:Callable, 
        dataloader:DataLoader, 
        metric_meter: MetricMeter, 
        device="cpu",
        verbose=False):
    loss_meter = AverageMeter()
    logs = {}
    metric_meter.reset()  # reset the metric meter to clear previous logs
    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train", mininterval=1., disable=not verbose)
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
            metric_meter.calculate_and_log(y, outputs)

        logs.update({"loss": loss_meter.avg})
        logs.update(metric_meter.summary())
        iterator.set_postfix_str(format_logs(logs))
        
        # delete these variables to reliese gpu cache
        del x, y, outputs, loss 
    logs.update(metric_meter.details())
    return logs


def valid_epoch(
        model:torch.nn.Module, 
        criterion:Callable, 
        dataloader:DataLoader, 
        metric_meter: MetricMeter, 
        device="cpu",
        verbose=False):
    loss_meter = AverageMeter()
    logs = {}
    metric_meter.reset()
    model.to(device).eval()

    iterator = tqdm(dataloader, desc="Valid", mininterval=1., disable=not verbose)
    for x, y, *_ in iterator:
        x = x.to(device)
        y = y.to(device)
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)
            loss = criterion(outputs, y)
            loss_meter.update(loss.item(), n=n)
            metric_meter.calculate_and_log(y, outputs)

        logs.update({"loss": loss_meter.avg})
        logs.update(metric_meter.summary())
        iterator.set_postfix_str(format_logs(logs))
        del x, y, outputs, loss
    logs.update(metric_meter.details())
    return logs

class Runner(object):
    '''Basic class for runner that trains and validates the given model'''

    def train(self, model, epochs=1, comment:str=None)->None:
        '''Train the given model for specific number of epochs'''
        pass

    def validate(self, model, comment:str=None)->Dict:
        '''validate the given model and return the metrics as a Dict'''
        pass

    def get_datasize(self, mode)->Any:
        '''
        return the size of training dataset or val dataset. 
        mode should be train or eval
        '''
        pass

class PytorchRunner():
    def __init__(self,
            optimizer:torch.optim.Optimizer, 
            criterion:Callable, 
            train_loader:DataLoader, 
            val_loader:DataLoader,
            metric_meter: MetricMeter,
            device="cpu",
            verbose=False):
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.metric_meter = metric_meter
        self.device = device
        self.verbose = verbose

    def train(self, model:torch.nn.Module, epochs=1, comment:str=None)->None:
        for epoch_i in range(epochs):
            train_log = train_epoch(
                model=model,
                optimizer=self.optimizer,
                criterion=self.criterion,
                dataloader=self.train_loader,
                metric_meter=self.metric_meter,
                device=self.device,
                verbose=self.verbose
            )
            if comment:
                LOGGER.debug({
                    "comment": comment,
                    "epoch": epoch_i,
                    "log": train_log})
            
    def validate(self, model:torch.nn.Module, comment:str=None)->Dict:
        eval_log = valid_epoch(
            model=model,
            criterion=self.criterion,
            dataloader=self.val_loader,
            metric_meter=self.metric_meter,
            device=self.device,
            verbose=self.verbose
            )
        if comment:
            LOGGER.debug({
                "comment": comment,
                "log": eval_log
            })
        return eval_log
        
    def get_datasize(self, mode:str):
        '''mode should be train or eval'''
        if mode == 'train':
            return len(self.train_loader.dataset)
        elif mode == 'eval':
            return len(self.val_loader.dataset)
        else:
            raise ValueError(f"Invalid mode {mode}, should be train or eval.")

