import flwr as fl
import torch
from torch.utils.data import DataLoader
import os
import warnings
warnings.filterwarnings("ignore")

from models.modelInterface import BDD100kModel
from lib.train.runners import Runner
from lib.data.tools import get_params, set_params
from lib.utils.logger import getLogger
from typing import Dict, Any

LOGGER = getLogger()

# Flower client, adapted from Pytorch quickstart example
class BDD100KClient(fl.client.NumPyClient):
    def __init__(self, 
                 cid: str,
                 model:BDD100kModel,
                 runner:Runner):
        self.cid = cid

        # Instantiate model
        self.model = model
        self.runner = runner

    def _process_log(self, log:Dict[str, Any]):
        metric_log = {}
        for k,v in log.items():
            if isinstance(v, (float, int, bool, bytes, str)):
                metric_log.setdefault(k,v)
        return metric_log

    def fit(self, parameters, config):
        LOGGER.info({"state": "Start fitting", "cid": self.cid})
        set_params(self.model, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory
        # Load data for this client and get trainloader

        # Train
        if not config.get("skip_train", False):
            epochs = config.get('epochs', 1)
            train_log = self.runner.train(self.model, epochs, f"round {config.get('round', 0)}")
            self.model.cpu()
            self.model.backbone.cpu()
            torch.cuda.empty_cache()
        else:
            LOGGER.info("Skip Training")
            train_log = {}

        # Return local model and statistics
        return get_params(self.model), self.runner.get_datasize("train"), self._process_log(train_log)

    def evaluate(self, parameters, config):
        LOGGER.info({"state": "Start evaluating", "cid": self.cid})
        set_params(self.model, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory

        # Evaluate
        eval_log = self.runner.validate(self.model, f"round {config.get('round', 0)}")

        # Return statistics
        return eval_log.get("loss", 1.), self.runner.get_datasize("eval"), self._process_log(eval_log)