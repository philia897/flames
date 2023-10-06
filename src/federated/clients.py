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

    def fit(self, parameters, config):
        LOGGER.info({"state": "Start fitting", "cid": self.cid})
        set_params(self.model, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory
        # Load data for this client and get trainloader

        # Train
        if not config.get("skip_train", False):
            LOGGER.info("Start training")
            epochs = config.get('epochs', 1)
            self.runner.train(self.model, epochs)

            self.model.cpu()
            self.model.backbone.cpu()
            torch.cuda.empty_cache()
        else:
            LOGGER.info("Skip Training")

        # Return local model and statistics
        return get_params(self.model), self.runner.get_datasize("train"), {}

    def evaluate(self, parameters, config):
        LOGGER.info({"state": "Start evaluating", "cid": self.cid})
        set_params(self.model, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory

        # Evaluate
        eval_log = self.runner.validate(self.model)
        metric_log = {}
        for k,v in eval_log.items():
            if isinstance(v, (float, int, bool, bytes, str)):
                metric_log.setdefault(k,v)
        # Return statistics
        return eval_log.get("Loss", 1), self.runner.get_datasize("eval"), metric_log