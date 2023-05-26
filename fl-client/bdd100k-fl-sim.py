import argparse
import flwr as fl
from flwr.common import EvaluateRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.typing import Scalar
from flwr.server import client_proxy
from flwr.server.client_proxy import ClientProxy
import ray
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim
from torch import nn
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List, Union
from torchvision import transforms
import time
import gc
import sys
import warnings
import json

warnings.filterwarnings("ignore")

sys.path.append(".")  # <= change path where you save code

from models.modelInterface import BDD100kModel

from lib import utils
from lib.runners import train_epoch, valid_epoch
from lib.bdd100kdataset import BDD100kDataset
from lib.simulation.split_dataset import Bdd100kDatasetSplitAgent, SplitMode
import os


IMAGE_PATH = os.path.join("data", "bdd100k", "images", "100k")
IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")

LABEL_PATH = os.path.join("data", "bdd100k", "labels", "drivable", "masks")
LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")


def get_dataloader(
    data: dict, batch_size, workers: int, output_size: tuple, is_train=True
):
    msk_fn_train = lambda fn: fn.replace(IMAGE_PATH_TRAIN, LABEL_PATH_TRAIN).replace(
        "jpg", "png"
    )
    msk_fn_val = lambda fn: fn.replace(IMAGE_PATH_VAL, LABEL_PATH_VAL).replace(
        "jpg", "png"
    )

    transform = transforms.Compose(
        [
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze())
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    if is_train:
        train_fns = data["train"]
        train_dataset = BDD100kDataset(
            train_fns,
            msk_fn_train,
            split="train",
            transform=transform,
            transform2=transform,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True
        )
        return train_loader
    else:
        val_fns = data["val"]
        val_dataset = BDD100kDataset(
            val_fns, msk_fn_val, split="val", transform=transform, transform2=transform
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False
        )
        return val_loader


def train_loop(model, train_data_loader, optimizer, criterion, epochs, device):

    model.to(device).train()
    # epoch_i = 1
    for epoch_i in range(epochs):
        # training
        print(f"\nEpoch: {epoch_i} / {epochs}\n-------------------------------")
        t1 = time.time()
        train_log = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=device,
        )
        t2 = time.time()

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


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_data: dict, model: BDD100kModel):
        self.cid = cid
        self.fed_data = fed_data

        # Instantiate model
        self.net = model
        self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        self.criterion = nn.CrossEntropyLoss()

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory
        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        self.fed_data = {
            "train": train_split_agent.get_partition(int(self.cid)),
            "val": val_split_agent.get_partition(int(self.cid)),
        }

        trainloader = get_dataloader(
            self.fed_data,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            output_size=output_size,
        )

        # Send model to device
        self.net.to(self.device)

        # Train
        train_loop(
            self.net,
            trainloader,
            self.optimizer,
            self.criterion,
            epochs=config["epochs"],
            device=self.device,
        )
        self.net.cpu()
        self.net.backbone.cpu()
        torch.cuda.empty_cache()


        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = get_dataloader(
            self.fed_data,
            is_train=False,
            batch_size=1,
            workers=num_workers,
            output_size=output_size,
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = validate_loop(
            self.net, valloader, self.criterion, device=self.device
        )

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 2,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.Module, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
    config_file: str,
):
    """Return an evaluation function for centralized evaluation."""

    attr_file = "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json"

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
        val_fns = utils.get_img_list_by_condition(condition, attr_file, IMAGE_PATH_VAL)
        val_loader = get_dataloader({"val": val_fns}, 1, 4, output_size, False)
        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = BDD100kModel(
            num_classes=3,
            backbone=utils.load_mmcv_checkpoint(config_file),
            size=output_size,
        )
        set_params(model, parameters)
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        loss, accuracy = validate_loop(model, val_loader, criterion, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate

def eval_metrics_aggregate(num_and_metrics:List[Tuple[int, Dict]]) -> Dict[str, Scalar]:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in num_and_metrics])
    weighted_acc = [num_examples * acc_dic['accuracy'] for num_examples, acc_dic in num_and_metrics]
    return {'accuracy': sum(weighted_acc) / num_total_evaluation_examples} 

from lib.utils import save_model


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Callable[
            [int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None
        ]
        | None = None,
        on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model = BDD100kModel(3, utils.load_mmcv_checkpoint(config_file), output_size)
        self.best_score = 0.0

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[Tuple[client_proxy.ClientProxy, fl.common.FitRes], BaseException]
        ],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        train_split_agent.split_list(
            pool_size, mode=SplitMode.RANDOM_SAMPLE, sample_n=300
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        print(f"Average score: {aggregated_metrics['accuracy']}\nCurrent Best Score: {self.best_score}")
        if float(aggregated_metrics['accuracy']) > float(self.best_score):
            self.best_score = aggregated_metrics['accuracy']
            # Save the model
            save_model(self.model.backbone, 5, 0, stored_model_name, OUTPUT_DIR)
            

        return aggregated_loss, aggregated_metrics


# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--num_client_cpus", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    # parse input arguments
    args = parser.parse_args()

    OUTPUT_DIR = "/home/zekun/drivable/outputs"

    with open("./config.json", "r") as f:
        configs = json.load(f)
    STORE_MODEL_NAME = configs["stored_model_name"]
    checkpoint_file = configs["checkpoint_file"]
    config_file = configs["model_config_file"]
    condition = configs["condition"]
    train_attr_file = (
        "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_train.json"
    )
    val_attr_file = (
        "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json"
    )

    output_size = (512, 1024)
    pool_size = 10  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": 1,
    }  # each client will get allocated 1 CPUs

    train_split_agent = Bdd100kDatasetSplitAgent(
        train_attr_file,
        utils.get_img_list_by_condition(condition, train_attr_file, IMAGE_PATH_TRAIN),
    )
    train_split_agent.split_list(
        pool_size, mode=SplitMode.RANDOM_SAMPLE, sample_n=300
    )
    val_split_agent = Bdd100kDatasetSplitAgent(
        val_attr_file,
        utils.get_img_list_by_condition(condition, val_attr_file, IMAGE_PATH_VAL),
    )
    val_split_agent.split_list(pool_size, mode=SplitMode.SIMPLE_SPLIT)

    init_model = BDD100kModel(
        num_classes=3,
        backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file),
        size=output_size,
    )

    import datetime

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    stored_model_name = f"{STORE_MODEL_NAME}-{timestamp}.pth"

    # configure the strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        # evaluate_fn=get_evaluate_fn(
        #     config_file
        # ),  # centralised evaluation of global model
        evaluate_metrics_aggregation_fn=eval_metrics_aggregate,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in init_model.state_dict().items()]
        ),
    )

    def client_fn(cid: str):
        warnings.filterwarnings("ignore")
        fed_data = {
            "train": train_split_agent.get_partition(int(cid)),
            "val": val_split_agent.get_partition(int(cid)),
        }
        model = BDD100kModel(
            num_classes=3,
            backbone=utils.load_mmcv_checkpoint(config_file),
            size=output_size,
        )
        # create a single client instance
        return FlowerClient(cid, fed_data, model)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    start = time.time()
    print(f"start at {time.ctime()}")
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))
    with open("./config.json", "r") as f:
        curr_configs = json.load(f)
    with open("./config.json", "w") as f:
        curr_configs['available_model_list'][STORE_MODEL_NAME] = {
            "checkpoint_file": os.path.join(OUTPUT_DIR, stored_model_name),
            "model_config_file": config_file,
            "condition": condition
        }
        json.dump(curr_configs, f, indent=4)


