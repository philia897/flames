import argparse
import datetime
import time
import warnings
from typing import Dict, List, Tuple

import flwr as fl
import ray
import torch
import torch.optim as optim
from flwr.common import Scalar
from flwr.common.typing import Scalar
from torch import nn

# warnings.filterwarnings("ignore")


import os

from lib import utils
from lib.simulation.env import (IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL,
                                get_label_paths, get_model_additional_configs)
from lib.simulation.split_dataset import Bdd100kDatasetSplitAgent, SplitMode
from lib.utils import get_dataloader
from lib.dbhandler import JsonDBHandler
from models.modelInterface import BDD100kModel

from federated.clients import FlowerClient
from federated.strategies import SaveModelStrategy

LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("sem_seg")
MODEL_META = get_model_additional_configs("sem_seg")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        # "batch_size": 2,
    }
    return config

# import logging

# # Configure the logger
# logging.basicConfig(level=logging.DEBUG)

# # Create a logger object
# logger = logging.getLogger(__name__)

# # Create a console handler and set its level
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)

# # Create a formatter and set it for the console handler
# formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# # Add the console handler to the logger
# logger.addHandler(console_handler)


# def get_evaluate_fn(
#     config_file: str,
# ):
#     """Return an evaluation function for centralized evaluation."""

#     attr_file = "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json"

#     def evaluate(
#         server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
#     ):
#         val_fns = utils.get_img_list_by_condition(condition, attr_file, IMAGE_PATH_VAL)
#         val_loader = get_dataloader({"val": val_fns}, 1, 4, output_size, False)
#         # determine device
#         device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#         model = BDD100kModel(
#             num_classes=3,
#             backbone=utils.load_mmcv_checkpoint(config_file),
#             size=output_size,
#         )
#         set_params(model, parameters)
#         model.to(device)

#         criterion = nn.CrossEntropyLoss()
#         loss, accuracy = validate_loop(model, val_loader, criterion, device=device)

#         # return statistics
#         return loss, {"accuracy": accuracy}

#     return evaluate


def eval_metrics_aggregate(
    num_and_metrics: List[Tuple[int, Dict]]
) -> Dict[str, Scalar]:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _ in num_and_metrics]
    )
    weighted_acc = [
        num_examples * acc_dic["accuracy"] for num_examples, acc_dic in num_and_metrics
    ]
    return {"accuracy": sum(weighted_acc) / num_total_evaluation_examples}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--num_client_cpus", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--learn_rate", type=float, default=0.0001)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--pool_size", type=int, default=10)  # number of dataset partions (= number of total clients)
    parser.add_argument("--train_model_type", type=str, default="deeplabv3+_backbone_fl10")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--attr_file_train", type=str, default="/home/zekun/drivable/data/bdd100k/labels/bdd100k_labels_images_attributes_train.json")
    parser.add_argument("--attr_file_val", type=str, default="/home/zekun/drivable/data/bdd100k/labels/bdd100k_labels_images_attributes_val.json")
    parser.add_argument("--condition_class_id", type=str, default="0")

    args = parser.parse_args()

    # parameter extraction
    STORE_MODEL_NAME = args.train_model_type
    condition_class_id = args.condition_class_id
    model_list_file = os.path.join(args.output_dir, 'db', 'model_list.json')
    checkpoint_file, config_file = JsonDBHandler.get_model_configs(model_list_file, condition_class_id)
    conditions = JsonDBHandler.get_conditions_from_class(os.path.join(args.output_dir, 'db', 'condition_classes.json'), condition_class_id)
    train_attr_file, val_attr_file = args.attr_file_train, args.attr_file_val
    output_size = (512, 1024)
    pool_size = args.pool_size
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": 1,
    }  # each client will get allocated 1 CPUs
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    stored_model_name = f"{STORE_MODEL_NAME}-class{condition_class_id}-{timestamp}.pth"
    # new_checkpoint_file = os.path.join(args.output_dir, "models", stored_model_name)
    new_checkpoint_file = None

    # define data split agent for pool_size clients
    train_split_agent = Bdd100kDatasetSplitAgent(
        train_attr_file,
        utils.get_img_paths_by_conditions(conditions, train_attr_file, IMAGE_PATH_TRAIN),
    )
    train_split_agent.split_list(pool_size, mode=SplitMode.RANDOM_SAMPLE, sample_n=300)
    val_split_agent = Bdd100kDatasetSplitAgent(
        val_attr_file,
        utils.get_img_paths_by_conditions(conditions, val_attr_file, IMAGE_PATH_VAL),
    )
    val_split_agent.split_list(pool_size, mode=SplitMode.SIMPLE_SPLIT)

    init_model = BDD100kModel(
        num_classes=MODEL_META["num_classes"],
        backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file),
        size=output_size
    )

    # configure the strategy
    strategy = SaveModelStrategy(
        init_model,
        new_checkpoint_file,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=eval_metrics_aggregate,
        initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_params(init_model)
        ),
    )

    def client_fn(cid: str):
        print("client created")
        # warnings.filterwarnings("ignore")
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader = get_dataloader(
            train_split_agent.get_partition(int(cid)),
            is_train=True,
            batch_size=args.batchsize,
            workers=num_workers,
            output_size=output_size,
            img_path=IMAGE_PATH_TRAIN,
            lbl_path=LABEL_PATH_TRAIN,
        )
        val_loader = get_dataloader(
            val_split_agent.get_partition(int(cid)),
            is_train=False,
            batch_size=args.batchsize,
            workers=num_workers,
            output_size=output_size,
            img_path=IMAGE_PATH_VAL,
            lbl_path=LABEL_PATH_VAL,
        )
        model = BDD100kModel(
            num_classes=MODEL_META["num_classes"],
            backbone=utils.load_mmcv_checkpoint(config_file),
            size=output_size
        )
        optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
        criterion = nn.CrossEntropyLoss()

        # create a single client instance
        return FlowerClient(
            cid, train_loader, val_loader, model, optimizer, criterion, DEVICE
        )

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

    # JsonDBHandler.save2model_list(model_list_file, new_checkpoint_file, config_file, condition_class_id)