import argparse
import time
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")

import flwr as fl
import torch
import torch.optim as optim
from flwr.common.typing import Scalar
from torch import nn
import os
import shutil

from lib.utils.logger import getLogger, create_id_by_timestamp
MODEL_NAME = f"model-{create_id_by_timestamp(include_ms=False)}.pth"
LOGGER_FILE = f"/home/zekun/drivable/outputs/semantic/logs/{MODEL_NAME.replace('.pth', '-svr-sim.log')}"
LOGGER = getLogger(logfile=LOGGER_FILE)

from lib.simulation.env import get_image_paths, get_label_paths, get_transforms
from lib.data.split_dataset import Bdd100kDatasetSplitAgent, SplitMode
from lib.data.tools import get_dataloader, load_mmcv_checkpoint, get_params, get_img_paths_by_conditions
from lib.utils.dbhandler import JsonDBHandler, Items
from models.modelInterface import BDD100kModel
from lib.train.metrics import IoUMetricMeter
from lib.train.runners import PytorchRunner

from federated.clients import BDD100KClient
from federated.strategies import BDD100KStrategy, aggregate_custom_metrics, update_modelinfo


def fit_config(server_round: int)->Dict[str,Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "round": server_round,
        "skip_train": False
    }
    return config

def eval_config(server_round: int)->Dict[str,Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "round": server_round
    }
    return config

def gen_client_fn(
        model_config_file,
        learn_rate,
        task_name,
        output_size,
        batchsize,
        img_path:Tuple,
        lbl_path:Tuple,
        num_classes,
        device,
        ignore_index=255,
    ):
    def client_fn(cid: str):
        LOGGER.info(f"client {cid} created")

        # num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        model = BDD100kModel(backbone=load_mmcv_checkpoint(model_config_file))
        optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

        train_transform = get_transforms(task_name, output_size, "train")
        val_transform = get_transforms(task_name, output_size, "test")

        train_loader = get_dataloader(
            train_split_agent.get_partition(cid),
            batch_size=batchsize,
            transform=train_transform,
            img_path=img_path[0],
            lbl_path=lbl_path[0],
            is_train=True
        )
        val_loader = get_dataloader(
            val_split_agent.get_partition(cid),
            batch_size=batchsize,
            transform=val_transform,
            img_path=img_path[1],
            lbl_path=lbl_path[1],        
            is_train=False
        )
        metric_meter = IoUMetricMeter(num_classes)

        runner = PytorchRunner(optimizer, criterion, train_loader, val_loader, metric_meter, device, verbose=False)
        # create a single client instance
        return BDD100KClient(cid=cid, model=model, runner=runner)
    return client_fn

client_resources = {
    "num_cpus": 2,
    "num_gpus": 1,
}  # each client will get allocated 1 CPUs

if __name__ == "__main__":
    start = time.time()

    # set environment and static experiment settings
    pkg_name = "10k" # 100k or 10k
    task_name = "sem_seg" # drivable, sem_seg
    outputsize = (512,1024)
    num_classes = 19
    sample_per_client = 500
    main_metric = "pAcc"

    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("-n", "--num_rounds", type=int, default=10)
    parser.add_argument("--learn_rate", type=float, default=1e-4) # 0.0001
    parser.add_argument("--batchsize", type=int, default=4)
    parser.add_argument("--client_num", type=int, default=10)  # number of dataset partions (= number of total clients)
    parser.add_argument("--db_dir", type=str, default="/home/zekun/drivable/outputs/semantic/db")
    parser.add_argument("--attr_file_train", type=str, default=f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_train.json")
    parser.add_argument("--attr_file_val", type=str, default=f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_val.json")
    parser.add_argument("-c", "--concls_id", type=str, default='0')
    args = parser.parse_args()

    # get environment info
    IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/", pkg_name)
    LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("/home/zekun/drivable/", task_name, pkg_name)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # parameter definition
    cls_id = args.concls_id
    lr = args.learn_rate
    batchsize = args.batchsize
    train_attr_file, val_attr_file = args.attr_file_train, args.attr_file_val
    client_num = args.client_num

    # read model item from DB
    handler = JsonDBHandler(args.db_dir)
    modelinfo = handler.read(cls_id, Items.MODEL_INFO)
    conditions = handler.read(cls_id, Items.CONDITION_CLASS).conditions
    LOGGER.debug(f"Read model item {cls_id}: {modelinfo}")

    init_model = BDD100kModel(backbone=load_mmcv_checkpoint(modelinfo.model_config_file, modelinfo.checkpoint_file))

    # set the new saved model name
    modelinfo.checkpoint_file = handler.suggest_model_save_path(MODEL_NAME)
    config_file = modelinfo.model_config_file

    # Extract old best score if exists:
    init_metric_value = modelinfo.meta.get("runtime_metrics", {}).get(main_metric, 0)

    # configure the strategy
    strategy = BDD100KStrategy(
        model=init_model,
        modelinfo=modelinfo,
        main_metric=main_metric,
        init_metric_value=init_metric_value,
        fraction_fit=0.5,
        fraction_evaluate=1.,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        fit_metrics_aggregation_fn=aggregate_custom_metrics,
        evaluate_metrics_aggregation_fn=aggregate_custom_metrics,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_params(init_model)
        ),
    )

    # Split the data for each client for training
    train_split_agent = Bdd100kDatasetSplitAgent(
        train_attr_file,
        get_img_paths_by_conditions(conditions, train_attr_file, IMAGE_PATH_TRAIN),
        min_size=client_num
    )
    train_split_agent.split_list(client_num, mode=SplitMode.RANDOM_SAMPLE, sample_n=sample_per_client)
    val_split_agent = Bdd100kDatasetSplitAgent(
        val_attr_file,
        get_img_paths_by_conditions(conditions, val_attr_file, IMAGE_PATH_VAL),
        min_size=client_num
    )
    val_split_agent.split_list(client_num, mode=SplitMode.SIMPLE_SPLIT)

    # client function to create a new client with cid
    client_fn = gen_client_fn(
        model_config_file=config_file,
        learn_rate=args.learn_rate,
        task_name=task_name,
        output_size=outputsize,
        batchsize=batchsize,
        img_path=(IMAGE_PATH_TRAIN, IMAGE_PATH_VAL),
        lbl_path=(LABEL_PATH_TRAIN, LABEL_PATH_VAL),
        num_classes=num_classes,
        device=DEVICE,
        ignore_index=255
    )
    # (optional) specify Ray config
    # ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=client_num,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        # ray_init_args=ray_init_args,
    )

    # Update the database item if the new model is saved
    if os.path.exists(modelinfo.checkpoint_file):
        modelinfo = update_modelinfo(modelinfo, num_classes, outputsize)
        handler.update(modelinfo, Items.MODEL_INFO, cls_id)
        LOGGER.debug(f"Update Model Item {cls_id}: {modelinfo}")
    else:
        LOGGER.debug("Original Model is better, nothing changed")

    # archive the client log: flames.log to DB 
    shutil.move("flames.log", LOGGER_FILE.replace("svr", "cli"))

    # Elapsed time
    LOGGER.info("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))