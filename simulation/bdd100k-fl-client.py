import flwr as fl
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
import os

from lib.utils.logger import getLogger
LOGGER = getLogger(logfile='flames-client.log')

from federated.clients import BDD100KClient
from lib.data.tools import (get_dataloader, get_img_paths_by_conditions, load_mmcv_checkpoint)
from lib.train.runners import PytorchRunner
from lib.train.metrics import IoUMetricMeter
from models.modelInterface import BDD100kModel
from lib.data.split_dataset import Bdd100kDatasetSplitAgent, SplitMode
from lib.simulation.env import (get_image_paths, get_label_paths, get_transforms)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_conditions = [
        # ["clear", "residential", "daytime"] ,
        # ["clear", "residential", "night"] ,
        # ["clear", "residential", "dawn/dusk"] ,
        # ["clear", "city street", "daytime"] ,
        # ["clear", "city street", "night"] ,
        # ["clear", "city street", "dawn/dusk"] ,
        # ["clear", "highway", "daytime"] ,
        # ["clear", "highway", "night"] ,
        # ["clear", "highway", "dawn/dusk"] ,
        # ["overcast", "residential", "daytime"] ,
        # ["overcast", "city street", "daytime"] ,
        # ["overcast", "city street", "dawn/dusk"] ,
        # ["overcast", "highway", "daytime"] ,
        # ["overcast", "highway", "dawn/dusk"] ,
        # ["undefined", "residential", "daytime"] ,
        # ["undefined", "city street", "daytime"] ,
        # ["undefined", "city street", "night"] ,
        # ["undefined", "city street", "dawn/dusk"] ,
        # ["undefined", "highway", "daytime"] ,
        # ["partly cloudy", "residential", "daytime"] ,
        # ["partly cloudy", "city street", "daytime"] ,
        # ["partly cloudy", "city street", "dawn/dusk"] ,
        # ["partly cloudy", "highway", "daytime"] ,
        # ["partly cloudy", "highway", "dawn/dusk"] ,
        # ["rainy", "residential", "daytime"] ,
        # ["rainy", "city street", "daytime"] ,
        # ["rainy", "city street", "night"] ,
        # ["rainy", "city street", "dawn/dusk"] ,
        # ["rainy", "highway", "daytime"] ,
        # ["rainy", "highway", "night"] ,
        # ["snowy", "residential", "daytime"] ,
        # ["snowy", "residential", "night"] ,
        # ["snowy", "city street", "daytime"] ,
        # ["snowy", "city street", "night"] ,
        # ["snowy", "city street", "dawn/dusk"] ,
        # ["snowy", "highway", "daytime"] ,
        # ["snowy", "highway", "night"]
    ]

def get_current_conditions(current_condition_fn):
    with open(current_condition_fn, "r") as f:
        lst = json.load(f)
    return lst

def create_client(
        cid: str,
        batchsize: int,
        classes_num: int,
        train_split_agent: Bdd100kDatasetSplitAgent,
        val_split_agent: Bdd100kDatasetSplitAgent,
        num_workers: int,
        model,
        optimizer,
        criterion,
        device
        ):
    LOGGER.info(f"client {cid} created")
    # warnings.filterwarnings("ignore")
    train_loader = get_dataloader(
        train_split_agent.get_partition(int(cid)),
        batch_size=batchsize,
        workers=num_workers,
        img_transform=img_transform,
        lbl_transform=lbl_transform,
        img_path=IMAGE_PATH_TRAIN,
        lbl_path=LABEL_PATH_TRAIN,
        is_train=True,
        classes_num=classes_num
    )
    val_loader = get_dataloader(
        val_split_agent.get_partition(int(cid)),
        batch_size=batchsize,
        workers=num_workers,
        img_transform=img_transform,
        lbl_transform=lbl_transform,
        img_path=IMAGE_PATH_VAL,
        lbl_path=LABEL_PATH_VAL,        
        is_train=False,
        classes_num=classes_num
    )
    metric_meter = IoUMetricMeter(classes_num)

    runner = PytorchRunner(optimizer, criterion, train_loader, val_loader, metric_meter, device, verbose=True)
    # create a single client instance
    return BDD100KClient(cid=cid, model=model, runner=runner)

if __name__ == "__main__":
    pkg_name = "10k" # 100k or 10k
    output_size = (512,1024)
    num_classes = 20

    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--cid", type=str, default="0")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--learn_rate", type=float, default=0.0001)
    parser.add_argument("--batchsize", type=int, default=2)
    # parser.add_argument("--train_model_type", type=str, default="deeplabv3+_backbone_fl10")
    # parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--model_config_file", type=str, default="/home/zekun/drivable/src/models/config-deeplabv3plus-sem_seg.py")
    parser.add_argument("--attr_file_train", type=str, default=f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_train.json")
    parser.add_argument("--attr_file_val", type=str, default=f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_val.json")
    parser.add_argument("--condition_class_id", type=str, default="0")
    parser.add_argument("--task_name", type=str, default="sem_seg")
    args = parser.parse_args()

    IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/", pkg_name)
    LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("/home/zekun/drivable/", args.task_name, pkg_name)
    img_transform, lbl_transform = get_transforms(args.task_name, output_size)
    # conditions = all_conditions
    env_path = __file__.replace(os.path.basename(__file__), "current_conditions.json")
    conditions = get_current_conditions(env_path)

    config_file = args.model_config_file
    client_num = 2
    cid = args.cid

    train_attr_file, val_attr_file = args.attr_file_train, args.attr_file_val
    # output_dir = args.output_dir

    train_split_agent = Bdd100kDatasetSplitAgent(
        train_attr_file,
        get_img_paths_by_conditions(conditions, train_attr_file, IMAGE_PATH_TRAIN),
    )
    train_split_agent.split_list(client_num, mode=SplitMode.RANDOM_SAMPLE, sample_n=30)
    val_split_agent = Bdd100kDatasetSplitAgent(
        val_attr_file,
        get_img_paths_by_conditions(conditions, val_attr_file, IMAGE_PATH_VAL),
    )
    val_split_agent.split_list(client_num, mode=SplitMode.SIMPLE_SPLIT)

    model = BDD100kModel(
        num_classes=num_classes,
        backbone=load_mmcv_checkpoint(config_file),
        size=output_size
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    client = create_client(cid, args.batchsize, num_classes,
                           train_split_agent, val_split_agent, 0, model, optimizer, criterion, DEVICE)
    
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)