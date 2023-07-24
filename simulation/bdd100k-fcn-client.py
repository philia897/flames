import flwr as fl
import torch
import torch.optim as optim
import torch.nn as nn
import argparse

from federated.clients import FlowerClient
from lib import utils
from lib.utils import get_dataloader
from models.modelInterface import BDD100kModel
from lib.simulation.split_dataset import Bdd100kDatasetSplitAgent, SplitMode
from lib.simulation.env import (get_image_paths, get_label_paths, get_model_additional_configs)

IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/")
LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("/home/zekun/drivable/", "sem_seg")
MODEL_META = get_model_additional_configs("sem_seg")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

all_conditions = [
        ["clear", "residential", "daytime"] ,
        ["clear", "residential", "night"] ,
        ["clear", "residential", "dawn/dusk"] ,
        ["clear", "city street", "daytime"] ,
        ["clear", "city street", "night"] ,
        ["clear", "city street", "dawn/dusk"] ,
        ["clear", "highway", "daytime"] ,
        ["clear", "highway", "night"] ,
        ["clear", "highway", "dawn/dusk"] ,
        ["overcast", "residential", "daytime"] ,
        ["overcast", "city street", "daytime"] ,
        ["overcast", "city street", "dawn/dusk"] ,
        ["overcast", "highway", "daytime"] ,
        ["overcast", "highway", "dawn/dusk"] ,
        ["undefined", "residential", "daytime"] ,
        ["undefined", "city street", "daytime"] ,
        ["undefined", "city street", "night"] ,
        ["undefined", "city street", "dawn/dusk"] ,
        ["undefined", "highway", "daytime"] ,
        ["partly cloudy", "residential", "daytime"] ,
        ["partly cloudy", "city street", "daytime"] ,
        ["partly cloudy", "city street", "dawn/dusk"] ,
        ["partly cloudy", "highway", "daytime"] ,
        ["partly cloudy", "highway", "dawn/dusk"] ,
        ["rainy", "residential", "daytime"] ,
        ["rainy", "city street", "daytime"] ,
        ["rainy", "city street", "night"] ,
        ["rainy", "city street", "dawn/dusk"] ,
        ["rainy", "highway", "daytime"] ,
        ["rainy", "highway", "night"] ,
        ["snowy", "residential", "daytime"] ,
        ["snowy", "residential", "night"] ,
        ["snowy", "city street", "daytime"] ,
        ["snowy", "city street", "night"] ,
        ["snowy", "city street", "dawn/dusk"] ,
        ["snowy", "highway", "daytime"] ,
        ["snowy", "highway", "night"]
    ]

def create_client(
        cid: str,
        batchsize: int,
        output_size: tuple,
        classes_num: int,
        train_split_agent: Bdd100kDatasetSplitAgent,
        val_split_agent: Bdd100kDatasetSplitAgent,
        num_workers: int,
        model,
        optimizer,
        criterion,
        device
        ):
    print("client created")
    # warnings.filterwarnings("ignore")
    train_loader = get_dataloader(
        train_split_agent.get_partition(int(cid)),
        is_train=True,
        batch_size=batchsize,
        workers=num_workers,
        output_size=output_size,
        img_path=IMAGE_PATH_TRAIN,
        lbl_path=LABEL_PATH_TRAIN,
        classes_num=classes_num
    )
    val_loader = get_dataloader(
        val_split_agent.get_partition(int(cid)),
        is_train=False,
        batch_size=batchsize,
        workers=num_workers,
        output_size=output_size,
        img_path=IMAGE_PATH_VAL,
        lbl_path=LABEL_PATH_VAL,
        classes_num=classes_num
    )

    # create a single client instance
    return FlowerClient(
        cid, train_loader, val_loader, model, optimizer, criterion, device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--learn_rate", type=float, default=0.0001)
    parser.add_argument("--batchsize", type=int, default=2)
    parser.add_argument("--train_model_type", type=str, default="deeplabv3+_backbone_fl10")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--model_config_file", type=str, default="/home/zekun/drivable/src/models/config-deeplabv3plus-sem_seg.py")
    parser.add_argument("--attr_file_train", type=str, default="/home/zekun/drivable/data/bdd100k/labels/bdd100k_labels_images_attributes_train.json")
    parser.add_argument("--attr_file_val", type=str, default="/home/zekun/drivable/data/bdd100k/labels/bdd100k_labels_images_attributes_val.json")
    parser.add_argument("--condition_class_id", type=str, default="0")
    args = parser.parse_args()

    conditions = all_conditions

    config_file = args.model_config_file
    output_size = (512,1024)
    client_num = 2
    condition_class_id = args.condition_class_id

    train_attr_file, val_attr_file = args.attr_file_train, args.attr_file_val
    output_dir = args.output_dir

    train_split_agent = Bdd100kDatasetSplitAgent(
        train_attr_file,
        utils.get_img_paths_by_conditions(conditions, train_attr_file, IMAGE_PATH_TRAIN),
    )
    train_split_agent.split_list(client_num, mode=SplitMode.RANDOM_SAMPLE, sample_n=300)
    val_split_agent = Bdd100kDatasetSplitAgent(
        val_attr_file,
        utils.get_img_paths_by_conditions(conditions, val_attr_file, IMAGE_PATH_VAL),
    )
    val_split_agent.split_list(client_num, mode=SplitMode.SIMPLE_SPLIT)

    model = BDD100kModel(
        num_classes=MODEL_META["num_classes"],
        backbone=utils.load_mmcv_checkpoint(config_file),
        size=output_size
    )
    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate)
    criterion = nn.CrossEntropyLoss()
    client = create_client(condition_class_id, args.batchsize, output_size, MODEL_META["num_classes"],
                           train_split_agent, val_split_agent, 2, model, optimizer, criterion, DEVICE)
    
    fl.client.start_numpy_client(server_address="[::]:8080", client=client)