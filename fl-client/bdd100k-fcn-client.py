from collections import OrderedDict

import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# from torchvision.datasets import CIFAR10
import sys
import os
from pathlib import Path
import numpy as np

import flwr as fl
sys.path.append('.') # <= change path where you save code

from lib.bdd100kdataset import BDD100kDataset

## Definition of parameters
BATCH_SIZE = 2
NUM_EPOCHS = 1
LR = 0.0001


condition = {
    "weather": ["snowy"],  
    # "clear", "rainy", "snowy", "overcast", "undefined", "partly cloudy", "foggy"
    "timeofday": ["daytime", "night", "dawn/dusk", "undefined"],
    # "daytime", "night", "dawn/dusk", "undefined"
    "scene": ["tunnel", "residential", "parking lot", "undefined", "city street", "gas stations"],  
    # "tunnel", "residential", "parking lot", "undefined", "city street", "gas stations", "highway"
}
# condition = None

sample_limit = 10000

output_size = (769,769)

BASE_PATH = "./"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

## -----------------------
## Get Images
## -----------------------

IMAGE_PATH = os.path.join("data", "bdd100k", "images", "100k")
IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")

LABEL_PATH = os.path.join("data", "bdd100k", "labels", "drivable", "masks")
LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")

msk_fn_train = lambda fn : fn.replace(IMAGE_PATH_TRAIN, LABEL_PATH_TRAIN).replace("jpg", "png")
msk_fn_val = lambda fn : fn.replace(IMAGE_PATH_VAL, LABEL_PATH_VAL).replace("jpg", "png")


if condition == None:
    train_fns = [str(f) for f in Path(IMAGE_PATH_TRAIN).rglob("*.jpg")]
    val_fns = [str(f) for f in Path(IMAGE_PATH_VAL).rglob("*.jpg")]
else:
    # Load the JSON file
    import json
    with open(f'{BASE_PATH}/data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_train.json') as f:
        data = json.load(f)

    # Extract the desired fields from the data
    result = []
    for entry in data:
        if entry["attributes"]["weather"] not in condition["weather"]:
            continue
        if entry["attributes"]["timeofday"] not in condition["timeofday"]:
            continue
        if entry["attributes"]["scene"] not in condition["scene"]:
            continue
        result.append(os.path.join(IMAGE_PATH_TRAIN, entry["name"]))

    train_fns = result

    # Load the JSON file
    with open(f'{BASE_PATH}/data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json') as f:
        data = json.load(f)

    # Extract the desired fields from the data
    result = []
    for entry in data:
        if entry["attributes"]["weather"] not in condition["weather"]:
            continue
        if entry["attributes"]["timeofday"] not in condition["timeofday"]:
            continue
        if entry["attributes"]["scene"] not in condition["scene"]:
            continue            
        result.append(os.path.join(IMAGE_PATH_VAL, entry["name"]))

    # print(result[1])
    # print(len(result))

    val_fns = result

import random
if sample_limit != 0:
    train_fns = random.sample(train_fns, min(sample_limit, len(train_fns)))
    val_fns = random.sample(val_fns, min(int(sample_limit/10), len(val_fns)))

print(f"train img: {len(train_fns)}")
print(f"val img: {len(val_fns)}")

print(train_fns[1])
print(msk_fn_train(train_fns[1]))

## -----------------------
## DataLoaders
## -----------------------

# Define transformation to be applied to both images and labels
transform = transforms.Compose([
    transforms.Resize(output_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze())
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.Resize(output_size),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.squeeze())
])

# Create training and validation datasets and data loaders
train_dataset = BDD100kDataset(train_fns, msk_fn_train, split='train', transform=transform, transform2=transform2)
val_dataset = BDD100kDataset(val_fns, msk_fn_val, split='val', transform=transform, transform2=transform2)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"train_loader: {len(train_loader)}")
print(f"val_loader: {len(val_loader)}")

num_examples = {"trainset" : len(train_dataset), "testset" : len(val_dataset)}

img, lbl = train_dataset[1]
# print(img.shape, lbl.shape)
print(np.unique(lbl.numpy()))

## -----------------------
## Model Definition
## -----------------------

config_file = 'models/config.py'
# checkpoint_file = '/content/fcn_r50-d8_769x769_40k_drivable_bdd100k.pth'
# checkpoint_file = f'outputs/{checkpoint_file_name}' # defined above
# img_path = '/content/bdd100k/images/100k/train/0000f77c-62c2a288.jpg'

from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
print(mmcv.version.version_info)
from mmengine import runner

backbone = init_model(config_file, device='cpu')

from drivable.models.modelInterface import BDD100kModel


model = BDD100kModel(num_classes=3, backbone=backbone, size=output_size)
# model = load_checkpoint(model, "test-20230402_144923.pth", OUTPUT_DIR)
model.to(DEVICE)

## -----------------------
## Model Training Settings
## -----------------------

import datetime
import torch.optim as optim
from torch import nn
from lib.utils import load_checkpoint

optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()


from lib.runners import train_epoch, valid_epoch
from lib.utils import save_model
import time

def train_loop(model, train_data_loader, optimizer, criterion, epochs):
    start = time.time()
    print(f"start at {time.ctime()}")

    model.to(DEVICE).train()
    # epoch_i = 1
    max_score = 0
    for epoch_i in range(epochs):
        # training
        print(f"\nEpoch: {epoch_i} / {NUM_EPOCHS}\n-------------------------------")
        t1 = time.time()
        train_log = train_epoch(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataloader=train_data_loader,
            device=DEVICE
        )
        t2 = time.time()
        print(f"\nEpoch {epoch_i} / {NUM_EPOCHS}: {t2-t1} unit time")

    print("Elapsed time: {:.3f} min".format((time.time() - start) / 60.0))

def validate_loop(model, val_data_loader, criterion):
    model.eval()

    valid_logs = valid_epoch(
        model=model,
        criterion=criterion,
        dataloader=val_data_loader,
        device=DEVICE,
    )

    print(valid_logs)
    return valid_logs["Loss"], valid_logs["Score"]

class BDD100kClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_loop(model, train_loader, optimizer, criterion, NUM_EPOCHS)
        return self.get_parameters(config={}), num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, score = validate_loop(model, val_loader, criterion)
        return float(loss), num_examples["testset"], {"accuracy": float(score)}


fl.client.start_numpy_client(server_address="[::]:8080", client=BDD100kClient())