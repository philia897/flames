import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from lib.data import tools
from lib.data.tools import get_dataloader
from lib.train.runners import PytorchRunner
from lib.train.metrics import IoUMetricMeter
from models.modelInterface import BDD100kModel
from lib.simulation.env import (get_image_paths, get_label_paths, get_transforms)

import segmentation_models_pytorch as smp


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pkg_name = "10k" # 100k or 10k
output_size = (512,1024)
task_name = "sem_seg" # "drivable", "sem_seg"
config_file = "/home/zekun/drivable/src/models/sem_seg/config-deeplabv3plus-sem_seg.py"

# checkpoint_file = "/home/zekun/drivable/outputs/semantic/test/deeplabv3+_r50-d8_512x1024_80k_sem_seg_bdd100k.pth"
# new_checkpoint_file = "/home/zekun/drivable/outputs/semantic/test/deeplabv3+_r50-d8-test.pth"
checkpoint_file = "/home/zekun/drivable/outputs/semantic/test/unet-resnet34_backbone-test.pth"
new_checkpoint_file = "/home/zekun/drivable/outputs/semantic/test/unet-resnet34_backbone-test.pth"

train_attr_file = f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_train.json"
val_attr_file = f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_val.json"

batchsize = 16
learn_rate = 2e-4  # 1e-4
epochs = 30
num_workers = 0

IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/", pkg_name)
LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("/home/zekun/drivable/", task_name, pkg_name)
train_transform = get_transforms(task_name, output_size, "train")
val_transform = get_transforms(task_name, output_size, "test")
classes_num = 19

conditions = []

from lib.data.condparser import BDD100KConditionParser, ConditionParserMode

all_conditions = tools.get_all_appeared_conditions(val_attr_file, ConditionParserMode.VALUE_LIST)

conditions = []
for condition in all_conditions:
    if "snowy" in condition or "rainy" in condition:
        conditions.append(condition)

# ## Data loaders

train_fns = tools.get_img_paths_by_conditions(conditions, train_attr_file, IMAGE_PATH_TRAIN)
val_fns = tools.get_img_paths_by_conditions(conditions, val_attr_file, IMAGE_PATH_VAL)

train_loader = get_dataloader(
    train_fns,
    batch_size=batchsize,
    workers=num_workers,
    transform=train_transform,
    img_path=IMAGE_PATH_TRAIN,
    lbl_path=LABEL_PATH_TRAIN,
    is_train=True,
)
val_loader = get_dataloader(
    val_fns,
    batch_size=batchsize,
    workers=num_workers,
    transform=val_transform,
    img_path=IMAGE_PATH_VAL,
    lbl_path=LABEL_PATH_VAL,        
    is_train=False,
)

print("train data size:", len(train_fns))
print("val data size:", len(val_fns))

# ## Model initialization
# model = BDD100kModel(backbone=tools.load_mmcv_checkpoint(config_file, checkpoint_file))
model = BDD100kModel(backbone=smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=classes_num,                      # model output channels (number of classes in your dataset)
))
tools.load_checkpoint(model.backbone, checkpoint_file)

# optimizer = optim.Adam(model.parameters(), lr=learn_rate)
optimizer = optim.SGD(model.parameters(), lr=learn_rate, momentum=0.9, weight_decay=0.0005)
criterion = nn.CrossEntropyLoss(ignore_index=255)

runner = PytorchRunner(optimizer, criterion, train_loader, val_loader, IoUMetricMeter(classes_num), DEVICE, True)

print("************* Initial State *************")
valid_logs = runner.validate(model)
print(valid_logs)
print("*****************************************")

best_metric = valid_logs["pAcc"]
# runner.train(model, epochs)
for i in range(epochs):
    train_log = runner.train(model)
    print(f"Epoch {i} Train: loss={train_log['loss']}, pAcc={train_log['pAcc']}, mIoU={train_log['mIoU']}")
    train_log = runner.validate(model)
    print(f"Epoch {i} Validate: loss={train_log['loss']}, pAcc={train_log['pAcc']}, mIoU={train_log['mIoU']}")
    if train_log['pAcc'] >= best_metric:
        tools.save_model(model.backbone, new_checkpoint_file, i, valid_logs['mIoU'])
        print(f"Metric {train_log['pAcc']} >= {best_metric}, Model saved to {new_checkpoint_file}")
        best_metric = train_log['pAcc']

valid_logs = runner.validate(model)
print(valid_logs)
for k,v in valid_logs.items():
    print(k, "=", v)