import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datetime

from lib.simulation.env import get_image_paths
from classification.utils import (
    get_img_list_by_condition,
    metric_fn,
    train_epoch,
    valid_epoch,
    save_model_info,
    load_model,
)
from classification.dataset import ConditionDataset

IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/")
train_attr_file = "/home/zekun/drivable/data/bdd100k/labels/100k/bdd100k_labels_images_attributes_train.json"
val_attr_file = (
    "/home/zekun/drivable/data/bdd100k/labels/100k/bdd100k_labels_images_attributes_val.json"
)
output_dir = "/home/zekun/drivable/outputs/classification"

# all_condition = {
#     "weather": [
#         "clear",
#         "overcast",
#         "partly cloudy",
#         "foggy",
#         "rainy",
#         "snowy",
#         "undefined",
#     ],
#     "timeofday": ["daytime", "night", "dawn/dusk", "undefined"],
#     "scene": [
#         "tunnel",
#         "residential",
#         "parking lot",
#         "city street",
#         "gas stations",
#         "highway",
#         "undefined",
#     ],
# }

condition = {
    "weather": [
        "clear",
        "overcast",
        "partly cloudy",
        "foggy",
        "rainy",
        "snowy",
        "undefined",
    ],
    "timeofday": ["daytime", "night", "dawn/dusk"],
    "scene": [
        "tunnel",
        "residential",
        "parking lot",
        "city street",
        "gas stations",
        "highway",
        "undefined",
    ],
}

model_type = "resnet18"
condition_attr = "timeofday"
num_classes = len(condition[condition_attr])
num_epochs = 10

cls_to_idx = {label: idx for idx, label in enumerate(condition[condition_attr])}
idx_to_cls = {idx: label for idx, label in cls_to_idx.items()}

train_imgs = get_img_list_by_condition(condition, train_attr_file, IMAGE_PATH_TRAIN)
val_imgs = get_img_list_by_condition(condition, val_attr_file, IMAGE_PATH_VAL)

# Define the transformations for data preprocessing and augmentation
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Create the dataset using CustomDataset
train_dataset = ConditionDataset(
    train_imgs, condition_attr, cls_to_idx, transform=transform
)
val_dataset = ConditionDataset(
    val_imgs, condition_attr, cls_to_idx, transform=transform
)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

model = load_model(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
stored_model_name = f"{condition_attr}-{timestamp}.pth"

max_acc = 0
details = []
for epoch in range(num_epochs):
    train_epoch(model, optimizer, criterion, train_loader, metric_fn, device)
    logs = valid_epoch(model, criterion, val_loader, num_classes, metric_fn, device)
    print(f"************ Valid in epoch {epoch} ******************")
    for k,v in logs.items():
        print(f"{k}: {v}")
    if logs['Score'] > max_acc:
        print(f"[MODEL SAVE] Acc {logs['Score']} > {max_acc}, save the model and update max acc")
        torch.save(model.state_dict(), os.path.join(output_dir, "models", stored_model_name))
        max_acc = logs['Score']
        details = logs['Details']
    print(f"******************************************************")
    if max_acc > 0.9 or epoch == num_epochs-1:
        print(f"Done with score {max_acc} : {details}")
        break

save_model_info(
    os.path.join(output_dir, "logs", "model_log.json"),
    stored_model_name,
    model_type,
    condition,
    condition_attr,
    max_acc,
    details,
    idx_to_cls,
)