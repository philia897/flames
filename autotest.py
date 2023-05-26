import sys
import os
sys.path.append('.') # <= change path where you save code
BASE_PATH = "./"
OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import json
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

import warnings
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np
from lib.bdd100kdataset import BDD100kDataset
import lib.utils as utils

IMAGE_PATH = os.path.join("data", "bdd100k", "images", "100k")
IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")

LABEL_PATH = os.path.join("data", "bdd100k", "labels", "drivable", "masks")
LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")

msk_fn_train = lambda fn : fn.replace(IMAGE_PATH_TRAIN, LABEL_PATH_TRAIN).replace("jpg", "png")
msk_fn_val = lambda fn : fn.replace(IMAGE_PATH_VAL, LABEL_PATH_VAL).replace("jpg", "png")

def run_test(condition, log_file="bdd100k-eval-result.json"):
    print(condition)

    attr_file = "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json"
    val_fns = utils.get_img_list_by_condition(condition, attr_file, IMAGE_PATH_VAL)

    num_val_samples = len(val_fns)
    print("val samples:", num_val_samples)

    from torch.utils.data import DataLoader

    # Define transformation to be applied to both images and labels
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze())
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create training and validation datasets and data loaders
    val_dataset = BDD100kDataset(val_fns, msk_fn_val, split='val', transform=transform, transform2=transform)

    import models.modelInterface
    reload(models.modelInterface)
    from models.modelInterface import BDD100kModel

    model = BDD100kModel(num_classes=3, 
                         backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file), 
                         size=output_size)
    # model = load_checkpoint(model, checkpoint_file, OUTPUT_DIR)
    model.to(DEVICE)

    TMP_DIR = "tmp"

    import shutil
    from tqdm import tqdm
    shutil.rmtree(TMP_DIR)
    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(f"{TMP_DIR}/mask", exist_ok=True)
    os.makedirs(f"{TMP_DIR}/pred", exist_ok=True)

    from torch.utils.data import DataLoader

    # Define dataloader for validation dataset
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    # Define transform to resize predicted mask to original image size
    resize = transforms.Resize((720, 1280))

    # Iterate over validation dataset
    iterator = tqdm(val_dataloader, desc="Predicting")
    for i, (image, _) in enumerate(iterator):
        # Move data to GPU if available
        image = image.to(DEVICE)
        # print(image.device)

        # Get predicted mask from model
        with torch.no_grad():
            output = model(image)
            mask = output.argmax(dim=1)

        # Resize mask to original image size and convert to PIL image
        mask = resize(mask.unsqueeze(1))
        mask_pil = transforms.ToPILImage()(mask.to(torch.uint8).squeeze().cpu())

        # print(mask_pil.size)
        
        msk_path = msk_fn_val(val_fns[i])
        name = os.path.basename(msk_path)
        shutil.copy(msk_path, f"{TMP_DIR}/mask/")
        # Save resized mask to disk with the same name as the input image
        mask_pil.save(f'{TMP_DIR}/pred/{name}')
        # break

    # %%
    import subprocess
    command = "python3 -m bdd100k.bdd100k.eval.run -t drivable -g ./tmp/mask/ -r ./tmp/pred/ --out-file ./tmp/result.json"
    subprocess.run(command, shell=True, check=True)

    if save_test:
        with open(f'{TMP_DIR}/result.json') as f:
            data = json.load(f)
            print(f.read())

        with open(log_file, "r") as f:
            results = json.load(f)
            results.append({
                "model": STORE_MODEL_NAME, 
                "condition": condition,
                "val samples": num_val_samples, 
                "metrics": data})

        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)
            print(f"Saved to {log_file}")


if __name__ == "__main__":
    test_mode = True
    save_test = False
    log_file = "bdd100k-eval-results-deeplabv3.json"
    benchmark_model = "/home/zekun/drivable/outputs/deeplabv3_backbone_refined_benchmark-20230427_225904.pth"

    sample_limit = 0
    output_size = (512,1024)

    with open("./config.json", "r") as f:
        configs = json.load(f)
    if test_mode:
        STORE_MODEL_NAME = configs["stored_model_name"]
        config_file = configs["model_config_file"]
        condition = configs["condition"]
        checkpoint_file = configs['checkpoint_file']
        run_test(condition, log_file=log_file)
        STORE_MODEL_NAME = "deeplabv3_backbone_refined_benchmark"
        checkpoint_file = benchmark_model
        run_test(condition, log_file)
    else:
        for name, item in configs['available_model_list'].items():
            STORE_MODEL_NAME = name
            config_file = item["model_config_file"]
            condition = item["condition"]
            checkpoint_file = item['checkpoint_file']
            run_test(condition, log_file=log_file)
            STORE_MODEL_NAME = "deeplabv3_backbone_refined_benchmark"
            checkpoint_file = benchmark_model
            run_test(condition, log_file)

    print("Complete")


