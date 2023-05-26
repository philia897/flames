import os
import numpy as np
import torch
from mmseg.apis import init_model
from mmengine import runner
import json
import random
from pathlib import Path

def load_mmcv_checkpoint(config_file: str, checkpoint_file:str|None=None):
    backbone = init_model(config_file, device='cpu')
    if checkpoint_file != None:
        runner.load_checkpoint(backbone, checkpoint_file)
    return backbone

def save_model(model, epoch: int, best_score: float, model_name: str, output_dir: str):
    """_summary_

    Args:
        model (_type_): torch model
        epoch (int): running epoch
        best_score (float): running best score
        model_name (str): model's name
        output_dir (str): path to the output directory
    """
    torch.save(
        {"state_dict": model.state_dict(), "epoch": epoch, "best_score": best_score},
        os.path.join(output_dir, model_name),
    )
    print("model saved")


def load_checkpoint(model, model_name: str, model_dir: str = "./"):
    """

    Args:
        checkpoint (path/str): Path to saved torch model
        model (object): torch model

    Returns:
        _type_: _description_
    """
    fn_model = os.path.join(model_dir, model_name)
    checkpoint = torch.load(fn_model, map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    print(
        "Loaded model:{} (Epoch={}, Score={:.3f})".format(
            model_name, checkpoint["epoch"], checkpoint["best_score"]
        )
    )
    return model

def get_img_list_by_condition(condition: dict, attr_file: str, prefix_dir: str, max_num=0):
    with open(attr_file) as f:
        data = json.load(f)

    # Extract the desired fields from the data
    results = []
    for entry in data:
        if entry["attributes"]["weather"] not in condition["weather"]:
            continue
        if entry["attributes"]["timeofday"] not in condition["timeofday"]:
            continue
        if entry["attributes"]["scene"] not in condition["scene"]:
            continue
        results.append(os.path.join(prefix_dir, entry["name"]))

    
    if max_num != 0:
        results = random.sample(results, min(max_num, len(results)))
    
    return results

def get_img_list_all(prefix_dir: str) -> list[str]:
    return [str(f) for f in Path(prefix_dir).rglob("*.jpg")]