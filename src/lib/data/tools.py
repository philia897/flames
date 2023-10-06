import os
import numpy as np
import torch
from mmseg.apis import init_model
from mmengine import runner
import json
import random
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List


from .bdd100kdataset import BDD100kDataset



def load_mmcv_checkpoint(config_file: str, checkpoint_file:str|None=None):
    backbone = init_model(config_file, device='cpu')
    if checkpoint_file != None:
        # runner.load_checkpoint(backbone, checkpoint_file)
        load_checkpoint(backbone, checkpoint_file)
    return backbone

def save_model(model, model_path, epoch: int=0, best_score: float=0):
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
        model_path,
    )
    print(f"model saved to {model_path}")


def load_checkpoint(model, checkpoint_path: str):
    """

    Args:
        checkpoint (path/str): Path to saved torch model
        model (object): torch model

    Returns:
        _type_: _description_
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    loaded_dict = checkpoint["state_dict"]
    sd = model.state_dict()
    for k in model.state_dict():
        if k in loaded_dict and sd[k].size() == loaded_dict[k].size():
            sd[k] = loaded_dict[k]
    loaded_dict = sd
    model.load_state_dict(loaded_dict)
    # print(
    #     "Loaded model:{} (Epoch={}, Score={:.3f})".format(
    #         model_name, checkpoint["epoch"], checkpoint["best_score"]
    #     )
    # )
    return model, checkpoint["epoch"], checkpoint['best_score']

def get_img_list_by_condition(condition: dict, attr_file: str, prefix_dir: str, max_num=0):
    '''
    Deprecated
    '''
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

def get_img_list_by_conditions(cond_list: list, attr_file: str, max_num=0):
    '''
    cond_list: List[[weather, scene, timeofday]]
    if empty, return all images by default
    '''
    def extract_attr(attr_dict:dict):
        return [attr_dict["weather"], attr_dict["scene"], attr_dict["timeofday"]]
    
    results = []
    with open(attr_file) as f:
        data = json.load(f)
        for entry in data:
            attr = extract_attr(entry['attributes'])
            if len(cond_list) == 0 or attr in cond_list:
                results.append(entry['name'])

    if max_num != 0:
        results = random.sample(results, min(max_num, len(results)))

    return results

def get_img_paths_by_conditions(cond_list: list, attr_file: str, prefix_dir: str,  max_num=0):
    '''
    cond_list: List[[weather, scene, timeofday]]
    if empty, return all images by default
    '''
    results = get_img_list_by_conditions(cond_list, attr_file, max_num)
    return [os.path.join(prefix_dir, fn) for fn in results]

def get_img_list_all(prefix_dir: str) -> list[str]:
    return [str(f) for f in Path(prefix_dir).rglob("*.jpg")]

def get_params(model) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_params(model: torch.nn.Module, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def get_dataloader(
        data: list, 
        batch_size: int, 
        workers: int, 
        img_transform: transforms.Compose, 
        lbl_transform: transforms.Compose,
        img_path:str, 
        lbl_path:str, 
        is_train=True, 
        classes_num=3
        ):
    
    msk_fn = lambda fn: fn.replace(img_path, lbl_path).replace(
        "jpg", "png"
    )

    if is_train:
        train_fns = data
        train_dataset = BDD100kDataset(
            train_fns, msk_fn, split="train", img_transform=img_transform, lbl_transform=lbl_transform, classes_num=classes_num
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True
        )
        return train_loader
    else:
        val_fns = data
        val_dataset = BDD100kDataset(
            val_fns, msk_fn, split="val", img_transform=img_transform, lbl_transform=lbl_transform, classes_num=classes_num
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False
        )
        return val_loader

# def get_dataloader(
#     data: dict, batch_size, workers: int, output_size: tuple, img_path:tuple, lbl_path:tuple, is_train=True
# ):
#     '''
#     img_path: tuple(img_path_train, img_path_val)
#     lbl_path: tuple(lbl_path_train, lbl_path_val)
#     '''
#     msk_fn_train = lambda fn: fn.replace(img_path[0], lbl_path[0]).replace(
#         "jpg", "png"
#     )
#     msk_fn_val = lambda fn: fn.replace(img_path[1], lbl_path[1]).replace(
#         "jpg", "png"
#     )

#     transform = transforms.Compose(
#         [
#             transforms.Resize(output_size),
#             transforms.ToTensor(),
#             transforms.Lambda(lambda x: x.squeeze())
#             # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ]
#     )

#     if is_train:
#         train_fns = data["train"]
#         train_dataset = BDD100kDataset(
#             train_fns,
#             msk_fn_train,
#             split="train",
#             transform=transform,
#             transform2=transform,
#         )
#         train_loader = DataLoader(
#             train_dataset, batch_size=batch_size, num_workers=workers, shuffle=True
#         )
#         return train_loader
#     else:
#         val_fns = data["val"]
#         val_dataset = BDD100kDataset(
#             val_fns, msk_fn_val, split="val", transform=transform, transform2=transform
#         )
#         val_loader = DataLoader(
#             val_dataset, batch_size=batch_size, num_workers=workers, shuffle=False
#         )
#         return val_loader