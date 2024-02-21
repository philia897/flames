import yaml
from lib.data import tools
import os
from lib.data.condparser import ConditionParserMode
import json

script_dir = os.path.dirname(os.path.abspath(__file__))  
os.chdir(script_dir)  

def cond_list_to_fname(l):
    return '-'.join(l).replace(" ", "_").replace("/", "_or_")

def cond_fname_to_list(s:str):
    return s.replace("_or_", "/").replace("_", " ").split('-')

pkg_name = "100k"
img_train_path = "/home/zekun/drivable/tmp/100k_yolo/images/train"
img_val_path = "/home/zekun/drivable/tmp/100k_yolo/images/val"
val_attr_file = f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_val.json"

all_conditions = tools.get_all_appeared_conditions(val_attr_file, ConditionParserMode.VALUE_LIST)

############################################
# subdataset = [
#     ['clear', 'city street', 'dawn/dusk'],
#     ['partly cloudy', 'highway', 'dawn/dusk'],
#     ['partly cloudy', 'residential', 'dawn/dusk'],
#     ['overcast', 'city street', 'dawn/dusk'],
#     ['rainy', 'highway', 'daytime'],
#     ['snowy', 'city street', 'dawn/dusk'],
#     ['rainy', 'residential', 'daytime'],
#     ['partly cloudy', 'residential', 'dawn/dusk'],
#     ['partly cloudy', 'highway', 'dawn/dusk'],
#     ['partly cloudy', 'city street', 'dawn/dusk'],
#     ['clear', 'parking lot', 'daytime']
# ]
# Specify the path for the YAML file
title = "no_snow_rain"
desc = "All the other conditions without snowy and rainy conditions"

############################################

yaml_file_path = f'./cfg/{title}.yaml'

train_lst = []
val_lst = []
num_map = {}
for condition in all_conditions:
    if "snowy" not in condition and "rainy" not in condition:
    # if condition in subdataset:
        print("Add ", condition)
        train_path = os.path.join(img_train_path, cond_list_to_fname(condition))
        val_path = os.path.join(img_val_path, cond_list_to_fname(condition))
        num_map[cond_list_to_fname(condition)] = f"Train: {len(os.listdir(train_path))}, Val: {len(os.listdir(val_path))}"
        train_lst.append(train_path)
        val_lst.append(val_path)

print("Condition Number:", len(train_lst))

# Data to be written to the YAML file
data = {
    "path": "/home/zekun/drivable/tmp/dataset",
    "train": train_lst,
    "val": val_lst,
    'names': {
        0: 'pedestrian',
        1: 'rider',
        2: 'car',
        3: 'truck',
        4: 'bus',
        5: 'train',
        6: 'motorcycle',
        7: 'bicycle',
        8: 'traffic light',
        9: 'traffic sign'
    }
}

# Write data to the YAML file
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data, yaml_file, default_flow_style=False)

with open('cfg_map.json', 'r+') as f:
    cfg_map = json.load(f)
    cfg_map[title] = {
        "desc": desc,
        "cfg_file": yaml_file_path,
        "details": num_map,
    }
    f.seek(0)
    json.dump(cfg_map, f, indent=4)
    # f.truncate()
    f.close()

print(f"YAML file '{yaml_file_path}' created successfully.")