# Note this file is just for temporary use (paper writing),
# It does not fit the real situation.
import json
import argparse
from typing import List, Callable
import torch
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

from lib.data.tools import (get_dataloader, get_img_paths_by_conditions, load_mmcv_checkpoint)
from lib.train.runners import PytorchRunner
from lib.train.metrics import IoUMetricMeter
from models.modelInterface import BDD100kModel
from lib.simulation.env import (get_image_paths, get_label_paths, get_transforms)
from lib.data.condparser import BDD100KConditionParser, ConditionParserMode
from lib.utils.dbhandler import JsonDBHandler, Items, EvalResultItem

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def eval_on_one_condition(
        condition:List,
        model:torch.nn.Module,
        criterion:Callable,
        val_attr_file:str,
        transform,
        img_path,
        lbl_path,
        class_num,
        device
        ):
    '''
    Evaluate the model on the specific condition
    Return eval_log:Dict[metric, Scalar], len(samples)
    '''
    images = get_img_paths_by_conditions([condition], val_attr_file, img_path)
    val_loader = get_dataloader(images, 4, transform, img_path, lbl_path, False, class_num)
    metric_meter = IoUMetricMeter(class_num)
    runner = PytorchRunner(
        optimizer=None, 
        criterion=criterion, 
        train_loader=None,
        val_loader=val_loader,
        metric_meter=metric_meter,
        device=device,
        verbose=True)
    
    eval_log = runner.validate(model)
    metric_log = {}
    for k,v in eval_log.items():
        if isinstance(v, (float, int, bool, bytes, str)):
            metric_log.setdefault(k,v)
    return metric_log, len(images)


if __name__ == '__main__':
    pkg_name = '10k'
    output_size = (512, 1024)
    num_classes = 19
    task_name = "sem_seg"

    parser = argparse.ArgumentParser(description="Evaluation Simulation")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--attr_file_val", type=str, default=f"/home/zekun/drivable/data/bdd100k/labels/{pkg_name}/bdd100k_labels_images_attributes_val.json")
    parser.add_argument("--cls_id", "-c", type=str, default="0")
    args = parser.parse_args()

    IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/", pkg_name)
    LABEL_PATH, LABEL_PATH_TRAIN, LABEL_PATH_VAL = get_label_paths("/home/zekun/drivable/", task_name, pkg_name)
    val_transform = get_transforms(task_name, output_size, "test")

    cls_id = args.cls_id
    handler = JsonDBHandler(os.path.join(args.output_dir, "db"))
    modelinfo = handler.read(cls_id, Items.MODEL_INFO)
    conditionclass = handler.read(cls_id, Items.CONDITION_CLASS)

    conditions:list = conditionclass.conditions
    val_attr_file = args.attr_file_val

    if len(conditions) == 0:
        val_cond_set = set()
        with open(val_attr_file, 'r') as f:
            attr_data = json.load(f)
            for entry in attr_data:
                attr = BDD100KConditionParser.convert(entry['attributes'],
                                ConditionParserMode.STRING_KEY)
                val_cond_set.add(attr)
        conditions = [BDD100KConditionParser.convert(v, ConditionParserMode.VALUE_LIST) for v in val_cond_set]

    print(conditions)
    print(f"Total length of conditions: {len(conditions)}")

    model = BDD100kModel(load_mmcv_checkpoint(modelinfo.model_config_file, modelinfo.checkpoint_file))

    eval_rst = {}
    for condition in conditions:
        eval_log, sz = eval_on_one_condition(
            condition=condition,
            model=model,
            criterion=torch.nn.CrossEntropyLoss(ignore_index=255),
            val_attr_file=val_attr_file,
            transform=val_transform,
            img_path=IMAGE_PATH_VAL,
            lbl_path=LABEL_PATH_VAL,
            class_num=num_classes,
            device=DEVICE
        )
        eval_rst[
            BDD100KConditionParser.convert(condition, ConditionParserMode.STRING_KEY)
            ] = {
                "sample_num": sz,
                "metrics": eval_log
            }
    handler.update(EvalResultItem(eval_rst), Items.EVAL_RESULT, cls_id)





