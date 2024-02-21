import pandas as pd
import argparse
import os
from typing import List
import torch
from PIL import Image
from torchvision.transforms import ToTensor, Resize

from algorithms.analyze import get_all_leaf_nodes, _calculate_avg_metrics, get_avg_metrics
from lib.data.tools import load_mmcv_checkpoint, get_img_paths_by_conditions
from lib.simulation.env import get_image_paths, get_transforms
from models.modelInterface import BDD100kModel
from lib.train.metrics import IoUMetricMeter

from lib.utils.dbhandler import (
    DBHandler,
    JsonDBHandler,
    Items,
    EvalResultItem,
    ConditionClassItem,
    ModelInfoItem,
)


IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL = get_image_paths("/home/zekun/drivable/", "10k")
ATTR_FILE = "/home/zekun/drivable/data/bdd100k/labels/10k/bdd100k_labels_images_attributes_train.json"
# Calculate the similarity of two pytorch model
# def _param_diff(model1:torch.nn.Module, model2:torch.nn.Module):  
#     total_diff = 0
#     for param1, param2 in zip(model1.parameters(), model2.parameters()):  
#         diff = torch.norm(param1 - param2)  
#         total_diff += diff  
#     return total_diff

# def _compare_diff_two_concls2(concls1, concls2, handler:JsonDBHandler):
#     modelinfo1 = handler.read(concls1, Items.MODEL_INFO)
#     modelinfo2 = handler.read(concls2, Items.MODEL_INFO)
#     model1 = load_mmcv_checkpoint(modelinfo1.model_config_file, modelinfo1.checkpoint_file)
#     model2 = load_mmcv_checkpoint(modelinfo2.model_config_file, modelinfo2.checkpoint_file)
#     return _param_diff(model1, model2)

def _prepare_eval(concls_id, handler:JsonDBHandler):
    '''
    Prepare for the requirements for one concls
    return: model, img_list, 
    '''
    modelinfo = handler.read(concls_id, Items.MODEL_INFO)
    conditions = handler.read(concls_id, Items.CONDITION_CLASS).conditions
    model = BDD100kModel(backbone=load_mmcv_checkpoint(modelinfo.model_config_file, modelinfo.checkpoint_file))
    img_list = get_img_paths_by_conditions(conditions, ATTR_FILE, IMAGE_PATH_TRAIN, 8)
    return model, img_list

def _compare_diff_two_concls(model1, model2, sample_lst, transform, device, meter:IoUMetricMeter):
    meter.reset()
    model1.to(device).eval()
    model2.to(device).eval()
    for image_name in sample_lst:
        # Load the image and preprocess it
        image = Image.open(image_name)
        image_tensor = transform(image)
        # image_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        # Pass the image through the model and obtain the predicted output
        with torch.no_grad():
            output = model1(image_tensor)
            output1 = output
            output = model2(image_tensor)
            output2 = output
            meter.calculate_and_log(torch.argmax(output1, dim=1), output2)
    return meter.details()

def _create_new_merged_concls(concls1, concls2, handler:JsonDBHandler):
    new_modelinfo:ModelInfoItem = handler.read(concls1, Items.MODEL_INFO)
    similar_modelinfo:ModelInfoItem = handler.read(concls2, Items.MODEL_INFO)
    if similar_modelinfo.parents == new_modelinfo.parents:  # No need for merge
        return None
    new_modelinfo.children = []
    new_modelinfo.parents = [concls1, concls2]
    new_modelinfo.meta = {
        "class_num": new_modelinfo.meta.get("class_num"),
        "output_size": new_modelinfo.meta.get("output_size"),
        "retrained_times": 0
    }

    conditions1 = handler.read(concls1, Items.CONDITION_CLASS).conditions
    conditions2 = handler.read(concls2, Items.CONDITION_CLASS).conditions
    conditions_union = conditions1 + conditions2
    return handler.create(new_modelinfo, ConditionClassItem(conditions_union))

def merge_conditions(concls_alternative:str, handler:JsonDBHandler, only_try=False):
    '''
    Find the most similar model to the concls_alternative
    If they are brothers, then no need to merge, else pair them to one new child.
    return new_child_id, similar_concls_id
    '''
    leaf_nodes = get_all_leaf_nodes(handler)
    leaf_nodes.remove(concls_alternative)
    modelinfo = handler.read(concls_alternative, Items.MODEL_INFO)
    model1, img_list1 = _prepare_eval(concls_alternative, handler)
    meter = IoUMetricMeter(modelinfo.meta['class_num'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_transforms("sem_seg", modelinfo.meta['output_size'], "test")
    similarity = []
    for node in leaf_nodes:
        model2, img_list2 = _prepare_eval(node, handler)
        logs = _compare_diff_two_concls(
            model1, model2, img_list1+ img_list2, transform, device, meter)
        similarity.append(logs.get('mIoU'))
        print(concls_alternative, node, round(logs.get('pAcc'), 4), round(logs.get('mIoU'), 4))
    idx = similarity.index(max(similarity))
    most_similar = leaf_nodes[idx]

    if not only_try:
        new_id = _create_new_merged_concls(concls_alternative, most_similar, handler)
    else:
        new_id = "try_only"
    return new_id, most_similar

def _get_avg_parent_grade(grades, parents):
    return _calculate_avg_metrics(grades, parents)

def test_merged_concls(child_concls_id:str, handler:JsonDBHandler, metric:str):

    '''
    Test the parents' average grade and compare it to the child's grade
    return [child_higher=True/False], child_grade, parent_avg_grade
    '''
    modelinfo: ModelInfoItem = handler.read(child_concls_id, Items.MODEL_INFO)
    parents = modelinfo.parents
    parents_grades = {}
    for parent in parents:
        avg_grade = get_avg_metrics(parent, handler)
        parents_grades[parent] = avg_grade
    avg_grade = _get_avg_parent_grade(parents_grades, parents)
    child_grade = get_avg_metrics(child_concls_id, handler)
    child_higher = False
    if avg_grade['metrics'][metric] < child_grade['metrics'][metric]:
        child_higher = True
    return child_higher, child_grade, avg_grade

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge Conditions")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--cls_id", "-c", type=str, default="0")
    args = parser.parse_args()

    concls_id = args.cls_id
    handler = JsonDBHandler(os.path.join(args.output_dir, "db"))

    print(merge_conditions(concls_id, handler, only_try=True))



