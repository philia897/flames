import lib.utils as utils
from models.modelInterface import BDD100kModel
import torch
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor, Resize
import json
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def iou_score(outputs, targets):
    # Get predicted and ground truth class labels
    outputs = outputs.argmax(1).cpu().numpy()
    targets = targets.argmax(1).cpu().numpy()

    # Calculate IoU for drivable and non-drivable classes
    iou_total = 0
    num_classes = 2
    scores = [0, 0]
    intersections = [0, 0]
    unions = [0, 0]
    for i, c in enumerate([0,1]):
        # Exclude background class with label 2
        # if c == 2:
        #     continue
        mask_pred = (outputs == c)
        mask_gt = (targets == c)
        intersection = (mask_pred & mask_gt).sum()
        union = (mask_pred | mask_gt).sum()
        if union > 0:
            iou = intersection / union
            iou_total += iou
            scores[i] += iou
            intersections[i] += intersection
            unions[i] += union

    # Calculate mean IoU over drivable and non-drivable classes
    miou = iou_total / num_classes
    return miou, scores, intersections, unions

def cal_similarity(output_size, config1, config2):

    config1_file = config1['model_config_file']
    config2_file = config2['model_config_file']

    model1_file = config1['checkpoint_file']
    model2_file = config2['checkpoint_file']

    condition1 = config1['condition']
    condition2 = config2['condition']

    model1 = BDD100kModel(
            num_classes=3,
            backbone=utils.load_mmcv_checkpoint(config1_file, model1_file),
            size=output_size,
        )
    model1.to(DEVICE).eval()
    model2 = BDD100kModel(
            num_classes=3,
            backbone=utils.load_mmcv_checkpoint(config2_file, model2_file),
            size=output_size,
        )
    model2.to(DEVICE).eval()

    prefix = "data/bdd100k/images/100k/val/"
    attr_file = 'data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json'

    img_list1 = utils.get_img_list_by_condition(condition1, attr_file, prefix, 10)
    img_list2 = utils.get_img_list_by_condition(condition2, attr_file, prefix, 10)
    img_list = img_list1 + img_list2

    similarity = 0
    for image_name in img_list:
        # Load the image and preprocess it
        image = Image.open(image_name)
        image_tensor = ToTensor()(image)
        image_tensor = Resize(output_size)(image_tensor)
        # image_tensor = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(DEVICE)

        # Pass the image through the model and obtain the predicted output
        with torch.no_grad():
            output = model1(image_tensor)
            output1 = output
            output = model2(image_tensor)
            output2 = output
            iou, _, _, _ = iou_score(output1, output2)
            similarity += iou
            # print(image_name, ":", iou)
    similarity = similarity/len(img_list)
    return similarity

if __name__ == '__main__':
    test_mode = False
    output_size = (512,1024)
    if test_mode:
        config1 = {
            "checkpoint_file": "/home/zekun/drivable/outputs/deeplabv3_backbone_rainysnowy_all_highway_fl10-20230509_204409.pth",
            "model_config_file": "/home/zekun/drivable/models/config-deeplabv3.py",
            "condition": {
                "weather": [
                    "rainy",
                    "snowy"
                ],
                "timeofday": [
                    "daytime",
                    "undefined",
                    "night",
                    "dawn/dusk"
                ],
                "scene": [
                    "highway"
                ]
            }
        }
        config2 = {
            "checkpoint_file": "/home/zekun/drivable/outputs/deeplabv3_backbone_cleartofoggy_all_highway_fl10-20230510_134557.pth",
            "model_config_file": "/home/zekun/drivable/models/config-deeplabv3.py",
            "condition": {
                "weather": [
                    "clear",
                    "overcast",
                    "undefined",
                    "partly cloudy",
                    "foggy"
                ],
                "timeofday": [
                    "daytime",
                    "undefined",
                    "night",
                    "dawn/dusk"
                ],
                "scene": [
                    "highway"
                ]
            }
        }
        print(cal_similarity(output_size, config1, config2))
    else:
        with open("./config.json", "r") as f:
            configs =  json.load(f)
        model_list = ["deeplabv3_backbone_cleartofoggy_all_highway_fl10", 
                      "deeplabv3_backbone_rainysnowy_all_highway_fl10",
                      "deeplabv3_backbone_cleartofoggy_dayundefined_tunneltogas_fl10",
                      "deeplabv3_backbone_cleartofoggy_nightdawn_tunneltogas_fl10",
                      "deeplabv3_backbone_rainy_all_nohighway_fl10",
                      "deeplabv3_backbone_cleartofoggy_all_city_fl10",
                      "deeplabv3_backbone_snowy_all_city_fl10",
                      "deeplabv3_backbone_snowy_all_residential_fl10"]
        sim_list = []
        # for (name1, config1) in configs['available_model_list'].items():
        #     for (name2, config2) in configs['available_model_list'].items():
        #         if name1 != name2:
        #             sim_item = {
        #                 "model1": name1,
        #                 "model2": name2,
        #                 "similarity": cal_similarity(output_size, config1, config2)
        #             }
        #             sim_list.append(sim_item)
        #             print(sim_item)
        for name1 in model_list:
            for name2 in model_list:
                if name1 != name2:
                    sim_item = {
                        "model1": name1,
                        "model2": name2,
                        "similarity": cal_similarity(output_size, 
                                                     configs['available_model_list'][name1], 
                                                     configs['available_model_list'][name2])
                    }
                    sim_list.append(sim_item)
                    print(sim_item)
        with open("./model-similarity-test.json", "w") as f:
            json.dump(sim_list, f)
        