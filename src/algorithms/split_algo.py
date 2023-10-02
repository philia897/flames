import lib.data.tools as utils
import os
from torchvision import transforms
from libdata.bdd100kdataset import BDD100kDataset
from models.modelInterface import BDD100kModel
from torch.utils.data import DataLoader
import torch
from torch import nn
from lib.train.runners import valid_epoch
import json
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = os.path.join("data", "bdd100k", "images", "100k")
IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")

LABEL_PATH = os.path.join("data", "bdd100k", "labels", "drivable", "masks")
LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")

def calculate_std(scores):
    # N = 0
    # mu = 0
    # sigma = 0
    # for s in scores:
    #     N += s[0]
    #     mu += s[1]*s[0]
    # mu = mu/N
    # for s in scores:
    #     sigma += (s[1]-mu)**2 * s[0]
    # sigma = sigma/N
    # return np.sqrt(sigma)
    ll = [score[1] for score in scores]
    return np.std(ll)

def run_iterative_tests():
    
    val_attr_file = "data/bdd100k/labels/drivable/bdd100k_labels_images_attributes_val.json"

    msk_fn_val = lambda fn: fn.replace(IMAGE_PATH_VAL, LABEL_PATH_VAL).replace(
        "jpg", "png"
    )

    transform = transforms.Compose(
        [
            transforms.Resize(output_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.squeeze())
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
    
    results = []
    scores = []
    for i in weather:
        for j in timeofday:
            for k in scene:
    # Extract the desired fields from the data
                condition = {"weather": [i], "timeofday": [j], "scene": [k]}
                val_imgs = utils.get_img_list_by_condition(condition, val_attr_file, IMAGE_PATH_VAL)
                if(len(val_imgs)>=min_sample_n):
                    print(f"{i},{j},{k}:",len(val_imgs))
                    val_dataset = BDD100kDataset(
                        val_imgs, msk_fn_val, split="val", transform=transform, transform2=transform
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=4, num_workers=4, shuffle=False
                    )
                    valid_logs = valid_epoch(
                        model=model,
                        criterion=criterion,
                        dataloader=val_loader,
                        device=DEVICE,
                    )
                    condition2 = {"weather": condition["weather"][0], 
                                  "timeofday": condition["timeofday"][0], 
                                  "scene": condition["scene"][0]}
                    results.append({
                        "condition": condition2,
                        "val samples": len(val_imgs), 
                        "metrics": valid_logs['Score']})
                    scores.append((len(val_imgs),valid_logs["Score"]))
    
    print("STD:", calculate_std(scores))
    with open('tmp_test.json', "w") as f:
        json.dump(results, f)
    if save_result:
        with open(log_file, "r") as f:
            logjson = json.load(f)
            logjson.append(
                {"model": STORE_MODEL_NAME, "Std": calculate_std(scores), "results": results}
            )
        with open(log_file, "w") as f:
            json.dump(logjson, f, indent=4)

def show_table():
    logs = []
    with open("tmp_test.json", 'r') as f:
        logs = json.load(f)

    from prettytable import prettytable

    table = prettytable.PrettyTable()
    table.field_names = ["weather", "timeofday", "scene","samples","mIoU"]
    tt = 0
    for log in logs:
        table.add_row([log['condition']['weather'], log['condition']['timeofday'], 
                    log['condition']['scene'], log['val samples'], 
                    "{:.2f}".format(log['metrics'])])
        tt += log['val samples']

    print("total samples:", tt)    
    table.sortby = "mIoU"
    table.reversesort = True
    return table

def kmeans_cluster(table):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # print(table.get_csv_string())
    with open('table.csv', 'w') as f:
        f.write(table.get_csv_string())
    # create a pandas DataFrame
    df = pd.read_csv('table.csv')

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # select fields to cluster
    X = df[['weather', 'timeofday', 'scene', 'mIoU']]

    # convert categorical variables to numeric
    X.loc[:,'weather'] = pd.factorize(X['weather'])[0]
    X.loc[:,'timeofday'] = pd.factorize(X['timeofday'])[0]
    X.loc[:,'scene'] = pd.factorize(X['scene'])[0]

    # scale numerical variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(X_scaled)
    # cluster the data
    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=20, random_state=0)
    pred_y = kmeans.fit_predict(X_scaled)

    # add cluster labels to dataframe
    df['Cluster'] = pred_y
    print(df)

if __name__ == '__main__':
    test_mode = True
    save_result = True
    log_file = "condition-bunch_test-results.json"
    output_size = (512,1024)
    min_sample_n = 20
    with open("./config.json", "r") as f:
        configs = json.load(f)
    if test_mode:
        STORE_MODEL_NAME = configs["stored_model_name"]
        config_file = configs["model_config_file"]
        condition_tmp = configs["condition"]
        checkpoint_file = configs['checkpoint_file']

        weather = condition_tmp['weather']
        timeofday =  condition_tmp['timeofday']
        scene = condition_tmp['scene']

        model = BDD100kModel(
                num_classes=3,
                backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file),
                size=output_size,
            )
        model.to(DEVICE).eval()

        criterion = nn.CrossEntropyLoss()
        run_iterative_tests()

        table = show_table()
        kmeans_cluster(table)
    else:
        with open(log_file, "w") as f: # Clean the log file
            json.dump([], f, indent=4)
        for (name, config) in configs['available_model_list'].items():
            STORE_MODEL_NAME = name
            config_file = config["model_config_file"]
            condition_tmp = config["condition"]
            checkpoint_file = config['checkpoint_file']

            weather = condition_tmp['weather']
            timeofday =  condition_tmp['timeofday']
            scene = condition_tmp['scene']

            model = BDD100kModel(
                    num_classes=3,
                    backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file),
                    size=output_size,
                )
            model.to(DEVICE).eval()

            criterion = nn.CrossEntropyLoss()
            run_iterative_tests()