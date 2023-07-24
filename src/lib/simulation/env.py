import os

TASK_LIST = ['drivable', 'sem_seg']

def get_image_paths(project_dir:str):
    IMAGE_PATH = os.path.join(project_dir,"data", "bdd100k", "images", "100k")
    IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
    IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")
    return IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL

def get_label_paths(project_dir:str, task_name:str):
    if task_name not in TASK_LIST:
        raise ValueError(f"Failed to get label paths: Task name must be within {TASK_LIST}")
    if task_name in ['drivable', 'sem_seg']:
        LABEL_PATH = os.path.join(project_dir, "data", "bdd100k", "labels", task_name, "masks")
        LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
        LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")
        return LABEL_PATH,LABEL_PATH_TRAIN, LABEL_PATH_VAL
    else:
        raise ValueError(f"Failed to get label paths: invalid task_name or task label missing.")

def get_model_additional_configs(task_name:str):
    if task_name not in TASK_LIST:
        raise ValueError(f"Failed to get label paths: Task name must be within {TASK_LIST}")
    if task_name == 'drivable':
        return {"num_classes": 3}
    if task_name == 'sem_seg':
        return {"num_classes": 19}
    else:
        raise ValueError(f"Failed to get label paths: invalid task_name or task label missing.")