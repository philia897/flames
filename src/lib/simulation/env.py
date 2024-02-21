import os
from . import transforms as local_transforms

TASK_LIST = ['drivable', 'sem_seg', "ins_seg"]

def get_image_paths(project_dir:str, pkg_name:str="100k"):
    IMAGE_PATH = os.path.join(project_dir,"data", "bdd100k", "images", pkg_name)
    IMAGE_PATH_TRAIN = os.path.join(IMAGE_PATH, "train")
    IMAGE_PATH_VAL = os.path.join(IMAGE_PATH, "val")
    return IMAGE_PATH, IMAGE_PATH_TRAIN, IMAGE_PATH_VAL

def get_label_paths(project_dir:str, task_name:str, pkg_name:str="100k"):
    if task_name not in TASK_LIST:
        raise ValueError(f"Failed to get label paths: Task name must be within {TASK_LIST}")
    if task_name in ['drivable', 'sem_seg']:
        LABEL_PATH = os.path.join(project_dir, "data", "bdd100k", "labels", pkg_name, task_name, "masks")
        LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
        LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")
        return LABEL_PATH,LABEL_PATH_TRAIN, LABEL_PATH_VAL
    if task_name in ['ins_seg']:
        LABEL_PATH = os.path.join(project_dir, "data", "bdd100k", "labels", pkg_name, task_name, "bitmasks")
        LABEL_PATH_TRAIN = os.path.join(LABEL_PATH, "train")
        LABEL_PATH_VAL = os.path.join(LABEL_PATH, "val")
        return LABEL_PATH,LABEL_PATH_TRAIN, LABEL_PATH_VAL
    else:
        raise ValueError(f"Failed to get label paths: invalid task_name or task label missing.")

# def replace_255(label):
#         label[label==255] = 20
#         return label



def get_transforms(task_name:str, output_size, mode:str):
    '''Get the transform for data, mode=train/test'''
    transform_fn = getattr(local_transforms, f"{local_transforms.PREFIX}_{task_name}_{mode}")
    return transform_fn(output_size)

    # if task_name not in TASK_LIST:
    #     raise ValueError(f"Failed to get label paths: Task name must be within {TASK_LIST}")
    # if task_name == 'drivable':
    #     return transform_1, transform_2
    # if task_name == 'sem_seg':
    #     return transform_1, transform_2
    # else:
    #     raise ValueError(f"Failed to get label paths: invalid task_name or task label missing.")

    # transform_1 = transforms.Compose([
    #     transforms.Resize(output_size),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.squeeze())
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # transform_2 = transforms.Compose([
    #     transforms.Resize(output_size, transforms.InterpolationMode.NEAREST),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: (x.squeeze()*255).to(torch.int64))
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    # transform_3 = transforms.Compose([
    #     transforms.Resize(output_size, transforms.InterpolationMode.NEAREST),
    #     transforms.ToTensor(),
    #     transforms.Lambda(lambda x: (x.squeeze()*255).to(torch.int64)),
    #     transforms.Lambda(replace_255)
    #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])