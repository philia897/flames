import warnings
warnings.filterwarnings("ignore")
from torchvision.transforms import v2
import torch

PREFIX = "create_transform"

def create_transform_drivable_train(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.RandomResize(720, 1440, interpolation=v2.InterpolationMode.NEAREST, antialias=False),
        v2.RandomCrop(size=output_size),
        v2.RandomHorizontalFlip(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def create_transform_drivable_test(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.Resize(size=(720, 1280), antialias=False),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=output_size),
        # v2.RandomPhotometricDistort(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def create_transform_sem_seg_train(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.RandomResize(720, 1440, interpolation=v2.InterpolationMode.NEAREST, antialias=False),
        v2.RandomCrop(size=output_size),
        v2.RandomHorizontalFlip(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def create_transform_sem_seg_test(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.Resize(size=(720, 1280), antialias=False),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=output_size),
        # v2.RandomPhotometricDistort(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def create_transform_ins_seg_train(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.RandomResize(720, 1440, interpolation=v2.InterpolationMode.NEAREST, antialias=False),
        v2.RandomCrop(size=output_size),
        v2.RandomHorizontalFlip(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

def create_transform_ins_seg_test(output_size):
    transform = v2.Compose([
        v2.Compose([v2.ToImageTensor(), v2.ConvertImageDtype()]),
        v2.Resize(size=(720, 1280), antialias=False),
        v2.RandomHorizontalFlip(),
        v2.RandomCrop(size=output_size),
        # v2.RandomPhotometricDistort(),
        v2.Lambda(lambda x: x.squeeze()),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform