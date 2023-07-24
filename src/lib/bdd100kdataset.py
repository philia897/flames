import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from importlib import reload
import matplotlib.pyplot as plt
import numpy as np

class BDD100kDataset(Dataset):
    def __init__(self, data_fns, msk_fn, split='train', transform=None, transform2=None, classes_num=3):
        super(BDD100kDataset, self).__init__()
        assert split in ['train', 'val', 'test'], "Invalid split provided. Expected 'train', 'val' or 'test'"
        
        self.image_fns = data_fns
        self.msk_fn = msk_fn
        self.split = split
        self.transform = transform
        self.transform2 = transform2
        self.classes_num = classes_num
        
        # Check that image file names and label file names match
        # assert len(self.image_file_names) == len(self.label_file_names), "Number of images and labels do not match"
        self.num_samples = len(self.image_fns)
    
    def __getitem__(self, index):
        # Load image and label
        image = Image.open(self.image_fns[index])
        label = Image.open(self.msk_fn(self.image_fns[index]))
        
        # Apply transformations if provided
        if self.transform is not None:
            image = self.transform(image)

        if self.transform2 is not None:
            label = self.transform2(label)
        
        # Convert label to tensor and convert from RGB to single channel (grayscale)
        label = torch.tensor(np.array(label)*255, dtype=torch.int64)
        label = torch.nn.functional.one_hot(label, num_classes=self.classes_num).permute(2, 0, 1).float()
        
        return image, label
    
    def __len__(self):
        return self.num_samples