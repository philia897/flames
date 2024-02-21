from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from typing import Callable, List
import torch

class BDD100kDataset(Dataset):
    lbl_formatter = v2.Lambda(lambda x: (x.squeeze()*255).to(torch.int64))

    def __init__(self, data_fns:List[str], msk_fn:Callable[[str], str], transform:v2.Compose=None):
        super(BDD100kDataset, self).__init__()
        # assert split in ['train', 'val', 'test'], "Invalid split provided. Expected 'train', 'val' or 'test'"
        
        self.image_fns = data_fns
        self.msk_fn = msk_fn
        self.transform = transform
        
        # Check that image file names and label file names match
        # assert len(self.image_file_names) == len(self.label_file_names), "Number of images and labels do not match"
        self.num_samples = len(self.image_fns)
    
    def __getitem__(self, index):
        # Load image and label
        image = Image.open(self.image_fns[index])
        label = Image.open(self.msk_fn(self.image_fns[index]))
        
        # Apply transformations if provided
        if self.transform is not None:
            image, label = self.transform(image, label)

        label = self.lbl_formatter(label)  # convert the label range from [0, 1] float to [0, 255] integer

        # Convert label to tensor and convert from RGB to single channel (grayscale)
        # label = torch.nn.functional.one_hot(label, num_classes=self.classes_num).permute(2, 0, 1).float()
        
        return image, label
    
    def __len__(self):
        return self.num_samples