from PIL import Image
import torch

class ConditionDataset(torch.utils.data.Dataset):
    def __init__(self, image_label_map, attr_name, cls_to_idx:dict, transform=None):
        self.image_label_map = image_label_map
        self.image_filenames = list(image_label_map.keys())
        self.labels = list(image_label_map.values())
        self.transform = transform
        self.attr_name = attr_name
        self.cls_to_idx = cls_to_idx

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image_path = self.image_filenames[index]
        image = Image.open(image_path)
        label = self.cls_to_idx[self.labels[index][self.attr_name]]

        if self.transform is not None:
            image = self.transform(image)

        return image, label