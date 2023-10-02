import math
import numpy as np
import torch
# import rasterio
from PIL import Image
from . import transforms





class ImageDataset(torch.utils.data.Dataset):

    """
    Image dataset

    Args:
        img_list (str list): List containing image names
        msk_fn: convert from img_path to corresponding msk_path
        classes (int list): list of class-code
        msk_map (dict): map mask value to class code
        testing (boolean): testing mode need no mask
        augm (Classes): transfromation pipeline (e.g. Rotate, Crop, etc.)
    """

    def __init__(self, img_list: list, msk_fn, classes:list, msk_map:dict=dict(), testing=False, augm=None):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [msk_fn(f) for f in self.fn_imgs]
        self.augm = augm
        self.testing = testing
        self.msk_map = msk_map
        self.classes = classes
        self.to_tensor = transforms.ToTensor(classes=self.classes)

    def __getitem__(self, idx):
        img = Image.open(self.fn_imgs[idx])

        if not self.testing:
            msk = Image.open(self.fn_msks[idx])
        else:
            msk = Image.fromarray(np.zeros(img.size[:2], dtype="uint8"))

        if self.augm is not None:
            data = self.augm({"image": img, "mask": msk})
        else:
            h, w = msk.size
            power_h = math.ceil(np.log2(h) / np.log2(2))
            power_w = math.ceil(np.log2(w) / np.log2(2))
            if 2**power_h != h or 2**power_w != w:
                img = img.resize((2**power_w, 2**power_h), resample=Image.BICUBIC)
                msk = msk.resize((2**power_w, 2**power_h), resample=Image.NEAREST)
            data = {"image": img, "mask": msk}

        data = self.to_tensor(
            {
                "image": np.array(data["image"], dtype="uint8"),
                "mask": self._mapping(np.array(data["mask"], dtype="uint8")),
            }
        )
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)

    def _mapping(self, msk):
      for k,v in self.msk_map.items():
        msk[msk==k] = v
      return msk
