import numpy as np
import os
import shutil
import json
import random
from enum import Enum

from lib import utils

class SplitMode(Enum):
    SIMPLE_SPLIT = 1
    RANDOM_SAMPLE = 2


class Bdd100kDatasetSplitAgent:
    def __init__(self, attr_file:str, image_list:list[str]):
        self.attr_obj = self._get_attr_jsonobj(attr_file)
        self.images = image_list
        self.partitions = []

    def _get_attr_jsonobj(self, attr_file):
        with open(attr_file) as f:
            return json.load(f)
        
    def clear_partitions(self):
        self.partitions.clear()

    def split_list(self, partition_num:int, mode:SplitMode, if_shuffle:bool=False, **kwargs):
        self.clear_partitions()
        l = self.images.copy()
        if if_shuffle:
            random.shuffle(l)
        if mode == SplitMode.SIMPLE_SPLIT:
            self.partitions = np.array_split(l, partition_num)
        elif mode == SplitMode.RANDOM_SAMPLE:
            if "sample_n" in kwargs:
                sample_n = int(kwargs["sample_n"])
            else:
                sample_n = int(len(l)/partition_num)
            for i in range(partition_num):
                self.partitions.append(np.array(random.sample(l, sample_n)))
        else:
            raise Exception(f"Mode is wrong, no such mode: {mode}")
    
    def partition_num(self):
        return len(self.partitions)

    def get_partition(self, id:int):
        if id < self.partition_num():
            return self.partitions[id]
        else:
            raise Exception(f"Invalid id {id} (0-{self.partition_num()-1})")

