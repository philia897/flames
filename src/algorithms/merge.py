import pandas as pd
import argparse
import os
from typing import List
import torch


from lib.utils.dbhandler import (
    DBHandler,
    JsonDBHandler,
    Items,
    EvalResultItem,
    ConditionClassItem,
    ModelInfoItem,
)
from lib.data.condparser import BDD100KConditionParser, ConditionParserMode

# Calculate the similarity of two pytorch model
def _param_diff(model1:torch.nn.Module, model2:torch.nn.Module):  
    total_diff = 0
    for param1, param2 in zip(model1.parameters(), model2.parameters()):  
        diff = torch.norm(param1 - param2)  
        total_diff += diff  
    return total_diff  


def merge_conditions(concls_alternatives:List, handler:JsonDBHandler):
    '''Find two most similar model items and merge them into one new item'''
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Merge Conditions")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--cls_id", "-c", type=str, default="0")
    args = parser.parse_args()

    concls_id = args.cls_id
    handler = JsonDBHandler(os.path.join(args.output_dir, "db"))



