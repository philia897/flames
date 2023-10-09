import pandas as pd
import argparse
import os
from typing import List


from lib.utils.dbhandler import (
    DBHandler,
    JsonDBHandler,
    Items,
    EvalResultItem,
    ConditionClassItem,
    ModelInfoItem,
)
from lib.data.condparser import BDD100KConditionParser, ConditionParserMode



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



