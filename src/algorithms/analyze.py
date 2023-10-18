import argparse
from lib.utils.dbhandler import (
    DBHandler,
    JsonDBHandler,
    Items,
    EvalResultItem,
    ConditionClassItem,
    ModelInfoItem,
)
from typing import Dict, Any, List
import os
from prettytable import PrettyTable

def _calculate_avg_metrics(evalresults:Dict[str, Dict[str, Any]], conditions:List[str]=None, metrics:List[str]=None):
    total_sample_n = sum([evalresults[c].get("sample_num") for c in conditions])
    rst = {"sample_num": total_sample_n}
    avg_results = {}
    if not conditions:
        conditions = list(evalresults.keys())
    if not metrics:  # metrics = None, then use all
        metrics = list(next(iter(evalresults.items()))[1].get('metrics').keys())
    for metric in metrics:
        # print(evalresults[conditions[1]].get("metrics").get())
        weighted_sum = sum([evalresults[c].get("sample_num")* evalresults[c].get("metrics")[metric]   for c in conditions])
        avg_results[metric] = weighted_sum / total_sample_n
    rst["metrics"] = avg_results
    return rst

def get_avg_metrics(concls_id, handler:JsonDBHandler):
    '''
    Get avg metrics of the condition class
    Return avg_result
    avg_result format: {"sample_num": N, "metrics": {"metric1": x, ...}}
    '''
    evalresults:dict = handler.read(concls_id, Items.EVAL_RESULT).eval_results_per_condition
    metrics = list(next(iter(evalresults.items()))[1].get('metrics').keys())
    conditions = list(evalresults.keys())
    return _calculate_avg_metrics(evalresults, conditions, metrics)

def _comb_two_lists(lst1, lst2):
    length = len(lst1)
    return [item for pair in [[lst1[i], lst2[i]] for i in range(length)] for item in pair]

def _get_detailed_table(evalresult1:Dict, evalresult2:Dict, conditions:List, metrics:List=None):
    if not metrics:
        metrics = list(next(iter(evalresult1.items()))[1].get('metrics').keys())
    table = PrettyTable()
    table.field_names = ['weather', 'scene', 'timeofday', 'val_samples'] + \
                _comb_two_lists([f"{m}_1" for m in metrics], [f"{m}_2" for m in metrics])
    for condition in conditions:
        weather, scene, time = condition.split('-')
        rst_1 = [round(evalresult1[condition]["metrics"][m], 4) for m in metrics]
        rst_2 = [round(evalresult2[condition]["metrics"][m], 4) for m in metrics]
        table.add_row([weather, scene, time, evalresult1[condition]["sample_num"]]+_comb_two_lists(rst_1, rst_2))
    table.sortby = 'val_samples'
    table.reversesort = True
    return table

def compare_evalresult_two_concls(concls_id_1:str, concls_id_2:str, handler:JsonDBHandler, metrics:List[str]=None):
    '''
    compare the eval results of two condition classes considering their intersection condtions
    Return avg_result_1, avg_result_2, intersection_conditions:List, detailed_table:PrettyTable
    avg_result format: {"sample_num": N, "metrics": {"metric1": x, ...}}
    '''
    evalresult1:dict = handler.read(concls_id_1, Items.EVAL_RESULT).eval_results_per_condition
    evalresult2:dict = handler.read(concls_id_2, Items.EVAL_RESULT).eval_results_per_condition
    conditions_1, conditions_2 = set(evalresult1.keys()), set(evalresult2.keys())
    conditions_intersection = list(conditions_1.intersection(conditions_2))
    detailed_table = _get_detailed_table(evalresult1, evalresult2, conditions_intersection, metrics)
    avg_result1 = _calculate_avg_metrics(evalresult1, conditions_intersection)
    avg_result2 = _calculate_avg_metrics(evalresult2, conditions_intersection)
    return avg_result1, avg_result2, conditions_intersection, detailed_table

def compare_evalresult_two_concls_and_show(concls_1, concls_2, handler, metrics=None):
    avg_result1, avg_result2, conditions, table = compare_evalresult_two_concls(
        concls_1, concls_2, handler, metrics)
    print("concls 1: ", concls_1, avg_result1)
    print("concls 2: ", concls_2, avg_result2)
    print("detailed table")
    print(table)

def get_all_leaf_nodes(handler:JsonDBHandler)->List[str]:
    all_nodes:dict = handler.read_all(Items.MODEL_INFO)
    leaf_nodes = []
    for k,v in all_nodes.items():
        if len(v['children']) == 0:
            leaf_nodes.append(k)
    return leaf_nodes

def test_avg_leaf_nodes(handler:JsonDBHandler):
    leaf_nodes = get_all_leaf_nodes(handler)
    grades = {}
    for node in leaf_nodes:
        avg_grade = get_avg_metrics(node, handler)
        grades[node] = avg_grade
    avg_grade = _calculate_avg_metrics(grades, leaf_nodes)
    return avg_grade, grades
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Condition")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--cls_id", "-c", type=str, default="2")
    args = parser.parse_args()
    output_dir = args.output_dir

    concls_id = args.cls_id
    handler = JsonDBHandler(os.path.join(output_dir, "db"))

    # print(get_avg_metrics("centralized-training", handler))
    # print(test_avg_leaf_nodes(handler))
    for node in get_all_leaf_nodes(handler):
        compare_evalresult_two_concls_and_show("full-fl-training", node, handler)

