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

def _calculate_avg_metrics(evalresults:Dict[str, Dict[str, Any]], conditions:List[str], metrics:List[str]):
    total_sample_n = sum([evalresults[c].get("sample_num") for c in conditions])
    rst = {"sample_num": total_sample_n}
    avg_results = {}
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

def compare_evalresult_two_concls(concls_id_1:str, concls_id_2:str, handler:JsonDBHandler):
    '''
    compare the eval results of two condition classes considering their intersection condtions
    Return avg_result_1, avg_result_2, intersection_conditions:List
    avg_result format: {"sample_num": N, "metrics": {"metric1": x, ...}}
    '''
    evalresult1:dict = handler.read(concls_id_1, Items.EVAL_RESULT).eval_results_per_condition
    evalresult2:dict = handler.read(concls_id_2, Items.EVAL_RESULT).eval_results_per_condition
    conditions_1, conditions_2 = set(evalresult1.keys()), set(evalresult2.keys())
    conditions_intersection = list(conditions_1.intersection(conditions_2))
    metrics = list(next(iter(evalresult1.items()))[1].get('metrics').keys())
    avg_result1 = _calculate_avg_metrics(evalresult1, conditions_intersection, metrics)
    avg_result2 = _calculate_avg_metrics(evalresult2, conditions_intersection, metrics)
    return avg_result1, avg_result2, conditions_intersection

def get_all_leaf_nodes(handler:JsonDBHandler):
    all_nodes:dict = handler.read_all(Items.MODEL_INFO)
    leaf_nodes = []
    for k,v in all_nodes.items():
        if len(v['children']) == 0:
            leaf_nodes.append(k)
    return leaf_nodes

if __name__ == "__main__":
    output_dir = "/home/zekun/drivable/outputs/semantic"

    handler = JsonDBHandler(os.path.join(output_dir, "db"))

    print(get_avg_metrics("0", handler))
