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

def calculate_avg_metrics(evalresults:Dict[str, Dict[str, Any]], conditions:List[str], metrics:List[str]):
    total_sample_n = sum([evalresults[c].get("sample_num") for c in conditions])
    rst = {"sample_num": total_sample_n}
    avg_results = {}
    for metric in metrics:
        weighted_sum = sum([evalresults[c].get("sample_num") * evalresults[c].get("metrics")[metric]  for c in conditions])
        avg_results[metric] = weighted_sum / total_sample_n
    rst["metrics"] = avg_results
    return rst

def compare_evalresult_two_conditions(concls_id_1:str, concls_id_2:str, handler:JsonDBHandler):
    evalresult1:dict = handler.read(concls_id_1, Items.EVAL_RESULT).eval_results_per_condition
    evalresult2:dict = handler.read(concls_id_2, Items.EVAL_RESULT).eval_results_per_condition
    conditions_1, conditions_2 = set(evalresult1.keys()), set(evalresult2.keys())
    conditions_union = conditions_1.union(conditions_2)
    metrics = set(evalresult1.items()[0][1].get('metrics').keys()),
    avg_result1 = calculate_avg_metrics(evalresult1, conditions_union, metrics)
    avg_result2 = calculate_avg_metrics(evalresult2, conditions_union, metrics)
    return avg_result1, avg_result2

if __name__ == "__main__":
    output_dir = "/home/zekun/drivable/outputs/semantic"

    handler = JsonDBHandler(os.path.join(output_dir, "db"))

    print(compare_evalresult_two_conditions("0", "1", handler))
