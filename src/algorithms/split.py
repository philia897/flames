import pandas as pd
import numpy as np
import argparse
import os

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from lib.utils.dbhandler import (
    DBHandler,
    JsonDBHandler,
    Items,
    EvalResultItem,
    ConditionClassItem,
    ModelInfoItem,
)
from lib.data.condparser import BDD100KConditionParser, ConditionParserMode
from algorithms.analyze import get_avg_metrics, _calculate_avg_metrics, get_all_leaf_nodes

def _eval_one_node(concls_id, handler:JsonDBHandler, metric:str):
    evalresults:dict = handler.read(concls_id, Items.EVAL_RESULT).eval_results_per_condition
    metric_list = [v['metrics'][metric] for _,v in evalresults.items()]
    # calculate the std of metric_list
    std = np.std(metric_list)
    return std

def find_concls_for_split(handler:JsonDBHandler, metric:str):
    '''
    Find the most proper condition class to split on.
    
    :param handler: JsonDBHandler
    :param metric: str
    :return: str
    '''
    leaf_nodes = get_all_leaf_nodes(handler)
    eval_values = []
    for concls_id in leaf_nodes:
        eval_value = _eval_one_node(concls_id, handler, metric)
        eval_values.append(eval_value)
    idx = eval_values.index(max(eval_values))
    return leaf_nodes[idx]

def split_condition(concls_id:str, handler:JsonDBHandler, cluster_num=2, only_try=False):
    evalresult:dict = handler.read(concls_id, Items.EVAL_RESULT).eval_results_per_condition
    df = pd.DataFrame()
    for k,v in evalresult.items():
        record = {"condition": k, "samples": v["sample_num"]}
        record.update(v["metrics"])
        new_item = pd.DataFrame([record])
        df = pd.concat([df, new_item], ignore_index=True)
    X = df.iloc[:, 1:] # except condition
    # scale numerical variables
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=cluster_num, init='k-means++', max_iter=300, n_init=20, random_state=0)
    pred_y = kmeans.fit_predict(X_scaled)

    # add cluster labels to dataframe
    df['Cluster'] = pred_y
    df = df.sort_values('Cluster', axis=0)
    # print(df[df['Cluster']==0])

    parser = BDD100KConditionParser()
    new_modelinfo:ModelInfoItem = handler.read(concls_id, Items.MODEL_INFO)
    new_modelinfo.parents = [concls_id]
    new_modelinfo.children = []
    new_modelinfo.meta = {
        "class_num": new_modelinfo.meta.get("class_num"),
        "output_size": new_modelinfo.meta.get("output_size"),
        "retrained_times": 0
    }
    children_concls_ids = []
    for cluster in range(cluster_num):
        conditions = []
        eval_results = {}
        for _, row in df[df['Cluster']==cluster].iterrows():
            # print(row['condition'])
            conditions.append(parser.convert(row['condition'], ConditionParserMode.VALUE_LIST))
            eval_results.setdefault(row['condition'], {
                "sample_num": row['samples'],
                "metrics": {k:row[k] for k in df.columns[2:-1]}
            }) 
        if not only_try:
            children_concls_ids.append(handler.create(new_modelinfo, ConditionClassItem(conditions), EvalResultItem(eval_results)))
    return children_concls_ids, df

def _get_avg_children_grade(grades, children):
    return _calculate_avg_metrics(grades, children)

def test_children(parent_concls_id, handler:JsonDBHandler, metric):
    '''
    Test the children's average grade and compare it to the parent's grade
    return [child_higher=True/False], child_avg_grade, parent_grade
    '''
    modelinfo: ModelInfoItem = handler.read(parent_concls_id, Items.MODEL_INFO)
    children = modelinfo.children
    children_grades = {}
    for child in children:
        avg_grade = get_avg_metrics(child, handler)
        children_grades[child] = avg_grade
    avg_grade = _get_avg_children_grade(children_grades, children)
    parent_grade = get_avg_metrics(parent_concls_id, handler)
    child_higher = False
    if avg_grade['metrics'][metric] > parent_grade['metrics'][metric]:
        child_higher = True
    return child_higher, avg_grade, parent_grade

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split Condition")
    parser.add_argument("--db_dir", type=str, default="/home/zekun/drivable/outputs/semantic/db")
    parser.add_argument("--cls_id", "-c", type=str, default="0")
    parser.add_argument("--mode", "-m", type=str, default="find") # find / split / test
    args = parser.parse_args()

    mode = args.mode
    concls_id = args.cls_id
    handler = JsonDBHandler(args.db_dir)

    if mode == "find":
        found = find_concls_for_split(handler, "loss")
        print(split_condition(found, handler, only_try=True))
        print("Found: ", found)
    elif mode == "split":
        new_ids, df = split_condition(concls_id, handler, only_try=False)
        print(df)
        print("New ids:", new_ids)
        # Add the new ids into the queue for later training.
        with open('simulation/waiting_model_queue.txt', 'w') as f:  
            for i in new_ids:
                f.write(i+'\n')
            print("New ids are written into waiting_model_queue.txt.")
    elif mode == "test":
        print(test_children(concls_id, handler, "pAcc"))
    elif mode == "try_split":
        new_ids, df = split_condition(concls_id, handler, only_try=True, cluster_num=3)
        print(df)
