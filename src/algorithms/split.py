import pandas as pd
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

def judge_split(handler:JsonDBHandler):
    pass

def split_condition(concls_id:str, handler:JsonDBHandler, cluster_num=2):
    evalresult:dict = handler.read(concls_id, Items.EVAL_RESULT).eval_results_per_condition
    # conditions = list(evalresult.keys())
    # metrics = set(evalresult.values()[0].get('metrics').keys())
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
    new_modelinfo.meta = {"retrained_times": 0} # means it has not be refined
    children_concls_ids = []
    for cluster in range(cluster_num):
        conditions = []
        eval_results = {}
        for _, row in df[df['Cluster']==cluster].iterrows():
            # print(row['condition'])
            conditions.append(parser.convert(row['condition'], ConditionParserMode.VALUE_LIST))
            eval_results.setdefault(row['condition'], {
                "sample_num": row['samples'],
                "metrics": {k:row[k] for k in df.columns[2:]}
            }) 
        children_concls_ids.append(handler.create(new_modelinfo, ConditionClassItem(conditions), EvalResultItem(eval_results)))
    return children_concls_ids

def eval_split()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split Condition")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--cls_id", "-c", type=str, default="0")
    args = parser.parse_args()

    concls_id = args.cls_id
    handler = JsonDBHandler(os.path.join(args.output_dir, "db"))

    print(split_condition(concls_id, handler))
