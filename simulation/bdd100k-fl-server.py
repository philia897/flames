import argparse
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common.typing import Scalar
import os
import json

from lib.utils.logger import getLogger, create_id_by_timestamp
MODEL_NAME = f"model-{create_id_by_timestamp(include_ms=False)}.pth"
LOGGER_FILE = f"/home/zekun/drivable/outputs/semantic/logs/{MODEL_NAME.replace('.pth', '-server.log')}"
LOGGER = getLogger(logfile=LOGGER_FILE)

from lib.data.tools import load_mmcv_checkpoint, get_params
from federated.strategies import BDD100KStrategy, aggregate_custom_metrics
from models.modelInterface import BDD100kModel
from lib.utils.dbhandler import JsonDBHandler, Items

def set_current_conditions(current_conditions_fn:str, conditions:List):
    with open(current_conditions_fn, 'w') as f:
        json.dump(conditions, f)

def fit_config(server_round: int)->Dict[str,Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "round": server_round
    }
    return config

def eval_config(server_round: int)->Dict[str,Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "round": server_round
    }
    return config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--num_rounds", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--cls_id", type=str, default='0')
    parser.add_argument("--task_name", type=str, default="sem_seg")
    args = parser.parse_args()

    cls_id = args.cls_id
    output_size = (512, 1024)
    handler = JsonDBHandler(os.path.join(args.output_dir, "db"))
    modelinfo = handler.read(cls_id, Items.MODEL_INFO)
    conditionclass = handler.read(cls_id, Items.CONDITION_CLASS)

    env_path = __file__.replace(os.path.basename(__file__), "current_conditions.json")
    set_current_conditions(env_path, conditionclass.conditions)

    init_model = BDD100kModel(
        num_classes=20,
        backbone=load_mmcv_checkpoint(modelinfo.model_config_file, modelinfo.checkpoint_file),
        size=output_size
    )

    modelinfo.checkpoint_file = handler.suggest_model_save_path(MODEL_NAME)

    # configure the strategy
    strategy = BDD100KStrategy(
        model=init_model,
        modelinfo=modelinfo,
        main_metric="mIoU",
        init_metric_value=0,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=aggregate_custom_metrics,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_params(init_model)
        ),
    )

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.num_rounds), 
                           strategy=strategy, 
                           server_address="[::]:8080")

    modelinfo.meta["retrained_times"] = modelinfo.meta.get("retrained_times", 0) + 1
    handler.update(modelinfo, Items.MODEL_INFO, cls_id)


    # After training, go to the evaluation phase:
    strategy = BDD100KStrategy(
        model=init_model,
        modelinfo=None,  # Done save model
        main_metric="mIoU",
        init_metric_value=0,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,  # All clients should be available
        on_fit_config_fn=None,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=aggregate_custom_metrics,
        initial_parameters=fl.common.ndarrays_to_parameters(
            get_params(init_model)
        ),

    )