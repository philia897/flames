import argparse
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common.typing import Scalar

from lib.utils.logger import getLogger
LOGGER = getLogger(logfile='flames-server.log')

from lib.data.tools import load_mmcv_checkpoint, get_params
from federated.strategies import BDD100KStrategy, aggregate_custom_metrics
from models.modelInterface import BDD100kModel
from lib.simulation.env import get_model_additional_configs

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
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--model_config_file", type=str, default="/home/zekun/drivable/src/models/config-deeplabv3plus-sem_seg.py")
    parser.add_argument("--checkpoint_file", type=str, default="/home/zekun/drivable/outputs/semantic/models/deeplabv3+_backbone_fl10-class0-initial.pth")
    parser.add_argument("--task_name", type=str, default="sem_seg")
    args = parser.parse_args()

    config_file = args.model_config_file
    checkpoint_file = args.checkpoint_file
    output_size = (512, 1024)
    MODEL_META = get_model_additional_configs(args.task_name)

    init_model = BDD100kModel(
        num_classes=MODEL_META["num_classes"],
        backbone=load_mmcv_checkpoint(config_file, checkpoint_file),
        size=output_size
    )

    # configure the strategy
    strategy = BDD100KStrategy(
        model=init_model,
        model_save_path=None,
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