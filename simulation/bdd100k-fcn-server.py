import argparse
from typing import Dict, List, Tuple
import flwr as fl
from flwr.common import Scalar
from flwr.common.typing import Scalar

from lib import utils
from federated.strategies import SaveModelStrategy
from models.modelInterface import BDD100kModel
from lib.simulation.env import get_model_additional_configs

MODEL_META = get_model_additional_configs("sem_seg")


def eval_metrics_aggregate(
    num_and_metrics: List[Tuple[int, Dict]]
) -> Dict[str, Scalar]:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _ in num_and_metrics]
    )
    weighted_acc = [
        num_examples * acc_dic["accuracy"] for num_examples, acc_dic in num_and_metrics
    ]
    return {"accuracy": sum(weighted_acc) / num_total_evaluation_examples}

def fit_config(server_round: int):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        # "batch_size": 2,
    }
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--train_model_type", type=str, default="deeplabv3+_backbone_fl10")
    parser.add_argument("--output_dir", type=str, default="/home/zekun/drivable/outputs/semantic")
    parser.add_argument("--model_config_file", type=str, default="/home/zekun/drivable/src/models/config-deeplabv3plus-sem_seg.py")
    parser.add_argument("--checkpoint_file", type=str, default="/home/zekun/drivable/outputs/semantic/models/deeplabv3+_backbone_fl10-class0-initial.pth")
    args = parser.parse_args()

    config_file = args.model_config_file
    checkpoint_file = args.checkpoint_file
    output_size = (512, 1024)

    init_model = BDD100kModel(
        num_classes=MODEL_META["num_classes"],
        backbone=utils.load_mmcv_checkpoint(config_file, checkpoint_file),
        size=output_size
    )

    # configure the strategy
    strategy = SaveModelStrategy(
        init_model,
        None,
        fraction_fit=0.5,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=2,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_metrics_aggregation_fn=eval_metrics_aggregate,
        initial_parameters=fl.common.ndarrays_to_parameters(
            utils.get_params(init_model)
        ),
    )

    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=args.num_rounds), strategy=strategy, server_address="[::]:8080")