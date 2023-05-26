import flwr as fl
from flwr.server import client_proxy
from typing import List, Tuple, Union, Dict, Optional
import torch
from collections import OrderedDict
import numpy as np
import sys
sys.path.append('.') # <= change path where you save code
# print(sys.path)

from lib.utils import save_model

STORE_MODEL_NAME = "fcn_backbone_snowy_anytime_others_fl"
checkpoint_file = "/home/zekun/drivable/outputs/fcn_backbone_refine_benchmark-20230422_112649.pth"

config_file = 'models/config.py'
# checkpoint_file = '/content/fcn_r50-d8_769x769_40k_drivable_bdd100k.pth'
# checkpoint_file = f'outputs/{checkpoint_file_name}' # defined above
# img_path = '/content/bdd100k/images/100k/train/0000f77c-62c2a288.jpg'

OUTPUT_DIR = "./outputs"
LOOP_NUM = 5


import datetime
now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
stored_model_name = f"{STORE_MODEL_NAME}-{timestamp}.pth"

## -------------
## Get model
## -------------
from mmseg.apis import init_model
import mmcv
# print(mmcv.version.version_info)
from mmengine import runner

backbone = init_model(config_file, device='cpu')
checkpoint = runner.load_checkpoint(backbone, checkpoint_file)

from drivable.models.modelInterface import BDD100kModel


output_size = (769,769)
model = BDD100kModel(num_classes=3, backbone=backbone, size=output_size)
# model = load_checkpoint(model, "test-20230402_144923.pth", OUTPUT_DIR)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            model.load_state_dict(state_dict, strict=True)

            # Save the model
            save_model(model.backbone, LOOP_NUM, 0, stored_model_name, OUTPUT_DIR)

        return aggregated_parameters, aggregated_metrics

strategy = SaveModelStrategy(
    initial_parameters = fl.common.ndarrays_to_parameters([val.cpu().numpy() for _, val in model.state_dict().items()])
    # fraction_fit=0.1,  # Sample 10% of available clients for the next round
    # min_fit_clients=10,  # Minimum number of clients to be sampled for the next round
    # min_available_clients=80,  # Minimum number of clients that need to be connected to the server before a training round can start
)

fl.server.start_server(server_address="[::]:8080", config=fl.server.ServerConfig(num_rounds=LOOP_NUM), strategy=strategy)

