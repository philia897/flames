import flwr as fl
from flwr.common import EvaluateRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.typing import Scalar
from flwr.server import client_proxy
from flwr.server.client_proxy import ClientProxy
import torch
import numpy as np
from collections import OrderedDict
from typing import Dict, Callable, Optional, Tuple, List, Union
import warnings

warnings.filterwarnings("ignore")



from lib.utils import save_model


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model,
        model_save_path: str | None = None,
        fraction_fit: float = 1,
        fraction_evaluate: float = 1,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Callable[
            [int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None
        ]
        | None = None,
        on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.model = model
        self.best_score = 0.0
        self.model_save_path = model_save_path

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[
            Union[Tuple[client_proxy.ClientProxy, fl.common.FitRes], BaseException]
        ],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_parameters...")

            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )

        print(f"Average score: {aggregated_metrics['accuracy']}\nCurrent Best Score: {self.best_score}")
        if float(aggregated_metrics['accuracy']) > float(self.best_score):
            self.best_score = aggregated_metrics['accuracy']
            # Save the model
            if isinstance(self.model_save_path, str):
                save_model(self.model.backbone, self.model_save_path)
            
        return aggregated_loss, aggregated_metrics