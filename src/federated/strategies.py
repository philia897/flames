import flwr as fl
from flwr.common import EvaluateRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.common.typing import Scalar
from flwr.server import client_proxy
from flwr.server.client_proxy import ClientProxy
import numpy as np
from typing import Dict, Callable, Optional, Tuple, List, Union
from models.modelInterface import BDD100kModel
import warnings

warnings.filterwarnings("ignore")

from lib.data.tools import save_model, set_params, get_params
from lib.utils.logger import getLogger
from lib.utils.dbhandler import ModelInfoItem

LOGGER = getLogger()

class FlwrStrategyException(Exception):
    pass

class BDD100KStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        model:BDD100kModel,
        modelinfo: ModelInfoItem | None = None,
        main_metric: str | None = None,
        init_metric_value: float = -1.,
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
        self.main_metric = main_metric
        self.highest_score = init_metric_value
        self.modelinfo = modelinfo
        LOGGER.info("Server Initialized")

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
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )
            set_params(self.model, aggregated_ndarrays)
            LOGGER.info(f"Aggregation Fit: round {server_round}")
            LOGGER.debug({"round": server_round, "aggregated_metrics": aggregated_metrics})

        return aggregated_parameters, aggregated_metrics

    def _update_modelinfo(self, loss, metrics, server_round):
        self.modelinfo.meta["loss"] = loss
        self.modelinfo.meta["last_server_round"] = server_round
        self.modelinfo.meta["runtime_metrics"] = metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> Tuple[float | None, Dict[str, Scalar]]:
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
            server_round, results, failures
        )
        if aggregated_loss == None:
            raise FlwrStrategyException("Evaluation Aggregation Failure, Aggregated_loss=None")
        if self.main_metric:
            metric_name = self.main_metric
            try:
                new_score = aggregated_metrics[self.main_metric]
            except Exception as e:
                raise ValueError(f'Received metrics: {aggregated_metrics}, but no main metric {self.main_metric}')
        else:
            metric_name = "loss"
            new_score = -1. * aggregated_loss
        LOGGER.info(f"Aggregation Eval: round {server_round}")
        LOGGER.debug(f"Average Score ({metric_name}): {new_score}\tCurrent Best: {self.highest_score}")
        LOGGER.debug({"round": server_round, "aggregated_metrics": aggregated_metrics})
        
        if float(new_score) >= self.highest_score:
            self.highest_score = new_score
            # Save the model
            if self.modelinfo:
                save_model(self.model.backbone, self.modelinfo.checkpoint_file, epoch=server_round, best_score=new_score)
                LOGGER.info(f"Model saved to {self.modelinfo.checkpoint_file}")
                self._update_modelinfo(aggregated_loss, aggregated_metrics, server_round)
            
        return aggregated_loss, aggregated_metrics


# class BDD100KEvalStrategy(fl.server.strategy.FedAvg):
#     '''Has not been completed
#     TODO: find a way to collect the eval results specifying conditions. See more on the todo notes
#     '''
#     def __init__(
#         self,
#         model:BDD100kModel,
#         evalresult: EvalResultItem,
#         fraction_fit: float = 1,
#         fraction_evaluate: float = 1,
#         min_fit_clients: int = 2,
#         min_evaluate_clients: int = 2,
#         min_available_clients: int = 2,
#         evaluate_fn: Callable[
#             [int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None
#         ]
#         | None = None,
#         on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
#         on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None,
#         accept_failures: bool = True,
#         initial_parameters: Parameters | None = None,
#         fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
#         evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
#     ) -> None:
#         super().__init__(
#             fraction_fit=fraction_fit,
#             fraction_evaluate=fraction_evaluate,
#             min_fit_clients=min_fit_clients,
#             min_evaluate_clients=min_evaluate_clients,
#             min_available_clients=min_available_clients,
#             evaluate_fn=evaluate_fn,
#             on_fit_config_fn=on_fit_config_fn,
#             on_evaluate_config_fn=on_evaluate_config_fn,
#             accept_failures=accept_failures,
#             initial_parameters=initial_parameters,
#             fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
#             evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
#         )
#         self.model = model
#         self.evalresult = evalresult
#         LOGGER.info("Eval Server Initialized")

#     def aggregate_fit(
#         self,
#         server_round: int,
#         results: List[Tuple[client_proxy.ClientProxy, fl.common.FitRes]],
#         failures: List[
#             Union[Tuple[client_proxy.ClientProxy, fl.common.FitRes], BaseException]
#         ],
#     ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:

#         aggregated_parameters = fl.common.ndarrays_to_parameters(get_params(self.model)) # Reset the model parameters

#         return aggregated_parameters, {}

#     def update_modelinfo(self, loss, metrics, server_round):
#         self.modelinfo.meta["loss"] = loss
#         self.modelinfo.meta["last_server_round"] = server_round
#         self.modelinfo.meta["accumulated_rounds"] = self.modelinfo.meta.get("accumulated_rounds", 0) + server_round
#         self.modelinfo.meta.update(metrics)

#     def aggregate_evaluate(
#         self,
#         server_round: int,
#         results: List[Tuple[ClientProxy, EvaluateRes]],
#         failures: List[Tuple[ClientProxy, EvaluateRes] | BaseException],
#     ) -> Tuple[float | None, Dict[str, Scalar]]:
#         aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
#             server_round, results, failures
#         )
#         LOGGER.debug({"aggregated_metrics": aggregated_metrics})
#         self.evalresult.eval_results_per_condition.update(aggregated_metrics)

#         return aggregated_loss, aggregated_metrics

def update_modelinfo(modelinfo, class_num, output_size):
    modelinfo.meta["retrained_times"] = modelinfo.meta.get("retrained_times", 0) + 1
    modelinfo.meta["accumulated_rounds"] = modelinfo.meta.get("accumulated_rounds", 0) + modelinfo.meta["last_server_round"]
    modelinfo.meta['class_num'] = class_num
    modelinfo.meta['output_size'] = output_size
    return modelinfo


def aggregate_custom_metrics(metrics:List[Tuple[int,Dict]]):
    num_total_examples = sum([n for n,_ in metrics])
    aggregated_metrics = dict()
    for name in metrics[0][1].keys():
        weighted_metric = sum([n * d[name] for n,d in metrics])/num_total_examples
        aggregated_metrics.setdefault(name, weighted_metric)
    return aggregated_metrics
