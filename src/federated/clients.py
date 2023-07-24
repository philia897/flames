import flwr as fl
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")

from models.modelInterface import BDD100kModel

from lib.runners import train_loop, validate_loop
from lib.utils import get_params, set_params

import os


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, 
                 cid: str, 
                 train_loader:DataLoader, 
                 val_loader:DataLoader, 
                 model: BDD100kModel,
                 optimizer,
                 criterion,
                 device):
        self.cid = cid

        # Instantiate model
        self.net = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Determine device
        self.device = device
        # print(self.device)

    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        print("start fitting")
        set_params(self.net, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory
        # Load data for this client and get trainloader

        # Send model to device
        self.net.to(self.device)

        # Train
        train_loop(
            self.net,
            self.train_loader,
            self.optimizer,
            self.criterion,
            epochs=config["epochs"],
            device=self.device,
        )
        self.net.cpu()
        self.net.backbone.cpu()
        torch.cuda.empty_cache()


        # Return local model and statistics
        return get_params(self.net), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)

        os.system(
            "pkill -9 ray::IDLE"
        )  # kill ray IDLE workers to avoid CUDA outofmemory

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = validate_loop(
            self.net, self.val_loader, self.criterion, device=self.device
        )

        # Return statistics
        return float(loss), len(self.val_loader.dataset), {"accuracy": float(accuracy)}

# # Flower client, adapted from Pytorch quickstart example
# class FlowerClient(fl.client.NumPyClient):
#     def __init__(self, cid: str, fed_data: dict, model: BDD100kModel):
#         self.cid = cid
#         self.fed_data = fed_data

#         # Instantiate model
#         self.net = model
#         self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
#         self.criterion = nn.CrossEntropyLoss()

#         # Determine device
#         self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#         print(self.device)

#     def get_parameters(self, config):
#         return get_params(self.net)

#     def fit(self, parameters, config):
#         set_params(self.net, parameters)

#         os.system(
#             "pkill -9 ray::IDLE"
#         )  # kill ray IDLE workers to avoid CUDA outofmemory
#         # Load data for this client and get trainloader
#         num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

#         self.fed_data = {
#             "train": train_split_agent.get_partition(int(self.cid)),
#             "val": val_split_agent.get_partition(int(self.cid)),
#         }

#         trainloader = get_dataloader(
#             self.fed_data,
#             is_train=True,
#             batch_size=config["batch_size"],
#             workers=num_workers,
#             output_size=output_size,
#             img_path=(IMAGE_PATH_TRAIN, IMAGE_PATH_VAL),
#             lbl_path=(LABEL_PATH_TRAIN, LABEL_PATH_VAL)
#         )

#         # Send model to device
#         self.net.to(self.device)

#         # Train
#         train_loop(
#             self.net,
#             trainloader,
#             self.optimizer,
#             self.criterion,
#             epochs=config["epochs"],
#             device=self.device,
#         )
#         self.net.cpu()
#         self.net.backbone.cpu()
#         torch.cuda.empty_cache()


#         # Return local model and statistics
#         return get_params(self.net), len(trainloader.dataset), {}

#     def evaluate(self, parameters, config):
#         set_params(self.net, parameters)

#         os.system(
#             "pkill -9 ray::IDLE"
#         )  # kill ray IDLE workers to avoid CUDA outofmemory

#         # Load data for this client and get trainloader
#         num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
#         valloader = get_dataloader(
#             self.fed_data,
#             is_train=False,
#             batch_size=1,
#             workers=num_workers,
#             output_size=output_size,
#             img_path=(IMAGE_PATH_TRAIN, IMAGE_PATH_VAL),
#             lbl_path=(LABEL_PATH_TRAIN, LABEL_PATH_VAL)
#         )

#         # Send model to device
#         self.net.to(self.device)

#         # Evaluate
#         loss, accuracy = validate_loop(
#             self.net, valloader, self.criterion, device=self.device
#         )

#         # Return statistics
#         return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
