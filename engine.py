import torch
from typing import Iterable


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer, epoch: int):
    model.train()
    criterion.train()

    for images, targets in data_loader:

        optimizer.zero_grad()
        optimizer.step()

    return {}
