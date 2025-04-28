import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from hpman.m import hp


def build_optimizer(model):
    trainable = filter(lambda x: x.requires_grad, model.parameters())
    kwargs_optimizer = {"lr": hp('base_lr', 1e-4)}

    optimizer = hp("train.optimizer.type", "AdamW")
    if optimizer == "SGD":
        optimizer_class = optim.SGD
        kwargs_optimizer["momentum"] = hp("train.optimizer.momentum", 0.9)
    elif optimizer == "Adadelta":
        optimizer_class = optim.Adadelta
    elif optimizer == "AdamW":
        optimizer_class = optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    return optimizer
