import torch.optim.lr_scheduler as lrs
from hpman.m import hp


def build_scheduler(optimizer):
    scheduler = hp("train.scheduler.type", "MultiStepLR")
    kwargs_scheduler = {}
    if scheduler == "StepLR":
        scheduler_class = lrs.StepLR
        kwargs_scheduler["step_size"] = hp("train.scheduler.step_size", 1)
        kwargs_scheduler["gamma"] = hp("train.scheduler.gamma", 0.1)
    elif scheduler == "MultiStepLR":
        scheduler_class = lrs.MultiStepLR
        kwargs_scheduler["milestones"] = hp("train.scheduler.milestones", [30, 80])
    elif scheduler == "CosineAnnealingWarmRestarts":
        scheduler_class = lrs.CosineAnnealingWarmRestarts
        kwargs_scheduler["T_0"] = hp("train.scheduler.T_0", 100)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    class CustomScheduler(scheduler_class):
        def __init__(self, optimizer, **kwargs):
            super(CustomScheduler, self).__init__(optimizer, **kwargs)

    scheduler = CustomScheduler(optimizer, **kwargs_scheduler)
    return scheduler
