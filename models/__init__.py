from torch import nn
import torch
import importlib
from hpman.m import hp


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss_l1(self, pred, label):
        return torch.abs(pred.reshape(-1)-label.reshape(-1)).mean()

    def get_loss_l2(self, pred, label):
        return torch.pow(pred.reshape(-1)-label.reshape(-1), 2).mean()

    def get_loss_dice(self, pred, label):
        dice = (2*pred*label + 1e-6) / (pred**2 + label**2+1e-6)
        return (1 - dice).mean()

    def get_loss(self, loss, pred, label):
        loss_map = {
            'loss_ce': nn.CrossEntropyLoss(),
            'loss_l1': self.get_loss_l1,
            'loss_l2': self.get_loss_l2,
            'loss_dice': self.get_loss_dice,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](pred, label)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update({loss: self.get_loss(loss, outputs, targets)})
        return losses


def build_model(args):
    device = torch.device(args.device)

    m = hp('model.network')
    network = importlib.import_module(f'models.{m}')
    model = getattr(network, m)() if hasattr(network, m) else None

    losses = []
    weight_dict = {}

    if hp('model.loss_ce_weight', None) is not None:
        losses.append('loss_ce')
        weight_dict['loss_ce'] = hp('model.loss_ce_weight')

    if hp('model.loss_l1_weight', None) is not None:
        losses.append('loss_l1')
        weight_dict['loss_l1'] = hp('model.loss_l1_weight')

    if hp('model.loss_l2_weight', None) is not None:
        losses.append('loss_l2')
        weight_dict['loss_l2'] = hp('model.loss_l2_weight')

    if hp('model.loss_dice_weight', None) is not None:
        losses.append('loss_dice')
        weight_dict['loss_dice'] = hp('model.loss_dice_weight')

    print(f'losses: {losses}')
    print(f'weight_dict: {weight_dict}')

    criterion = SetCriterion(weight_dict, losses)
    criterion.to(device)

    return model, criterion
