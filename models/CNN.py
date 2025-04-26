from torch import nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.output = nn.Linear(32*7*7, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(out.size(0), -1)
        out = self.output(out)
        return out


class SetCriterion(nn.Module):
    def __init__(self, weight_dict, losses):
        super(SetCriterion, self).__init__()
        self.weight_dict = weight_dict
        self.losses = losses

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'loss_ce': nn.CrossEntropyLoss(),
        }
        assert loss in loss_map, f"Unknown loss: {loss}"
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        losses = {}
        for loss in self.losses:
            losses.update({loss: self.get_loss(loss, outputs, targets)})
        return losses


def build(args):
    device = torch.device(args.device)

    model = CNN()

    weight_dict = {'loss_ce': 1.0}
    losses = ['loss_ce']
    criterion = SetCriterion(weight_dict, losses)
    criterion.to(device)

    return model, criterion
