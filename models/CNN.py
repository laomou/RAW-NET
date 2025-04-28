from torch import nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


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

    model = Net()

    weight_dict = {'loss_ce': 1.0}
    losses = ['loss_ce']
    criterion = SetCriterion(weight_dict, losses)
    criterion.to(device)

    return model, criterion
