import torch.nn as nn

class LinearModel(nn.Module):
    num_classes = 2
    settings = {}

    def __init__(self, num_weights) -> None:
        super().__init__()
        self.lin = nn.Linear(num_weights - 1, 1)

    def forward(self, x):
        return {'y': self.lin(x)}


class BettingLinearModel(nn.Module):
    settings = {}
    num_classes = 2

    def __init__(self, num_weights) -> None:
        super().__init__()
        self.lin = nn.Linear(num_weights - 1, 1)
        self.lin_q = nn.Linear(num_weights - 1, 1)

    def forward(self, x):
        return {'y': self.lin(x),
                'yhat': self.lin_q(x)}


def get_model(loss):
    return BettingLinearModel if loss in ['Betting', 'CrossBet'] else LinearModel
