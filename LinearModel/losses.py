import torch
import torch.nn.functional as F


class BCELoss:
    losses = ['BCE']

    def __call__(self, outputs, targets):
        y = outputs['y']
        BCE = F.binary_cross_entropy_with_logits(y, targets)
        return {'Criterion': BCE,
                'BCE': BCE}


class MSELoss:
    losses = ['MSE', 'BCE']

    def __call__(self, outputs, targets):
        y = outputs['y']
        probs = torch.sigmoid(y)
        MSE = ((targets - probs) ** 2).sum()
        BCE = F.binary_cross_entropy_with_logits(y, targets)
        return {'Criterion': MSE,
                'MSE': MSE,
                'BCE': BCE
                }


class MAELoss:
    losses = ['MAE', 'BCE']

    def __call__(self, outputs, targets):
        y = outputs['y']
        probs = torch.sigmoid(y)
        MAE = torch.abs(targets - probs).sum()
        BCE = F.binary_cross_entropy_with_logits(y, targets)
        return {'Criterion': MAE,
                'MAE': MAE,
                'BCE': BCE
                }


class NaiveBetLoss:
    losses = ['Naive', 'BCE']

    def __call__(self, outputs, targets):
        y = outputs['y']
        BCE = F.binary_cross_entropy_with_logits(y, targets)
        probs = torch.sigmoid(y)
        #  counteracts bias in loss
        # probs = (probs + 1/2) / 2
        naive = ((1 / 2 - probs) * (targets - probs)).sum()
        return {'Criterion': naive,
                'Naive': naive,
                'BCE': BCE}


class BettingLoss:
    """
    A mode is divided into two detached classifiers.
    This loss makes the two classifier compete with each other.
    One classifier is the book and makes a bet while
    freezing the proposed probabilities p.
    The other classifier is the bettor and gives a separate evaluation q
    according to which it buys or sells the bet.
    The backpropagation algorithm works on the two classifiers independently.
    """
    losses = ['Book Loss', 'Bettor Loss', "BCEp", "BCEq"]

    def __call__(self, outputs, targets):
        y = outputs['y']
        yhat = outputs['yhat']
        BCEp = F.binary_cross_entropy_with_logits(y, targets)
        BCEq = F.binary_cross_entropy_with_logits(yhat, targets)
        probs = torch.sigmoid(y)
        q = torch.sigmoid(yhat)
        p_detached = probs.detach()
        bettor_loss = ((q - p_detached) * (p_detached - targets)).sum()
        book_loss = ((q.detach() - probs) * (targets - probs)).sum()
        backprop = book_loss + bettor_loss
        return {'Criterion': backprop,
                'Book Loss': book_loss,
                'Bettor Loss': bettor_loss,
                'BCEp': BCEp,
                'BCEq': BCEq
                }


def get_loss(loss_name):
    loss_dict = {
        "BCE": BCELoss,
        "MAE": MAELoss,
        "MSE": MSELoss,
        "Naive": NaiveBetLoss,
        "Betting": BettingLoss,
    }
    return loss_dict[loss_name]()
