import torch.nn.functional as F


class CELoss:
    losses = ['CE']

    def loss(self, outputs, targets):
        y = outputs['y']
        CE = F.cross_entropy(y, targets)
        return {'Criterion': CE,
                'CE': CE}


class MSELoss:
    losses = ['MSE']

    def loss(self, outputs, targets):
        y = outputs['y']
        probs = F.softmax(y, dim=-1)
        targets = F.one_hot(targets, num_classes=self.model.num_classes).float()
        MSE = ((targets - probs) ** 2).sum()
        return {'Criterion': MSE,
                'MSE': MSE,
                }


class NaiveBettingLoss:
    losses = ['Naive']

    def loss(self, outputs, inputs, targets):
        y = outputs['y']
        probs = F.softmax(y, dim=-1)
        #  counteracts bias in loss
        # probs = (probs + 1/self.model.num_classes) / 2
        targets = F.one_hot(targets, num_classes=self.model.num_classes).float()
        naive = ((1 / self.model.num_classes - probs) * (targets - probs)).sum()
        return {'Criterion': naive,
                'Naive': naive}


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

    losses = ['Book Loss', 'Bettor Loss', "CEp", "CEq"]

    def loss(self, outputs, inputs, targets, eps=0):
        y = outputs['y']
        yhat = outputs['yhat']
        CEp = F.cross_entropy(y, targets)
        CEq = F.cross_entropy(yhat, targets)
        targets = F.one_hot(targets, num_classes=self.model.num_classes).float()
        probs = F.softmax(y, dim=-1)
        q = F.softmax(yhat, dim=-1)
        p_detached = probs.detach()
        bettor_loss = ((q - p_detached) * (p_detached - targets - eps)).sum()
        book_loss = ((q.detach() - probs) * (targets - probs - eps)).sum()
        return {'Criterion': book_loss + bettor_loss,
                'Book Loss': book_loss,
                'Bettor Loss': bettor_loss,
                'CEp': CEp,
                'CEq': CEq
                }


class BettingCrossEntropyLoss(BettingLoss):

    def loss(self, outputs, inputs, targets, eps=0):
        y = outputs['y']
        yhat = outputs['yhat']
        CEp = F.cross_entropy(y, targets)
        CEq = F.cross_entropy(yhat, targets)
        targets = F.one_hot(targets, num_classes=self.model.num_classes).float()
        probs = F.softmax(y, dim=-1)
        q = F.softmax(yhat, dim=-1)
        bettor_loss = CEq
        book_loss = ((q.detach() - probs) * (targets - probs - eps)).sum()
        return {'Criterion': book_loss + bettor_loss,
                'Book Loss': book_loss,
                'Bettor Loss': bettor_loss,
                'CEp': CEp,
                'CEq': CEq
                }
