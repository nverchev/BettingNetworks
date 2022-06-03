import numpy as np
import torch
import matplotlib.pyplot as plt
from Utils.trainer import Trainer
from losses import get_loss

class ClassificationTrainer(Trainer):
    def __init__(self, model, exp_name, block_args):
        super().__init__(model, exp_name, **block_args)
        self.test_probs = None
        self.targets = None
        self.test_pred = None
        self.bins = None
        self.wrong_indices == []

        return

    wrong_indices = []

    # overwrites Trainer method
    def test(self, on='test', prob='book'):
        super().test(on=on)  # stored in RAM
        if prob == 'book':  # standard or book probabilities
            y = torch.stack(self.test_outputs['y'])
            self.test_probs = torch.sigmoid(y)
        elif 'q' not in self.test_outputs.keys():
            print('Bettor probabilities not available')
            return
        elif prob == 'bettor':  # bettor probabilities
            yhat = torch.stack(self.test_outputs['yhat'])
            self.test_probs = torch.sigmoid(yhat)
        else:
            raise ValueError('prob = ' + prob + ' not defined')
        self.test_pred = torch.where(self.test_probs > .5, 1, 0)
        self.targets = torch.stack(self.test_targets)
        right_pred = (self.test_pred == self.targets)
        self.wrong_indices = torch.nonzero(~right_pred)[:, 0]
        acc = 1 - self.wrong_indices.size()[0] / self.targets.size()[0]
        weights_err = self.get_weights_err()
        if not self.quiet_mode:
            print('Accuracy' + f' : {100 * acc:.4f}', end='\t')
            print(f"MSE with true model weights: {weights_err: .2f}")
        return acc, weights_err

    def get_weights_err(self):
        infer_weights = self.model.state_dict()['lin.weight'].squeeze()
        infer_weights = torch.hstack([self.model.state_dict()['lin.bias'], infer_weights])
        target_weights = self.train_loader.dataset.weights.to(self.device)
        weights_err = ((infer_weights - target_weights) ** 2).sum()
        return weights_err

    def prob_analysis(self, on='val', bins=100, prob='book'):  # call after test
        print(self.exp_name)
        if len(self.wrong_indices) == 0:
            self.test(on=on, prob=prob)
        self.bins = bins
        self.uniform_calibration_prediction()
        self.quantile_calibration_prediction()

    def uniform_calibration_prediction(self):

        confidence = np.linspace(1 / (2 * self.bins), 1 - 1 / (2 * self.bins), self.bins)
        wrong_conf = self.test_probs[self.wrong_indices]
        hist = torch.histc(self.test_probs, bins=self.bins, min=0, max=1)
        wrong_hist = torch.histc(wrong_conf, bins=self.bins, min=0, max=1)
        right_hist = hist - wrong_hist
        # Must inverse the prediction for p < 0.5
        class1_hist = np.where(confidence > 0.5, right_hist, hist - right_hist)
        # Empty bins are given expected probability
        obs_prob = np.divide(class1_hist, hist, \
                             out=np.zeros(self.bins), where=hist != 0)

        plt.figure(figsize=(30, 6))
        plt.subplot(1, 4, 1)
        plt.bar(confidence, hist / hist.sum(), width=.9 / self.bins)
        plt.xticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        plt.title('confidence distribution')

        plt.subplot(1, 4, 2)

        plt.bar(confidence, wrong_hist / wrong_hist.sum(), width=.9 / self.bins)
        plt.title('misclassified by conf (normalized)')
        plt.xticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))

        plt.subplot(1, 4, 3)
        plt.bar(confidence, right_hist / right_hist.sum(), width=.9 / self.bins)
        plt.title('correct by conf (normalized)')
        plt.xticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))

        plt.subplot(1, 4, 4)
        plt.title('frequency probability by conf')
        plt.bar(confidence, 1, width=.9 / self.bins, color='w', edgecolor='b', alpha=0.1)
        plt.bar(confidence, obs_prob, width=.9 / self.bins)
        plt.xticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        plt.plot(np.linspace(0, 1, self.bins), np.linspace(0, 1, self.bins), color='seagreen')
        plt.show()

        nonzero = np.nonzero(hist)
        ece = np.abs(obs_prob[nonzero] - confidence[nonzero]).mean()
        print('ECE_pred: ', ece.item())

    def quantile_calibration_prediction(self):
        right_conf = torch.ones_like(self.test_probs)
        right_conf[self.wrong_indices] = 0
        print(f'Brier Score: {torch.mean((self.test_probs - right_conf) ** 2):.2f}')
        confidence, obs_prob = self.quantile_binning(self.test_probs, right_conf, self.bins)
        obs_prob = torch.where(confidence > 0.5, obs_prob, 1 - obs_prob)
        plt.figure(figsize=(30, 6))
        plt.subplot(1, 2, 1)
        plt.title('frequency probability by conf')
        plt.bar(confidence, 1, width=.9 / self.bins, color='w', edgecolor='b', alpha=0.3)
        plt.bar(confidence, obs_prob, width=.9 / self.bins)
        plt.xticks(np.linspace(0, 1, 5), np.linspace(0, 1, 5))
        plt.plot(np.linspace(0, 1, self.bins), np.linspace(0, 1, self.bins), color="seagreen")
        plt.subplot(1, 2, 2)
        plt.title('frequency probability by conf (remapped)')
        plt.bar(np.linspace(0, 1, self.bins), 1, width=.9 / self.bins, color='w', edgecolor='b', alpha=0.3)
        plt.bar(np.linspace(0, 1, self.bins), obs_prob, width=.9 / self.bins)
        ticks = [round(confidence[i].item(), 2) for i in range(0, self.bins, self.bins // 4)]
        plt.xticks(np.linspace(0, 1, 5), ticks + [round(confidence[-1].item(), 2)])
        plt.plot(np.linspace(0, 1, self.bins), confidence, color="seagreen")
        plt.show()

        ece = np.abs(obs_prob - confidence).mean()
        print('Quantile ECE_pred: ', ece.item())
        return

    def coverage(self, crossentropy=False):
        l_test = len(self.test_outputs['probs'])

        if crossentropy:
            assert 'q' in self.test_outputs.keys(), "valid for betting only"

        err = len(self.wrong_indices) / l_test
        acc = 1 - err
        wrong_p = []
        right_p = []
        for i in range(l_test):
            if i in self.wrong_indices:
                wrong_p.append(self.test_outputs['probs'][i])
            else:
                right_p.append(self.test_outputs['probs'][i])

        p = torch.stack(self.test_outputs['probs'])
        wrong_p = torch.stack(wrong_p)
        right_p = torch.stack(right_p)
        pred_p, _ = p.max(dim=1)
        pred_p, _ = pred_p.sort()
        pred_wrong_p, _ = wrong_p.max(dim=1)
        pred_wrong_p, _ = pred_wrong_p.sort()
        pred_right_p, _ = right_p.max(dim=1)
        pred_right_p, _ = pred_right_p.sort()

        new_accs_pred = []
        for i in range(1, 30):
            thresh = 1 - i / 100
            new_l_test = int(thresh * l_test)
            pred_thresh = pred_p[-new_l_test]
            new_right = (pred_right_p > pred_thresh).sum()
            new_accs_pred.append(new_right / new_l_test)
        new_accs_pred = np.array(new_accs_pred)
        impr_new_accs_pred = new_accs_pred / acc
        new_errs = 1 - new_accs_pred
        new_errs_perc = 100 * new_errs / err
        impr_errs = 100 - new_errs_perc
        return impr_new_accs_pred, impr_errs

    @staticmethod
    def quantile_binning(conf, targets, bins):
        conf, order = conf.sort(dim=0)
        targets = targets[order].view(-1, 1)
        tensor = torch.hstack([conf, targets])
        N = tensor.size(0)
        # to cover the case where bins divide N, we add and subtract 1
        large_bins, small_bin = divmod(N + 1, bins - 1)
        avg_conf, avg_corr = [], []
        for t in tensor.split((bins - 1) * [large_bins] + [small_bin - 1]):
            aconf, acorr = t.mean(axis=0)
            avg_conf.append(aconf)
            avg_corr.append(acorr)
        return torch.stack(avg_conf), torch.stack(avg_corr)


def get_trainer(model, exp_name, loss, block_args):
    Loss = get_loss(loss)

    class FinalTrainer(Loss, ClassificationTrainer):
        pass

    return FinalTrainer(model, exp_name, block_args)