import numpy as np
import torch
import matplotlib.pyplot as plt
from Utils.trainer import Trainer
from losses import get_loss
from sklearn import metrics
import torch.nn.functional as F


class ClassificationTrainer(Trainer):
    _metrics = {}
    average = "macro"
    quiet_mode = True
    bin = 'bettingnetworksefficient'

    def __init__(self, model, loss_name, exp_name, block_args):
        super().__init__(model, exp_name, **block_args)
        self._loss = get_loss(loss_name)
        self.test_probs = None
        self.targets = None
        self.test_pred = None
        self.bins = None
        self.wrong_indices = []
        return

    def loss(self, output, inputs, targets):
        return self._loss(output, targets)

    # overwrites Trainer method
    def test(self, partition='test', prob='book'):
        super().test(partition=partition)  # stored in RAM
        if prob == 'book':  # standard or book probabilities
            y = torch.stack(self.test_outputs['y'])
            self.test_probs = F.softmax(y, dim=-1)
        elif 'q' not in self.test_outputs.keys():
            print('Bettor probabilities not available')
            return
        elif prob == 'bettor':  # bettor probabilities
            yhat = torch.stack(self.test_outputs['yhat'])
            self.test_probs = F.softmax(yhat, dim=-1)
        else:
            raise ValueError('prob = ' + prob + ' not defined')
        self.test_pred = torch.argmax(self.test_probs, dim=1)
        self.targets = torch.stack(self.test_targets)
        right_pred = (self.test_pred == self.targets)
        self.wrong_indices = torch.nonzero(~right_pred).squeeze()
        self.calculate_metrics()
        return

    @property
    def metrics(self):
        self.test(partition='val')
        return self._metrics

    def calculate_metrics(self, print_flag=True):

        avg_type = self.average.capitalize() + ' ' if self.test_probs.size(1) > 1 else ""
        # calculates common and also gives back the indices of the wrong guesses

        self._metrics['Accuracy'] = \
            metrics.accuracy_score(self.targets, self.test_pred)
        self._metrics[avg_type + 'F1 Score'] = \
            metrics.f1_score(self.targets, self.test_pred, average=self.average)
        self._metrics[avg_type + 'Jaccard Score'] = \
            metrics.jaccard_score(self.targets,
                                  self.test_pred, average=self.average)
        self._metrics[avg_type + 'AUC ROC'] = \
            metrics.roc_auc_score(self.targets, self.test_probs,
                                  average=self.average, multi_class='ovr')
        if print_flag:
            for metric, value in self._metrics.items():
                print(metric + f' : {value:.4f}', end='\t')
            print('')
        return

    def prob_analysis(self, partition='val', bins=100, prob='book'):  # call after test
        self.bins = bins
        print(self.version)
        if len(self.wrong_indices) == 0:
            self.test(partition=partition, prob=prob)
        self.uniform_calibration_prediction()
        self.uniform_calibration()
        self.quantile_calibration_prediction()
        self.quantile_calibration()

    def uniform_calibration_prediction(self):
        confidence = np.linspace(1 / (2 * self.bins), 1 - 1 / (2 * self.bins), self.bins)
        highest_conf = torch.max(self.test_probs, dim=1)[0]
        wrong_conf = highest_conf[self.wrong_indices]
        hist = torch.histc(highest_conf, bins=self.bins, min=0, max=1)
        wrong_hist = torch.histc(wrong_conf, bins=self.bins, min=0, max=1)
        right_hist = hist - wrong_hist

        # Empty bins are given expected probability
        obs_prob = np.divide(right_hist, hist, \
                             out=np.zeros(self.bins), where=hist != 0)
        print('Max confidence')
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

    def uniform_calibration(self):
        confidence = np.linspace(1 / (2 * self.bins), 1 - 1 / (2 * self.bins), self.bins)
        ece = 0
        for cls in range(self.model.num_classes):
            conf = self.test_probs[:, cls]
            right_indices = torch.nonzero(self.targets == cls)
            right_conf = conf[right_indices]
            hist = torch.histc(conf, bins=self.bins, min=0, max=1)
            right_hist = torch.histc(right_conf, bins=self.bins, min=0, max=1)
            nonzero = np.nonzero(hist)
            obs_prob = np.divide(right_hist, hist, where=hist != 0)
            ece += torch.abs(obs_prob[nonzero] - confidence[nonzero]).mean()

        targets = F.one_hot(self.targets, num_classes=self.model.num_classes).float()
        brier_multi = torch.mean(torch.sum((self.test_probs - targets) ** 2, axis=1))
        print('ECE: ', (ece / self.model.num_classes).item(), ' Brier Score: ', brier_multi.item())
        return

    def quantile_calibration_prediction(self):
        highest_conf = torch.max(self.test_probs, dim=1)[0]
        right_conf = torch.ones_like(highest_conf)
        right_conf[self.wrong_indices] = 0
        confidence, obs_prob = self.quantile_binning(highest_conf, right_conf, self.bins)
        print('Max conf quantile')
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
        plt.xticks(np.linspace(0, 1, 5), [round(confidence[i].item(), 2) for i in range(0, self.bins, self.bins // 5)])
        plt.plot(np.linspace(0, 1, self.bins), confidence, color="seagreen")
        plt.show()

        ece = np.abs(obs_prob - confidence).mean()
        print('Quantile ECE_pred: ', ece.item())
        return

    def quantile_calibration(self):
        ece = 0
        for cls in range(self.model.num_classes):
            conf = self.test_probs[:, cls]
            right_conf = (self.targets == cls).float()
            confidence, obs_prob = self.quantile_binning(conf, right_conf, self.bins)
            ece += np.abs(obs_prob - confidence).mean()
        print('Quantile ECE: ', (ece / self.model.num_classes).item())
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
        conf, order = conf.sort()
        targets = targets[order]
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


def get_trainer(model, loss_name, exp_name, block_args):
    return ClassificationTrainer(model, loss_name, exp_name, block_args)
