import numpy as np
import torch
from torch.utils.data import Dataset


class LinearRegressionDataset(Dataset):

    def __init__(self, weights, noise_data=0., noise_label=0.):
        self.num_weights = len(weights)
        self.weights = weights
        self.noise_data = noise_data
        self.noise_label = noise_label

    def __len__(self):
        return 64 * 64

    def __getitem__(self, index):
        datum = torch.randn(self.num_weights - 1)
        biased_datum = torch.hstack([torch.tensor(1.), datum])
        out = (biased_datum * self.weights).sum().unsqueeze(0)
        noised_datum = datum + self.noise_data * torch.randn(self.num_weights - 1)
        noised_out = out + self.noise_label * torch.randn(1)
        label = torch.sigmoid(noised_out)
        return [noised_datum, label]


class LinearClassificationDataset(Dataset):

    def __init__(self, weights, noise_data=0., noise_label=0.):
        self.num_weights = len(weights)
        self.weights = weights
        self.noise_data = noise_data
        self.noise_label = noise_label

    def __len__(self):
        return 64 * 64

    def __getitem__(self, index):
        datum = torch.randn(self.num_weights - 1)
        biased_datum = torch.hstack([torch.tensor(1.), datum])
        out = (biased_datum * self.weights).sum().unsqueeze(0)
        noised_datum = datum + self.noise_data * torch.randn(self.num_weights - 1)
        noised_out = out + self.noise_label * torch.randn(1)
        label = (noised_out > 0).float()
        return [noised_datum, label]


def get_dataset(num_weights, batch_size, noise_data, noise_label, classification=True):
    np.random.seed(1337)
    weights = torch.from_numpy(np.random.randn(num_weights))
    weights = weights / (weights ** 2).sum()
    LinearDataset = LinearClassificationDataset if classification else LinearRegressionDataset
    pin_memory = torch.cuda.is_available()
    train_dataset = LinearDataset(weights, noise_data, noise_label)
    test_dataset = LinearDataset(weights, 0, 0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, pin_memory=pin_memory)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=pin_memory)
    return train_loader, test_loader