import numpy as np
import torch
import os
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets.cifar import CIFAR100
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def A_transform(transform):
    return lambda img: transform(image=np.array(img))['image']


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, always_apply=True),
    A.Sharpen((0.1, 0.2)),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.SmallestMaxSize(256, interpolation=2),
    A.Affine(shear=(-5, 5)),
    A.ShiftScaleRotate(rotate_limit=10, p=0.5),
    A.RandomCrop(224, 224),
    A.CoarseDropout(max_holes=1, max_height=56, max_width=56, p=0.3),
    ToTensorV2(),
])
test_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    A.SmallestMaxSize(224, interpolation=2),
    A.pytorch.ToTensorV2(),
])


def get_dataset(experiment, batch_size, dir_path="./", Loader=True):
    experiment_strings = experiment.split("_")
    final = experiment_strings[0] == 'final'
    noised_label_perc = 0
    if experiment_strings[-1] == 'noise':
        if experiment_strings[-2] == "extreme":
            noised_label_perc = 10
        if experiment_strings[-2] == "heavy":
            noised_label_perc = 5
        if experiment_strings[-2] == "medium":
            noised_label_perc = 3
        if experiment_strings[-2] == "light":
            noised_label_perc = 1
    class_imbalanced = experiment_strings[-1] == "imbalanced"
    data_path = os.path.join(dir_path, 'Cifar100')
    split = 1 / 6
    pin_memory = torch.cuda.is_available()
    train_dataset = CIFAR100(root=data_path, train=True, transform=A_transform(train_transform), download=Loader)
    val_dataset = CIFAR100(root=data_path, train=True, transform=A_transform(test_transform), download=False)
    test_dataset = CIFAR100(root=data_path, train=False, transform=A_transform(test_transform), download=Loader)
    num_train = len(train_dataset)
    split = int(np.floor(split * num_train))
    indices = list(range(num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # adding noise to targets during training
    if noised_label_perc > 0:
        np.random.seed = 123
        jump = 100 // noised_label_perc
        for i in range(0, len(train_dataset), jump):
            train_dataset.targets[i] = np.random.randint(100)

    # class imbalance
    if class_imbalanced:
        for i in range(len(train_dataset)):
            train_dataset.targets[i] = min(49, train_dataset.targets[i])
        for i in range(len(test_dataset)):
            test_dataset.targets[i] = min(49, test_dataset.targets[i])

    if final:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=pin_memory
        )
        val_loader = None
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size, sampler=train_sampler, drop_last=True,
                                                   pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, sampler=valid_sampler, pin_memory=pin_memory
        )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                              shuffle=False, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader
