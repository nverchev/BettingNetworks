import torch
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import re
from torch import optim
import torch.cuda.amp as amp
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from abc import ABCMeta, abstractmethod
import gc
from sklearn import metrics
import torchvision
from PIL import Image
from torchvision.datasets.cifar import CIFAR100
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from PIL.Image import BICUBIC
from collections import OrderedDict


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):

    def forward(self, x):
        return x.flatten(1)


class SqueezeExcitation(nn.Module):

    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


class MBConv(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride,
                 expand_rate=1.0, se_rate=0.25,
                 drop_connect_rate=0.2):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish()
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(inplanes, expand_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=expand_planes,
                      bias=False),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish()
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = drop_connect_rate

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        # Add identity skip
        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode='fan_out')
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):

    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(self, num_classes=100,
                 width_coefficient=1.0,
                 depth_coefficient=1.0,
                 se_rate=0.25,
                 dropout_rate=0.2,
                 drop_connect_rate=0.2,
                 list_channels=[32, 16, 24, 40, 80, 112, 192, 320, 1280]):
        super(EfficientNet, self).__init__()
        self.settings = {}
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8
        self.num_classes = num_classes

        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        # Define stem:
        self.stem = nn.Sequential(
            nn.Conv2d(3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish()
        )

        # Define MBConv blocks
        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):

            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append((
                name,
                MBConv(num_channels, next_num_channels,
                       kernel_size=kernel_size, stride=stride, expand_rate=expand_rate,
                       se_rate=se_rate, drop_connect_rate=drop_rate)
            ))
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append((
                    name,
                    MBConv(next_num_channels, next_num_channels,
                           kernel_size=kernel_size, stride=1, expand_rate=expand_rate,
                           se_rate=se_rate, drop_connect_rate=drop_rate)
                ))
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.head_gen = lambda: nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1],
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten()
        )
        self.classifier_gen = lambda: nn.Sequential(nn.Dropout(p=dropout_rate),
                                                    nn.Linear(list_channels[-1], num_classes)
                                                    )

        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return x


class BaselineClassifier(EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self.head_gen()
        self.classifier = self.classifier_gen()

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
        y = self.classifier(x)
        return {'y': y,
                'probs': F.softmax(y, dim=1)}


class BettingNetworks(EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head = self.head_gen()
        self.book = self.classifier_gen()
        self.bettor = self.classifier_gen()

    def forward(self, x):
        x = super().forward(x)
        x = self.head(x)
        hidden = x.detach().clone()
        y = self.book(x)
        p = F.softmax(y, dim=1)
        yhat = self.bettor(hidden)
        q = F.softmax(yhat, dim=1)
        return {'y': y,
                'yhat': yhat,
                'probs': p,
                'q': q
                }


class BettingNetworksTwoHeaded(EfficientNet):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.head1 = self.head_gen()
        self.book = self.classifier_gen()
        self.head2 = self.head_gen()
        self.bettor = self.classifier_gen()

    def forward(self, x):
        x = super().forward(x)
        detached = x.detach().clone()
        x = self.head1(x)
        y = self.book(x)
        p = F.softmax(y, dim=1)
        x = self.head2(detached)
        yhat = self.bettor(x)
        q = F.softmax(yhat, dim=1)
        return {'y': y,
                'yhat': yhat,
                'probs': p,
                'q': q
                }


def get_model(model_name, experiment, dirpath="./"):
    model_class = {"BaselineClassifier": BaselineClassifier,
                   "BettingNetworks": BettingNetworks,
                   "BettingNetworksTwoHeaded": BettingNetworksTwoHeaded,
                   "BettingCrossEntropy": BettingNetworks,
                   "BaselineClassifier_no_momentum": BaselineClassifier,
                   }

    num_classes = 50 if experiment.split("./") == "imbalanced" else 100
    model = model_class[model_name](num_classes=num_classes)
    efficient_path = os.path.join(dirpath, 'EfficientNet')
    B0_state = torch.load(os.path.join(efficient_path, "efficientnet-b0-08094119.pth"))

    # A basic remapping is required
    mapping = {
        k: v for k, v in zip(B0_state.keys(), model.state_dict().keys())
    }

    mapped_B0_state = OrderedDict([
        (mapping[k], v) for k, v in B0_state.items()
    ])
    # the last layer has different dimensions
    mapped_B0_state.popitem()
    mapped_B0_state.popitem()

    mapped_model_state = OrderedDict()
    for key, value in mapped_B0_state.items():
        if key[:5] == 'head1':
            # has effect only with TwoHeaded Betting Networks
            layername = key[5:]
            mapped_model_state['head2' + layername] = value
            mapped_model_state[key] = value
        else:
            mapped_model_state[key] = value

    model.load_state_dict(mapped_model_state, strict=False)
    return model
