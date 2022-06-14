import argparse
import numpy as np
import torch
from torch.utils.data import Dataset
from dataset import get_dataset
from model import get_model
from trainer import get_trainer
from optimisation import get_opt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Utils.Scheduling import CosineSchedule


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--loss', default='BCE', choices=["BCE", "MAE", "MSE", "HuberLoss", "Naive", "Betting"])
    parser.add_argument('--classification', action=argparse.BooleanOptionalAction, default=True, help='classification '
                                                                                                      'setup')
    parser.add_argument('--num_weights', type=int, default=64, help='weights of the linear models')
    parser.add_argument('--noise_data', type=float, default=0., help='standard deviation noise samples')
    parser.add_argument('--noise_label', type=float, default=0., help='standard deviation noise label (before '
                                                                      'thresholding)')
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment.')
    parser.add_argument('--epochs', default=60, type=int, help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--n_tests', type=int, default=10, help='repeats the whole training and aggregates metrics')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"]
                        , help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--cuda', action=argparse.BooleanOptionalAction, default=True, help='enables CUDA training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    loss_name = args.loss
    classification = args.classification
    print(classification)
    noise_data = args.noise_data
    noise_label = args.noise_label
    num_weights = args.num_weights
    epochs = args.epochs
    n_tests = args.n_tests
    batch_size = args.batch_size
    opt = args.optimizer
    initial_learning_rate = args.lr
    weight_decay = args.wd
    exp_name = args.exp_name
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    print(args.cuda)
    train_loader, test_loader = get_dataset(num_weights, batch_size, noise_data, noise_label)
    optimizer, optim_args = get_opt(opt, initial_learning_rate, weight_decay)
    block_args = {
        'optim_name': opt,
        'optim': optimizer,
        'optim_args': optim_args,
        'train_loader': train_loader,
        'device': device,
        'test_loader': test_loader,
        'batch_size': batch_size,
        'schedule': CosineSchedule(decay_steps=epochs, min_decay=0.1)
    }
    accs, weights_errs = [], []
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)
    trainer = None
    for _ in range(n_tests):
        model = get_model(loss_name)(num_weights)
        trainer = get_trainer(model, loss_name, exp_name, classification, block_args)
        trainer.train(epochs)
        acc, weights_err = trainer.test(partition="test")
        accs.append(acc)
        weights_errs.append(weights_err.cpu())
    if classification:
        trainer.prob_analysis(partition='test')
        print(f"Overall accuracy: {np.array(accs).mean():.4f}", end="")
    print(f"\nOverall weight_err: {np.array(weights_errs).mean():.4f}")

