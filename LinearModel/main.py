import argparse
import torch
from torch.utils.data import Dataset
from dataset import get_dataset
from model import get_model
from trainer import get_trainer
from Utils.optim import get_opt, CosineSchedule


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--loss', default='BCELoss', choices=["BCE", "MAE", "MSE", "Naive", "Betting", "CrossBet"])
    parser.add_argument('--classification', type=bool, default=True, help='classification setup')
    parser.add_argument('--num_weights', type=int, default=64, help='weights of the linear models')
    parser.add_argument('--noise_data', type=float, default=0., help='standard deviation noise samples')
    parser.add_argument('--noise_label', type=float, default=0., help='standard deviation noise label (before '
                                                                      'thresholding)')
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment.')
    parser.add_argument('--epochs', default=60, type=int, help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='decay rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"]
                        , help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    loss = args.loss
    classification = args.classification
    noise_data = args.noise_data
    noise_label = args.noise_label
    num_weights = args.num_weights
    epochs = args.epochs
    batch_size = args.batch_size
    opt = args.optimizer
    initial_learning_rate = args.lr
    weight_decay = args.wd
    exp_name = args.exp_name
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    train_loader, test_loader = get_dataset(num_weights, batch_size, noise_data, \
                                            noise_label, classification)
    model = get_model(loss)(num_weights)
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
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)

    trainer = get_trainer(model, exp_name, loss, block_args)

    trainer.silent_mode = True
    trainer.train(epochs)
