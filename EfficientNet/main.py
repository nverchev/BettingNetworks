import argparse
import torch
from minio import Minio
from dataset import get_dataset
from model import get_model
from trainer import get_trainer
from optimisation import get_opt
from Utils.Scheduling import CosineSchedule


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--model_name', default='BaselineClassifier',
                        choices=['BaselineClassifier', 'BettingNetworks', 'BettingNetworksTwoHeaded'])
    parser.add_argument('--loss', default='BCELoss', choices=["BCE", "MAE", "MSE", "Naive", "Betting", "CrossBet"])
    parser.add_argument('--dir_path', type=str, default='./', help='Directory for storing data and models')
    parser.add_argument('--exp_name', type=str, default='', help='Name of the experiment.')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model (exp_name needs to start with "final")')
    parser.add_argument('--epochs', default=60, type=int, help='number of epoch in training')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size in training')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='decay rate')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=["SGD", "SGD_nesterov", "Adam", "AdamW"]
                        , help='SGD has no momentum, otherwise momentum = 0.9')
    parser.add_argument('--wd', type=float, default=0., help='weight decay')
    parser.add_argument('--cuda', type=bool, default=True, help='enables CUDA training')
    parser.add_argument('--minio_credential', type=str, default='',
                        help='path of file with written server.access_key.secret_key')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    loss_name = args.loss
    model_name = args.model_name
    model_eval = args.eval
    dir_path = args.dir_path
    training_epochs = args.epochs
    batch_size = args.batch_size
    opt = args.optimizer
    learning_rate = args.lr
    weight_decay = args.wd
    experiment = args.experiment
    minio_credential = args.minio_credential
    if minio_credential:
        with open(minio_credential) as f:
            server, access_key, secret_key = f.readline().split(';')
            secret_key = secret_key.strip()
            minioClient = Minio(server, access_key=access_key, secret_key=secret_key, secure=True)
    else:
        minio_credential = None

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    train_loader, val_loader, test_loader = get_dataset(experiment, batch_size, dirpath="./")
    optimizer, optim_args = get_opt(model_name, opt, learning_rate, weight_decay)
    block_args = {
        'optim_name': opt,
        'optim': optimizer,
        'optim_args': optim_args,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'device': device,
        'batch_size': batch_size,
        'schedule': CosineSchedule(),
        'minioClient': minioClient,
        'dir_path': dir_path
    }
    for k, v in block_args.items():
        if not isinstance(v, (type, torch.utils.data.dataloader.DataLoader)):
            print(k, ': ', v)

    model = get_model(model_name, experiment)
    exp_name = '_'.join([model_name, experiment])
    trainer = get_trainer(model, loss_name, exp_name, block_args)

    if not model_eval:
        for _ in range(training_epochs // 10):
            trainer.train(10)
            if experiment[:5] != 'final':
                trainer.test(partition="val", m=512)
            trainer.save()

    if experiment[:5] == 'final':
        trainer.test(partition='test')
