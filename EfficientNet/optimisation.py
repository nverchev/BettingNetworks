import torch.optim as optim


def helper_lr(initial_learning_rate, model_name):
    lr = {'stem': initial_learning_rate * 0.1, 'blocks': initial_learning_rate * 0.1}

    if model_name in ["BaselineClassifier"]:
        lr['head'] = initial_learning_rate * 0.2
        lr['classifier'] = initial_learning_rate
    elif model_name in ["BettingNetworks"]:
        lr['head'] = initial_learning_rate * 0.2
        lr['book'] = initial_learning_rate
        lr['bettor'] = initial_learning_rate
    elif model_name == "BettingNetworksTwoHeaded":
        lr['head1'] = initial_learning_rate * 0.2
        lr['head2'] = initial_learning_rate * 0.2
        lr['book'] = initial_learning_rate
        lr['bettor'] = initial_learning_rate
    return lr


def get_opt(model_name, opt, initial_learning_rate, weight_decay=0):
    lr = helper_lr(initial_learning_rate, model_name)

    optimizer = {'Adam': optim.Adam,
                 'AdamW': optim.AdamW,
                 'SGD': optim.SGD,
                 'SGD_nesterov': optim.SGD
                 }

    optimi_args = {'Adam': {'weight_decay': weight_decay, 'lr': lr},
                   'Adam': {'weight_decay': weight_decay, 'lr': lr},
                   'SGD': {'weight_decay': weight_decay, 'lr': lr},
                   'SGD_nesterov': {'weight_decay': weight_decay, 'lr': lr,
                                    'momentum': 0.9, 'nesterov': True}}
    return optimizer[opt], optimi_args[opt]
