import torch.optim as optim


def get_opt(opt, initial_learning_rate, weight_decay=0):
    optimizer = {'Adam': optim.Adam,
                 'AdamW': optim.AdamW,
                 'SGD': optim.SGD,
                 'SGD_nesterov': optim.SGD
                 }

    optimi_args = {'Adam': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'Adam': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'SGD': {'weight_decay': weight_decay, 'lr': initial_learning_rate},
                   'SGD_nesterov': {'weight_decay': weight_decay, 'lr': initial_learning_rate,
                                    'momentum': 0.9, 'nesterov': True}}
    return optimizer[opt], optimi_args[opt]
