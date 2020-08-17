import json
from os.path import join, exists

from PyInquirer import prompt, Validator, ValidationError


def main():
    questions = [
        {
            'type': 'input',
            'name': 'train_data',
            'message': 'Path to the training data:'
        },
        {
            'type': 'input',
            'name': 'epochs',
            'message': 'Number of training epochs:',
            'default': '200',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'batch_size',
            'message': 'Training batch size:',
            'default': '16',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'unet_kern_size',
            'message': 'U-Net convolution kernel size:',
            'default': '3',
            'validate': lambda val: int(val) > 0 and int(val) % 2 == 1,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'unet_n_depth',
            'message': 'U-Net depth:',
            'default': '3',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'unet_n_first',
            'message': 'U-Net number of initial channels:',
            'default': '16',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'learning_rate',
            'message': 'Learning rate:',
            'default': '0.0004',
            'validate': lambda val: float(val) > 0,
            'filter': lambda val: float(val)
        },
        {
            'type': 'input',
            'name': 'model_name',
            'message': 'Model name:'
        },
        {
            'type': 'input',
            'name': 'path',
            'message': 'Path:',
            'validate': lambda val: exists(val)
        }
    ]

    config = prompt(questions)
    with open(join(config['path'], 'train_config.json'), 'w+') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()