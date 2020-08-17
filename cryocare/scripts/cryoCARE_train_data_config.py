import json
from os.path import join, exists

from PyInquirer import prompt, Validator, ValidationError


def main():
    questions = [
        {
            'type': 'input',
            'name': 'even',
            'message': 'Path to the even tomogram:'
        },
        {
            'type': 'input',
            'name': 'odd',
            'message': 'Path to the odd tomogram:'
        },
        {
            'type': 'input',
            'name': 'patch_shape',
            'message': 'Shape of the training volumes:',
            'default': '72,72,72',
            'filter': lambda val: [int(x) for x in val.split(',')]
        },
        {
            'type': 'input',
            'name': 'num_slices',
            'message': 'Number of patches to extract:',
            'default': '1200',
            'validate': lambda val: int(val) > 0,
            'filter': lambda val: int(val)
        },
        {
            'type': 'input',
            'name': 'split',
            'message': 'Train-Validation split:',
            'default': '0.9',
            'validate': lambda val: float(val) > 0.0 and float(val) < 1.0,
            'filter': lambda val: float(val)
        },
        {
            'type': 'input',
            'name': 'path',
            'message': 'Save path:',
            'validate': lambda val: exists(val)
        }
    ]

    config = prompt(questions)
    with open(join(config['path'], 'train_data_config.json'), 'w+') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()