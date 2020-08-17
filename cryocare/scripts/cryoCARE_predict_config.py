import json
from os.path import join, exists

from PyInquirer import prompt, Validator, ValidationError


def main():
    questions = [
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
        },
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
            'name': 'output_name',
            'message': 'Name of the denoised file:',
            'default': 'denoised.mrc'
        },
        {
            'type': 'input',
            'name': 'mrc_slice_shape',
            'message': 'Tomo sub-volume size loaded into memory:',
            'default': '1200, 1200, 1200',
            'filter': lambda val: [int(x) for x in val.split(',')]
        }
    ]

    config = prompt(questions)
    with open(join(config['path'], 'predict_config.json'), 'w+') as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    main()