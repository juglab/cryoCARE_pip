#! python
import argparse
import json
import warnings
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning


def main():
    parser = argparse.ArgumentParser(description='Load training data generation config.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    dm = CryoCARE_DataModule()
    dm.setup(config['odd'], config['even'], n_samples_per_tomo=config['num_slices'],
                             validation_fraction=(1.0 - config['split']), sample_shape=config['patch_shape'],
                             tilt_axis=config['tilt_axis'], n_normalization_samples=config['n_normalization_samples'])
    dm.save(config['path'])


if __name__ == "__main__":
    # execute only if run as a script
    main()
