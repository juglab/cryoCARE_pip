#! python
import argparse
import json
from os.path import join
import datetime
import mrcfile
import numpy as np

from cryocare.internals.CryoCARE import CryoCARE
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule

import psutil


def main():
    parser = argparse.ArgumentParser(description='Run cryoCARE prediction.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    dm = CryoCARE_DataModule()
    dm.load(config['path'])

    model = CryoCARE(None, config['model_name'], basedir=config['path'])

    even = mrcfile.mmap(config['even'], mode='r', permissive=True)
    odd = mrcfile.mmap(config['odd'], mode='r', permissive=True)
    denoised = mrcfile.new_mmap(join(config['path'], config['output_name']), even.data.shape, mrc_mode=2,
                                overwrite=True)

    even.data.shape += (1,)
    odd.data.shape += (1,)
    denoised.data.shape += (1,)

    mean, std = dm.train_dataset.mean, dm.train_dataset.std

    model.predict(even.data, odd.data, denoised.data, axes='ZYXC', normalizer=None, mean=mean, std=std,
                  n_tiles=config['n_tiles'] + [1, ])

    for l in even.header.dtype.names:
        if l == 'label':
            new_label = np.concatenate((even.header[l][1:-1], np.array([
                'cryoCARE                                                ' + datetime.datetime.now().strftime(
                    "%d-%b-%y  %H:%M:%S") + "     "]),
                                        np.array([''])))
            print(new_label)
            denoised.header[l] = new_label
        else:
            denoised.header[l] = even.header[l]
    denoised.header['mode'] = 2


if __name__ == "__main__":
    main()
