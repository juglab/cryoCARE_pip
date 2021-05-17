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

    def file_size(z, y, x):
        return (x * y * z * 32) / (8 * 1024 * 1024 * 1024)

    def get_available_memory():
        return psutil.virtual_memory().available / (1024 * 1024 * 1024)

    def get_n_tiles(z, y, x):
        n_tiles = [1, 1, 1, 1]
        cz = z / n_tiles[0]
        cy = y / n_tiles[1]
        cx = x / n_tiles[2]
        while file_size(cz, cy, cx) > (get_available_memory() / 2.):
            if cz > cy and cz > cx:
                n_tiles[0] *= 2
            elif cy > cx:
                n_tiles[1] *= 2
            else:
                n_tiles[2] *= 2

            cz = z / n_tiles[0]
            cy = y / n_tiles[1]
            cx = x / n_tiles[2]

        return n_tiles

    model.predict(even.data, odd.data, denoised.data, axes='ZYXC', normalizer=None, mean=mean, std=std,
                  n_tiles=get_n_tiles(even.header.nz, even.header.ny, even.header.nx))

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
