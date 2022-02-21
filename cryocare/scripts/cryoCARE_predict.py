#! python
import argparse
import json
from os.path import join
import datetime
import mrcfile
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

from cryocare.internals.CryoCARE import CryoCARE
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule

import psutil

def pad(volume: NDArray, div_by: Tuple) -> NDArray:
    pads = []
    for axis_index, axis_size in enumerate(volume.shape):
        pad_by = axis_size%div_by[axis_index]
        pads.append([0,pad_by])
    volume_padded = np.pad(volume, pads, mode='mean')

    return volume_padded



def denoise(config: dict, mean: float, std: float, even: str, odd: str, output_file: str):
    model = CryoCARE(None, config['model_name'], basedir=config['path'])

    even = mrcfile.mmap(even, mode='r', permissive=True)
    odd = mrcfile.mmap(odd, mode='r', permissive=True)
    shape_before_pad = even.data.shape
    even_vol = even.data
    odd_vol = odd.data
    even_vol = even_vol
    odd_vol = odd_vol

    div_by = model._axes_div_by('XYZ')

    even_vol = pad(even_vol,div_by=div_by)
    odd_vol = pad(odd_vol, div_by=div_by)

    denoised = np.zeros(even_vol.shape)

    even_vol.shape += (1,)
    odd_vol.shape += (1,)
    denoised.shape += (1,)

    model.predict(even_vol, odd_vol, denoised, axes='ZYXC', normalizer=None, mean=mean, std=std,
                  n_tiles=config['n_tiles'] + [1, ])

    denoised = denoised[slice(0, shape_before_pad[0]), slice(0, shape_before_pad[1]), slice(0, shape_before_pad[2])]
    mrc = mrcfile.new_mmap(output_file, denoised.shape, mrc_mode=2, overwrite=True)
    mrc.data[:] = denoised





    for l in even.header.dtype.names:
        if l == 'label':
            new_label = np.concatenate((even.header[l][1:-1], np.array([
                'cryoCARE                                                ' + datetime.datetime.now().strftime(
                    "%d-%b-%y  %H:%M:%S") + "     "]),
                                        np.array([''])))
            print(new_label)
            mrc.header[l] = new_label
        else:
            mrc.header[l] = even.header[l]
    mrc.header['mode'] = 2

def main():
    parser = argparse.ArgumentParser(description='Run cryoCARE prediction.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    import os
    import tarfile
    import tempfile
    if os.path.isfile(config['path']):
        with tempfile.TemporaryDirectory() as tmpdirname:
            tar = tarfile.open(config['path'], "r:gz")
            tar.extractall(tmpdirname)
            tar.close()
            config['model_name'] = os.listdir(tmpdirname)[0]
            config['path'] = os.path.join(tmpdirname)
            with open(os.path.join(tmpdirname,config['model_name'],"norm.json")) as f:
                norm_data = json.load(f)
                mean = norm_data["mean"]
                std = norm_data["std"]
            os.makedirs(config['output'])

            from glob import glob

            if os.path.isdir(config['even']) and os.path.isdir(config['odd']):
                all_even = glob(os.path.join(config['even'],"*.mrc"))
                all_odd = glob(os.path.join(config['odd'],"*.mrc"))
            else:
                all_even = [config['even']]
                all_odd = [config['odd']]

            for even,odd in zip(all_even,all_odd):
                out_filename = os.path.join(config['output'], os.path.basename(even))
                denoise(config, mean, std, even=even, odd=odd, output_file=out_filename)
    else:
        dm = CryoCARE_DataModule()
        dm.load(config['path'])
        mean, std = dm.train_dataset.mean, dm.train_dataset.std
        denoise(config, mean, std, even=config['even'], odd=config['odd'], output_file=join(config['path'], config['output_name']))




if __name__ == "__main__":
    main()
