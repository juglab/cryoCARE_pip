#! python
import argparse
import json
from os.path import join
import os
import tarfile
import tempfile
import datetime
import mrcfile
import numpy as np
import sys
import tensorflow as tf
from typing import Tuple

from cryocare.internals.CryoCARE import CryoCARE
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule

import psutil

def set_gpu_id(config: dict):
    if 'gpu_id' in config:
        if type(config['gpu_id']) is list:
            gpu_ids = config['gpu_id']
            if len(gpu_ids) == 0:
                raise RuntimeError('ERROR: List of GPU IDs is empty')
        elif type(config['gpu_id']) is int:
            gpu_ids = [config['gpu_id']]
        else:
            raise RuntimeError('gpu_id in json is neither a list nor an integer')
    else:
        if len(tf.config.list_physical_devices('GPU')) > 0:
            gpu_ids = list(range(0,len(tf.config.list_physical_devices('GPU'))))
        else:
            print('WARNING: No GPUs found by tensorflow')
    
    #Check GPUs given by IDs exist and set_memory_growth to True
    physical_devices = []
    try:
        for gpu in gpu_ids:
            print(f'Looking for GPU with ID: {gpu}')
            physical_devices = physical_devices + [tf.config.list_physical_devices('GPU')[gpu]]
            print(f'GPU {gpu} successfully found')
            tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[gpu], True)
    except IndexError:
        print(f'WARNING: GPU {gpu} not found')
    
    if len(physical_devices) > 0:
        tf.config.set_visible_devices(physical_devices, 'GPU') 

def pad(volume: np.array, div_by: Tuple) -> np.array:
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

    try:
        os.makedirs(config['output'])
    except OSError:
        if 'overwrite' in config and config['overwrite']:
            os.makedirs(config['output'], exist_ok=True)
        else:
            print("Output directory already exists. Please choose a new output directory or set 'overwrite' to 'true' in your configuration file.")
            sys.exit(1)
    
    set_gpu_id(config)
    
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



            from glob import glob
            if type(config['even']) is list:
                all_even=tuple(config['even'])
                all_odd=tuple(config['odd'])
            elif os.path.isdir(config['even']) and os.path.isdir(config['odd']):
                all_even = glob(os.path.join(config['even'],"*.mrc"))
                all_odd = glob(os.path.join(config['odd'],"*.mrc"))
            else:
                all_even = [config['even']]
                all_odd = [config['odd']]

            for even,odd in zip(all_even,all_odd):
                out_filename = os.path.join(config['output'], os.path.basename(even))
                denoise(config, mean, std, even=even, odd=odd, output_file=out_filename)
    else:
        # Fall back to original cryoCARE implmentation
        s = f" {config['path']} is not a file"
        if os.path.exists(config['path']):
            s = f" {config['path']} does not exist"
        print(f"The specified 'path' {s}. Your config is not in the format that cryoCARE >=0.2 requires. Fallback to cryCARE 0.1 format.")
        if 'output_name' not in config or os.path.isfile(config['path']):
            print("Invalid config format.")
            sys.exit(1)

        dm = CryoCARE_DataModule()
        dm.load(config['path'])
        mean, std = dm.train_dataset.mean, dm.train_dataset.std

        denoise(config, mean, std, even=config['even'], odd=config['odd'], output_file=join(config['path'], config['output_name']))




if __name__ == "__main__":
    main()
