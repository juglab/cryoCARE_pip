import argparse
import json
import mrcfile
import warnings
from tqdm import tqdm
import numpy as np
from os.path import join

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

    even = mrcfile.mmap(config['even'], mode='r', permissive=True)
    odd = mrcfile.mmap(config['odd'], mode='r', permissive=True)

    slices = sample_patch_slices(even.data.shape, patch_shape=config['patch_shape'], num_slices=config['num_slices'])
    patches = []
    for c in tqdm(slices):
        patches.append([
                            even.data[c[0], c[1], c[2]][...,np.newaxis],
                            odd.data[c[0], c[1], c[2]][...,np.newaxis]
                       ])

    patches = np.array(patches)
    mean = np.mean(patches)
    std = np.std(patches)

    np.savez(join(config['path'], "mean_std.npz"), mean=mean, std=std)
    patches = (patches - mean)/std

    np.random.shuffle(patches)

    split = int(len(patches) * config['split'])
    np.savez(join(config['path'], "train_data.npz"),
             X=patches[:split, 0],
             Y=patches[:split, 1],
             X_val=patches[split:, 0],
             Y_val=patches[split:, 1])



def sample_patch_slices(img_shape, patch_shape=(72,72,72), num_slices=1200):
    slices = []
    for z in range(0, img_shape[0] - patch_shape[0], patch_shape[0]):
        for y in range(0, img_shape[1] - patch_shape[1], patch_shape[1]):
            for x in range(0, img_shape[2] - patch_shape[2], patch_shape[2]):
                slices.append([slice(z, z + patch_shape[0]),
                               slice(y, y + patch_shape[1]),
                               slice(x, x + patch_shape[2])])

    if len(slices) < num_slices:
        warnings.warn("Requested {} patches but only {} patches are available.".format(num_slices, len(slices)))
        return slices

    return slices[:num_slices]


if __name__ == "__main__":
    # execute only if run as a script
    main()