import argparse
import json
from os.path import join
import datetime
import mrcfile
import numpy as np
from csbdeep.internals.predict import tile_overlap

from cryocare.internals.CryoCARE import CryoCARE


def main():
    parser = argparse.ArgumentParser(description='Run cryoCARE prediction.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    model = CryoCARE(None, config['model_name'], basedir=config['path'])

    even = mrcfile.mmap(config['even'], mode='r', permissive=True)
    odd = mrcfile.mmap(config['odd'], mode='r', permissive=True)
    denoised = mrcfile.new_mmap(join(config['path'], config['output_name']), even.data.shape, mrc_mode=2, overwrite=True)

    mean_std = np.load(join(config['path'], 'mean_std.npz'))
    mean, std = mean_std['mean'], mean_std['std']

    overlap = tile_overlap(model.config.unet_n_depth, model.config.unet_kern_size)

    mrc_slice_shape = config['mrc_slice_shape']

    print(even.data.shape)
    print(denoised.data.shape)
    for z in range(0, even.data.shape[0], mrc_slice_shape[0] - 2 * overlap):
        for y in range(0, even.data.shape[1], mrc_slice_shape[1] - 2 * overlap):
            for x in range(0, even.data.shape[2], mrc_slice_shape[2] - 2 * overlap):
                start_z, end_z = fix_slice(z, z + mrc_slice_shape[0], even.data.shape[0])
                start_y, end_y = fix_slice(y, y + mrc_slice_shape[1], even.data.shape[1])
                start_x, end_x = fix_slice(x, x + mrc_slice_shape[2], even.data.shape[2])

                even_slice = even.data[start_z:end_z, start_y:end_y, start_x:end_x]
                odd_slice = odd.data[start_z:end_z, start_y:end_y, start_x:end_x]

                even_slice = (even_slice - mean) / std
                odd_slice = (odd_slice - mean) / std
                even_slice = (model.predict(even_slice, axes='ZYX', normalizer=None) * std) + mean
                odd_slice = (model.predict(odd_slice, axes='ZYX', normalizer=None) * std) + mean

                img_slice_z, slice_slice_z = get_slices(start_z, end_z, overlap,
                                                        even.data.shape[0])
                img_slice_y, slice_slice_y = get_slices(start_y, end_y, overlap,
                                                        even.data.shape[1])
                img_slice_x, slice_slice_x = get_slices(start_x, end_x, overlap,
                                                        even.data.shape[2])

                print(img_slice_x, slice_slice_x)
                even_slice = even_slice[slice_slice_z, slice_slice_y, slice_slice_x]
                odd_slice = odd_slice[slice_slice_z, slice_slice_y, slice_slice_x]
                print(even_slice.shape)
                denoised.data[img_slice_z, img_slice_y, img_slice_x] = (even_slice + odd_slice) / 2.0

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


def get_slices(start, end, overlap, img_shape):
    img_start = start + overlap if start > 0 else 0
    img_end = end - overlap if end < img_shape - overlap else -1

    if img_start == 0 and img_end == img_shape:
        slice_start = 0
        slice_end = -1
    elif img_start == 0 and img_end < img_shape:
        slice_start = 0
        slice_end = -overlap
    elif img_start > 0 and img_end == img_shape:
        slice_start = overlap
        slice_end = -1
    else:
        slice_start = overlap
        slice_end = -overlap

    return slice(img_start, img_end), slice(slice_start, slice_end)


def fix_slice(start, end, max_len):
    if end > max_len:
        end = -1

    return start, end


if __name__ == "__main__":
    main()
