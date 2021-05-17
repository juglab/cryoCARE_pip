import numpy as np
import tensorflow as tf

import mrcfile
import tqdm
import numpy as np

from os.path import join


class CryoCARE_Dataset(tf.keras.utils.Sequence):
    def __init__(self, tomo_paths_odd=None, tomo_paths_even=None, n_samples_per_tomo=None,
                 extraction_shapes=None, mean=None, std=None,
                 sample_shape=(64, 64, 64), shuffle=True, n_normalization_samples=500):
        self.tomo_paths_odd = tomo_paths_odd
        self.tomo_paths_even = tomo_paths_even
        self.n_samples_per_tomo = n_samples_per_tomo

        self.extraction_shapes = extraction_shapes
        self.mean = mean
        self.std = std

        self.sample_shape = np.array(list(sample_shape))
        self.shuffle = shuffle
        self.coords = None

        self.tomos_odd = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_odd]
        self.tomos_even = [mrcfile.mmap(p, mode='r', permissive=True) for p in self.tomo_paths_even]
        self.n_tomos = len(self.tomo_paths_odd)

        self.create_coordinate_lists()
        self.length = sum([c.shape[0] for c in self.coords])

        if self.shuffle:
            self.indices = np.random.permutation(self.length)
        else:
            self.indices = np.arange(self.length)

        if self.mean == None or self.std == None:
            self.compute_mean_std(n_samples=n_normalization_samples)

    def save(self, path):
        np.savez(path,
                 tomo_paths_odd=self.tomo_paths_odd,
                 tomo_paths_even=self.tomo_paths_even,
                 mean=self.mean,
                 std=self.std,
                 n_samples_per_tomo=self.n_samples_per_tomo,
                 extraction_shapes=self.extraction_shapes,
                 sample_shape=self.sample_shape,
                 shuffle=self.shuffle,
                 coords=self.coords)

    @classmethod
    def load(cls, path):
        tmp = np.load(path)
        tomo_paths_odd = [str(p) for p in tmp['tomo_paths_odd']]
        tomo_paths_even = [str(p) for p in tmp['tomo_paths_even']]
        mean = tmp['mean']
        std = tmp['std']
        n_samples_per_tomo = tmp['n_samples_per_tomo']
        extraction_shapes = tmp['extraction_shapes']
        sample_shape = tmp['sample_shape']
        shuffle = tmp['shuffle']
        coords = tmp['coords']

        ds = cls(tomo_paths_odd=tomo_paths_odd,
                 tomo_paths_even=tomo_paths_even,
                 mean=mean,
                 std=std,
                 n_samples_per_tomo=n_samples_per_tomo,
                 extraction_shapes=extraction_shapes,
                 sample_shape=sample_shape,
                 shuffle=shuffle)
        ds.coords = coords
        return ds

    def compute_mean_std(self, n_samples=2000):
        samples = []
        print('Computing normalization parameters:')
        for i in tqdm.trange(n_samples):
            x, _ = self.__getitem__(i)
            samples.append(x)

        self.mean = np.mean(samples)
        self.std = np.std(samples)
        del (samples)

    def create_coordinate_lists(self):
        self.coords = []
        for odd, even, es in zip(self.tomo_paths_odd, self.tomo_paths_even, self.extraction_shapes):
            self.coords.append(self.__create_coords_for_tomo__(even, odd, es))

        self.coords = np.array(self.coords)

    def __create_coords_for_tomo__(self, even_path, odd_path, extraction_shape):
        even = mrcfile.mmap(even_path, mode='r')
        odd = mrcfile.mmap(odd_path, mode='r')

        assert even.data.shape == odd.data.shape, '{} and {} tomogram have different shapes.'.format(even_path,
                                                                                                     odd_path)
        assert even.data.shape[0] > 2 * self.sample_shape[0]
        assert even.data.shape[1] > 2 * self.sample_shape[1]
        assert even.data.shape[2] > 2 * self.sample_shape[2]

        coords = self.create_random_coords(extraction_shape[0],
                                           extraction_shape[1],
                                           extraction_shape[2],
                                           n_samples=self.n_samples_per_tomo)

        even.close()
        odd.close()

        return coords

    def create_random_coords(self, z, y, x, n_samples):
        z_coords = np.random.randint(z[0], z[1] - self.sample_shape[0], size=n_samples)
        y_coords = np.random.randint(y[0], y[1] - self.sample_shape[0], size=n_samples)
        x_coords = np.random.randint(x[0], x[1] - self.sample_shape[0], size=n_samples)

        return np.stack([z_coords, y_coords, x_coords], -1)

    def random_swapper(self, x, y):
        if np.random.rand() > 0.5:
            return y, x
        else:
            return x, y

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        tomo_index, coord_index = idx // self.n_samples_per_tomo, idx % self.n_samples_per_tomo
        z, y, x = self.coords[tomo_index][coord_index]

        even_subvolume = self.tomos_even[tomo_index].data[z:z + self.sample_shape[0],
                         y:y + self.sample_shape[1],
                         x:x + self.sample_shape[2]]

        odd_subvolume = self.tomos_odd[tomo_index].data[z:z + self.sample_shape[0],
                        y:y + self.sample_shape[1],
                        x:x + self.sample_shape[2]]

        return self.random_swapper(np.array(even_subvolume)[..., np.newaxis], np.array(odd_subvolume)[..., np.newaxis])

    def __iter__(self):
        for idx in self.indices:
            yield self.__getitem__(idx)
        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            self.indices = np.random.permutation(self.length)

    def close(self):
        for even, odd in zip(self.tomos_even, self.tomos_odd):
            even.close()
            odd.close()


class CryoCARE_DataModule(object):
    def __init__(self):
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, tomo_paths_odd, tomo_paths_even, n_samples_per_tomo, validation_fraction=0.1,
              sample_shape=(64, 64, 64), tilt_axis='Y', n_normalization_samples=500):
        train_extraction_shapes = []
        val_extraction_shapes = []
        for e, o in zip(tomo_paths_even, tomo_paths_odd):
            tes, ves = self.__compute_extraction_shapes__(e, o, tilt_axis_index=['Z', 'Y', 'X'].index(tilt_axis),
                                                          sample_shape=sample_shape,
                                                          validation_fraction=validation_fraction)
            train_extraction_shapes.append(tes)
            val_extraction_shapes.append(ves)

        self.train_dataset = CryoCARE_Dataset(tomo_paths_odd=tomo_paths_odd,
                                              tomo_paths_even=tomo_paths_even,
                                              mean=None,
                                              std=None,
                                              n_samples_per_tomo=int(
                                                  n_samples_per_tomo * (1 - validation_fraction)),
                                              extraction_shapes=train_extraction_shapes,
                                              sample_shape=sample_shape,
                                              shuffle=True, n_normalization_samples=n_normalization_samples)

        self.val_dataset = CryoCARE_Dataset(tomo_paths_odd=tomo_paths_odd,
                                            tomo_paths_even=tomo_paths_even,
                                            mean=self.train_dataset.mean,
                                            std=self.train_dataset.std,
                                            n_samples_per_tomo=int(n_samples_per_tomo * validation_fraction),
                                            extraction_shapes=val_extraction_shapes,
                                            sample_shape=sample_shape,
                                            shuffle=False)

    def save(self, path):
        self.train_dataset.save(join(path, 'train_data.npz'))
        self.val_dataset.save(join(path, 'val_data.npz'))

    def load(self, path):
        self.train_dataset = CryoCARE_Dataset.load(join(path, 'train_data.npz'))
        self.val_dataset = CryoCARE_Dataset.load(join(path, 'val_data.npz'))

    def __compute_extraction_shapes__(self, even_path, odd_path, tilt_axis_index, sample_shape, validation_fraction):
        even = mrcfile.mmap(even_path, mode='r')
        odd = mrcfile.mmap(odd_path, mode='r')

        assert even.data.shape == odd.data.shape, '{} and {} tomogram have different shapes.'.format(even_path,
                                                                                                     odd_path)
        assert even.data.shape[0] > 2 * sample_shape[0]
        assert even.data.shape[1] > 2 * sample_shape[1]
        assert even.data.shape[2] > 2 * sample_shape[2]

        val_cut_off = int(even.data.shape[tilt_axis_index] * validation_fraction)
        if even.data.shape[tilt_axis_index] - val_cut_off < sample_shape[tilt_axis_index]:
            val_cut_off = even.data.shape[tilt_axis_index] - sample_shape[tilt_axis_index]

        extraction_shape_train = [[0, even.data.shape[0]], [0, even.data.shape[1]], [0, even.data.shape[2]]]
        extraction_shape_val = [[0, even.data.shape[0]], [0, even.data.shape[1]], [0, even.data.shape[2]]]
        extraction_shape_train[tilt_axis_index] = [0, val_cut_off]
        extraction_shape_val[tilt_axis_index] = [val_cut_off, even.data.shape[tilt_axis_index]]

        return extraction_shape_train, extraction_shape_val

    def get_normalizer(self, mean, std):
        def normalize(x, y):
            x = (x - mean) / std
            y = (y - mean) / std
            return x, y

        return normalize

    def get_train_dataset(self):
        sample_shape = self.train_dataset.sample_shape
        ds = tf.data.Dataset.from_generator(self.train_dataset.__iter__,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(
                                                tuple(sample_shape) + (1,), tuple(sample_shape) + (1,)))
        return ds.map(self.get_normalizer(self.train_dataset.mean, self.train_dataset.std)).prefetch(tf.data.experimental.AUTOTUNE).repeat()

    def get_val_dataset(self):
        sample_shape = self.val_dataset.sample_shape
        ds = tf.data.Dataset.from_generator(self.val_dataset.__iter__,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(
                                                tuple(sample_shape) + (1,), tuple(sample_shape) + (1,)))
        return ds.map(self.get_normalizer(self.train_dataset.mean, self.train_dataset.std))

    def close(self):
        self.train_dataset.close()
        self.val_dataset.close()
