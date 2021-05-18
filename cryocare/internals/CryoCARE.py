from csbdeep.data import PadAndCropResizer, PercentileNormalizer, NoResizer
from csbdeep.internals.predict import Progress, total_n_tiles, tile_iterator_1d, to_tensor, from_tensor
from csbdeep.models import CARE
from csbdeep.utils import _raise, axes_check_and_normalize, axes_dict
import warnings

import numpy as np
import tensorflow as tf


class CryoCARE(CARE):

    def train(self, train_dataset, val_dataset, epochs=None, steps_per_epoch=None):
        """Train the neural network with the given data.
        Parameters
        ----------
        X : :class:`numpy.ndarray`
            Array of source images.
        Y : :class:`numpy.ndarray`
            Array of target images.
        validation_data : tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray`)
            Tuple of arrays for source and target validation images.
        epochs : int
            Optional argument to use instead of the value from ``config``.
        steps_per_epoch : int
            Optional argument to use instead of the value from ``config``.
        Returns
        -------
        ``History`` object
            See `Keras training history <https://keras.io/models/model/#fit>`_.
        """

        axes = axes_check_and_normalize('S' + self.config.axes, len(train_dataset.element_spec[0].shape) + 1)
        ax = axes_dict(axes)

        train_shape = (1,) + train_dataset.element_spec[0].shape
        for a, div_by in zip(axes, self._axes_div_by(axes)):
            n = train_shape[ax[a]]
            print(ax[a], n)
            if n % div_by != 0:
                raise ValueError(
                    "training images must be evenly divisible by %d along axis %s"
                    " (which has incompatible size %d)" % (div_by, a, n)
                )

        if epochs is None:
            epochs = self.config.train_epochs
        if steps_per_epoch is None:
            steps_per_epoch = self.config.train_steps_per_epoch

        if not self._model_prepared:
            self.prepare_for_training()

        history = self.keras_model.fit(train_dataset.batch(self.config.train_batch_size),
                                       validation_data=val_dataset.batch(self.config.train_batch_size),
                                       epochs=epochs, steps_per_epoch=steps_per_epoch,
                                       callbacks=self.callbacks, verbose=1)

        if self.basedir is not None:
            self.keras_model.save_weights(str(self.logdir / 'weights_last.h5'))

            if self.config.train_checkpoint is not None:
                print()
                self._find_and_load_weights(self.config.train_checkpoint)
                try:
                    # remove temporary weights
                    (self.logdir / 'weights_now.h5').unlink()
                except FileNotFoundError:
                    pass

        return history

    def predict(self, even, odd, output, axes, normalizer=PercentileNormalizer(), resizer=PadAndCropResizer(), mean=0,
                std=1, n_tiles=None):
        """Apply neural network to raw image to predict restored image.

                Parameters
                ----------
                img : :class:`numpy.ndarray`
                    Raw input image
                axes : str
                    Axes of the input ``img``.
                normalizer : :class:`csbdeep.data.Normalizer` or None
                    Normalization of input image before prediction and (potentially) transformation back after prediction.
                resizer : :class:`csbdeep.data.Resizer` or None
                    If necessary, input image is resized to enable neural network prediction and result is (possibly)
                    resized to yield original image size.
                n_tiles : iterable or None
                    Out of memory (OOM) errors can occur if the input image is too large.
                    To avoid this problem, the input image is broken up into (overlapping) tiles
                    that can then be processed independently and re-assembled to yield the restored image.
                    This parameter denotes a tuple of the number of tiles for every image axis.
                    Note that if the number of tiles is too low, it is adaptively increased until
                    OOM errors are avoided, albeit at the expense of runtime.
                    A value of ``None`` denotes that no tiling should initially be used.

                Returns
                -------
                :class:`numpy.ndarray`
                    Returns the restored image. If the model is probabilistic, this denotes the `mean` parameter of
                    the predicted per-pixel Laplace distributions (i.e., the expected restored image).
                    Axes semantics are the same as in the input image. Only if the output is multi-channel and
                    the input image didn't have a channel axis, then output channels are appended at the end.

                """
        self._predict_mean_and_scale(self._crop(even), self._crop(odd), self._crop(output), axes, normalizer, resizer=NoResizer(), mean=mean, std=std,
                                     n_tiles=n_tiles)

    def _crop(self, data):
        div_by = self._axes_div_by('XYZ')
        data_shape = data.shape
        slices = ()
        for i in range(3):
            if data_shape[i] % div_by[i] == 0:
                slices += (slice(None),)
            else:
                slices += (slice(0, -(data_shape[i]%div_by[i])),)
        return data[slices]

    def _predict_mean_and_scale(self, even, odd, output, axes, normalizer, resizer, mean, std, n_tiles=None):
        """Apply neural network to raw image to predict restored image.

        See :func:`predict` for parameter explanations.

        Returns
        -------
        tuple(:class:`numpy.ndarray`, :class:`numpy.ndarray` or None)
            If model is probabilistic, returns a tuple `(mean, scale)` that defines the parameters
            of per-pixel Laplace distributions. Otherwise, returns the restored image via a tuple `(restored,None)`

        """
        print(even.shape)
        normalizer, resizer = self._check_normalizer_resizer(normalizer, resizer)
        # axes = axes_check_and_normalize(axes,img.ndim)

        # different kinds of axes
        # -> typical case: net_axes_in = net_axes_out, img_axes_in = img_axes_out
        img_axes_in = axes_check_and_normalize(axes, even.ndim)
        net_axes_in = self.config.axes
        net_axes_out = axes_check_and_normalize(self._axes_out)
        set(net_axes_out).issubset(set(net_axes_in)) or _raise(ValueError("different kinds of output than input axes"))
        net_axes_lost = set(net_axes_in).difference(set(net_axes_out))
        img_axes_out = ''.join(a for a in img_axes_in if a not in net_axes_lost)
        # print(' -> '.join((img_axes_in, net_axes_in, net_axes_out, img_axes_out)))
        tiling_axes = net_axes_out.replace('C', '')  # axes eligible for tiling

        _permute_axes = self._make_permute_axes(img_axes_in, net_axes_in, net_axes_out, img_axes_out)
        # _permute_axes: (img_axes_in -> net_axes_in), undo: (net_axes_out -> img_axes_out)
        even = _permute_axes(even)
        odd = _permute_axes(odd)
        # x has net_axes_in semantics
        x_tiling_axis = tuple(axes_dict(net_axes_in)[a] for a in tiling_axes)  # numerical axis ids for x

        channel_in = axes_dict(net_axes_in)['C']
        channel_out = axes_dict(net_axes_out)['C']
        net_axes_in_div_by = self._axes_div_by(net_axes_in)
        net_axes_in_overlaps = self._axes_tile_overlap(net_axes_in)
        self.config.n_channel_in == even.shape[channel_in] or _raise(ValueError())

        # TODO: refactor tiling stuff to make code more readable

        def _total_n_tiles(n_tiles):
            n_block_overlaps = [int(np.ceil(1. * tile_overlap / block_size)) for tile_overlap, block_size in
                                zip(net_axes_in_overlaps, net_axes_in_div_by)]
            return total_n_tiles(even, n_tiles=n_tiles, block_sizes=net_axes_in_div_by,
                                 n_block_overlaps=n_block_overlaps, guarantee='size')

        _permute_axes_n_tiles = self._make_permute_axes(img_axes_in, net_axes_in)

        # _permute_axes_n_tiles: (img_axes_in <-> net_axes_in) to convert n_tiles between img and net axes
        def _permute_n_tiles(n, undo=False):
            # hack: move tiling axis around in the same way as the image was permuted by creating an array
            return _permute_axes_n_tiles(np.empty(n, np.bool), undo=undo).shape

        # to support old api: set scalar n_tiles value for the largest tiling axis
        if np.isscalar(n_tiles) and int(n_tiles) == n_tiles and 1 <= n_tiles:
            largest_tiling_axis = [i for i in np.argsort(even.shape) if i in x_tiling_axis][-1]
            _n_tiles = [n_tiles if i == largest_tiling_axis else 1 for i in range(x.ndim)]
            n_tiles = _permute_n_tiles(_n_tiles, undo=True)
            warnings.warn("n_tiles should be a tuple with an entry for each image axis")
            print("Changing n_tiles to %s" % str(n_tiles))

        if n_tiles is None:
            n_tiles = [1] * even.ndim
        try:
            n_tiles = tuple(n_tiles)
            even.ndim == len(n_tiles) or _raise(TypeError())
        except TypeError:
            raise ValueError("n_tiles must be an iterable of length %d" % even.ndim)

        all(np.isscalar(t) and 1 <= t and int(t) == t for t in n_tiles) or _raise(
            ValueError("all values of n_tiles must be integer values >= 1"))
        n_tiles = tuple(map(int, n_tiles))
        n_tiles = _permute_n_tiles(n_tiles)
        (all(n_tiles[i] == 1 for i in range(even.ndim) if i not in x_tiling_axis) or
         _raise(ValueError("entry of n_tiles > 1 only allowed for axes '%s'" % tiling_axes)))
        # n_tiles_limited = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
        # if any(np.array(n_tiles) != np.array(n_tiles_limited)):
        #     print("Limiting n_tiles to %s" % str(_permute_n_tiles(n_tiles_limited,undo=True)))
        # n_tiles = n_tiles_limited
        n_tiles = list(n_tiles)

        # normalize & resize
        even = resizer.before(even, net_axes_in, net_axes_in_div_by)
        odd = resizer.before(odd, net_axes_in, net_axes_in_div_by)

        done = False
        progress = Progress(_total_n_tiles(n_tiles), 1)
        c = 0
        while not done:
            try:
                # raise tf.errors.ResourceExhaustedError(None,None,None) # tmp
                pred = predict_tiled(self.keras_model, even, odd, output, [4 * (slice(None),)], 4 * (slice(None),),
                                     mean=mean, std=std,
                                     axes_in=net_axes_in, axes_out=net_axes_out,
                                     n_tiles=n_tiles, block_sizes=net_axes_in_div_by,
                                     tile_overlaps=net_axes_in_overlaps, pbar=progress)
                output = pred
                # x has net_axes_out semantics
                done = True
                progress.close()
            except tf.errors.ResourceExhaustedError:
                # TODO: how to test this code?
                # n_tiles_prev = list(n_tiles) # make a copy
                tile_sizes_approx = np.array(even.shape) / np.array(n_tiles)
                t = [i for i in np.argsort(tile_sizes_approx) if i in x_tiling_axis][-1]
                n_tiles[t] *= 2
                # n_tiles = self._limit_tiling(x.shape,n_tiles,net_axes_in_div_by)
                # if all(np.array(n_tiles) == np.array(n_tiles_prev)):
                # raise MemoryError("Tile limit exceeded. Memory occupied by another process (notebook)?")
                if c >= 16:
                    raise MemoryError(
                        "Giving up increasing number of tiles. Memory occupied by another process (notebook)?")
                print('Out of memory, retrying with n_tiles = %s' % str(_permute_n_tiles(n_tiles, undo=True)))
                progress.total = _total_n_tiles(n_tiles)
                c += 1

        n_channel_predicted = self.config.n_channel_out * (2 if self.config.probabilistic else 1)

        output.data.shape[channel_out] == n_channel_predicted or _raise(ValueError())

        resizer.after(output, net_axes_out)


def predict_tiled(keras_model, even, odd, output, s_src_out, s_dst_out, mean, std, n_tiles, block_sizes, tile_overlaps,
                  axes_in,
                  axes_out=None, pbar=None, **kwargs):
    """TODO."""
    if all(t == 1 for t in n_tiles):
        even_pred = predict_direct(keras_model, even, mean, std, axes_in, axes_out, **kwargs)
        odd_pred = predict_direct(keras_model, odd, mean, std, axes_in, axes_out, **kwargs)
        pred = (even_pred + odd_pred) / 2.
        for src in s_src_out:
            pred = pred[src]
        if pbar is not None:
            pbar.update()
        return pred

    ###

    if axes_out is None:
        axes_out = axes_in
    axes_in, axes_out = axes_check_and_normalize(axes_in, even.ndim), axes_check_and_normalize(axes_out)
    assert 'S' not in axes_in
    assert 'C' in axes_in and 'C' in axes_out
    ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
    channel_in, channel_out = ax_in['C'], ax_out['C']

    assert set(axes_out).issubset(set(axes_in))
    axes_lost = set(axes_in).difference(set(axes_out))

    def _to_axes_out(seq, elem):
        # assumption: prediction size is same as input size along all axes, except for channel (and lost axes)
        assert len(seq) == len(axes_in)
        # 1. re-order 'seq' from axes_in to axes_out semantics
        seq = [seq[ax_in[a]] for a in axes_out]
        # 2. replace value at channel position with 'elem'
        seq[ax_out['C']] = elem
        return tuple(seq)

    ###

    assert even.ndim == len(n_tiles) == len(block_sizes)
    assert odd.ndim == len(n_tiles) == len(block_sizes)
    assert n_tiles[channel_in] == 1
    assert all(n_tiles[ax_in[a]] == 1 for a in axes_lost)
    assert all(np.isscalar(t) and 1 <= t and int(t) == t for t in n_tiles)

    # first axis > 1
    axis = next(i for i, t in enumerate(n_tiles) if t > 1)

    block_size = block_sizes[axis]
    tile_overlap = tile_overlaps[axis]
    n_block_overlap = int(np.ceil(1. * tile_overlap / block_size))

    # print(f"axis={axis},n_tiles={n_tiles[axis]},block_size={block_size},tile_overlap={tile_overlap},n_block_overlap={n_block_overlap}")

    n_tiles_remaining = list(n_tiles)
    n_tiles_remaining[axis] = 1

    for ((even_tile, s_src, s_dst), (odd_tile, _, _), (output_tile, _, _)) in zip(
            tile_iterator_1d(even, axis=axis, n_tiles=n_tiles[axis], block_size=block_size,
                             n_block_overlap=n_block_overlap),
            tile_iterator_1d(odd, axis=axis, n_tiles=n_tiles[axis], block_size=block_size,
                             n_block_overlap=n_block_overlap),
            tile_iterator_1d(output, axis=axis, n_tiles=n_tiles[axis], block_size=block_size,
                             n_block_overlap=n_block_overlap)):
        pred = predict_tiled(keras_model, even_tile, odd_tile, output_tile[s_src_out[-1]], s_src_out + [s_src], s_dst,
                             mean, std, n_tiles_remaining, block_sizes, tile_overlaps, axes_in, axes_out, pbar=pbar,
                             **kwargs)

        s_dst = _to_axes_out(s_dst, slice(None))
        output[s_dst][s_src_out[-1]] = pred

    return output[s_src_out[-1]]


def predict_direct(keras_model, x, mean, std, axes_in, axes_out=None, **kwargs):
    """TODO."""
    if axes_out is None:
        axes_out = axes_in
    ax_in, ax_out = axes_dict(axes_in), axes_dict(axes_out)
    channel_in, channel_out = ax_in['C'], ax_out['C']
    single_sample = ax_in['S'] is None
    len(axes_in) == x.ndim or _raise(ValueError())
    x = (x - mean) / std
    x = to_tensor(x, channel=channel_in, single_sample=single_sample)
    pred = from_tensor(keras_model.predict(x, **kwargs), channel=channel_out, single_sample=single_sample)
    len(axes_out) == pred.ndim or _raise(ValueError())
    pred = (pred * std) + mean
    return pred
