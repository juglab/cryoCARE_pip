import numpy as np
from keras.utils import Sequence


def augment(x, y):
    rot_k = np.random.randint(0, 4, x.shape[0])

    X = x.copy()
    Y = y.copy()

    for i in range(X.shape[0]):
        if np.random.rand() < 0.5:
            X[i] = np.rot90(x[i], k=rot_k[i], axes=(0, 2))
            Y[i] = np.rot90(y[i], k=rot_k[i], axes=(0, 2))
        else:
            X[i] = np.rot90(y[i], k=rot_k[i], axes=(0, 2))
            Y[i] = np.rot90(x[i], k=rot_k[i], axes=(0, 2))

    return X, Y


class CryoDataWrapper(Sequence):

    def __init__(self, X, Y, batch_size):
        self.X, self.Y = X, Y
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i * self.batch_size, (i + 1) * self.batch_size)
        idx = self.perm[idx]

        return self.__augment__(self.X[idx], self.Y[idx])

    def __augment__(self, x, y):
        return augment(x, y)
