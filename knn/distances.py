import numpy as np


def euclidean_distance(x, y):
    a = np.sum(x * x, axis=1)
    b = np.sum(y * y, axis=1)
    c = np.dot(x, y.T)

    return np.sqrt(a[:, np.newaxis] + b - 2 * c)


def cosine_distance(x, y):
    raise NotImplementedError()
