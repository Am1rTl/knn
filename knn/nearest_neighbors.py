import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self._X = None
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        distances = self._metric_func(X, self._X)
        partition = np.partition(distances, self.n_neighbors-1, axis=1)
        arg_partition = np.argpartition(distances, self.n_neighbors-1, axis=1)
        return (partition[:, :self.n_neighbors], arg_partition[:, :self.n_neighbors]) if return_distance \
            else arg_partition[:, :self.n_neighbors]
