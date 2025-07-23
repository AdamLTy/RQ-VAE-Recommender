import numpy as np
import paddle

from einops import rearrange
from typing import NamedTuple


def kmeans_init_(tensor: paddle.Tensor, x: paddle.Tensor):
    assert tensor.ndim == 2
    assert x.ndim == 2

    with paddle.no_grad():
        k, _ = tensor.shape
        kmeans_out = Kmeans(k=k).run(x)
        tensor.set_value(kmeans_out.centroids)


class KmeansOutput(NamedTuple):
    centroids: paddle.Tensor
    assignment: paddle.Tensor


class Kmeans:
    def __init__(self,
                 k: int,
                 max_iters: int = None,
                 stop_threshold: float = 1e-10) -> None:
        self.k = k
        self.iters = max_iters
        self.stop_threshold = stop_threshold
        self.centroids = None
        self.assignment = None

    def _init_centroids(self, x: paddle.Tensor) -> None:
        B, D = x.shape
        init_idx = np.random.choice(B, self.k, replace=False)
        self.centroids = x[init_idx, :]
        self.assignment = None

    def _update_centroids(self, x) -> paddle.Tensor:
        squared_pw_dist = (
            rearrange(x, "b d -> b 1 d") - rearrange(self.centroids, "b d -> 1 b d")
        )**2
        centroid_idx = paddle.argmin(squared_pw_dist.sum(axis=2), axis=1)
        assigned = (
            rearrange(paddle.arange(self.k), "d -> d 1") == centroid_idx
        )

        for cluster in range(self.k):
            is_assigned_to_c = assigned[cluster]
            if not is_assigned_to_c.any():
                if x.shape[0] > 0:
                    self.centroids[cluster, :] = x[paddle.randint(0, x.shape[0], (1,))].squeeze(0)
                else:
                    raise ValueError("Can not choose random element from x, x is empty")
            else:
                self.centroids[cluster, :] = x[is_assigned_to_c, :].mean(axis=0)
        self.assignment = centroid_idx

    def run(self, x):
        self._init_centroids(x)

        i = 0
        while self.iters is None or i < self.iters:
            old_c = self.centroids.clone()
            self._update_centroids(x)
            if paddle.norm(self.centroids - old_c, p=2, axis=1).max() < self.stop_threshold:
                break
            i += 1

        return KmeansOutput(
            centroids=self.centroids,
            assignment=self.assignment
        )
