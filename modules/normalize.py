import paddle
from paddle import nn
from paddle import Tensor
from paddle.nn import functional as F


def l2norm(x, axis=-1, eps=1e-12):
    return F.normalize(x, p=2, axis=axis, epsilon=eps)


class L2NormalizationLayer(nn.Layer):
    def __init__(self, dim=-1, eps=1e-12) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x) -> Tensor:
        return l2norm(x, axis=self.dim, eps=self.eps)


class RMSNorm(nn.Layer):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = paddle.create_parameter(shape=[dim], dtype='float32', 
                                            default_initializer=paddle.nn.initializer.Constant(1.0))

    def _norm(self, x):
        return x * paddle.rsqrt(paddle.pow(x, 2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x) -> Tensor:
        output = self._norm(x.astype('float32')).astype(x.dtype)
        return output * self.weight
