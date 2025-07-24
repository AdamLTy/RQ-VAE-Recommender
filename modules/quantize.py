import gin
import paddle

from distributions.gumbel import gumbel_softmax_sample
from einops import rearrange
from enum import Enum
from init.kmeans import kmeans_init_
from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer
from typing import NamedTuple
from paddle import nn
from paddle import Tensor
from paddle.nn import functional as F


@gin.constants_from_enum
class QuantizeForwardMode(Enum):
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple):
    embeddings: Tensor
    ids: Tensor
    loss: Tensor


def efficient_rotation_trick_transform(u, q, e):
    """
    4.2 in https://arxiv.org/abs/2410.06424
    """
    e = rearrange(e, 'b d -> b 1 d')
    w = F.normalize(u + q, p=2, axis=1, epsilon=1e-6).detach()

    return (
        e -
        2 * paddle.matmul(paddle.matmul(e, rearrange(w, 'b d -> b d 1')), rearrange(w, 'b d -> b 1 d')) +
        2 * paddle.matmul(paddle.matmul(e, rearrange(u, 'b d -> b d 1').detach()), rearrange(q, 'b d -> b 1 d').detach())
    ).squeeze()


class Quantize(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        do_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        sim_vq: bool = False,  # https://arxiv.org/pdf/2411.02038
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        distance_mode: QuantizeDistance = QuantizeDistance.L2
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(n_embed, embed_dim)
        self.forward_mode = forward_mode
        self.distance_mode = distance_mode
        self.do_kmeans_init = do_kmeans_init
        self.kmeans_initted = False

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias_attr=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if codebook_normalize else nn.Identity()
        )

        self.quantize_loss = QuantizeLoss(commitment_weight)
        self._init_weights()

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    @property
    def device(self):
        return paddle.get_device()

    def _init_weights(self) -> None:
        for m in self.sublayers():
            if isinstance(m, nn.Embedding):
                paddle.nn.initializer.Uniform()(m.weight)
    
    @paddle.no_grad()
    def _kmeans_init(self, x) -> None:
        kmeans_init_(self.embedding.weight, x=x)
        self.kmeans_initted = True

    def get_item_embeddings(self, item_ids) -> Tensor:
        return self.out_proj(self.embedding(item_ids))

    def forward(self, x, temperature) -> QuantizeOutput:
        assert x.shape[-1] == self.embed_dim

        if self.do_kmeans_init and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)
        
        if self.distance_mode == QuantizeDistance.L2:
            dist = (
                paddle.sum(x**2, axis=1, keepdim=True) +
                paddle.sum(codebook.T**2, axis=0, keepdim=True) -
                2 * paddle.matmul(x, codebook.T)
            )
        elif self.distance_mode == QuantizeDistance.COSINE:
            dist = -(
                paddle.matmul(
                    x / paddle.norm(x, axis=1, keepdim=True),
                    (codebook.T) / paddle.norm(codebook.T, axis=0, keepdim=True)
                )
            )
        else:
            raise Exception("Unsupported Quantize distance mode.")

        ids = (dist.detach()).argmin(axis=1)

        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = paddle.matmul(weights, codebook)
                emb_out = emb
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            elif self.forward_mode == QuantizeForwardMode.ROTATION_TRICK:
                emb = self.get_item_embeddings(ids)
                emb_out = efficient_rotation_trick_transform(
                    x / (paddle.norm(x, p=2, axis=-1, keepdim=True) + 1e-8),
                    emb / (paddle.norm(emb, p=2, axis=-1, keepdim=True) + 1e-8),
                    x
                )
                emb_out = emb_out * (
                    paddle.norm(emb, p=2, axis=1, keepdim=True) / (paddle.norm(x, p=2, axis=1, keepdim=True) + 1e-6)
                ).detach()
            else:
                raise Exception("Unsupported Quantize forward mode.")
            
            loss = self.quantize_loss(query=x, value=emb)
        
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return QuantizeOutput(
            embeddings=emb_out,
            ids=ids,
            loss=loss
        )
