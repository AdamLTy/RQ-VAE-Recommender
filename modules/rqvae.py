import paddle

from data.schemas import SeqBatch
from einops import rearrange
from functools import cached_property
from modules.encoder import MLP
from modules.loss import CategoricalReconstuctionLoss
from modules.loss import ReconstructionLoss
from modules.loss import QuantizeLoss
from modules.normalize import l2norm
from modules.quantize import Quantize
from modules.quantize import QuantizeForwardMode
from typing import List
from typing import NamedTuple
from paddle import nn
from paddle import Tensor


class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    quantize_loss: Tensor


class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


class RqVae(nn.Layer):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        codebook_kmeans_init: bool = True,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.GUMBEL_SOFTMAX,
        n_layers: int = 3,
        commitment_weight: float = 0.25,
        n_cat_features: int = 18,
    ) -> None:
        # Store config without 'self' to avoid circular references
        config_locals = locals()
        config_locals.pop('self')
        self._config = config_locals
        
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.commitment_weight = commitment_weight
        self.n_cat_feats = n_cat_features

        self.layers = nn.LayerList(sublayers=[
            Quantize(
                embed_dim=embed_dim,
                n_embed=codebook_size,
                forward_mode=codebook_mode,
                do_kmeans_init=codebook_kmeans_init,
                codebook_normalize=i == 0 and codebook_normalize,
                sim_vq=codebook_sim_vq,
                commitment_weight=commitment_weight
            ) for i in range(n_layers)
        ])

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize
        )

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=True
        )

        self.reconstruction_loss = (
            CategoricalReconstuctionLoss(n_cat_features) if n_cat_features != 0
            else ReconstructionLoss()
        )
    
    @cached_property
    def config(self) -> dict:
        return self._config
    
    @property
    def device(self):
        return paddle.get_device()
    
    def load_pretrained(self, path: str) -> None:
        state = paddle.load(path)
        self.set_state_dict(state["model"])

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)

    def get_semantic_ids(
        self,
        x: Tensor,
        gumbel_t: float = 0.001
    ) -> RqVaeOutput:
        res = self.encode(x)
        
        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.layers:
            residuals.append(res)
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized.loss
            emb, id = quantized.embeddings, quantized.ids
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)

        # Handle both 2D and 3D tensor cases
        if len(embs[0].shape) == 3:  # [batch_size, seq_len, embed_dim]
            # Flatten sequence dimension for stacking
            embs_flat = [emb.reshape([-1, emb.shape[-1]]) for emb in embs]
            residuals_flat = [res.reshape([-1, res.shape[-1]]) for res in residuals]
            sem_ids_flat = [id.reshape([-1]) for id in sem_ids]
            
            embeddings = rearrange(paddle.stack(embs_flat), "h b d -> h d b")
            residuals = rearrange(paddle.stack(residuals_flat), "h b d -> h d b")
            sem_ids = rearrange(paddle.stack(sem_ids_flat), "h b -> b h")
        else:  # [batch_size, embed_dim]
            embeddings = rearrange(paddle.stack(embs), "h b d -> h d b")
            residuals = rearrange(paddle.stack(residuals), "h b d -> h d b")
            sem_ids = rearrange(paddle.stack(sem_ids), "h b -> b h")
            
        return RqVaeOutput(
            embeddings=embeddings,
            residuals=residuals,
            sem_ids=sem_ids,
            quantize_loss=quantize_loss
        )

    def forward(self, batch: SeqBatch, gumbel_t: float) -> RqVaeComputedLosses:
        x = batch.x
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized.embeddings, quantized.residuals
        x_hat = self.decode(embs.sum(axis=0).transpose([1, 0]))
        x_hat = paddle.concat([l2norm(x_hat[...,:-self.n_cat_feats]), x_hat[...,-self.n_cat_feats:]], axis=-1)

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized.quantize_loss
        loss = (reconstuction_loss + rqvae_loss).mean()

        with paddle.no_grad():
            # Compute debug ID statistics
            embs_norm = paddle.norm(embs, p=2, axis=1)
            p_unique_ids = (~paddle.triu(
                (rearrange(quantized.sem_ids, "b d -> b 1 d") == rearrange(quantized.sem_ids, "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized.sem_ids.shape[0]

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=reconstuction_loss.mean(),
            rqvae_loss=rqvae_loss.mean(),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids
        )
