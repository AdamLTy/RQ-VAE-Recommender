import gin
import paddle

from einops import rearrange
from enum import Enum
from data.schemas import TokenizedSeqBatch
from modules.embedding.id_embedder import SemIdEmbedder
from modules.embedding.id_embedder import UserIdEmbedder
from modules.normalize import RMSNorm
from modules.transformer.attention import AttentionInput
from modules.transformer.model import TransformerDecoder
from modules.transformer.model import TransformerEncoderDecoder
from modules.utils import eval_mode
from modules.utils import maybe_repeat_interleave
from modules.utils import reset_encoder_cache
from modules.utils import reset_kv_cache
from modules.utils import select_columns_per_row
from ops.triton.jagged import jagged_to_flattened_tensor
from ops.triton.jagged import padded_to_jagged_tensor
from typing import NamedTuple
from paddle import nn
from paddle import Tensor
from paddle.nn import functional as F


class ModelOutput(NamedTuple):
    loss: Tensor
    logits: Tensor
    loss_d: Tensor


class GenerationOutput(NamedTuple):
    sem_ids: Tensor
    log_probas: Tensor


class EncoderDecoderRetrievalModel(nn.Layer):
    def __init__(
        self,
        embedding_dim,
        attn_dim,
        dropout,
        num_heads,
        n_layers,
        num_embeddings,
        sem_id_dim,
        inference_verifier_fn,
        max_pos=2048,
        jagged_mode: bool = True,
    ) -> None:
        super().__init__()

        self.jagged_mode = jagged_mode
        self.num_embeddings = num_embeddings
        self.sem_id_dim = sem_id_dim
        self.attn_dim = attn_dim
        self.inference_verifier_fn = inference_verifier_fn
        self.enable_generation = False

        self.bos_emb = paddle.create_parameter(shape=[embedding_dim], dtype='float32', default_initializer=paddle.nn.initializer.Uniform())
        self.norm = RMSNorm(embedding_dim)
        self.norm_cxt = RMSNorm(embedding_dim)
        self.do = nn.Dropout(p=0.5)

        self.sem_id_embedder = SemIdEmbedder(
            num_embeddings=num_embeddings,
            sem_ids_dim=sem_id_dim,
            embeddings_dim=embedding_dim
        )
        self.user_id_embedder = UserIdEmbedder(2000, embedding_dim)
        
        self.wpe = nn.Embedding(num_embeddings=max_pos, embedding_dim=embedding_dim)
        self.tte = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)
        self.tte_fut = nn.Embedding(num_embeddings=sem_id_dim, embedding_dim=embedding_dim)

        self.transformer = TransformerEncoderDecoder(
            d_in=attn_dim,
            d_out=attn_dim,
            dropout=dropout,
            num_heads=num_heads,
            encoder_layers=n_layers // 2,
            decoder_layers=n_layers // 2
        ) if self.jagged_mode else nn.Transformer(
            d_model=attn_dim,
            nhead=num_heads,
            num_encoder_layers=n_layers // 2,
            num_decoder_layers=n_layers // 2,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )

        self.in_proj = nn.Linear(embedding_dim, attn_dim, bias_attr=False)
        self.in_proj_context = nn.Linear(embedding_dim, attn_dim, bias_attr=False)
        self.out_proj = nn.Linear(attn_dim, num_embeddings, bias_attr=False)
    
    def _predict(self, batch: TokenizedSeqBatch) -> AttentionInput:
        user_emb = self.user_id_embedder(batch.user_ids).unsqueeze(1)
        sem_ids_emb = self.sem_id_embedder(batch)
        sem_ids_emb, sem_ids_emb_fut = sem_ids_emb.seq, sem_ids_emb.fut
        seq_lengths = batch.seq_mask.sum(axis=1)
        
        B, N, D = sem_ids_emb.shape

        pos_max = N // self.sem_id_dim
          
        pos = paddle.arange(N).unsqueeze(0)
        wpe = self.wpe(pos)

        input_embedding = paddle.concat([user_emb, paddle.add(wpe, sem_ids_emb)], axis=1)
        input_embedding_fut = self.bos_emb.tile([B, 1, 1])
        if sem_ids_emb_fut is not None:
            tte_fut = self.tte(batch.token_type_ids_fut)
            input_embedding_fut = paddle.concat([
                input_embedding_fut, 
                sem_ids_emb_fut + tte_fut
                ], axis=1
            )

        if self.jagged_mode:
            try:
                input_embedding = padded_to_jagged_tensor(input_embedding, lengths=seq_lengths+1, max_len=input_embedding.shape[1])

                seq_lengths_fut = paddle.to_tensor(input_embedding_fut.shape[1], dtype='int64').tile([B])
                input_embedding_fut = padded_to_jagged_tensor(input_embedding_fut, lengths=seq_lengths_fut, max_len=input_embedding_fut.shape[1])
            except RuntimeError as e:
                if "active drivers" in str(e):
                    print("Warning: Triton/CUDA not available, falling back to non-jagged mode")
                    self.jagged_mode = False
                else:
                    raise e
        
        if not self.jagged_mode:
            mem_mask = paddle.concat([
                paddle.ones([B, 1], dtype='bool'),
                batch.seq_mask
            ], axis=1)
            f_mask = paddle.zeros_like(mem_mask, dtype='float32')
            f_mask[~mem_mask] = float("-inf")
        
        transformer_context = self.in_proj_context(self.do(self.norm(input_embedding)))
        transformer_input = self.in_proj(self.do(self.norm_cxt(input_embedding_fut)))
        
        if self.jagged_mode:
            transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=batch.seq_mask, jagged=self.jagged_mode)
        else:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(self.transformer, transformer_input.shape[1])
            transformer_output = self.transformer(x=transformer_input, context=transformer_context, padding_mask=f_mask, jagged=self.jagged_mode)

        return transformer_output

    @eval_mode
    @reset_encoder_cache
    @paddle.no_grad()
    def generate_next_sem_id(
        self,
        batch: TokenizedSeqBatch,
        temperature: int = 1,
        top_k: bool = True
    ) -> GenerationOutput:
        
        assert self.enable_generation, "Model generation is not enabled"

        B, N = batch.sem_ids.shape
        generated = None
        log_probas = paddle.zeros([B, 1], dtype='float32')
        k = 32 if top_k else 1
        n_top_k_candidates = 200 if top_k else 1

        input_batch = TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=batch.sem_ids,
            sem_ids_fut=None,
            seq_mask=batch.seq_mask,
            token_type_ids=batch.token_type_ids,
            token_type_ids_fut=None
        )

        for i in range(self.sem_id_dim):
            logits = self.forward(input_batch).logits
            probas_batched = F.softmax(logits / temperature, axis=-1)
            samples_batched = paddle.multinomial(probas_batched, num_samples=n_top_k_candidates)

            if generated is None:
                is_valid_prefix = self.inference_verifier_fn(samples_batched.unsqueeze(-1))
            else:
                prefix = paddle.concat([generated.flatten([0,1]).unsqueeze(1).tile([1, n_top_k_candidates, 1]), samples_batched.unsqueeze(-1)], axis=-1)
                is_valid_prefix = self.inference_verifier_fn(prefix).reshape(B, -1)
            
            # Create batch indices for gather operation
            batch_indices = paddle.arange(B).unsqueeze(1).tile([1, n_top_k_candidates])
            sampled_log_probas = paddle.log(probas_batched[batch_indices, samples_batched]).reshape([B, -1])
            samples = samples_batched.reshape([B, -1])

            # Get top-K:
            invalid_mask = paddle.logical_not(is_valid_prefix).astype('float32')
            repeated_log_probas = maybe_repeat_interleave(log_probas, n_top_k_candidates, dim=1)
            
            combined_scores = paddle.add(
                paddle.add(-10000 * invalid_mask, sampled_log_probas),
                repeated_log_probas
            )
            sorted_indices = combined_scores.argsort(-1, descending=True)
            sorted_log_probas = paddle.gather(combined_scores, sorted_indices, axis=-1)

            top_k_log_probas, top_k_indices = sorted_log_probas[:, :k], sorted_indices[:, :k]
            top_k_samples = paddle.gather(samples, top_k_indices, axis=1)
            
            if generated is not None:
                parent_id = paddle.gather(generated, (top_k_indices // n_top_k_candidates).unsqueeze(2).expand([-1,-1,i]), axis=1)
                top_k_samples = paddle.concat([parent_id, top_k_samples.unsqueeze(-1)], axis=-1)

                next_sem_ids = top_k_samples.flatten(end_dim=1)

                input_batch = TokenizedSeqBatch(
                    user_ids=input_batch.user_ids,
                    sem_ids=input_batch.sem_ids,
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=paddle.arange(next_sem_ids.shape[1]).tile([next_sem_ids.shape[0], 1]),
                    seq_mask=input_batch.seq_mask,
                    token_type_ids=input_batch.token_type_ids
                )

                generated = top_k_samples.detach().clone()
                log_probas = top_k_log_probas.detach().clone()
            else:
                next_sem_ids = top_k_samples.reshape(-1, 1)

                # Explode encoder cache on dim 0 to match input size B*k
                # TODO: Figure out how to avoid jagged - padded conversions 
                # (E.g. Implement repeat_interleave jagged kernel)
                if self.jagged_mode:
                    try:
                        cache = paddle.zeros([input_batch.sem_ids.shape[0], input_batch.sem_ids.shape[1]+1, self.attn_dim])
                        cache_mask = paddle.concat([paddle.ones([input_batch.sem_ids.shape[0], 1], dtype='bool'), input_batch.seq_mask], axis=1)
                        cache[cache_mask] = self.transformer.cached_enc_output.values()
                        lengths = maybe_repeat_interleave(self.transformer.cached_enc_output.offsets().diff(), k, dim=0)
                        cache = maybe_repeat_interleave(cache, k, dim=0)
                        self.transformer.cached_enc_output = padded_to_jagged_tensor(cache, lengths, max_len=cache.shape[1])
                    except RuntimeError as e:
                        if "active drivers" in str(e):
                            print("Warning: Triton/CUDA not available during beam search, falling back to non-jagged mode")
                            self.jagged_mode = False
                        else:
                            raise e

                input_batch = TokenizedSeqBatch(
                    user_ids=maybe_repeat_interleave(input_batch.user_ids, k, dim=0),
                    sem_ids=maybe_repeat_interleave(input_batch.sem_ids, k, dim=0),
                    sem_ids_fut=next_sem_ids,
                    token_type_ids_fut=paddle.zeros_like(next_sem_ids),
                    seq_mask=maybe_repeat_interleave(input_batch.seq_mask, k, dim=0),
                    token_type_ids=maybe_repeat_interleave(input_batch.token_type_ids, k, dim=0)
                )

                generated = top_k_samples.unsqueeze(-1)
                log_probas = top_k_log_probas.detach().clone()
        
        return GenerationOutput(
            sem_ids=generated.squeeze(),
            log_probas=log_probas.squeeze()
        )
            
    def forward(self, batch: TokenizedSeqBatch) -> ModelOutput:
        seq_mask = batch.seq_mask
        B, N = seq_mask.shape

        trnsf_out = self._predict(batch)
        
        if self.training or not self.enable_generation:
            predict_out = self.out_proj(trnsf_out)
            if self.jagged_mode:
                # This works because batch.sem_ids_fut is fixed length, no padding.
                logits = rearrange(jagged_to_flattened_tensor(predict_out), "(b n) d -> b n d", b=B)[:,:-1,:].flatten(end_dim=1)
                target = batch.sem_ids_fut.flatten(end_dim=1)
                unred_loss = rearrange(F.cross_entropy(logits, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            else:
                logits = predict_out
                out = logits[:, :-1, :].flatten(start_axis=0, stop_axis=1)
                target = batch.sem_ids_fut.flatten(start_axis=0, stop_axis=1)
                unred_loss = rearrange(F.cross_entropy(out, target, reduction="none", ignore_index=-1), "(b n) -> b n", b=B)
                loss = unred_loss.sum(axis=1).mean()
            if not self.training and self.jagged_mode:
                self.transformer.cached_enc_output = None
            loss_d = unred_loss.mean(axis=0)
        elif self.jagged_mode:
            trnsf_out = trnsf_out.contiguous()
            trnsf_out_flattened = rearrange(jagged_to_flattened_tensor(trnsf_out), "(b n) d -> b n d", b=B)[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None
        else:
            trnsf_out_flattened = trnsf_out[:,-1,:]
            logits = self.out_proj(trnsf_out_flattened)
            loss = None
            loss_d = None

        return ModelOutput(loss=loss, logits=logits, loss_d=loss_d)
