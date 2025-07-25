import math
import os
import paddle

from data.processed import ItemData
from data.processed import SeqData
from data.schemas import SeqBatch
from data.schemas import TokenizedSeqBatch
from data.utils import batch_to
from einops import rearrange
from einops import pack
from modules.utils import eval_mode
from modules.rqvae import RqVae
from typing import List
from typing import Optional
from paddle import nn
from paddle import Tensor
from paddle.io import DataLoader

BATCH_SIZE = 16

class SemanticIdTokenizer(nn.Layer):
    """
        Tokenizes a batch of sequences of item features into a batch of sequences of semantic ids.
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        codebook_size: int,
        n_layers: int = 3,
        n_cat_feats: int = 18,
        commitment_weight: float = 0.25,
        rqvae_weights_path: Optional[str] = None,
        rqvae_codebook_normalize: bool = False,
        rqvae_sim_vq: bool = False
    ) -> None:
        super().__init__()

        self.rq_vae = RqVae(
            input_dim=input_dim,
            embed_dim=output_dim,
            hidden_dims=hidden_dims,
            codebook_size=codebook_size,
            codebook_kmeans_init=False,
            codebook_normalize=rqvae_codebook_normalize,
            codebook_sim_vq=rqvae_sim_vq,
            n_layers=n_layers,
            n_cat_features=n_cat_feats,
            commitment_weight=commitment_weight,
        )
        
        if rqvae_weights_path is not None:
            self.rq_vae.load_pretrained(rqvae_weights_path)

        self.rq_vae.eval()

        self.codebook_size = codebook_size
        self.n_layers = n_layers
        self.reset()
    
    def _get_hits(self, query: Tensor, key: Tensor) -> Tensor:
        return (rearrange(key, "b d -> 1 b d") == rearrange(query, "b d -> b 1 d")).all(axis=-1)
    
    def reset(self):
        self.cached_ids = None
    
    def save_cached_ids(self, cache_path: str):
        """Save precomputed corpus IDs to file."""
        if self.cached_ids is None:
            raise ValueError("No cached IDs to save. Run precompute_corpus_ids first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        # Save the cached IDs
        paddle.save(self.cached_ids, cache_path)
        print(f"Cached IDs saved to {cache_path}")
    
    def load_cached_ids(self, cache_path: str) -> bool:
        """Load precomputed corpus IDs from file. Returns True if successful."""
        if not os.path.exists(cache_path):
            return False
        
        try:
            self.cached_ids = paddle.load(cache_path)
            print(f"Cached IDs loaded from {cache_path}, shape: {self.cached_ids.shape}")
            return True
        except Exception as e:
            print(f"Failed to load cached IDs from {cache_path}: {e}")
            return False
    
    @property
    def sem_ids_dim(self):
        return self.n_layers + 1
    
    @paddle.no_grad()
    @eval_mode
    def precompute_corpus_ids(self, movie_dataset: ItemData, cache_path: Optional[str] = None) -> Tensor:
        print(f"[DEBUG] precompute_corpus_ids: Starting with dataset size {len(movie_dataset)}")
        
        # Try to load from cache first
        if cache_path and self.load_cached_ids(cache_path):
            print(f"[DEBUG] precompute_corpus_ids: Successfully loaded cached IDs from {cache_path}")
            return self.cached_ids
        
        print(f"[DEBUG] precompute_corpus_ids: No cache found or cache loading failed, computing corpus IDs...")
        cached_ids = None
        dedup_dim = []
        
        # Define collate function for H5 dataset compatibility
        def collate_fn(batch):
            if len(batch) == 1:
                return batch[0]
            # Check if using H5Dataset that needs item-level batching
            from data.h5_dataset import H5PretrainedDataset
            if isinstance(movie_dataset, H5PretrainedDataset):
                # Concatenate item-level tensors for H5 dataset
                from data.schemas import SeqBatch
                user_ids = paddle.concat([item.user_ids for item in batch], axis=0)
                ids = paddle.concat([item.ids for item in batch], axis=0)
                ids_fut = paddle.concat([item.ids_fut for item in batch], axis=0)
                x = paddle.concat([item.x for item in batch], axis=0)
                x_fut = paddle.concat([item.x_fut for item in batch], axis=0)
                seq_mask = paddle.concat([item.seq_mask for item in batch], axis=0)
                return SeqBatch(user_ids=user_ids, ids=ids, ids_fut=ids_fut, x=x, x_fut=x_fut, seq_mask=seq_mask)
            else:
                # For other datasets, assume batch contains SeqBatch objects and return first one
                # since we're processing items individually in precompute_corpus_ids
                return batch[0] if batch else None
        
        dataloader = DataLoader(movie_dataset, batch_size=512, shuffle=False, collate_fn=collate_fn)
        print(f"[DEBUG] precompute_corpus_ids: Created dataloader with batch_size=512")
        
        total_batches = len(dataloader)
        print(f"[DEBUG] precompute_corpus_ids: Total batches to process: {total_batches}")
        
        batch_count = 0
        for batch in dataloader:
            batch_count += 1
            if batch_count % 100 == 1 or batch_count <= 5:
                print(f"[DEBUG] precompute_corpus_ids: Processing batch {batch_count}/{total_batches}")
            if batch_count % 100 == 1 or batch_count <= 5:
                print(f"[DEBUG] precompute_corpus_ids: Running forward pass for batch {batch_count}")
            batch_ids = self.forward(batch_to(batch, self.rq_vae.device)).sem_ids
            if batch_count % 100 == 1 or batch_count <= 5:
                print(f"[DEBUG] precompute_corpus_ids: Forward pass completed, batch_ids shape: {batch_ids.shape}")
            # Detect in-batch duplicates
            if batch_count % 100 == 1 or batch_count <= 5:
                print(f"[DEBUG] precompute_corpus_ids: Detecting in-batch duplicates for batch {batch_count}")
            is_hit = self._get_hits(batch_ids, batch_ids)
            hits = paddle.tril(is_hit, diagonal=-1).sum(axis=-1)
            assert hits.min() >= 0
            if batch_count % 100 == 1 or batch_count <= 5:
                print(f"[DEBUG] precompute_corpus_ids: In-batch duplicates detected, hits shape: {hits.shape}")
            if cached_ids is None:
                if batch_count % 100 == 1 or batch_count <= 5:
                    print(f"[DEBUG] precompute_corpus_ids: Initializing cached_ids for batch {batch_count}")
                cached_ids = batch_ids.clone()
            else:
                # Detect batch-cache duplicates
                if batch_count % 100 == 1 or batch_count <= 5:
                    print(f"[DEBUG] precompute_corpus_ids: Detecting batch-cache duplicates for batch {batch_count}, cached_ids shape: {cached_ids.shape}")
                is_hit = self._get_hits(batch_ids, cached_ids)
                hits += is_hit.sum(axis=-1)
                cached_ids = pack([cached_ids, batch_ids], "* d")[0]
                if batch_count % 100 == 1 or batch_count <= 5:
                    print(f"[DEBUG] precompute_corpus_ids: Updated cached_ids shape: {cached_ids.shape}")
            dedup_dim.append(hits)
            
            if batch_count % 500 == 0:
                print(f"[DEBUG] precompute_corpus_ids: Progress update - completed {batch_count}/{total_batches} batches")
        print(f"[DEBUG] precompute_corpus_ids: Completed all {batch_count} batches, concatenating results")
        # Concatenate new column to deduplicate ids
        print(f"[DEBUG] precompute_corpus_ids: Packing dedup_dim with {len(dedup_dim)} elements")
        dedup_dim_tensor = pack(dedup_dim, "*")[0]
        print(f"[DEBUG] precompute_corpus_ids: dedup_dim_tensor shape: {dedup_dim_tensor.shape}")
        print(f"[DEBUG] precompute_corpus_ids: Final packing with cached_ids shape: {cached_ids.shape}")
        self.cached_ids = pack([cached_ids, dedup_dim_tensor], "b *")[0]
        print(f"[DEBUG] precompute_corpus_ids: Final cached_ids shape: {self.cached_ids.shape}")
        
        # Save to cache if cache_path is provided
        if cache_path:
            self.save_cached_ids(cache_path)
        
        print(f"[DEBUG] precompute_corpus_ids: Completed successfully")
        
        return self.cached_ids

    @paddle.no_grad()
    @eval_mode
    def exists_prefix(self, sem_id_prefix: Tensor) -> Tensor:
        if self.cached_ids is None:
            raise Exception("No match can be found in empty cache.")

        # Reshape flattened semantic ID sequences to match cache structure
        # sem_id_prefix shape: [B, seq_len] where seq_len = n_items * n_layers
        # We need to reshape to [B, n_items, n_layers] to match cached_ids [num_items, n_layers]
        batch_size = sem_id_prefix.shape[0]
        seq_len = sem_id_prefix.shape[-1]
        n_layers = self.cached_ids.shape[-1]  # This should be 4
        
        # Check if sequence length is divisible by n_layers
        if seq_len % n_layers != 0:
            return paddle.zeros([batch_size], dtype=paddle.bool)
        
        n_items = seq_len // n_layers
        sem_id_prefix_reshaped = sem_id_prefix.reshape([batch_size, n_items, n_layers])
        
        out = paddle.zeros([batch_size], dtype=paddle.bool)
        
        # Check each item in the sequence to see if it exists in cache
        for item_idx in range(n_items):
            item_ids = sem_id_prefix_reshaped[:, item_idx, :]  # [B, n_layers]
            
            # Batch matching to avoid OOM
            batches = math.ceil(batch_size // BATCH_SIZE)
            for i in range(batches):
                start_idx = i * BATCH_SIZE
                end_idx = min((i + 1) * BATCH_SIZE, batch_size)
                batch_ids = item_ids[start_idx:end_idx]  # [batch_size_chunk, n_layers]
                
                # Check if any cached item matches
                matches = (batch_ids.unsqueeze(1) == self.cached_ids.unsqueeze(0)).all(axis=-1).any(axis=-1)
                out[start_idx:end_idx] = out[start_idx:end_idx] | matches
        
        return out
    
    def _tokenize_seq_batch_from_cached(self, ids: Tensor) -> Tensor:
        # Handle both 1D (item-level) and 2D (sequence-level) ids
        if ids.ndim == 1:
            # H5 dataset: item-level data, N=1 
            n = 1
        else:
            # Regular sequence data
            n = ids.shape[1]
        return rearrange(self.cached_ids[ids.flatten(), :], "(b n) d -> b (n d)", n=n)
    
    @paddle.no_grad()
    @eval_mode
    def forward(self, batch: SeqBatch) -> TokenizedSeqBatch:
        # TODO: Handle output inconstency in If-else.
        # If block has to return 3-sized ids for use in precompute_corpus_ids
        # Else block has to return deduped 4-sized ids for use in decoder training.
        if self.cached_ids is None or batch.ids.max() >= self.cached_ids.shape[0]:
            # Handle both sequence data (2D) and item data (1D) from H5 dataset
            if batch.ids.ndim == 1:
                # H5 dataset: item-level data, treat each item as single-item sequence
                B = batch.ids.shape[0]
                N = 1
            else:
                # Regular sequence data
                B, N = batch.ids.shape
            sem_ids = self.rq_vae.get_semantic_ids(batch.x).sem_ids
            D = sem_ids.shape[-1]
            # Reshape sem_ids to match expected format [B, N*D]
            if batch.ids.ndim == 1:
                # For item-level data, sem_ids should be [B, D]
                sem_ids = sem_ids.reshape([B, D])
            else:
                # For sequence data, sem_ids should be [B, N*D]
                sem_ids = sem_ids.reshape([B, N * D])
            # Create proper seq_mask when not using cached tokenization
            if batch.ids.ndim == 1:
                # For item-level data, create a simple mask of ones
                seq_mask = paddle.ones([B, D], dtype='bool')
            else:
                # For sequence data, expand the original sequence mask to semantic dimensions
                if hasattr(batch, 'seq_mask') and batch.seq_mask is not None:
                    seq_mask = batch.seq_mask.cast('int32').repeat_interleave(D, axis=1).cast('bool')
                else:
                    # Create default mask if no seq_mask in batch
                    seq_mask = paddle.ones([B, N * D], dtype='bool')
            # Generate semantic IDs for future items
            if hasattr(batch, 'x_fut') and batch.x_fut is not None:
                sem_ids_fut = self.rq_vae.get_semantic_ids(batch.x_fut).sem_ids
                # Use the actual batch size from sem_ids_fut, not from batch.ids
                B_fut = sem_ids_fut.shape[0]
                if batch.ids.ndim == 1:
                    # For item-level data, sem_ids_fut should be [B_fut, D]
                    sem_ids_fut = sem_ids_fut.reshape([B_fut, D])
                else:
                    # For sequence data, sem_ids_fut should be [B_fut, D] (single future item)
                    sem_ids_fut = sem_ids_fut.reshape([B_fut, D])
            else:
                sem_ids_fut = None
        else:
            # Handle both sequence data (2D) and item data (1D) from H5 dataset  
            if batch.ids.ndim == 1:
                # H5 dataset: item-level data
                B = batch.ids.shape[0]
                N = 1
            else:
                # Regular sequence data
                B, N = batch.ids.shape
            _, D = self.cached_ids.shape
            sem_ids = self._tokenize_seq_batch_from_cached(batch.ids)
            seq_mask = batch.seq_mask.cast('int32').repeat_interleave(D, axis=1).cast('bool')
            sem_ids[~seq_mask] = -1

            sem_ids_fut = self._tokenize_seq_batch_from_cached(batch.ids_fut)
        
        # Handle both sequence data (2D) and item data (1D) from H5 dataset
        if batch.ids.ndim == 1:
            # For item-level data
            token_type_ids = paddle.arange(D).unsqueeze(0).tile([B, 1])
        else:
            # For sequence data, repeat [0,1,2,...,D-1] for each sequence position
            token_type_ids = paddle.arange(D).tile([N]).unsqueeze(0).tile([B, 1])
        token_type_ids_fut = paddle.arange(D).unsqueeze(0).tile([B, 1])
        return TokenizedSeqBatch(
            user_ids=batch.user_ids,
            sem_ids=sem_ids,
            sem_ids_fut=sem_ids_fut,
            seq_mask=seq_mask,
            token_type_ids=token_type_ids,
            token_type_ids_fut=token_type_ids_fut
        )

if __name__ == "__main__":
    dataset = ItemData("dataset/ml-1m-movie")
    tokenizer = SemanticIdTokenizer(18, 32, [32], 32)
    tokenizer.precompute_corpus_ids(dataset)
    
    seq_data = SeqData("dataset/ml-1m")
    batch = seq_data[:10]
    tokenized = tokenizer(batch)
    import pdb; pdb.set_trace()
