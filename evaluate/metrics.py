from collections import defaultdict
from einops import rearrange
from paddle import Tensor


class TopKAccumulator:
    def __init__(self, ks=[1, 5, 10]):
        self.ks = ks
        self.reset()

    def reset(self):
        self.total = 0
        self.metrics = defaultdict(int)

    def accumulate(self, actual: Tensor, top_k: Tensor) -> None:
        B, D_actual = actual.shape
        B_topk, K, D_topk = top_k.shape
        
        # Handle dimension mismatch by taking minimum length
        D = min(D_actual, D_topk)
        actual_truncated = actual[:, :D]
        top_k_truncated = top_k[:, :, :D]
        
        pos_match = (rearrange(actual_truncated, "b d -> b 1 d") == top_k_truncated)
        for i in range(D):
            match_found, rank = pos_match[...,:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_slice_:{i+1}"] += len(matched_rank[matched_rank < k])
            
            match_found, rank = pos_match[...,i:i+1].all(axis=-1).max(axis=-1)
            matched_rank = rank[match_found]
            for k in self.ks:
                self.metrics[f"h@{k}_pos_{i}"] += len(matched_rank[matched_rank < k])
        self.total += B
        
    def reduce(self) -> dict:
        return {k: v/self.total for k, v in self.metrics.items()}
