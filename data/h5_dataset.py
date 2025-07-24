import h5py
import numpy as np
import paddle
from typing import Dict, List, Optional, Tuple
from data.schemas import SeqBatch
from data.utils import batch_to
import os


class H5PretrainedDataset:
    """
    Dataset for loading pre-trained item embeddings from H5 files for RQ-VAE training.
    
    This dataset is optimized for RQ-VAE training and provides item-level train/eval splitting:
    - 95% of items for training
    - 5% of items for evaluation
    
    Expected file structure:
    - item_data.h5: Contains item embeddings and features
    """
    
    def __init__(
        self,
        item_data_path: str,
        train_test_split: str = "train",
        test_ratio: float = 0.05
    ):
        """
        Initialize H5 pretrained dataset for RQ-VAE training.
        
        Args:
            item_data_path: Path to item_data.h5 file
            train_test_split: "train" or "eval" for item-level splitting
            test_ratio: Ratio of items to use for evaluation (default: 0.05 = 5%)
        """
        self.item_data_path = item_data_path
        self.train_test_split = train_test_split
        self.test_ratio = test_ratio
        
        # Verify file exists
        if not os.path.exists(item_data_path):
            raise FileNotFoundError(f"Item data file not found: {item_data_path}")
            
        # Load item data
        self._load_item_data()
        
        # Create item-level train/test split
        self._create_train_test_split()
        
    def _load_item_data(self):
        """Load item embeddings and create item mapping."""
        print(f"Loading item data from {self.item_data_path}")
        
        with h5py.File(self.item_data_path, 'r') as f:
            self.item_ids = f['item_ids'][:]
            self.item_embeddings = f['embeddings'][:]
            
            # Load metadata
            self.n_items = f.attrs['n_items']
            self.embedding_dim = f.attrs['embedding_dim']
            
        # Create item_id to index mapping
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        print(f"Loaded {self.n_items} items with {self.embedding_dim}D embeddings")
        
        
    def _create_train_test_split(self):
        """Create item-level train/test split for RQ-VAE training."""
        # Create item-level train/test split: 95% train, 5% eval
        paddle.seed(42)  # For reproducibility
        self.item_is_train = paddle.rand([self.n_items]) > self.test_ratio
        
        # Determine active items based on train_test_split mode
        if self.train_test_split == "train":
            self.active_item_mask = self.item_is_train
            print(f"Using {self.item_is_train.sum().item()}/{self.n_items} items for training")
        elif self.train_test_split == "eval":
            self.active_item_mask = ~self.item_is_train
            print(f"Using {(~self.item_is_train).sum().item()}/{self.n_items} items for evaluation")
        else:
            raise ValueError(f"Unsupported train_test_split: {self.train_test_split}. Use 'train' or 'eval'.")
        
    def __len__(self):
        """Return number of active items."""
        return self.active_item_mask.sum().item()
        
    def __getitem__(self, idx) -> SeqBatch:
        """Get individual item sample for RQ-VAE training."""
        # Handle batch indices (tensor input)
        if isinstance(idx, paddle.Tensor):
            if idx.ndim == 0:  # Single tensor index
                idx = idx.item()
            else:  # Multiple indices
                return [self[i.item()] for i in idx]
        
        # Get active item indices
        active_item_indices = paddle.where(self.active_item_mask)[0]
        
        if idx >= len(active_item_indices):
            raise IndexError(f"Index {idx} out of range for item-level access")
        
        # Get the actual item index
        item_idx = active_item_indices[idx].item()
        
        # Create tensors for single item (optimized for RQ-VAE)
        ids = paddle.to_tensor([item_idx], dtype=paddle.int64)
        user_ids = paddle.to_tensor([-1], dtype=paddle.int64)  # Placeholder
        ids_fut = paddle.to_tensor([-1], dtype=paddle.int64)  # Placeholder
        
        # Get item embedding - this is what RQ-VAE actually uses
        x = paddle.to_tensor(self.item_embeddings[item_idx], dtype=paddle.float32).unsqueeze(0)
        x_fut = paddle.zeros_like(x)  # Placeholder
        
        # Single item mask
        seq_mask = paddle.to_tensor([True], dtype=paddle.bool)
        
        return SeqBatch(
            user_ids=user_ids,
            ids=ids,
            ids_fut=ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=seq_mask
        )


def create_h5_dataloader(
    item_data_path: str,
    batch_size: int = 64,
    train_test_split: str = "train",
    test_ratio: float = 0.05,
    shuffle: bool = True
) -> paddle.io.DataLoader:
    """
    Create DataLoader for H5 pretrained dataset (RQ-VAE training).
    
    Args:
        item_data_path: Path to item_data.h5
        batch_size: Batch size
        train_test_split: "train" or "eval" for item-level splitting
        test_ratio: Ratio of items for evaluation (default: 0.05 = 5%)
        shuffle: Whether to shuffle data
        
    Returns:
        paddle.io.DataLoader
    """
    dataset = H5PretrainedDataset(
        item_data_path=item_data_path,
        train_test_split=train_test_split,
        test_ratio=test_ratio
    )
    
    def collate_fn(batch):
        """
        Collate function for item-level batching (RQ-VAE training).
        
        Each sample in batch is a single item with shape [1, embedding_dim].
        Output batch will have shape [batch_size, embedding_dim].
        """
        if len(batch) == 1:
            return batch[0]
            
        # Concatenate item-level tensors
        user_ids = paddle.concat([item.user_ids for item in batch], axis=0)  # [batch_size]
        ids = paddle.concat([item.ids for item in batch], axis=0)  # [batch_size]
        ids_fut = paddle.concat([item.ids_fut for item in batch], axis=0)  # [batch_size]
        x = paddle.concat([item.x for item in batch], axis=0)  # [batch_size, embedding_dim]
        x_fut = paddle.concat([item.x_fut for item in batch], axis=0)  # [batch_size, embedding_dim]
        seq_mask = paddle.concat([item.seq_mask for item in batch], axis=0)  # [batch_size]
        
        return SeqBatch(
            user_ids=user_ids,
            ids=ids,
            ids_fut=ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=seq_mask
        )
    
    sampler = paddle.io.RandomSampler(dataset) if shuffle else None
    
    return paddle.io.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 for debugging, increase for production
    )