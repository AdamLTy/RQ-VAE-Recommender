import h5py
import numpy as np
import paddle
from typing import Dict, List, Optional, Tuple
from data.schemas import SeqBatch
from data.utils import batch_to
import os


class H5PretrainedDataset:
    """
    Dataset for loading pre-trained embeddings from H5 files.
    
    Expected file structure:
    - item_data.h5: Contains item embeddings and features
    - sequence.h5: Contains user interaction sequences
    """
    
    def __init__(
        self,
        item_data_path: str,
        sequence_data_path: str,
        max_seq_len: int = 200,
        train_test_split: str = "train",
        test_ratio: float = 0.2
    ):
        """
        Initialize H5 pretrained dataset.
        
        Args:
            item_data_path: Path to item_data.h5 file
            sequence_data_path: Path to sequence.h5 file  
            max_seq_len: Maximum sequence length for training
            train_test_split: "train", "test", or "all"
            test_ratio: Ratio of data to use for testing
        """
        self.item_data_path = item_data_path
        self.sequence_data_path = sequence_data_path
        self.max_seq_len = max_seq_len
        self.train_test_split = train_test_split
        self.test_ratio = test_ratio
        
        # Verify files exist
        if not os.path.exists(item_data_path):
            raise FileNotFoundError(f"Item data file not found: {item_data_path}")
        if not os.path.exists(sequence_data_path):
            raise FileNotFoundError(f"Sequence data file not found: {sequence_data_path}")
            
        # Load item data
        self._load_item_data()
        
        # Load and process sequence data
        self._load_sequence_data()
        
        # Create train/test split
        self._create_train_test_split()
        
    def _load_item_data(self):
        """Load item embeddings and create item mapping."""
        print(f"Loading item data from {self.item_data_path}")
        
        with h5py.File(self.item_data_path, 'r') as f:
            self.item_ids = f['item_ids'][:]
            self.item_embeddings = f['embeddings'][:]
            self.item_features = f['features'][:]
            self.item_feature_lengths = f['feature_lengths'][:]
            
            # Load metadata
            self.n_items = f.attrs['n_items']
            self.embedding_dim = f.attrs['embedding_dim']
            self.max_feature_len = f.attrs['max_feature_len']
            
        # Create item_id to index mapping
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        
        print(f"Loaded {self.n_items} items with {self.embedding_dim}D embeddings")
        
    def _load_sequence_data(self):
        """Load user interaction sequences."""
        print(f"Loading sequence data from {self.sequence_data_path}")
        
        with h5py.File(self.sequence_data_path, 'r') as f:
            # Load raw sequences
            user_ids_raw = [f['user_ids'][i].decode('utf-8') for i in range(len(f['user_ids']))]
            sequences_raw = [f['sequences'][i] for i in range(len(f['sequences']))]
            sequence_lengths = f['sequence_lengths'][:]
            
            # Load metadata
            self.n_sequences = f.attrs['n_sequences']
            self.max_sequence_length = f.attrs['max_sequence_length']
            
        # Filter sequences and convert item IDs to indices
        self.processed_sequences = []
        self.user_ids = []
        
        for user_id, seq, seq_len in zip(user_ids_raw, sequences_raw, sequence_lengths):
            # Convert item IDs to indices, skip unknown items
            item_indices = []
            for item_id in seq:
                if item_id in self.item_id_to_idx:
                    item_indices.append(self.item_id_to_idx[item_id])
                    
            # Only keep sequences with sufficient length
            if len(item_indices) >= 2:  # Need at least 2 items for training
                self.processed_sequences.append(item_indices)
                self.user_ids.append(user_id)
                
        print(f"Processed {len(self.processed_sequences)} valid sequences")
        
    def _create_train_test_split(self):
        """Create train/test split based on configuration."""
        n_sequences = len(self.processed_sequences)
        n_test = int(n_sequences * self.test_ratio)
        n_train = n_sequences - n_test
        
        if self.train_test_split == "train":
            self.active_sequences = self.processed_sequences[:n_train]
            self.active_user_ids = self.user_ids[:n_train] 
        elif self.train_test_split == "test":
            self.active_sequences = self.processed_sequences[n_train:]
            self.active_user_ids = self.user_ids[n_train:]
        else:  # "all"
            self.active_sequences = self.processed_sequences
            self.active_user_ids = self.user_ids
            
        print(f"Active dataset size: {len(self.active_sequences)} sequences")
        
    def __len__(self):
        """Return number of training samples."""
        total_samples = 0
        for seq in self.active_sequences:
            # Each sequence can generate multiple training samples
            seq_len = min(len(seq), self.max_seq_len + 1)
            if seq_len >= 2:
                total_samples += seq_len - 1
        return total_samples
        
    def __getitem__(self, idx: int) -> SeqBatch:
        """
        Get a single training sample.
        
        Returns a SeqBatch with:
        - user_ids: User ID tensor
        - ids: Item ID sequence 
        - ids_fut: Future item IDs
        - x: Item embedding sequence
        - x_fut: Future item embeddings
        - seq_mask: Sequence mask
        """
        # Map flat index to sequence and position
        current_idx = 0
        for seq_idx, seq in enumerate(self.active_sequences):
            seq_len = min(len(seq), self.max_seq_len + 1)
            if seq_len < 2:
                continue
                
            n_samples = seq_len - 1
            if current_idx + n_samples > idx:
                # Found the sequence
                pos_in_seq = idx - current_idx
                break
            current_idx += n_samples
        else:
            raise IndexError(f"Index {idx} out of range")
            
        # Extract sequence segment
        seq = self.active_sequences[seq_idx]
        user_id = self.active_user_ids[seq_idx]
        
        # Create input and target sequences
        start_pos = max(0, pos_in_seq + 1 - self.max_seq_len)
        end_pos = pos_in_seq + 1
        target_pos = pos_in_seq + 1
        
        input_seq = seq[start_pos:end_pos]
        target_item = seq[target_pos] if target_pos < len(seq) else -1
        
        # Pad sequence if needed
        actual_len = len(input_seq)
        if actual_len < self.max_seq_len:
            input_seq = input_seq + [-1] * (self.max_seq_len - actual_len)
        
        # Create tensors
        ids = paddle.to_tensor(input_seq, dtype=paddle.int64)
        ids_fut = paddle.to_tensor([target_item], dtype=paddle.int64)
        user_ids = paddle.to_tensor([hash(user_id) % (2**31)], dtype=paddle.int64)  # Simple hash for user ID
        
        # Get embeddings for valid items
        x = paddle.zeros([self.max_seq_len, self.embedding_dim], dtype=paddle.float32)
        x_fut = paddle.zeros([1, self.embedding_dim], dtype=paddle.float32)
        
        for i, item_idx in enumerate(input_seq):
            if item_idx >= 0:
                x[i] = paddle.to_tensor(self.item_embeddings[item_idx], dtype=paddle.float32)
                
        if target_item >= 0:
            x_fut[0] = paddle.to_tensor(self.item_embeddings[target_item], dtype=paddle.float32)
            
        # Create sequence mask
        seq_mask = paddle.to_tensor([1 if item_idx >= 0 else 0 for item_idx in input_seq], dtype=paddle.bool)
        
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
    sequence_data_path: str,
    batch_size: int = 64,
    max_seq_len: int = 200,
    train_test_split: str = "train",
    test_ratio: float = 0.2,
    shuffle: bool = True
) -> paddle.io.DataLoader:
    """
    Create DataLoader for H5 pretrained dataset.
    
    Args:
        item_data_path: Path to item_data.h5
        sequence_data_path: Path to sequence.h5
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        train_test_split: "train", "test", or "all"
        test_ratio: Ratio for train/test split
        shuffle: Whether to shuffle data
        
    Returns:
        paddle.io.DataLoader
    """
    dataset = H5PretrainedDataset(
        item_data_path=item_data_path,
        sequence_data_path=sequence_data_path,
        max_seq_len=max_seq_len,
        train_test_split=train_test_split,
        test_ratio=test_ratio
    )
    
    def collate_fn(batch):
        """Collate function to batch SeqBatch objects."""
        if len(batch) == 1:
            return batch[0]
            
        # Stack all tensors in the batch
        user_ids = paddle.concat([item.user_ids for item in batch], axis=0)
        ids = paddle.stack([item.ids for item in batch], axis=0)
        ids_fut = paddle.stack([item.ids_fut for item in batch], axis=0)
        x = paddle.stack([item.x for item in batch], axis=0)
        x_fut = paddle.stack([item.x_fut for item in batch], axis=0)
        seq_mask = paddle.stack([item.seq_mask for item in batch], axis=0)
        
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