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
            
            # Try to load timestamps if available
            self.has_timestamps = 'timestamps' in f
            if self.has_timestamps:
                timestamps_raw = [f['timestamps'][i] for i in range(len(f['timestamps']))]
                print("Found timestamps in sequence data - will use time-based splitting")
            else:
                timestamps_raw = None
                print("No timestamps found - using simple index-based splitting")
            
            # Load metadata
            self.n_sequences = f.attrs['n_sequences']
            self.max_sequence_length = f.attrs['max_sequence_length']
            
        # Filter sequences and convert item IDs to indices
        self.processed_sequences = []
        self.user_ids = []
        self.sequence_timestamps = []
        
        timestamp_iter = timestamps_raw if self.has_timestamps else [None] * len(user_ids_raw)
        
        for user_id, seq, seq_len, timestamps in zip(user_ids_raw, sequences_raw, sequence_lengths, timestamp_iter):
            # Convert item IDs to indices, skip unknown items
            item_indices = []
            for item_id in seq:
                if item_id in self.item_id_to_idx:
                    item_indices.append(self.item_id_to_idx[item_id])
                    
            # Only keep sequences with sufficient length
            if len(item_indices) >= 2:  # Need at least 2 items for training
                self.processed_sequences.append(item_indices)
                self.user_ids.append(user_id)
                
                # Store max timestamp for this sequence (for time-based splitting)
                if self.has_timestamps and timestamps is not None:
                    max_timestamp = max(timestamps) if len(timestamps) > 0 else 0
                    self.sequence_timestamps.append(max_timestamp)
                else:
                    # Use sequence index as fallback timestamp
                    self.sequence_timestamps.append(len(self.processed_sequences) - 1)
                
        print(f"Processed {len(self.processed_sequences)} valid sequences")
        
    def _create_train_test_split(self):
        """Create train/test split like original dataset - based on sequence-internal temporal order."""
        # In the original dataset, the split logic is:
        # - For training: use full sequences with target=-1 
        # - For eval: use sequence[:-1] with target=sequence[-1]
        # - All sequences are used in both training and eval, but processed differently
        
        # Keep all sequences active since we'll handle train/test split at sample level
        self.active_sequences = self.processed_sequences
        self.active_user_ids = self.user_ids
        
        print(f"Active dataset size: {len(self.active_sequences)} sequences")
        print("Using sequence-internal temporal splitting (like original dataset)")
        
    def __len__(self):
        """Return number of samples based on train_test_split mode."""
        if self.train_test_split == "train":
            # Training mode: each sequence generates 1 sample (full sequence with target=-1)
            return len([seq for seq in self.active_sequences if len(seq) >= 2])
        elif self.train_test_split == "test":
            # Test mode: each sequence generates 1 sample (sequence[:-1] with target=sequence[-1])
            return len([seq for seq in self.active_sequences if len(seq) >= 2])
        else:  # "all"
            # All mode: each sequence generates 2 samples (1 train + 1 eval)
            return 2 * len([seq for seq in self.active_sequences if len(seq) >= 2])
        
    def __getitem__(self, idx: int) -> SeqBatch:
        """
        Get a single sample based on train_test_split mode (like original dataset).
        
        Training mode: full sequence with target=-1
        Test mode: sequence[:-1] with target=sequence[-1]
        All mode: alternates between train and test samples
        """
        # Filter valid sequences
        valid_sequences = [(i, seq) for i, seq in enumerate(self.active_sequences) if len(seq) >= 2]
        
        if self.train_test_split == "train":
            # Training mode: each sequence -> 1 training sample
            if idx >= len(valid_sequences):
                raise IndexError(f"Index {idx} out of range")
            seq_idx, seq = valid_sequences[idx]
            is_train_sample = True
            
        elif self.train_test_split == "test":
            # Test mode: each sequence -> 1 test sample  
            if idx >= len(valid_sequences):
                raise IndexError(f"Index {idx} out of range")
            seq_idx, seq = valid_sequences[idx]
            is_train_sample = False
            
        else:  # "all"
            # All mode: each sequence generates 2 samples
            n_valid = len(valid_sequences)
            if idx >= 2 * n_valid:
                raise IndexError(f"Index {idx} out of range")
            seq_idx, seq = valid_sequences[idx // 2]
            is_train_sample = (idx % 2 == 0)  # Even indices are training samples
        
        user_id = self.active_user_ids[seq_idx]
        
        if is_train_sample:
            # Training sample: use full sequence (up to max_seq_len), target=-1
            input_seq = seq[-self.max_seq_len:] if len(seq) > self.max_seq_len else seq
            target_item = -1
        else:
            # Test sample: use sequence[:-1] (up to max_seq_len), target=sequence[-1]
            seq_without_last = seq[:-1]
            input_seq = seq_without_last[-self.max_seq_len:] if len(seq_without_last) > self.max_seq_len else seq_without_last
            target_item = seq[-1]
        
        # Pad sequence if needed
        actual_len = len(input_seq)
        if actual_len < self.max_seq_len:
            input_seq = input_seq + [-1] * (self.max_seq_len - actual_len)
        
        # Create tensors
        ids = paddle.to_tensor(input_seq, dtype=paddle.int64)
        ids_fut = paddle.to_tensor([target_item], dtype=paddle.int64)
        user_ids = paddle.to_tensor([hash(user_id) % (2**31)], dtype=paddle.int64)
        
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