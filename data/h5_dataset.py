import h5py
import numpy as np
import paddle
import random
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
            else:  # Multiple indices - return concatenated batch
                samples = [self[i.item()] for i in idx]
                return self._batch_samples(samples)
        
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
    
    def _batch_samples(self, samples):
        """Batch multiple SeqBatch samples into a single batch."""
        user_ids = paddle.concat([sample.user_ids for sample in samples], axis=0)
        ids = paddle.concat([sample.ids for sample in samples], axis=0)
        ids_fut = paddle.concat([sample.ids_fut for sample in samples], axis=0)
        x = paddle.concat([sample.x for sample in samples], axis=0)
        x_fut = paddle.concat([sample.x_fut for sample in samples], axis=0)
        seq_mask = paddle.concat([sample.seq_mask for sample in samples], axis=0)
        
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
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2
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
        num_workers=num_workers,
        use_shared_memory=False,
        prefetch_factor=prefetch_factor
    )


class H5SequenceDataset:
    """
    用于Decoder训练的H5序列数据集，完全独立于物品级数据集。
    
    该数据集处理用户行为序列，支持训练RQ-VAE Decoder模型。
    
    预期的H5文件结构：
    - sequence_data.h5: 包含用户行为序列和物品embeddings
      - user_ids: 用户ID数组 [n_sequences]  
      - item_sequences: 每个用户的物品ID序列 (变长)
      - sequence_lengths: 每个序列的长度 [n_sequences]
      - item_embeddings: 物品embedding矩阵 [n_items, embedding_dim]
      - item_id_mapping: 物品ID到embedding索引的映射
    """
    
    def __init__(
        self,
        sequence_data_path: str,
        item_data_path: str,
        is_train: bool = True,
        max_seq_len: int = 200,
        test_ratio: float = 0.2,  # 保留参数以兼容现有代码，但实际上不使用
        subsample: bool = False
    ):
        """
        初始化H5序列数据集。
        
        Args:
            sequence_data_path: 序列数据H5文件路径
            item_data_path: 物品数据H5文件路径 (包含item embeddings)
            is_train: 是否为训练集
            max_seq_len: 最大序列长度
            test_ratio: 测试集比例 (保留参数兼容性，但不使用)
            subsample: 是否对训练序列进行子采样
        
        注意: 验证集通过每个序列的最后一个位置自动构建，不需要test_ratio分割
        """
        self.sequence_data_path = sequence_data_path
        self.item_data_path = item_data_path
        self.is_train = is_train
        self.max_seq_len = max_seq_len
        self.test_ratio = test_ratio
        self.subsample = subsample
        
        # 验证文件存在
        print(f"🔍 Checking sequence file exists: {sequence_data_path}")
        if not os.path.exists(sequence_data_path):
            raise FileNotFoundError(f"Sequence data file not found: {sequence_data_path}")
        print("✅ Sequence file exists")
        
        print(f"🔍 Checking item data file exists: {item_data_path}")
        if not os.path.exists(item_data_path):
            raise FileNotFoundError(f"Item data file not found: {item_data_path}")
        print("✅ Item data file exists")
            
        # 加载序列数据
        print("🔄 Starting _load_sequence_data()...")
        self._load_sequence_data()
        print("✅ _load_sequence_data() completed")
        
        # 创建训练/测试分割
        print("🔄 Starting _create_train_test_split()...")
        self._create_train_test_split()
        print("✅ _create_train_test_split() completed")
        
    def _load_sequence_data(self):
        """加载序列数据和物品embeddings。"""
        print(f"Loading sequence data from {self.sequence_data_path}")
        
        with h5py.File(self.sequence_data_path, 'r') as f:
            # 加载用户序列数据
            print("🔄 Loading user_ids...")
            self.user_ids = f['user_ids'][:]
            print(f"✅ Loaded {len(self.user_ids)} user_ids")
            
            print("🔄 Loading sequence_lengths...")
            self.sequence_lengths = f['sequence_lengths'][:]
            print(f"✅ Loaded sequence_lengths")
            
            # 加载变长序列 (存储为vlen数据类型)
            print("🔄 Loading sequences data (this may take time)...")
            sequences_data = f['sequences'][:]  # 使用正确的字段名
            print(f"✅ Loaded sequences data, shape: {len(sequences_data)}")
            
            print("🔄 Converting sequences to list format...")
            self.item_sequences = [seq.tolist() for seq in sequences_data]
            print(f"✅ Converted to list format")
            
        # 从item_data.h5加载物品embeddings
        print(f"🔄 Loading item embeddings from {self.item_data_path}...")
        with h5py.File(self.item_data_path, 'r') as item_f:
            self.item_ids = item_f['item_ids'][:]
            self.item_embeddings = item_f['embeddings'][:]
            self.embedding_dim = item_f.attrs['embedding_dim']
            
            print(f"✅ Loaded item_embeddings: {self.item_embeddings.shape}")
            print(f"✅ Embedding dimension: {self.embedding_dim}")
            
            # 创建item_id到embedding index的映射
            self.item_id_to_embedding_idx = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
            print(f"✅ Created item_id_to_embedding_idx mapping: {len(self.item_id_to_embedding_idx)} items")
            
        # 加载元数据
        print("🔄 Loading metadata...")
        with h5py.File(self.sequence_data_path, 'r') as f:
            if hasattr(f, 'attrs'):
                self.n_items = f.attrs.get('total_items', len(self.item_id_to_embedding_idx))
                self.total_items = f.attrs.get('total_items', len(self.item_id_to_embedding_idx))
                print(f"✅ Loaded metadata from sequence file attrs: n_items={self.n_items}, total_items={self.total_items}")
            else:
                self.n_items = len(self.item_id_to_embedding_idx)
                self.total_items = self.n_items
                print(f"✅ Using default metadata: n_items={self.n_items}, total_items={self.total_items}")
            
        self.n_sequences = len(self.user_ids)
        
        print(f"🎉 Successfully loaded {self.n_sequences} sequences")
        print(f"🎉 Found {self.total_items} unique items with {self.embedding_dim}D embeddings")
        print("🎉 _load_sequence_data completed successfully")
        
    def _create_train_test_split(self):
        """创建训练/验证数据分割。
        
        注意: 由于验证是通过序列内部的最后一个位置实现的，
        训练和验证都使用所有序列，区别仅在于序列的处理方式。
        """
        print(f"🔄 Creating active indices for {self.n_sequences} sequences...")
        # 使用所有序列，因为验证通过序列内部的最后位置实现
        self.active_indices = paddle.arange(self.n_sequences)
        print(f"✅ Created active_indices: {len(self.active_indices)} items")
        
        if self.is_train:
            print(f"🎓 Using all {self.n_sequences} sequences for training (input: first n-1 items, target: last item)")
        else:
            print(f"📊 Using all {self.n_sequences} sequences for evaluation (input: first n-1 items, target: last item)")
    
    def __len__(self):
        """返回活跃序列的数量。"""
        return len(self.active_indices)
    
    def _get_item_embedding(self, item_id: int) -> np.ndarray:
        """根据物品ID获取embedding。"""
        if item_id == -1:  # padding token
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # 使用映射获取embedding索引
        if item_id in self.item_id_to_embedding_idx:
            emb_idx = self.item_id_to_embedding_idx[item_id]
            return self.item_embeddings[emb_idx]
        else:
            # 未知物品，返回零向量
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def __getitem__(self, idx) -> SeqBatch:
        """获取单个序列样本。"""
        # 获取实际序列索引
        seq_idx = self.active_indices[idx].item()
        
        user_id = self.user_ids[seq_idx]
        sequence = self.item_sequences[seq_idx]
        
        if self.subsample and self.is_train:
            # 对训练序列进行子采样
            if len(sequence) >= 3:
                start_idx = random.randint(0, max(0, len(sequence) - 3))
                end_idx = random.randint(start_idx + 3, 
                                       min(start_idx + self.max_seq_len + 1, len(sequence) + 1))
                sequence = sequence[start_idx:end_idx]
        
        # 截断到最大长度
        if len(sequence) > self.max_seq_len:
            sequence = sequence[-self.max_seq_len:]
        
        # 分离当前序列和未来物品
        if len(sequence) > 1:
            item_ids = sequence[:-1]
            item_ids_fut = [sequence[-1]]
        else:
            item_ids = sequence
            item_ids_fut = [-1]  # 没有未来物品
        
        # Padding到固定长度
        while len(item_ids) < self.max_seq_len:
            item_ids.append(-1)
        
        # 转换为tensor
        item_ids = paddle.to_tensor(item_ids[:self.max_seq_len], dtype=paddle.int64)
        item_ids_fut = paddle.to_tensor(item_ids_fut, dtype=paddle.int64)
        user_ids = paddle.to_tensor([user_id], dtype=paddle.int64)
        
        # 获取物品embeddings
        x = []
        for item_id in item_ids.numpy():
            x.append(self._get_item_embedding(int(item_id)))
        x = paddle.to_tensor(np.stack(x), dtype=paddle.float32)
        
        x_fut = []
        for item_id in item_ids_fut.numpy():
            x_fut.append(self._get_item_embedding(int(item_id)))
        x_fut = paddle.to_tensor(np.stack(x_fut), dtype=paddle.float32)
        
        # 创建序列mask (有效位置为True)
        seq_mask = (item_ids >= 0)
        
        return SeqBatch(
            user_ids=user_ids,
            ids=item_ids,
            ids_fut=item_ids_fut,
            x=x,
            x_fut=x_fut,
            seq_mask=seq_mask
        )


def create_h5_sequence_dataloader(
    sequence_data_path: str,
    item_data_path: str,
    batch_size: int = 64,
    is_train: bool = True,
    max_seq_len: int = 200,
    test_ratio: float = 0.2,  # 保留参数兼容性，但不使用
    subsample: bool = False,
    shuffle: bool = True,
    num_workers: int = 0,
    prefetch_factor: int = 2
) -> paddle.io.DataLoader:
    """
    创建H5序列数据集的DataLoader (用于Decoder训练)。
    
    Args:
        sequence_data_path: 序列数据H5文件路径
        item_data_path: 物品数据H5文件路径 (包含item embeddings)
        batch_size: 批次大小
        is_train: 是否为训练集
        max_seq_len: 最大序列长度
        test_ratio: 测试集比例 (保留参数兼容性，但不使用)
        subsample: 是否对训练序列进行子采样
        shuffle: 是否打乱数据
        
    Returns:
        paddle.io.DataLoader
        
    注意: 验证集通过每个序列的最后一个位置自动构建，不需要test_ratio分割
    """
    print(f"🔄 Creating H5SequenceDataset...")
    dataset = H5SequenceDataset(
        sequence_data_path=sequence_data_path,
        item_data_path=item_data_path,
        is_train=is_train,
        max_seq_len=max_seq_len,
        test_ratio=test_ratio,  # 传递但不使用
        subsample=subsample
    )
    print(f"✅ H5SequenceDataset created successfully, dataset size: {len(dataset)}")
    
    def collate_fn(batch):
        """
        序列级批处理函数。
        
        输入: 列表of SeqBatch, 每个包含一个序列
        输出: 批次化的SeqBatch, 形状为 [batch_size, seq_len, ...]
        """
        if len(batch) == 1:
            return batch[0]
            
        # 批次化序列级tensors
        user_ids = paddle.concat([item.user_ids for item in batch], axis=0)  # [batch_size]
        ids = paddle.stack([item.ids for item in batch], axis=0)  # [batch_size, seq_len]
        ids_fut = paddle.concat([item.ids_fut for item in batch], axis=0)  # [batch_size]
        x = paddle.stack([item.x for item in batch], axis=0)  # [batch_size, seq_len, embed_dim]
        x_fut = paddle.stack([item.x_fut for item in batch], axis=0)  # [batch_size, embed_dim]
        seq_mask = paddle.stack([item.seq_mask for item in batch], axis=0)  # [batch_size, seq_len]
        
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
        num_workers=num_workers,
        use_shared_memory=False,
        prefetch_factor=prefetch_factor
    )