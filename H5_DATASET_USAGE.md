# H5 预训练数据集使用指南

本指南说明如何在RQ-VAE训练中使用预训练的H5格式数据集。

## 快速开始

### 1. 准备H5数据文件

确保您有以下两个H5文件：

```
data/preprocessed/
├── item_data.h5      # 商品embedding和特征数据
└── sequence.h5       # 用户行为序列数据
```

### 2. 使用预配置的训练脚本

```bash
# 使用H5数据集训练RQ-VAE
python train_rqvae.py configs/rqvae_h5_pretrained.gin
```

### 3. 自定义配置

您也可以通过修改配置文件或使用命令行参数来自定义训练：

```bash
# 自定义H5文件路径
python train_rqvae.py configs/rqvae_h5_pretrained.gin \
    --gin_param="train.h5_item_data_path='path/to/your/item_data.h5'" \
    --gin_param="train.h5_sequence_data_path='path/to/your/sequence.h5'"
```

## 配置参数说明

### H5数据集相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_h5_dataset` | bool | False | 是否使用H5数据集 |
| `h5_item_data_path` | str | "data/preprocessed/item_data.h5" | 商品数据H5文件路径 |
| `h5_sequence_data_path` | str | "data/preprocessed/sequence.h5" | 序列数据H5文件路径 |
| `h5_max_seq_len` | int | 200 | 最大序列长度 |
| `h5_test_ratio` | float | 0.2 | 测试集比例 |

### 自动配置

当使用H5数据集时，以下配置会自动设置：

- `vae_input_dim`: 自动从H5文件中的embedding维度读取
- 数据加载器: 自动使用H5数据加载器而非原始数据集

## H5数据文件格式要求

### item_data.h5 结构

```python
# 数据集
'item_ids': shape=(n_items,), dtype=np.int64
'embeddings': shape=(n_items, embedding_dim), dtype=np.float32  
'features': shape=(n_items, max_feature_len), dtype=np.int32
'feature_lengths': shape=(n_items,), dtype=np.int32

# 属性
attrs['n_items']: int
attrs['embedding_dim']: int  
attrs['max_feature_len']: int
attrs['original_order_preserved']: bool
```

### sequence.h5 结构

```python
# 数据集  
'user_ids': shape=(n_sequences,), dtype=vlen string
'sequences': shape=(n_sequences,), dtype=vlen np.int64
'sequence_lengths': shape=(n_sequences,), dtype=np.int32
'timestamps': shape=(n_sequences,), dtype=vlen np.int64  # 可选，用于时间划分

# 属性
attrs['n_sequences']: int
attrs['max_sequence_length']: int
attrs['total_items']: int
attrs['avg_sequence_length']: float
attrs['order_preserved']: bool
```

**重要说明**: 
- `timestamps`字段是**可选的**，如果存在，会使用基于时间的数据划分（推荐）
- 如果没有`timestamps`字段，会退化为基于索引的简单划分
- `timestamps[i]`包含第i个序列中每个交互的时间戳数组

## 数据划分策略

### 序列内部时间划分（与原始数据集一致）
H5数据集现在使用与原始数据集完全相同的划分策略：

- **训练样本**: 使用完整序列，目标设为-1
  - 输入: `[item1, item2, ..., itemN]`
  - 目标: `-1`
  
- **验证样本**: 使用序列前N-1个items，目标为最后一个item  
  - 输入: `[item1, item2, ..., itemN-1]`
  - 目标: `itemN`

**优势**:
- ✅ 完全符合原始数据集的时间顺序逻辑
- ✅ 避免数据泄露，符合推荐系统真实场景
- ✅ 每个用户序列都参与训练和验证

## 训练流程

1. **数据加载**: H5数据集加载器读取预训练embeddings和用户序列
2. **序列划分**: 每个序列生成训练样本和验证样本
3. **样本生成**: 
   - 训练时: 使用完整序列学习重构
   - 验证时: 使用前N-1个items预测最后一个item
4. **Embedding获取**: 直接使用预训练的embedding，无需重新计算
5. **模型训练**: RQ-VAE模型学习将embeddings量化为语义ID

## 性能优势

使用H5预训练数据集的优势：

- **更快的训练启动**: 跳过文本编码步骤，直接使用预训练embeddings
- **内存效率**: H5格式支持lazy loading，减少内存使用
- **灵活性**: 可以使用任何预训练的embedding模型
- **可重复性**: 固定的embedding确保训练结果可重复

## 故障排除

### 常见问题

1. **文件找不到**: 确保H5文件路径正确且文件存在
2. **格式错误**: 检查H5文件是否包含所需的数据集和属性
3. **内存不足**: 考虑减少batch_size或使用数据采样

### 调试提示

```python
# 检查H5文件内容
import h5py

with h5py.File('data/preprocessed/item_data.h5', 'r') as f:
    print("Item数据集:", list(f.keys()))
    print("Item属性:", dict(f.attrs))
    print("Item数量:", f.attrs['n_items'])
    print("Embedding维度:", f.attrs['embedding_dim'])

with h5py.File('data/preprocessed/sequence.h5', 'r') as f:
    print("Sequence数据集:", list(f.keys()))
    print("Sequence属性:", dict(f.attrs))
    print("是否包含timestamps:", 'timestamps' in f)
    if 'timestamps' in f:
        print("将使用时间划分")
    else:
        print("将使用索引划分")
```

## 示例代码

```python
# 创建自定义H5数据加载器
from data.h5_dataset import create_h5_dataloader

dataloader = create_h5_dataloader(
    item_data_path="data/preprocessed/item_data.h5",
    sequence_data_path="data/preprocessed/sequence.h5",
    batch_size=64,
    max_seq_len=200,
    train_test_split="train",
    shuffle=True
)

# 获取一个batch
for batch in dataloader:
    print("Batch结构:")
    print(f"  user_ids: {batch.user_ids.shape}")
    print(f"  ids: {batch.ids.shape}")  
    print(f"  x: {batch.x.shape}")
    break
```