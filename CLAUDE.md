# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a PyTorch implementation of a generative retrieval recommender system using RQ-VAE (Residual Quantized Variational AutoEncoder) semantic IDs. The system has two main training stages:

1. **RQ-VAE Stage**: Items are mapped to semantic ID tuples using an RQ-VAE model
2. **Decoder Stage**: A transformer-based model is trained on sequences of semantic IDs for retrieval

## Core Commands

### Installation
```bash
pip install -r requirements.txt
```

### Training Commands

#### RQ-VAE Tokenizer Training
```bash
# Amazon Reviews dataset
python train_rqvae.py configs/rqvae_amazon.gin

# MovieLens 32M dataset  
python train_rqvae.py configs/rqvae_ml32m.gin
```

#### Retrieval Model Training
```bash
# Amazon Reviews dataset
python train_decoder.py configs/decoder_amazon.gin

# MovieLens 32M dataset
python train_decoder.py configs/decoder_ml32m.gin
```

### Development and Testing

No formal test suite is provided. The training scripts include evaluation metrics during training:
- RQ-VAE training includes reconstruction loss and quantization metrics
- Decoder training includes top-k accuracy metrics

### Monitoring Training
Both training scripts support W&B logging when `wandb_logging=True` is set in gin configs.

## Architecture Overview

### Key Components

**Data Processing** (`data/`):
- `processed.py`: Core dataset abstraction with `RecDataset` enum (AMAZON, ML_1M, ML_32M)
- `amazon.py`, `ml1m.py`, `ml32m.py`: Dataset-specific loaders
- `schemas.py`: Data structures for batching (`SeqBatch`, `TokenizedSeqBatch`)
- `utils.py`: Batching and data utilities

**RQ-VAE Model** (`modules/rqvae.py`):
- `RqVae`: Main model class with configurable encoder/decoder architecture
- Supports multiple quantization modes: Gumbel-Softmax, rotation trick
- Integrates with HuggingFace Hub for model sharing

**Retrieval Model** (`modules/model.py`):
- `EncoderDecoderRetrievalModel`: Transformer-based generative retrieval
- Uses frozen RQ-VAE for semantic ID tokenization
- Supports jagged tensor operations for variable-length sequences

**Core Modules**:
- `modules/quantize.py`: Vector quantization implementations
- `modules/transformer/`: Attention mechanisms and transformer blocks
- `modules/embedding/`: Semantic ID and user ID embedders
- `modules/loss.py`: Reconstruction and quantization losses

**Optimizations** (`ops/triton/`):
- Custom Triton kernels for jagged tensor operations
- Optimized for variable-length sequence handling

### Configuration System

Uses `gin-config` for experiment configuration:
- Training functions decorated with `@gin.configurable`
- All hyperparameters specified in `.gin` files under `configs/`
- Sample configs provided for Amazon and MovieLens datasets

### Model Checkpoints

Pre-trained models stored in `trained_models/`:
- RQ-VAE checkpoints for different datasets
- Models available on HuggingFace Hub (e.g., `edobotta/rqvae-amazon-beauty`)

## Dataset Support

**Supported Datasets**:
- Amazon Reviews (Beauty, Sports, Toys)
- MovieLens 1M
- MovieLens 32M

**Data Flow**:
1. Raw datasets automatically downloaded and cached
2. Preprocessing creates item features and user sequences
3. RQ-VAE training maps items to semantic IDs
4. Decoder training uses tokenized sequences

## Development Notes

- Built with PyTorch 2.5+ and Accelerate for distributed training
- Uses mixed precision training (fp16) by default
- Supports both CPU and GPU training with automatic device detection
- Triton kernels require CUDA-compatible GPU
- Training typically requires significant GPU memory (16GB+ recommended)