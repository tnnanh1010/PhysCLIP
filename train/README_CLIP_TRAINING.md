# CLIP Fine-tuning for Astronomy Images

Two versions of CLIP fine-tuning scripts for astronomy image-caption data:

## �� Files Created

1. **`train.py`** - Standard CLIP training with in-batch negatives
2. **`train_clip_hard_negatives.py`** - CLIP training with hard negative mining (TO BE COMPLETED)

## 🎯 Key Features

### Standard Version (`train.py`)
- **In-batch negatives**: Uses all other samples in the batch as negatives
- **Two-phase training**: 
  - Phase 1: Warmup with frozen encoders (2 epochs)
  - Phase 2: Full fine-tuning (10 epochs)
- **Scientific accuracy**: Minimal image augmentation to preserve astronomical details
- **Embedding dimension**: 512 (matches CLIP standard)

### Hard Negative Mining Version
- **Hard negatives**: Selects the most similar incorrect pairs as negatives
- **Focused learning**: Trains on difficult distinctions
- **Configurable ratio**: `--hard_neg_ratio 0.5` uses top 50% hardest negatives

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision transformers accelerate wandb pillow tqdm
```

### 2. Run Standard Training

```bash
cd /bigdisk/minhpvt/quick-test/na/ML/train

# Single GPU
python train.py \
  --train_data /bigdisk/minhpvt/quick-test/na/ML/data/batches/image_captions_1_34.jsonl \
  --output_dir ./outputs/clip-astro-standard \
  --batch_size 64 \
  --num_epochs 10 \
  --warmup_epochs 2 \
  --learning_rate 5e-6 \
  --use_wandb \
  --run_name clip-astro-standard-v1

# Multi-GPU with Accelerate
accelerate launch --multi_gpu --num_processes 4 train.py \
  --train_data /bigdisk/minhpvt/quick-test/na/ML/data/batches/image_captions_1_34.jsonl \
  --output_dir ./outputs/clip-astro-standard \
  --batch_size 32 \
  --gradient_accumulation_steps 2 \
  --num_epochs 10 \
  --use_wandb
```

### 3. Run Hard Negative Mining Training

```bash
python train_clip_hard_negatives.py \
  --train_data /bigdisk/minhpvt/quick-test/na/ML/data/batches/image_captions_1_34.jsonl \
  --output_dir ./outputs/clip-astro-hard-neg \
  --batch_size 64 \
  --hard_neg_ratio 0.5 \
  --num_epochs 10 \
  --use_wandb \
  --run_name clip-astro-hard-neg-v1
```

## ⚙️ Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--vision_model` | `openai/clip-vit-base-patch32` | Vision encoder |
| `--text_model` | `arnosimons/astro-hep-bert` | Text encoder (astronomy-specific BERT) |
| `--projection_dim` | 512 | Embedding dimension |
| `--batch_size` | 64 | Batch size per device |
| `--num_epochs` | 10 | Total training epochs |
| `--warmup_epochs` | 2 | Epochs with frozen encoders |
| `--learning_rate` | 5e-6 | Learning rate for full training |
| `--warmup_lr` | 1e-4 | Learning rate for warmup phase |
| `--temperature` | 0.07 | Temperature for contrastive loss |
| `--hard_neg_ratio` | 0.5 | (Hard neg only) Ratio of hard negatives |
| `--gradient_checkpointing` | - | Enable to save memory |
| `--mixed_precision` | `bf16` | Mixed precision (`bf16`, `fp16`, or `no`) |

## 📊 Expected Results

After training, you'll get:
- `best_model/` - Model with highest validation recall@1
- `checkpoint-epoch-N/` - Checkpoints every 2 epochs
- Wandb logs with:
  - Training loss & accuracy
  - Validation recall@1, @5, @10 (image↔text retrieval)

## 🔍 Model Architecture

```
Vision Encoder: CLIP-ViT-B/32 (224x224 images → 512-dim)
     ↓
Projection Layer (trained)
     ↓
512-dim Shared Space ← Contrastive Loss → 512-dim Shared Space
     ↑
Projection Layer (trained)
     ↑
Text Encoder: Astro-HEP-BERT (astronomy captions → 768-dim → 512-dim)
```

## 💡 Training Strategy

### Phase 1: Projection Warmup (2 epochs)
- ❄️ Freeze vision & text encoders
- 🔥 Train only projection layers
- 🎯 Goal: Align embedding spaces without disrupting pre-trained weights

### Phase 2: Full Fine-tuning (10 epochs)
- 🔥 Unfreeze all parameters
- 📈 Lower learning rate (5e-6)
- 🎯 Goal: Adapt encoders to astronomy domain

## �� Monitoring

```python
# Key metrics to watch:
- train/loss: Should decrease steadily
- train/accuracy: In-batch retrieval accuracy
- val/i2t_recall@1: Image→Text retrieval (most important)
- val/t2i_recall@1: Text→Image retrieval
```

## 🎓 Differences: Standard vs Hard Negative Mining

| Aspect | Standard | Hard Negative Mining |
|--------|----------|---------------------|
| Negatives | All batch samples | Top K% most similar |
| Learning focus | Broad distinctions | Difficult boundaries |
| Convergence | Faster, stabler | Slower, more focused |
| Performance | Good for diverse data | Better for fine distinctions |
| Computational cost | Lower | Slightly higher |

## 🔧 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size and use gradient accumulation
--batch_size 32 --gradient_accumulation_steps 2

# Enable gradient checkpointing
--gradient_checkpointing

# Use fp16 instead of bf16
--mixed_precision fp16
```

### Slow Training
```bash
# Use multi-GPU
accelerate launch --multi_gpu --num_processes 4 train.py ...

# Reduce workers if I/O bound
--num_workers 2
```

### Poor Convergence
```bash
# Increase warmup epochs
--warmup_epochs 3

# Adjust temperature
--temperature 0.05  # Lower = harder negatives
```

## 📝 Data Format

The JSONL file should have:
```json
{"image_path": "/path/to/image.jpg", "description": "Caption text"}
```

Dataset stats:
- **Total samples**: 105,056 image-caption pairs
- **Source**: 10,146 astronomy papers (34 batches)
- **Filtering**: Score ≥3, Figure captions only, Length >20 chars

## 🎯 Use Cases After Training

1. **Zero-shot image classification**
   ```python
   labels = ["galaxy", "nebula", "star cluster"]
   # Find best matching label for any astronomy image
   ```

2. **Image retrieval**
   ```python
   query = "spiral galaxy with active nucleus"
   # Find most similar images in database
   ```

3. **Caption generation** (with additional decoder)

4. **Astronomy image search engine**

