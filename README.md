# PhysCLIP Benchmark & Evaluation (Notebooks)

This repo provides a reproducible benchmark pipeline for PhysCLIP and baselines on:
1) **Image Classification** (Top-1 Accuracy) under zero-shot and linear probing (1% / 10% / 100% labels)
2) **Cross-Modal Retrieval** (Recall@1/5/10) for text→image and image→text

The benchmarking protocol follows standard CLIP-style evaluation: compute normalized embeddings, measure cosine similarity, and report Recall@K for retrieval.

---

## Benchmark Overview

### Tasks
- **Image Classification:** CIFAR-10 (general-domain) and SpaceNet (astronomy-domain) with:
  - zero-shot (prompt-based, where supported)
  - linear probing with 1% / 10% / 100% labels
- **Cross-Modal Retrieval:** AstroFig (figure–caption pairs), evaluated as:
  - text→image retrieval
  - image→text retrieval

### Models / Baselines
- CLIP (OpenAI)
- AstroCLIP (image-only baseline; no free-form text encoder → no prompt-based zero-shot classification)
- PhysCLIP variants (AstroBERT-based and PhysBERT-based)

---

## Data and Model Sources

Use the following sources as configured in the notebooks:

```text
Datasets:
- AstroFig (retrieval): https://huggingface.co/datasets/tnnanh1005/AstroFig
- SpaceNet (classification): https://www.kaggle.com/datasets/razaimam45/spacenet-an-optimally-distributed-astronomy-data
- CIFAR-10 (classification): https://www.cs.toronto.edu/~kriz/cifar.html

Baselines / Checkpoints:
- OpenAI CLIP ViT-B/32: https://huggingface.co/openai/clip-vit-base-patch32
- AstroCLIP (repo): https://github.com/PolymathicAI/AstroCLIP
- PhysCLIP-AstroBERT: https://huggingface.co/tnnanh1005/physclip_astrobert
- PhysCLIP-PhysBERT: https://huggingface.co/tnnanh1005/PhysBert_Model
