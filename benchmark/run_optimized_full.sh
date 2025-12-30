#!/bin/bash

# Optimized benchmark - all 75 epochs per model
# With dataset pre-loaded in memory and larger batches

python benchmark_spacenet.py \
  --dataset_dir /bigdisk/minhpvt/quick-test/na/ML/benchmark/data/spacenet/SpaceNet.FLARE.imam_alam \
  --astrobert_dir /bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_astrobert \
  --physbert_dir /bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_physbert \
  --output_dir /bigdisk/minhpvt/quick-test/na/ML/benchmark/results \
  --batch_size 1024
