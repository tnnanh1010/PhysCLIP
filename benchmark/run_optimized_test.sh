#!/bin/bash

# Optimized benchmark - test with 5 epochs

python benchmark_spacenet.py \
  --dataset_dir /bigdisk/minhpvt/quick-test/na/ML/benchmark/data/spacenet/SpaceNet.FLARE.imam_alam \
  --astrobert_dir /bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_astrobert \
  --physbert_dir /bigdisk/minhpvt/quick-test/na/ML/train/outputs/physclip_physbert \
  --output_dir /bigdisk/minhpvt/quick-test/na/ML/benchmark/results_test \
  --batch_size 1024 \
  --epochs "2,4,6,8,10"
