python train.py \
  --train_data /bigdisk/minhpvt/quick-test/na/ML/data/batches/image_captions_1_34.jsonl \
  --output_dir ./outputs/clip-astro_physbert_uncased \
  --batch_size 256 \
  --num_epochs 150 \
  --use_wandb \
  --run_name clip-astronomy-v1_physbert_uncased \
  --warmup_epochs 10 \
  --text_model thellert/physbert_uncased \