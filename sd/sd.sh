#!/bin/bash
PYTHON=python3

mkdir -p models
wget -O models/v1-5-pruned.ckpt https://huggingface.co/kngoc281/sd-v1-5-pruned/resolve/main/v1-5-pruned.ckpt
wget -O models/v1-inference.yaml https://huggingface.co/kngoc281/sd-v1-5-pruned/resolve/main/v1-inference.yaml

CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
  --prompt "A futuristic city with flying cars, ultra realistic" \
  --ckpt models/v1-5-pruned.ckpt \
  --config models/v1-inference.yaml \
  --local_image_path /content/BayesDiff/laion_subset \
  --laion_art_path /content/drive/MyDrive/sd/laion-art.txt \
  --H 256 --W 256 --scale 3 \
  --train_la_data_size 50 --train_la_batch_size 2 \
  --sample_batch_size 1 --total_n_samples 4 --timesteps 20 \
  --precision autocast

CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
  --prompt "A futuristic city with flying cars, ultra realistic" \
  --ckpt models/v1-5-pruned.ckpt \
  --config models/v1-inference.yaml \
  --local_image_path /content/BayesDiff/laion_subset \
  --laion_art_path /content/drive/MyDrive/sd/laion-art.txt \
  --H 256 --W 256 --scale 3 \
  --train_la_data_size 50 --train_la_batch_size 2 \
  --sample_batch_size 1 --total_n_samples 4 --timesteps 20 \
  --precision autocast
