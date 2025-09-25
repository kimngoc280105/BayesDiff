#!/bin/bash
PYTHON=python3

# ============================
# 1. Tải model từ Hugging Face
# ============================
mkdir -p models
wget -O models/v1-5-pruned.ckpt https://huggingface.co/kngoc281/sd-v1-5-pruned/resolve/main/v1-5-pruned.ckpt
wget -O models/v1-inference.yaml https://huggingface.co/kngoc281/sd-v1-5-pruned/resolve/main/v1-inference.yaml

# ============================
# 2. Chạy với ddim_skipUQ.py
# ============================
CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt models/v1-5-pruned.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# ============================
# 3. Chạy với dpmsolver_skipUQ.py
# ============================
CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt models/v1-5-pruned.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
