#!/bin/bash
PYTHON=/content/bayesdiff_env/bin/python

# =============================
# 1. Tải model từ Hugging Face
# =============================
MODEL_DIR=/content/BayesDiff/sd/models
mkdir -p $MODEL_DIR

# tải ckpt và yaml (thay <username> <repo> bằng repo của bạn)
wget -nc https://huggingface.co/<username>/<repo>/resolve/main/v1-5-pruned.ckpt -O $MODEL_DIR/v1-5-pruned.ckpt
wget -nc https://huggingface.co/<username>/<repo>/resolve/main/v1-inference.yaml -O $MODEL_DIR/v1-inference.yaml

# =============================
# 2. Chạy ddim_skipUQ.py
# =============================
CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt $MODEL_DIR/v1-5-pruned.ckpt \
--local_image_path /content/BayesDiff/sd/laion_subset \
--laion_art_path /content/BayesDiff/sd/laion_art \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# =============================
# 3. Chạy dpmsolver_skipUQ.py
# =============================
CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt $MODEL_DIR/v1-5-pruned.ckpt \
--local_image_path /content/BayesDiff/sd/laion_subset \
--laion_art_path /content/BayesDiff/sd/laion_art \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
