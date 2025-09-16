#!/bin/bash
set -e

# Về thư mục gốc
cd /content

# Kích hoạt môi trường ảo
source bayesdiff_env/bin/activate

# Vào thư mục Stable Diffusion
cd BayesDiff/sd

# ============================
# 1. Cooperate UQ into ddim sampler
# ============================
CUDA_VISIBLE_DEVICES=0 python ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# ============================
# 2. Cooperate UQ into dpm-solver-2 sampler
# ============================
CUDA_VISIBLE_DEVICES=0 python dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
