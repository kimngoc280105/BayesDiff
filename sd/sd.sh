#!/bin/bash
PYTHON=python3

python3 ddim_skipUQ.py --prompt "a hamster working in a police office, professional photography, photo realistic" \
--ckpt models/v1-5-pruned.ckpt --local_image_path /content/BayesDiff/laion_subset --laion_art_path /content/drive/MyDrive/laion-art \
--H 512 --W 512 --scale 1 --train_la_data_size 10 --train_la_batch_size 1 \
--sample_batch_size 1 --total_n_samples 1 --timesteps 50
# CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
#   --prompt "A futuristic city with flying cars, ultra realistic" \
#   --ckpt models/v1-5-pruned.ckpt \
#   --config models/v1-inference.yaml \
#   --local_image_path /content/BayesDiff/laion_subset \
#   --laion_art_path /content/drive/MyDrive/sd/laion-art.txt \
#   --H 256 --W 256 --scale 3 \
#   --train_la_data_size 50 --train_la_batch_size 2 \
#   --sample_batch_size 1 --total_n_samples 4 --timesteps 20 \
#   --precision autocast

# CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
#   --prompt "A futuristic city with flying cars, ultra realistic" \
#   --ckpt models/v1-5-pruned.ckpt \
#   --config models/v1-inference.yaml \
#   --local_image_path /content/BayesDiff/laion_subset \
#   --laion_art_path /content/drive/MyDrive/sd/laion-art.txt \
#   --H 256 --W 256 --scale 3 \
#   --train_la_data_size 50 --train_la_batch_size 2 \
#   --sample_batch_size 1 --total_n_samples 4 --timesteps 20 \
#   --precision autocast
