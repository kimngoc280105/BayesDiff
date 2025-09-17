#!/bin/bash
PYTHON=/content/bayesdiff_env/bin/python

# Cooperate UQ into ddim sampler
CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/models/v1-5-pruned.ckpt \
--config /content/BayesDiff/models/v1-inference.yaml \
--local_image_path /content/BayesDiff/laion_subset \
--laion_art_path /content/BayesDiff/laion_subset \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50 \
--outdir /content/BayesDiff/outputs

# Cooperate UQ into dpm-solver-2 sampler
CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/models/v1-5-pruned.ckpt \
--config /content/BayesDiff/models/v1-inference.yaml \
--local_image_path /content/BayesDiff/laion_subset \
--laion_art_path /content/BayesDiff/laion_subset \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50 \
--outdir /content/BayesDiff/outputs
