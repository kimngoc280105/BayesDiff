#!/bin/bash

# Cooperate UQ into ddim sampler
../bayesdiff_env/bin/python ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# Cooperate UQ into dpm-solver-2 sampler
../bayesdiff_env/bin/python dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
