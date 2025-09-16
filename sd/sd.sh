#!/bin/bash
PYTHON=../bayesdiff_env/bin/python

# Kiểm tra môi trường Python có tồn tại
if [ ! -f "$PYTHON" ]; then
  echo "❌ Không tìm thấy Python tại $PYTHON"
  exit 1
fi

# Kiểm tra PyTorch đã cài chưa
$PYTHON -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
  echo "⚠️ PyTorch chưa được cài trong môi trường $PYTHON"
  echo "👉 Chạy lệnh sau để cài: $PYTHON -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118"
  exit 1
fi

# Cooperate UQ into ddim sampler
CUDA_VISIBLE_DEVICES=0 $PYTHON ddim_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50

# Cooperate UQ into dpm-solver-2 sampler
CUDA_VISIBLE_DEVICES=0 $PYTHON dpmsolver_skipUQ.py \
--prompt "A futuristic city with flying cars, ultra realistic" \
--ckpt /content/BayesDiff/sd/your_local_model_path/sd-v1-5.ckpt \
--local_image_path /content/BayesDiff/sd/your_local_image_path \
--laion_art_path /content/BayesDiff/sd/your_laion_art_path \
--H 512 --W 512 --scale 3 \
--train_la_data_size 1000 --train_la_batch_size 10 \
--sample_batch_size 2 --total_n_samples 48 --timesteps 50
