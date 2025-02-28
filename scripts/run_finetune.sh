#!/usr/bin/env bash
# Example usage of the toxic_gpt2_finetune_and_distill.py script

# 1) Activate conda environment
conda activate unidetox  # name matches environment.yml

# 2) Run the Python script with arguments
python -m unidetox.toxic_gpt2_finetune_and_distill \
  --base_model_name gpt2-xl \
  --output_dir ./toxic_model \
  --epochs 3 \
  --lr 1e-5 \
  --batch_size 4 \
  --random_seed 123 \
  --alpha_list 0.1 \
  --beta_list inf
