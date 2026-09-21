#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

run_id="CM001-muon-dense-llama60m-c4-1p1b-ws4-s1243"
output_dir="output/${run_id}"
mkdir -p "${output_dir}"

export CUDA_VISIBLE_DEVICES=2,3,6,7
export WANDB_MODE=online
export HF_HOME=/dev/shm/wyr_tmp/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH=.

.venv/bin/torchrun \
    --standalone \
    --nproc_per_node=4 \
    c4/run_llama_pretraining.py \
    --model_config c4/configs/llama_60m.json \
    --dataset_path /dev/shm/wyr_tmp/c4/en-30-shards \
    --max_length 256 \
    --dtype float32 \
    --num_training_steps 8393 \
    --warmup_steps 1000 \
    --scheduler cosine \
    --min_lr_ratio 0.1 \
    --total_batch_size 512 \
    --batch_size 32 \
    --gradient_accumulation 4 \
    --workers 4 \
    --seed 1243 \
    --optimizer muon \
    --lr 0.02 \
    --weight_decay 0.0 \
    --muon_mu 0.95 \
    --muon_epsilon 1e-8 \
    --muon_scalar_lr 0.001 \
    --muon_scalar_beta1 0.9 \
    --muon_scalar_beta2 0.999 \
    --muon_scalar_eps 1e-8 \
    --muon_scalar_weight_decay 0.0 \
    --muon_adjust_lr spectral_norm \
    --muon_compile \
    --compressor none \
    --grad_clipping 1.0 \
    --eval_tokens 10000000 \
    --wandb_project ARC-TopK-LLaMA-60M \
    --wandb_job_type formal-pretraining \
    --output_dir "${output_dir}" \
    2>&1 | tee "${output_dir}/train.log"
