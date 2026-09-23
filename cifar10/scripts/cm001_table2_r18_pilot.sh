#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/wyr/ARC-TopK-release"
TORCHRUN_BIN="${REPO_DIR}/.venv/bin/torchrun"
LOG_DIR="${REPO_DIR}/output/CM006-table2-r18-gpu2-5"
TARGET_GPUS="2,3,4,5"
SEED=1410

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}"

echo "[CM006] Waiting for CIFAR-10 extraction..."
while ! "${REPO_DIR}/.venv/bin/python" -c \
    "from torchvision.datasets import CIFAR10; CIFAR10(root='data', train=True, download=False); CIFAR10(root='data', train=False, download=False)" \
    >/dev/null 2>&1; do
    sleep 30
done
echo "[CM006] CIFAR-10 is ready."

wait_for_target_gpus() {
    echo "[CM006] Waiting for GPU ${TARGET_GPUS} with utilization < 30% and free memory >= 4 GiB..."
    while true; do
        READY_COUNT="$(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits \
            | awk -F', ' '($1 == 2 || $1 == 3 || $1 == 4 || $1 == 5) && $2 >= 4096 && $3 < 30 { count++ } END { print count + 0 }')"
        if [ "${READY_COUNT}" -eq 4 ]; then
            echo "[CM006] Selected CUDA_VISIBLE_DEVICES=${TARGET_GPUS}"
            return
        fi
        sleep 30
    done
}

COMMON_ARGS=(
    --per_device_train_batch_size 16
    --num_train_epochs 200
    --seed "${SEED}"
    --start_compress_iter 1000
    --compress_ratio 0.2
    --col_rank 4
    --use_wandb 1
)

wait_for_target_gpus
CUDA_VISIBLE_DEVICES="${TARGET_GPUS}" WANDB_MODE=online PYTHONPATH=. \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 --master-port=29618 \
    cifar10/run_cifar10.py \
    --optimizer adamw --lr 1e-3 --weight_decay 5e-4 \
    --compressor none --use_error_feedback noef \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/01-dense-adam.log"

wait_for_target_gpus
CUDA_VISIBLE_DEVICES="${TARGET_GPUS}" WANDB_MODE=online PYTHONPATH=. \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 --master-port=29619 \
    cifar10/run_cifar10.py \
    --optimizer muon --lr 0.02 --weight_decay 5e-4 \
    --muon_scalar_lr 1e-3 --muon_scalar_weight_decay 5e-4 \
    --muon_mu 0.95 --muon_epsilon 1e-8 --muon_scalar_eps 1e-8 \
    --muon_adjust_lr spectral_norm \
    --compressor none --use_error_feedback noef \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/02-dense-muon.log"

wait_for_target_gpus
CUDA_VISIBLE_DEVICES="${TARGET_GPUS}" WANDB_MODE=online PYTHONPATH=. \
    "${TORCHRUN_BIN}" --standalone --nproc_per_node=4 --master-port=29620 \
    cifar10/run_cifar10.py \
    --optimizer muon --lr 0.02 --weight_decay 5e-4 \
    --muon_scalar_lr 1e-3 --muon_scalar_weight_decay 5e-4 \
    --muon_mu 0.95 --muon_epsilon 1e-8 --muon_scalar_eps 1e-8 \
    --muon_adjust_lr spectral_norm \
    --compressor group_topk_no_reshape --use_error_feedback ef14 \
    "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG_DIR}/03-arctopk-ef14-muon.log"

echo "[CM006] All three runs finished. No checkpoints were requested."
