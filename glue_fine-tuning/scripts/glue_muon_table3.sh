#!/usr/bin/env bash

set -euo pipefail

# Reproduce the Table III RoBERTa-base/GLUE setting with the repository's Muon
# implementation.  By default this runs one seed and two arms:
#   1. Muon + dense gradients
#   2. Muon + ARC-TopK gradients
#
# This script only starts training when invoked normally.  Use DRY_RUN=1 to
# print all commands without launching them.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-FacebookAI/roberta-base}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
SEED="${SEED:-1240}"
MUON_MATRIX_LR="${MUON_MATRIX_LR:-0.02}"
MUON_SCALAR_LR="${MUON_SCALAR_LR:-0.001}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/glue_muon_table3_seed${SEED}}"
DRY_RUN="${DRY_RUN:-0}"
WITH_TRACKING="${WITH_TRACKING:-0}"
REPORT_TO="${REPORT_TO:-wandb}"

# The paper uses local/per-device batch sizes of 32 for CoLA/MRPC and 16 for
# the other tasks.  The Muon matrix and scalar learning rates follow the
# supplied Muon configuration image rather than the paper's Adam LR values.
TASKS=(cola sst2 mrpc stsb qqp mnli qnli rte)

if [[ "${WITH_TRACKING}" == "1" ]]; then
    TRACKING_ARGS=(--with_tracking --report_to "${REPORT_TO}")
else
    TRACKING_ARGS=()
fi

run_one() {
    local task="$1"
    local arm="$2"
    local local_batch_size="$3"
    local compressor="$4"
    local error_feedback="$5"
    local run_dir="${OUTPUT_ROOT}/${task}/${arm}"

    local -a command=(
        env PYTHONPATH=.
        accelerate launch --num_processes "${NUM_PROCESSES}"
        glue_fine-tuning/run_glue_no_trainer_new.py
        --model_name_or_path "${MODEL_NAME_OR_PATH}"
        --task_name "${task}"
        --max_length 512
        --learning_rate "${MUON_MATRIX_LR}"
        --muon_scalar_lr "${MUON_SCALAR_LR}"
        --weight_decay 0
        --muon_scalar_weight_decay 0
        --optimizer muon
        --muon_mu 0.95
        --muon_epsilon 1e-8
        --muon_scalar_beta1 0.9
        --muon_scalar_beta2 0.999
        --muon_scalar_eps 1e-8
        --muon_adjust_lr spectral_norm
        --compressor "${compressor}"
        --use_error_feedback "${error_feedback}"
        --start_compress_iter 1000
        --compress_ratio 0.2
        --r 4
        --per_device_train_batch_size "${local_batch_size}"
        --num_train_epochs 30
        --lr_scheduler_type linear
        --seed "${SEED}"
        --output_dir "${run_dir}"
        "${TRACKING_ARGS[@]}"
    )

    printf '\n[%s/%s] matrix_lr=%s scalar_lr=%s local_batch_size=%s\n' \
        "${task}" "${arm}" "${MUON_MATRIX_LR}" "${MUON_SCALAR_LR}" "${local_batch_size}"
    printf 'output_dir=%s\n' "${run_dir}"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    mkdir -p "${run_dir}"
    "${command[@]}" 2>&1 | tee "${run_dir}/train.log"
}

for task in "${TASKS[@]}"; do
    case "${task}" in
        cola|mrpc)
            local_batch_size=32
            ;;
        *)
            local_batch_size=16
            ;;
    esac

    # Dense Muon control: no gradient compressor or error feedback.
    run_one "${task}" muon_dense "${local_batch_size}" none noef

    # ARC-TopK + Muon: the paper's sparsity/rank/error-feedback configuration.
    run_one "${task}" muon_arctopk "${local_batch_size}" \
        group_topk_no_reshape ef21
done

printf '\nPrepared/runs completed under %s\n' "${OUTPUT_ROOT}"
