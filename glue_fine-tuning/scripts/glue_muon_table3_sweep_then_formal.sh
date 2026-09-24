#!/usr/bin/env bash

set -euo pipefail

# One invocation performs the complete workflow:
#   1. Sweep three paired Muon learning rates on SST-2 for five epochs.
#   2. Select the pair with the best dense validation accuracy.
#   3. Reuse that pair for the formal eight-task, two-arm experiment for ten epochs.
#
# Use DRY_RUN=1 to print three sweep commands and sixteen formal commands without
# launching training. Normal execution waits for the requested physical GPUs
# to be below the configured memory/utilization threshold before starting.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

GPU_IDS="${GPU_IDS:-2,3,4,5}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "${GPU_IDS}" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s does not match GPU_IDS=%s\n' \
        "${CUDA_VISIBLE_DEVICES}" "${GPU_IDS}" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
IFS=',' read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-FacebookAI/roberta-base}"
NUM_PROCESSES="${NUM_PROCESSES:-${#GPU_ID_ARRAY[@]}}"
SEED="${SEED:-1240}"
SWEEP_TASK="${SWEEP_TASK:-sst2}"

SWEEP_MATRIX_LRS=(0.02 0.002 0.0002)
SWEEP_SCALAR_LRS=(0.001 0.0001 0.00005)

# SST-2 paper-global batch is 16 * 4 = 64. The sweep uses 8 * 4 * 2 = 64
# to reduce peak memory while preserving the global batch. The sweep is a
# short screening run; formal training retains the longer epoch count.
SWEEP_TRAIN_BATCH_SIZE="${SWEEP_TRAIN_BATCH_SIZE:-8}"
SWEEP_EVAL_BATCH_SIZE="${SWEEP_EVAL_BATCH_SIZE:-8}"
SWEEP_GRADIENT_ACCUMULATION_STEPS="${SWEEP_GRADIENT_ACCUMULATION_STEPS:-2}"
SWEEP_EPOCHS="${SWEEP_EPOCHS:-5}"

FORMAL_EVAL_BATCH_SIZE="${FORMAL_EVAL_BATCH_SIZE:-8}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-10}"
FORMAL_BATCH_DIVISOR="${FORMAL_BATCH_DIVISOR:-1}"
FORMAL_GRADIENT_ACCUMULATION_STEPS="${FORMAL_GRADIENT_ACCUMULATION_STEPS:-${FORMAL_BATCH_DIVISOR}}"
COMPRESSION_WARMUP_FRACTION="${COMPRESSION_WARMUP_FRACTION:-0.1}"

if ! [[ "${FORMAL_BATCH_DIVISOR}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'FORMAL_BATCH_DIVISOR must be a positive integer\n' >&2
    exit 2
fi
if ! [[ "${FORMAL_GRADIENT_ACCUMULATION_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'FORMAL_GRADIENT_ACCUMULATION_STEPS must be a positive integer\n' >&2
    exit 2
fi

SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-${REPO_DIR}/outputs/glue_muon_table3_sweep_${SWEEP_TASK}_seed${SEED}}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-${REPO_DIR}/outputs/glue_muon_table3_formal_seed${SEED}}"
SELECTION_FILE="${SELECTION_FILE:-${SWEEP_OUTPUT_ROOT}/selected_config.json}"
SWEEP_MANIFEST="${SWEEP_MANIFEST:-${SWEEP_OUTPUT_ROOT}/sweep_results.tsv}"

DRY_RUN="${DRY_RUN:-0}"
SKIP_SWEEP="${SKIP_SWEEP:-0}"
SWEEP_WITH_TRACKING="${SWEEP_WITH_TRACKING:-0}"
FORMAL_WITH_TRACKING="${FORMAL_WITH_TRACKING:-1}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_IDLE_THRESHOLD_PERCENT="${GPU_IDLE_THRESHOLD_PERCENT:-20}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-30}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-21600}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${REPO_DIR}/.venv/bin/accelerate}"

TASKS=(cola sst2 mrpc stsb qqp mnli qnli rte)

# For a continuation run, optionally start the formal sequence at a later
# task while preserving the canonical task order and all other settings.
FORMAL_START_TASK="${FORMAL_START_TASK:-}"
if [[ -n "${FORMAL_START_TASK}" ]]; then
    formal_start_found=0
    formal_tasks=()
    for task in "${TASKS[@]}"; do
        if [[ "${task}" == "${FORMAL_START_TASK}" ]]; then
            formal_start_found=1
        fi
        if [[ "${formal_start_found}" == "1" ]]; then
            formal_tasks+=("${task}")
        fi
    done
    if [[ "${formal_start_found}" != "1" ]]; then
        printf 'FORMAL_START_TASK=%s is not in the formal task list\n' \
            "${FORMAL_START_TASK}" >&2
        exit 2
    fi
    TASKS=("${formal_tasks[@]}")
fi

lr_tag() {
    printf '%s' "$1" | tr '.' 'p' | tr '-' 'm'
}

wait_for_gpus() {
    if [[ "${DRY_RUN}" == "1" || "${WAIT_FOR_GPUS}" == "0" ]]; then
        return 0
    fi
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf 'nvidia-smi is required for GPU waiting\n' >&2
        return 1
    }

    local deadline=$((SECONDS + GPU_WAIT_TIMEOUT_SECONDS))
    while (( SECONDS < deadline )); do
        local stats
        stats="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
        local all_idle=1
        local status_line=""
        local gpu used total util mem_pct idle

        for gpu in "${GPU_ID_ARRAY[@]}"; do
            read -r used total util < <(
                awk -F',' -v target="${gpu}" '$1 + 0 == target {
                    gsub(/[[:space:]]/, "", $2); gsub(/[[:space:]]/, "", $3); gsub(/[[:space:]]/, "", $4);
                    print $2, $3, $4; exit
                }' <<< "${stats}"
            )
            if [[ -z "${used:-}" || -z "${total:-}" || -z "${util:-}" ]]; then
                printf 'Could not read GPU %s status\n' "${gpu}" >&2
                return 1
            fi
            mem_pct="$(awk -v used="${used}" -v total="${total}" 'BEGIN { printf "%.2f", 100 * used / total }')"
            idle="$(awk -v mem="${mem_pct}" -v util="${util}" -v threshold="${GPU_IDLE_THRESHOLD_PERCENT}" \
                'BEGIN { print (mem < threshold && util < threshold) ? 1 : 0 }')"
            status_line+=" GPU${gpu}:mem=${mem_pct}%/util=${util}%"
            if [[ "${idle}" != "1" ]]; then
                all_idle=0
            fi
        done

        printf '[gpu-wait] threshold=%s%%%s\n' "${GPU_IDLE_THRESHOLD_PERCENT}" "${status_line}"
        if [[ "${all_idle}" == "1" ]]; then
            printf '[gpu-wait] all requested GPUs are below threshold\n'
            return 0
        fi
        sleep "${GPU_WAIT_INTERVAL_SECONDS}"
    done

    printf 'Timed out waiting for GPUs %s to fall below %s%%\n' \
        "${GPU_IDS}" "${GPU_IDLE_THRESHOLD_PERCENT}" >&2
    return 1
}

run_one() {
    local task="$1"
    local run_name="$2"
    local matrix_lr="$3"
    local scalar_lr="$4"
    local local_batch_size="$5"
    local eval_batch_size="$6"
    local gradient_accumulation_steps="$7"
    local compressor="$8"
    local error_feedback="$9"
    local start_compress_iter="${10}"
    local num_epochs="${11}"
    local output_root="${12}"
    local with_tracking="${13}"
    local run_dir="${output_root}/${task}/${run_name}"
    local -a tracking_args=()
    if [[ "${with_tracking}" == "1" ]]; then
        tracking_args=(--with_tracking --report_to "${REPORT_TO}")
    fi

    local -a command=(
        env PYTHONPATH=. WANDB_MODE="${WANDB_MODE}"
        "${ACCELERATE_BIN}" launch
        --num_processes "${NUM_PROCESSES}"
        --num_machines 1
        --mixed_precision no
        glue_fine-tuning/run_glue_no_trainer_new.py
        --model_name_or_path "${MODEL_NAME_OR_PATH}"
        --task_name "${task}"
        --max_length 512
        --learning_rate "${matrix_lr}"
        --muon_scalar_lr "${scalar_lr}"
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
        --start_compress_iter "${start_compress_iter}"
        --compression_warmup_fraction "${COMPRESSION_WARMUP_FRACTION}"
        --compression_warmup_reference_epochs "${FORMAL_EPOCHS}"
        --compress_ratio 0.2
        --r 4
        --per_device_train_batch_size "${local_batch_size}"
        --per_device_eval_batch_size "${eval_batch_size}"
        --gradient_accumulation_steps "${gradient_accumulation_steps}"
        --num_train_epochs "${num_epochs}"
        --lr_scheduler_type linear
        --seed "${SEED}"
        --output_dir "${run_dir}"
        "${tracking_args[@]}"
    )

    printf '\n[%s/%s] matrix_lr=%s scalar_lr=%s train_bs=%s eval_bs=%s accum=%s\n' \
        "${task}" "${run_name}" "${matrix_lr}" "${scalar_lr}" \
        "${local_batch_size}" "${eval_batch_size}" "${gradient_accumulation_steps}"
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

read_accuracy() {
    local result_file="$1"
    [[ -f "${result_file}" ]] || {
        printf 'Missing result file: %s\n' "${result_file}" >&2
        return 1
    }
    "${PYTHON_BIN}" - "${result_file}" <<'PY'
import json
import math
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
value = result.get("eval_accuracy")
if not isinstance(value, (int, float)) or not math.isfinite(value):
    raise SystemExit("eval_accuracy is missing or non-finite")
print(f"{value:.10f}")
PY
}

prefetch_formal_datasets() {
    printf '\nPrefetching GLUE datasets for the formal runs.\n'
    "${PYTHON_BIN}" - "${TASKS[@]}" <<'PY'
import sys
from datasets import load_dataset

for task in sys.argv[1:]:
    dataset = load_dataset("glue", task)
    print(f"cached {task}: train={len(dataset['train'])}", flush=True)
PY
}

run_sweep() {
    mkdir -p "${SWEEP_OUTPUT_ROOT}"
    : > "${SWEEP_MANIFEST}"

    for i in "${!SWEEP_MATRIX_LRS[@]}"; do
        local matrix_lr="${SWEEP_MATRIX_LRS[$i]}"
        local scalar_lr="${SWEEP_SCALAR_LRS[$i]}"
        local tag="m$(lr_tag "${matrix_lr}")_s$(lr_tag "${scalar_lr}")"
        local dense_score

        run_one "${SWEEP_TASK}" "sweep_${tag}_muon_dense" \
            "${matrix_lr}" "${scalar_lr}" "${SWEEP_TRAIN_BATCH_SIZE}" \
            "${SWEEP_EVAL_BATCH_SIZE}" "${SWEEP_GRADIENT_ACCUMULATION_STEPS}" \
            none noef 0 "${SWEEP_EPOCHS}" "${SWEEP_OUTPUT_ROOT}" \
            "${SWEEP_WITH_TRACKING}"
        dense_score="$(read_accuracy "${SWEEP_OUTPUT_ROOT}/${SWEEP_TASK}/sweep_${tag}_muon_dense/all_results.json")"

        printf '%s\t%s\t%s\n' "${matrix_lr}" "${scalar_lr}" "${dense_score}" \
            | tee -a "${SWEEP_MANIFEST}"
    done
}

select_best_pair() {
    "${PYTHON_BIN}" - "${SWEEP_MANIFEST}" "${SELECTION_FILE}" "${SWEEP_TASK}" <<'PY'
import json
import sys

manifest, selection_file, task = sys.argv[1:]
rows = []
with open(manifest, encoding="utf-8") as handle:
    for line in handle:
        matrix_lr, scalar_lr, dense = line.rstrip("\n").split("\t")
        dense = float(dense)
        rows.append({
            "matrix_lr": matrix_lr,
            "scalar_lr": scalar_lr,
            "dense_accuracy": dense,
        })
if len(rows) != 3:
    raise SystemExit(f"expected three sweep rows, got {len(rows)}")
best = max(rows, key=lambda row: row["dense_accuracy"])
payload = {"task": task, "selection_metric": "dense_accuracy", "selected": best, "all": rows}
with open(selection_file, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2)
    handle.write("\n")
print(json.dumps(best, sort_keys=True))
PY
}

run_formal() {
    local matrix_lr="$1"
    local scalar_lr="$2"
    for task in "${TASKS[@]}"; do
        local local_batch_size
        case "${task}" in
            cola|mrpc) local base_batch_size=32 ;;
            *) local base_batch_size=16 ;;
        esac
        if (( base_batch_size % FORMAL_BATCH_DIVISOR != 0 )); then
            printf 'FORMAL_BATCH_DIVISOR=%s does not divide base batch size %s for %s\n' \
                "${FORMAL_BATCH_DIVISOR}" "${base_batch_size}" "${task}" >&2
            exit 2
        fi
        local_batch_size=$((base_batch_size / FORMAL_BATCH_DIVISOR))

        run_one "${task}" muon_dense "${matrix_lr}" "${scalar_lr}" \
            "${local_batch_size}" "${FORMAL_EVAL_BATCH_SIZE}" "${FORMAL_GRADIENT_ACCUMULATION_STEPS}" \
            none noef 0 "${FORMAL_EPOCHS}" "${FORMAL_OUTPUT_ROOT}" \
            "${FORMAL_WITH_TRACKING}"
        run_one "${task}" muon_arctopk "${matrix_lr}" "${scalar_lr}" \
            "${local_batch_size}" "${FORMAL_EVAL_BATCH_SIZE}" "${FORMAL_GRADIENT_ACCUMULATION_STEPS}" \
            group_topk_no_reshape ef21 0 "${FORMAL_EPOCHS}" "${FORMAL_OUTPUT_ROOT}" \
            "${FORMAL_WITH_TRACKING}"
    done
}

if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] sweep commands:\n'
    for i in "${!SWEEP_MATRIX_LRS[@]}"; do
        matrix_lr="${SWEEP_MATRIX_LRS[$i]}"
        scalar_lr="${SWEEP_SCALAR_LRS[$i]}"
        tag="m$(lr_tag "${matrix_lr}")_s$(lr_tag "${scalar_lr}")"
        run_one "${SWEEP_TASK}" "sweep_${tag}_muon_dense" \
            "${matrix_lr}" "${scalar_lr}" "${SWEEP_TRAIN_BATCH_SIZE}" \
            "${SWEEP_EVAL_BATCH_SIZE}" "${SWEEP_GRADIENT_ACCUMULATION_STEPS}" \
            none noef 0 "${SWEEP_EPOCHS}" "${SWEEP_OUTPUT_ROOT}" \
            "${SWEEP_WITH_TRACKING}"
    done
    printf '[dry-run] formal commands use the pair selected at runtime; first pair is shown as placeholder.\n'
    run_formal "${SWEEP_MATRIX_LRS[0]}" "${SWEEP_SCALAR_LRS[0]}"
    exit 0
fi

wait_for_gpus
if [[ "${SKIP_SWEEP}" == "1" ]]; then
    [[ -f "${SELECTION_FILE}" ]] || {
        printf 'SKIP_SWEEP=1 requires an existing selection file: %s\n' "${SELECTION_FILE}" >&2
        exit 2
    }
    printf '\nReusing completed sweep selection: %s\n' "${SELECTION_FILE}"
else
    printf '\nStarting integrated SST-2 sweep.\n'
    run_sweep

    printf '\nSelecting the best LR pair from the completed sweep.\n'
    select_output="$(select_best_pair)"
    printf 'selected=%s\n' "${select_output}"
fi
selected_matrix_lr="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["matrix_lr"])' "${SELECTION_FILE}")"
selected_scalar_lr="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["scalar_lr"])' "${SELECTION_FILE}")"

printf '\nStarting formal Table III runs with matrix_lr=%s scalar_lr=%s.\n' \
    "${selected_matrix_lr}" "${selected_scalar_lr}"
prefetch_formal_datasets
run_formal "${selected_matrix_lr}" "${selected_scalar_lr}"

printf '\nIntegrated sweep and formal runs completed.\nSweep outputs: %s\nFormal outputs: %s\n' \
    "${SWEEP_OUTPUT_ROOT}" "${FORMAL_OUTPUT_ROOT}"
