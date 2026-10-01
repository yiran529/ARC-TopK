#!/usr/bin/env bash

set -euo pipefail

# Standalone GLUE launcher derived from glue_muon_table3_sweep_then_formal.sh.
# This file intentionally contains no learning-rate sweep and never modifies
# the original sweep-then-formal launcher.
#
# Fixed settings inherited from the completed SST-2 Muon sweep:
#   matrix LR = 0.0002, scalar LR = 0.00005
#
# Per task, run two compressed arms:
#   1. muon_topk  : tensor Top-K + EF21
#   2. muon_randk : tensor Rand-K + EF21
#
# DRY_RUN=1 prints the eight pending commands (4 tasks x 2 arms) without prefetching,
# waiting, or launching training.

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

GPU_IDS="${GPU_IDS:-0,1,2,7}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-FacebookAI/roberta-base}"
SEED="${SEED:-1240}"
MUON_MATRIX_LR="${MUON_MATRIX_LR:-0.0002}"
MUON_SCALAR_LR="${MUON_SCALAR_LR:-0.00005}"
FORMAL_EPOCHS="${FORMAL_EPOCHS:-10}"
FORMAL_EVAL_BATCH_SIZE="${FORMAL_EVAL_BATCH_SIZE:-8}"
FORMAL_OUTPUT_ROOT="${FORMAL_OUTPUT_ROOT:-$REPO_DIR/outputs/CM009-m003-glue-topk-randk-ga1-seed$SEED-gpu0127}"

# Resume only the unfinished task arms. The completed CoLA, SST-2, MRPC, and
# STS-B results remain in the original output root.
COMPRESSION_WARMUP_FRACTION="${COMPRESSION_WARMUP_FRACTION:-0.1}"
COMPRESS_RATIO="${COMPRESS_RATIO:-0.2}"
SPARSE_TYPE="${SPARSE_TYPE:-tensor}"

DRY_RUN="${DRY_RUN:-0}"
FORMAL_WITH_TRACKING="${FORMAL_WITH_TRACKING:-1}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_MODE

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_IDLE_THRESHOLD_PERCENT="${GPU_IDLE_THRESHOLD_PERCENT:-20}"
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-0}"
REQUIRED_IDLE_POLLS="${REQUIRED_IDLE_POLLS:-2}"
GPU_WAIT_INTERVAL_SECONDS="${GPU_WAIT_INTERVAL_SECONDS:-300}"
GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-21600}"

PYTHON_BIN="${PYTHON_BIN:-$REPO_DIR/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$REPO_DIR/.venv/bin/accelerate}"

TASKS=(qqp mnli qnli rte)
FORMAL_START_TASK="${FORMAL_START_TASK:-qqp}"
start_found=0
remaining_tasks=()
for task in "${TASKS[@]}"; do
    if [[ "$task" == "$FORMAL_START_TASK" ]]; then
        start_found=1
    fi
    if (( start_found )); then
        remaining_tasks+=("$task")
    fi
done
if (( ! start_found )); then
    printf 'FORMAL_START_TASK=%s is not in the pending task list\n' "$FORMAL_START_TASK" >&2
    exit 2
fi
TASKS=("${remaining_tasks[@]}")

if ! [[ "$REQUIRED_IDLE_POLLS" =~ ^[1-9][0-9]*$ ]]; then
    printf 'REQUIRED_IDLE_POLLS must be a positive integer\n' >&2
    exit 2
fi
if ! [[ "$GPU_MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    printf 'GPU_MIN_FREE_MIB must be a nonnegative integer\n' >&2
    exit 2
fi
if [[ "$NUM_PROCESSES" -ne 4 ]]; then
    printf 'This launcher expects four GPUs; NUM_PROCESSES=%s\n' "$NUM_PROCESSES" >&2
    exit 2
fi

IFS=',' read -r GPU0 GPU1 GPU2 GPU7 <<< "$GPU_IDS"
if [[ "$GPU0,$GPU1,$GPU2,$GPU7" != "$GPU_IDS" ]]; then
    printf 'GPU_IDS must contain four comma-separated IDs; got %s\n' "$GPU_IDS" >&2
    exit 2
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != "$GPU_IDS" ]]; then
    printf 'CUDA_VISIBLE_DEVICES=%s does not match GPU_IDS=%s\n' "$CUDA_VISIBLE_DEVICES" "$GPU_IDS" >&2
    exit 2
fi
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

wait_for_gpus() {
    if [[ "$DRY_RUN" == "1" || "$WAIT_FOR_GPUS" == "0" ]]; then
        return 0
    fi
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf 'nvidia-smi is required for GPU waiting\n' >&2
        return 1
    }

    local deadline=$((SECONDS + GPU_WAIT_TIMEOUT_SECONDS))
    local idle_polls=0
    while (( SECONDS < deadline )); do
        local stats
        stats="$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
        local all_idle=1
        local status_line=""
        local gpu used total util mem_pct free_mib

        for gpu in "$GPU0" "$GPU1" "$GPU2" "$GPU7"; do
            read -r used total util < <(
                awk -F',' -v target="$gpu" '$1 + 0 == target {
                    gsub(/[[:space:]]/, "", $2)
                    gsub(/[[:space:]]/, "", $3)
                    gsub(/[[:space:]]/, "", $4)
                    print $2, $3, $4
                    exit
                }' <<< "$stats"
            )
            if [[ -z "${used:-}" || -z "${total:-}" || -z "${util:-}" ]]; then
                printf 'Could not read GPU %s status\n' "$gpu" >&2
                return 1
            fi
            mem_pct="$(awk -v used="$used" -v total="$total" 'BEGIN { printf "%.2f", 100 * used / total }')"
            free_mib=$((total - used))
            status_line+=" GPU$gpu:free=${free_mib}MiB/mem=$mem_pct%/util=$util%"
            if (( GPU_MIN_FREE_MIB > 0 )); then
                if (( free_mib < GPU_MIN_FREE_MIB )); then
                    all_idle=0
                fi
            else
                if ! awk -v mem="$mem_pct" -v util="$util" -v threshold="$GPU_IDLE_THRESHOLD_PERCENT" 'BEGIN { exit !(mem < threshold && util < threshold) }'; then
                    all_idle=0
                fi
            fi
        done

        if (( all_idle )); then
            idle_polls=$((idle_polls + 1))
        else
            idle_polls=0
        fi
        printf '[gpu-wait] threshold=%s%% min_free=%sMiB%s; consecutive eligible polls=%s/%s\n' "$GPU_IDLE_THRESHOLD_PERCENT" "$GPU_MIN_FREE_MIB" "$status_line" "$idle_polls" "$REQUIRED_IDLE_POLLS"
        if (( idle_polls >= REQUIRED_IDLE_POLLS )); then
            return 0
        fi
        sleep "$GPU_WAIT_INTERVAL_SECONDS"
    done

    printf 'Timed out waiting for GPUs %s to become idle\n' "$GPU_IDS" >&2
    return 1
}

run_one() {
    RUN_ONE_TRAIN_LAUNCHED=0
    local task="$1"
    local run_name="$2"
    local compressor="$3"
    local error_feedback="$4"
    local local_batch_size="$5"
    local gradient_accumulation_steps="$6"
    local run_dir="$FORMAL_OUTPUT_ROOT/$task/$run_name"
    local result_file="$run_dir/all_results.json"
    local -a tracking_args=()

    if [[ "$FORMAL_WITH_TRACKING" == "1" ]]; then
        tracking_args=(--with_tracking --report_to "$REPORT_TO")
    fi

    if [[ -f "$result_file" ]]; then
        "$PYTHON_BIN" -c 'import json,sys; json.load(open(sys.argv[1], encoding="utf-8"))' "$result_file"
        printf '[skip] valid completed result exists: %s\n' "$result_file"
        return 0
    fi
    if [[ -d "$run_dir" ]] && [[ -n "$(find "$run_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Refusing to reuse non-empty incomplete directory: %s\n' "$run_dir" >&2
        return 2
    fi

    local -a command=(
        env PYTHONPATH=. WANDB_MODE="$WANDB_MODE"
        "$ACCELERATE_BIN" launch
        --num_processes "$NUM_PROCESSES"
        --num_machines 1
        --mixed_precision no
        glue_fine-tuning/run_glue_no_trainer_new.py
        --model_name_or_path "$MODEL_NAME_OR_PATH"
        --task_name "$task"
        --max_length 512
        --learning_rate "$MUON_MATRIX_LR"
        --muon_scalar_lr "$MUON_SCALAR_LR"
        --weight_decay 0
        --muon_scalar_weight_decay 0
        --optimizer muon
        --muon_mu 0.95
        --muon_epsilon 1e-8
        --muon_scalar_beta1 0.9
        --muon_scalar_beta2 0.999
        --muon_scalar_eps 1e-8
        --muon_adjust_lr spectral_norm
        --compressor "$compressor"
        --use_error_feedback "$error_feedback"
        --start_compress_iter 0
        --compression_warmup_fraction "$COMPRESSION_WARMUP_FRACTION"
        --compression_warmup_reference_epochs "$FORMAL_EPOCHS"
        --disable_compression_warmup
        --compress_ratio "$COMPRESS_RATIO"
        --sparse_type "$SPARSE_TYPE"
        --r 4
        --per_device_train_batch_size "$local_batch_size"
        --per_device_eval_batch_size "$FORMAL_EVAL_BATCH_SIZE"
        --gradient_accumulation_steps "$gradient_accumulation_steps"
        --num_train_epochs "$FORMAL_EPOCHS"
        --lr_scheduler_type linear
        --warmup_fraction 0.0
        --seed "$SEED"
        --output_dir "$run_dir"
        "${tracking_args[@]}"
    )

    printf '\n[%s/%s] matrix_lr=%s scalar_lr=%s train_bs=%s eval_bs=%s GA=%s compressor=%s EF=%s\n' "$task" "$run_name" "$MUON_MATRIX_LR" "$MUON_SCALAR_LR" "$local_batch_size" "$FORMAL_EVAL_BATCH_SIZE" "$gradient_accumulation_steps" "$compressor" "$error_feedback"
    printf 'output_dir=%s\n' "$run_dir"
    printf 'command='
    printf '%q ' "${command[@]}"
    printf '\n'

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi

    wait_for_gpus || return $?
    mkdir -p "$run_dir"
    RUN_ONE_TRAIN_LAUNCHED=1
    "${command[@]}" 2>&1 | tee "$run_dir/train.log"
}

run_one_or_continue_on_oom() {
    local task="$1"
    local run_name="$2"
    local run_log="$FORMAL_OUTPUT_ROOT/$task/$run_name/train.log"
    local status

    if run_one "$@"; then
        return 0
    else
        status=$?
    fi

    if [[ "$RUN_ONE_TRAIN_LAUNCHED" == "1" && -f "$run_log" ]] && grep -Eq 'torch\.OutOfMemoryError:|CUDA out of memory' "$run_log"; then
        printf '%s\t%s\t%s\n' "$task" "$run_name" "$run_log" | tee -a "$FORMAL_OUTPUT_ROOT/oom_skipped.tsv"
        printf '[oom-skip] %s/%s failed with CUDA OOM (exit %s); continuing\n' "$task" "$run_name" "$status" >&2
        return 0
    fi
    return "$status"
}

run_task() {
    local task="$1"
    local base_batch_size
    local gradient_accumulation_steps
    base_batch_size=16
    gradient_accumulation_steps=1
    local local_batch_size="$base_batch_size"

    run_one_or_continue_on_oom "$task" muon_topk topk_sync ef21 "$local_batch_size" "$gradient_accumulation_steps"
    run_one_or_continue_on_oom "$task" muon_randk randk_sync ef21 "$local_batch_size" "$gradient_accumulation_steps"
}

if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run] no dataset access, GPU wait, or training\n'
else
    printf '[data] using the existing local GLUE cache\n'
fi

for task in "${TASKS[@]}"; do
    run_task "$task"
done

printf '\nRand-K/Top-K Muon scheduled runs ended; check oom_skipped.tsv for failed arms.\nOutputs: %s\n' "$FORMAL_OUTPUT_ROOT"
