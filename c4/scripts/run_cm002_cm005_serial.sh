#!/bin/bash

set -uo pipefail

cd "$(dirname "$0")/../.."

run_mode="${1:-all}"
if [ "${run_mode}" = "arc_rerun" ]; then
    suite_id="CM003-CM005-arc-rerun"
else
    suite_id="CM002-CM005-serial"
fi
suite_dir="output/${suite_id}"
status_file="${suite_dir}/status.tsv"
mkdir -p "${suite_dir}"

minimum_free_mib=14000
maximum_utilization=50

while true; do
    eligible_gpus=()
    while IFS=', ' read -r gpu_index free_mib utilization; do
        if [ "${free_mib}" -ge "${minimum_free_mib}" ] && [ "${utilization}" -le "${maximum_utilization}" ]; then
            eligible_gpus+=("${gpu_index}")
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits)

    if [ "${#eligible_gpus[@]}" -ge 4 ]; then
        selected_gpus="$(IFS=,; echo "${eligible_gpus[*]:0:4}")"
        break
    fi
    sleep 60
done

printf 'selected_gpus\t%s\n' "${selected_gpus}" > "${status_file}"
printf 'suite_started_at\t%s\n' "$(date --iso-8601=seconds)" >> "${status_file}"

export CUDA_VISIBLE_DEVICES="${selected_gpus}"
export WANDB_MODE=online
export HF_HOME=/dev/shm/wyr_tmp/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH=.

run_experiment() {
    run_id="$1"
    model_name="$2"
    batch_size="$3"
    gradient_accumulation="$4"
    num_training_steps="$5"
    optimizer_name="$6"
    compressor_name="$7"

    if [ "${model_name}" = "llama_60m" ]; then
        wandb_project="ARC-TopK-LLaMA-60M"
    else
        wandb_project="ARC-TopK-LLaMA-130M"
    fi

    output_dir="output/${run_id}"
    mkdir -p "${output_dir}"
    printf '%s\tstarted\t%s\n' "${run_id}" "$(date --iso-8601=seconds)" >> "${status_file}"

    command=(
        .venv/bin/torchrun
        --standalone
        --nproc_per_node=4
        --
        c4/run_llama_pretraining.py
        --model_config "c4/configs/${model_name}.json"
        --dataset_path /dev/shm/wyr_tmp/c4/en-30-shards
        --max_length 256
        --dtype float32
        --num_training_steps "${num_training_steps}"
        --warmup_steps 1000
        --scheduler cosine
        --min_lr_ratio 0.1
        --total_batch_size 512
        --batch_size "${batch_size}"
        --gradient_accumulation "${gradient_accumulation}"
        --workers 4
        --seed 1243
        --weight_decay 0.0
        --grad_clipping 1.0
        --eval_tokens 10000000
        --compressor "${compressor_name}"
        --wandb_project "${wandb_project}"
        --wandb_job_type formal-pretraining
        --output_dir "${output_dir}"
    )

    if [ "${optimizer_name}" = "adamw" ]; then
        command+=(
            --optimizer adamw
            --lr 0.002
            --beta1 0.9
            --beta2 0.999
            --eps 1e-8
        )
    else
        command+=(
            --optimizer muon
            --lr 0.02
            --muon_mu 0.95
            --muon_epsilon 1e-8
            --muon_scalar_lr 0.001
            --muon_scalar_beta1 0.9
            --muon_scalar_beta2 0.999
            --muon_scalar_eps 1e-8
            --muon_scalar_weight_decay 0.0
            --muon_adjust_lr spectral_norm
            --muon_compile
        )
    fi

    if [ "${compressor_name}" = "group_topk_no_reshape" ]; then
        command+=(
            --start_compress_iter 1000
            --use_error_feedback ef14
            --compress_ratio 0.2
            --r 4
        )
    fi

    "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    exit_code="${PIPESTATUS[0]}"
    printf '%s\tfinished\t%s\texit_code=%s\n' \
        "${run_id}" "$(date --iso-8601=seconds)" "${exit_code}" >> "${status_file}"
}

if [ "${run_mode}" = "all" ]; then
    run_experiment "CM002-dense-adam-llama60m-c4-1p1b-ws4-s1243" \
        "llama_60m" 32 4 8393 "adamw" "none"
fi

if [ "${run_mode}" = "arc_rerun" ]; then
    run_experiment "CM003-m002-arctopk-muon-llama60m-c4-1p1b-ws4-s1243-rerun1" \
        "llama_60m" 32 4 8393 "muon" "group_topk_no_reshape"
else
    run_experiment "CM003-m002-arctopk-muon-llama60m-c4-1p1b-ws4-s1243" \
        "llama_60m" 32 4 8393 "muon" "group_topk_no_reshape"
fi

if [ "${run_mode}" = "all" ]; then
    run_experiment "CM004-m001-dense-muon-llama130m-c4-2p2b-ws4-s1243" \
        "llama_130m" 16 8 16785 "muon" "none"
fi

if [ "${run_mode}" = "arc_rerun" ]; then
    run_experiment "CM005-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243-rerun1" \
        "llama_130m" 16 8 16785 "muon" "group_topk_no_reshape"
else
    run_experiment "CM005-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243" \
        "llama_130m" 16 8 16785 "muon" "group_topk_no_reshape"
fi

printf 'suite_finished_at\t%s\n' "$(date --iso-8601=seconds)" >> "${status_file}"
