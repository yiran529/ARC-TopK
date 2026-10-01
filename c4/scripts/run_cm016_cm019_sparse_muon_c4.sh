#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

suite_id="CM016-CM019-sparse-muon-c4-serial"
suite_dir="output/${suite_id}"
data_dir="${suite_dir}/c4-data"
status_file="${suite_dir}/status.tsv"
source_en_dir="/home/wyr/greedy_lore/c4/c4_en/en"

if [[ -e "${suite_dir}" ]]; then
    echo "Suite directory already exists: ${suite_dir}" >&2
    exit 1
fi
mkdir -p "${data_dir}/en"

# Use the same 30 training and 8 validation shard numbers as the ARC runs.
for index in $(seq -w 0 29); do
    shard="c4-train.000${index}-of-01024.json.gz"
    test -f "${source_en_dir}/${shard}"
    ln -s "${source_en_dir}/${shard}" "${data_dir}/en/${shard}"
done
for index in $(seq -w 0 7); do
    shard="c4-validation.0000${index}-of-00008.json.gz"
    test -f "${source_en_dir}/${shard}"
    ln -s "${source_en_dir}/${shard}" "${data_dir}/en/${shard}"
done

export WANDB_MODE=online
export HF_HOME=/dev/shm/wyr_tmp/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH=.

printf 'suite\tstarted\t%s\n' "$(date --iso-8601=seconds)" > "${status_file}"

wait_for_idle_gpus() {
    local run_id="$1" index uuid free_mib utilization
    local -a selected active_uuids
    printf '%s\twaiting_for_4_idle_gpus\t%s\n' \
        "${run_id}" "$(date --iso-8601=seconds)" >> "${status_file}"
    while true; do
        selected=()
        mapfile -t active_uuids < <(
            nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | sort -u
        )
        while IFS=',' read -r index uuid free_mib utilization; do
            index="${index//[[:space:]]/}"
            uuid="${uuid//[[:space:]]/}"
            free_mib="${free_mib//[[:space:]]/}"
            utilization="${utilization//[[:space:]]/}"
            [[ "${free_mib}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || continue
            if (( free_mib >= 23000 && utilization <= 5 )) &&
               [[ ! " ${active_uuids[*]} " =~ " ${uuid} " ]]; then
                selected+=("${index}")
            fi
        done < <(nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits)
        if (( ${#selected[@]} >= 4 )); then
            CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${selected[*]:0:4}")"
            export CUDA_VISIBLE_DEVICES
            printf '%s\tselected_gpus=%s\t%s\n' \
                "${run_id}" "${CUDA_VISIBLE_DEVICES}" "$(date --iso-8601=seconds)" >> "${status_file}"
            return
        fi
        sleep 60
    done
}

run_experiment() {
    local run_id="$1" model_name="$2" batch_size="$3" ga="$4"
    local steps="$5" matrix_lr="$6" compressor="$7" wandb_project="$8"
    local output_dir="output/$1" exit_code
    if [[ -e "${output_dir}" ]]; then
        echo "Run directory already exists: ${output_dir}" >&2
        return 1
    fi
    wait_for_idle_gpus "${run_id}"
    mkdir -p "${output_dir}"
    local -a command=(
        .venv/bin/torchrun --standalone --nproc_per_node=4 --
        c4/run_llama_pretraining.py
        --model_config "c4/configs/${model_name}.json"
        --dataset_path "${data_dir}"
        --max_length 256
        --dtype float32
        --num_training_steps "${steps}"
        --warmup_steps 1000
        --scheduler cosine
        --min_lr_ratio 0.1
        --total_batch_size 512
        --batch_size "${batch_size}"
        --gradient_accumulation "${ga}"
        --workers 4
        --seed 1243
        --optimizer muon
        --lr "${matrix_lr}"
        --weight_decay 0.0
        --muon_mu 0.95
        --muon_epsilon 1e-8
        --muon_scalar_lr 0.001
        --muon_scalar_beta1 0.9
        --muon_scalar_beta2 0.999
        --muon_scalar_eps 1e-8
        --muon_scalar_weight_decay 0.0
        --muon_adjust_lr spectral_norm
        --muon_compile
        --compressor "${compressor}"
        --start_compress_iter 1000
        --use_error_feedback ef14
        --compress_ratio 0.2
        --sparse_type tensor
        --disable_compression_warmup
        --grad_clipping 1.0
        --eval_tokens 10000000
        --save_every 0
        --wandb_project "${wandb_project}"
        --wandb_job_type formal-pretraining
        --output_dir "${output_dir}"
    )
    printf '%s\tstarted\t%s\n' "${run_id}" "$(date --iso-8601=seconds)" >> "${status_file}"
    set +e
    "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    exit_code="${PIPESTATUS[0]}"
    set -e
    printf '%s\tfinished\t%s\texit_code=%s\n' \
        "${run_id}" "$(date --iso-8601=seconds)" "${exit_code}" >> "${status_file}"
    return "${exit_code}"
}

failures=0
run_experiment "CM016-m003-topk-muon-llama60m-c4-1p1b-ws4-s1243-lr002" \
    llama_60m 32 4 8393 0.02 topk_sync ARC-TopK-LLaMA-60M || failures=$((failures + 1))
run_experiment "CM017-m003-randk-muon-llama60m-c4-1p1b-ws4-s1243-lr002" \
    llama_60m 32 4 8393 0.02 randk_sync ARC-TopK-LLaMA-60M || failures=$((failures + 1))
run_experiment "CM018-m003-topk-muon-llama130m-c4-2p62b-ws4-s1243-lr001" \
    llama_130m 16 8 20000 0.01 topk_sync ARC-TopK-LLaMA-130M || failures=$((failures + 1))
run_experiment "CM019-m003-randk-muon-llama130m-c4-2p62b-ws4-s1243-lr001" \
    llama_130m 16 8 20000 0.01 randk_sync ARC-TopK-LLaMA-130M || failures=$((failures + 1))
printf 'suite\tfinished_failures=%s\t%s\n' "${failures}" "$(date --iso-8601=seconds)" >> "${status_file}"
(( failures == 0 ))
