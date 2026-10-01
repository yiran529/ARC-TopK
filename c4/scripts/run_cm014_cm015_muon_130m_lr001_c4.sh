#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

suite_id="CM014-CM015-muon-llama130m-c4-lr001-serial"
suite_dir="output/${suite_id}"
data_dir="${suite_dir}/c4-data"
status_file="${suite_dir}/status.tsv"
source_en_dir="/home/wyr/greedy_lore/c4/c4_en/en"

if [[ -e "${suite_dir}" ]]; then
    echo "Suite directory already exists: ${suite_dir}" >&2
    exit 1
fi
mkdir -p "${data_dir}/en"

# Match the CM004/CM005 data selection: first 30 train shards, all 8 val shards.
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

printf 'suite\twaiting_for_4_idle_gpus\t%s\n' "$(date --iso-8601=seconds)" > "${status_file}"
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
        selected_gpus="$(IFS=,; echo "${selected[*]:0:4}")"
        break
    fi
    sleep 60
done

printf 'suite\tselected_gpus=%s\t%s\n' "${selected_gpus}" "$(date --iso-8601=seconds)" >> "${status_file}"
export CUDA_VISIBLE_DEVICES="${selected_gpus}"
export WANDB_MODE=online
export HF_HOME=/dev/shm/wyr_tmp/hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONPATH=.

run_experiment() {
    local run_id="$1" compressor="$2" output_dir="output/$1" exit_code
    local -a command=(
        .venv/bin/torchrun --standalone --nproc_per_node=4 --
        c4/run_llama_pretraining.py
        --model_config c4/configs/llama_130m.json
        --dataset_path "${data_dir}"
        --max_length 256
        --dtype float32
        --num_training_steps 20000
        --warmup_steps 1000
        --scheduler cosine
        --min_lr_ratio 0.1
        --total_batch_size 512
        --batch_size 16
        --gradient_accumulation 8
        --workers 4
        --seed 1243
        --weight_decay 0.0
        --grad_clipping 1.0
        --eval_tokens 10000000
        --compressor "${compressor}"
        --wandb_project ARC-TopK-LLaMA-130M
        --wandb_job_type formal-pretraining
        --output_dir "${output_dir}"
        --optimizer muon
        --lr 0.01
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
    if [[ "${compressor}" == "group_topk_no_reshape" ]]; then
        command+=(
            --start_compress_iter 1000
            --use_error_feedback ef14
            --compress_ratio 0.2
            --r 4
        )
    fi
    if [[ -e "${output_dir}" ]]; then
        echo "Run directory already exists: ${output_dir}" >&2
        return 1
    fi
    mkdir -p "${output_dir}"
    printf '%s\tstarted\t%s\n' "${run_id}" "$(date --iso-8601=seconds)" >> "${status_file}"
    set +e
    "${command[@]}" 2>&1 | tee "${output_dir}/train.log"
    exit_code="${PIPESTATUS[0]}"
    set -e
    printf '%s\tfinished\t%s\texit_code=%s\n' \
        "${run_id}" "$(date --iso-8601=seconds)" "${exit_code}" >> "${status_file}"
    return "${exit_code}"
}

run_experiment "CM014-m001-dense-muon-llama130m-c4-2p62b-ws4-s1243-lr001" none
run_experiment "CM015-m002-arctopk-muon-llama130m-c4-2p62b-ws4-s1243-lr001" group_topk_no_reshape
printf 'suite\tfinished\t%s\n' "$(date --iso-8601=seconds)" >> "${status_file}"
