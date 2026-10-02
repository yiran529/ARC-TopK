"""Run paired 350M Muon timings on eight GPUs with one NCCL SHM channel."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "output/CM042-CM043-table-v-muon-350m-shm1-ws8"
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
ARMS = (
    (42, "dense", "none", "m001"),
    (43, "arctopk", "group_topk_no_reshape", "m002"),
)


def timestamp():
    return datetime.now().astimezone().isoformat()


def command_for(number, arm, compressor, method):
    run_id = (
        f"CM{number:03d}-{method}-{arm}-muon-llama350m-c4-table-v-"
        "fp32-ws8-shm1-s1243"
    )
    command = [
        str(ROOT / ".venv/bin/torchrun"),
        "--standalone",
        "--nproc_per_node=8",
        "--",
        "c4/table_v_timing.py",
        "--model_config",
        "c4/configs/llama_350m.json",
        "--dataset_path",
        str(DATASET),
        "--batch_size",
        "1",
        "--max_length",
        "256",
        "--dtype",
        "float32",
        "--warmup_iterations",
        "100",
        "--measured_iterations",
        "50",
        "--workers",
        "4",
        "--seed",
        "1243",
        "--optimizer",
        "muon",
        "--lr",
        "0.01",
        "--muon_scalar_lr",
        "0.001",
        "--muon_mu",
        "0.95",
        "--muon_adjust_lr",
        "spectral_norm",
        "--weight_decay",
        "0",
        "--muon_scalar_weight_decay",
        "0",
        "--grad_clipping",
        "1",
        "--compressor",
        compressor,
        "--start_compress_iter",
        "0" if arm == "dense" else "100",
        "--use_error_feedback",
        "ef14",
        "--compress_ratio",
        "0.2",
        "--sparse_type",
        "tensor",
        "--disable_compression_warmup",
        "--r",
        "4",
        "--output_dir",
        str(ARTIFACTS / run_id),
    ]
    return {"run_id": run_id, "arm": arm, "command": command}


def selected_gpus_are_idle(gpus):
    active = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    inventory = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    occupied = {
        line.split(",")[0].strip() for line in active.splitlines() if line.strip()
    }
    selected = [
        line.split(",")
        for line in inventory.splitlines()
        if line.split(",")[0].strip() in gpus
    ]
    return len(selected) == 8 and all(
        uuid.strip() not in occupied and int(free.strip()) >= 23000
        for _, uuid, free in selected
    )


def main():
    gpus = [str(index) for index in range(8)]
    matrix = [command_for(*arm) for arm in ARMS]
    os.chdir(ROOT)
    if not DATASET.joinpath("en").is_dir():
        raise FileNotFoundError(f"C4 data directory does not exist: {DATASET / 'en'}")
    if not selected_gpus_are_idle(gpus):
        raise RuntimeError("GPUs 0-7 are not idle")
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(gpus),
        NCCL_P2P_DISABLE="1",
        NCCL_SHM_DISABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_MAX_NCHANNELS="1",
        NCCL_MIN_NCHANNELS="1",
        NCCL_DEBUG="INFO",
        HF_HOME="/dev/shm/wyr_tmp/hf",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        PYTHONPATH=str(ROOT),
    )
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": timestamp(),
                "gpus": gpus,
                "world_size": 8,
                "nccl_max_nchannels": 1,
                "nccl_min_nchannels": 1,
                "cells": matrix,
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []

    def record_status(run_id, state, detail=""):
        with (ARTIFACTS / "status.tsv").open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp()}\t{run_id}\t{state}\t{detail}\n")
        print(f"{run_id}: {state} {detail}", flush=True)

    record_status("queue", "started", "serial; no polling; world_size=8; SHM channels=1")
    for cell in matrix:
        run_id = cell["run_id"]
        record_status(run_id, "started")
        (ARTIFACTS / f"{run_id}.command.txt").write_text(
            shlex.join(cell["command"]) + "\n", encoding="utf-8"
        )
        with (ARTIFACTS / f"{run_id}.log").open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                ["timeout", "--signal=TERM", "--kill-after=60s", "2h", *cell["command"]],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result_path = ARTIFACTS / run_id / "all_results.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        accepted = (
            completed.returncode == 0
            and result is not None
            and result.get("status") == "completed"
            and result.get("measured_steps") == 50
            and result.get("optimizer") == "muon"
        )
        row = {
            "run_id": run_id,
            "arm": cell["arm"],
            "exit_code": completed.returncode,
            "status": "completed" if accepted else "failed",
            "mean_iteration_seconds": result["mean_iteration_seconds"] if accepted else None,
            "parameter_count": result["parameter_count"] if accepted else None,
        }
        rows.append(row)
        record_status(
            run_id,
            row["status"],
            f'exit_code={completed.returncode}; seconds={row["mean_iteration_seconds"]}',
        )
        (ARTIFACTS / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

    with (ARTIFACTS / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record_status(
        "queue",
        "finished",
        f'completed={sum(row["status"] == "completed" for row in rows)}/2',
    )


if __name__ == "__main__":
    main()
