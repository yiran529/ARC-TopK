"""Run paired 350M Muon experiments with backward/communication overlap removed."""

import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "output/CM054-CM057-table-v-muon-350m-no-overlap"
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
CELLS = (
    (54, "bucket2048", "dense", "none", "m001", 2048, False),
    (55, "bucket2048", "arctopk", "group_topk_no_reshape", "m002", 2048, False),
    (56, "blocking", "dense", "none", "m001", None, True),
    (57, "blocking", "arctopk", "group_topk_no_reshape", "m002", None, True),
)


def timestamp():
    return datetime.now().astimezone().isoformat()


def command_for(number, mode, arm, compressor, method, bucket_cap_mb, blocking):
    run_id = (
        f"CM{number:03d}-{method}-{arm}-muon-llama350m-c4-table-v-"
        f"fp32-ws8-{mode}-s1243"
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
    if bucket_cap_mb is not None:
        command.extend(("--ddp_bucket_cap_mb", str(bucket_cap_mb)))
    if blocking:
        command.append("--blocking_communication")
    return {
        "run_id": run_id,
        "mode": mode,
        "arm": arm,
        "bucket_cap_mb": bucket_cap_mb,
        "blocking_communication": blocking,
        "command": command,
    }


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
    return len(selected) == len(gpus) and all(
        uuid.strip() not in occupied and int(free.strip()) >= 23000
        for _, uuid, free in selected
    )


def environment_for(gpus):
    return dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=",".join(gpus),
        NCCL_P2P_DISABLE="1",
        NCCL_SHM_DISABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_MAX_CTAS="1",
        NCCL_MIN_CTAS="1",
        NCCL_MAX_NCHANNELS="1",
        NCCL_MIN_NCHANNELS="1",
        NCCL_DEBUG="INFO",
        HF_HOME="/dev/shm/wyr_tmp/hf",
        HF_HUB_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4",
        PYTHONPATH=str(ROOT),
    )


def main():
    gpus = [str(index) for index in range(8)]
    matrix = [command_for(*cell) for cell in CELLS]
    os.chdir(ROOT)
    if not DATASET.joinpath("en").is_dir():
        raise FileNotFoundError(f"C4 data directory does not exist: {DATASET / 'en'}")
    if not selected_gpus_are_idle(gpus):
        raise RuntimeError("GPUs 0-7 are not idle")
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": timestamp(),
                "gpus": gpus,
                "world_size": 8,
                "dtype": "float32",
                "cells": matrix,
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rows = []
    environment = environment_for(gpus)

    def record_status(run_id, state, detail=""):
        with (ARTIFACTS / "status.tsv").open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp()}\t{run_id}\t{state}\t{detail}\n")
        print(f"{run_id}: {state} {detail}", flush=True)

    record_status("queue", "started", "serial; no polling; modes=bucket2048,blocking")
    for cell in matrix:
        run_id = cell["run_id"]
        record_status(run_id, "started", cell["mode"])
        (ARTIFACTS / f"{run_id}.command.txt").write_text(
            shlex.join(cell["command"]) + "\n", encoding="utf-8"
        )
        (ARTIFACTS / f"{run_id}.environment.json").write_text(
            json.dumps(
                {
                    key: value
                    for key, value in environment.items()
                    if key.startswith("NCCL_") or key == "CUDA_VISIBLE_DEVICES"
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
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
            "mode": cell["mode"],
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
        f'completed={sum(row["status"] == "completed" for row in rows)}/4',
    )


if __name__ == "__main__":
    main()
