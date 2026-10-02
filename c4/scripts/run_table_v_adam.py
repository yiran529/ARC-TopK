"""Run the four dense AdamW controls paired with the Table V Muon timings."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "output/CM036-CM039-table-v-adam-fp32"
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
MODELS = (
    ("60m", "llama_60m.json"),
    ("130m", "llama_130m.json"),
    ("350m", "llama_350m.json"),
    ("1b", "llama_1b.json"),
)


def cells():
    for number, (model, config) in enumerate(MODELS, start=36):
        run_id = f"CM{number:03d}-m004-dense-adamw-llama{model}-c4-table-v-fp32-ws4-s1243"
        command = [
            str(ROOT / ".venv/bin/torchrun"),
            "--standalone",
            "--nproc_per_node=4",
            "--",
            "c4/table_v_timing.py",
            "--model_config",
            f"c4/configs/{config}",
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
            "adamw",
            "--lr",
            "0.002",
            "--beta1",
            "0.9",
            "--beta2",
            "0.999",
            "--eps",
            "1e-8",
            "--weight_decay",
            "0",
            "--grad_clipping",
            "1",
            "--compressor",
            "none",
            "--start_compress_iter",
            "0",
            "--use_error_feedback",
            "noef",
            "--output_dir",
            str(ARTIFACTS / run_id),
        ]
        yield {"run_id": run_id, "model": model, "command": command}


def timestamp():
    return datetime.now().astimezone().isoformat()


def selected_gpus_are_idle(gpus):
    active = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader",
        ],
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
    return len(selected) == 4 and all(
        uuid.strip() not in occupied and int(free.strip()) >= 23000
        for _, uuid, free in selected
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or any(not gpu.isdigit() for gpu in gpus):
        parser.error("--gpus must contain four distinct GPU indices")
    matrix = list(cells())
    if args.dry_run:
        for cell in matrix:
            print(shlex.join(cell["command"]))
        return

    os.chdir(ROOT)
    if not DATASET.joinpath("en").is_dir():
        raise FileNotFoundError(f"C4 data directory does not exist: {DATASET / 'en'}")
    if not selected_gpus_are_idle(gpus):
        raise RuntimeError(f"Selected GPUs are not idle: {args.gpus}")
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=args.gpus,
        NCCL_P2P_DISABLE="1",
        NCCL_SHM_DISABLE="0",
        NCCL_CUMEM_HOST_ENABLE="0",
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
                "gpus": args.gpus,
                "cells": matrix,
                "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (ARTIFACTS / "gpu_inventory.txt").write_text(
        subprocess.check_output(["nvidia-smi", "-L"], text=True), encoding="utf-8"
    )
    rows = []

    def record_status(run_id, state, detail=""):
        with (ARTIFACTS / "status.tsv").open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp()}\t{run_id}\t{state}\t{detail}\n")
        print(f"{run_id}: {state} {detail}", flush=True)

    record_status("queue", "started", "serial; no polling; warmup=100; measured=50")
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
            and result.get("optimizer") == "adamw"
        )
        row = {
            "run_id": run_id,
            "model": cell["model"],
            "exit_code": completed.returncode,
            "status": "completed" if accepted else "failed",
            "mean_iteration_seconds": result["mean_iteration_seconds"] if accepted else None,
            "parameter_count": result["parameter_count"] if accepted else None,
            "peak_allocated_gib": (
                max(result["rank_peak_allocated_bytes"]) / 2**30 if accepted else None
            ),
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
