"""Run the 16-cell ARC-TopK Table V matrix with Muon serially."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "output/CM020-CM035-table-v-muon-fp32-rerun1"
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
MODELS = (
    ("60m", "llama_60m.json"),
    ("130m", "llama_130m.json"),
    ("350m", "llama_350m.json"),
    ("1b", "llama_1b.json"),
)
ARMS = (
    ("dense", "none", "m001"),
    ("topk", "topk_sync", "m003"),
    ("randk", "randk_sync", "m003"),
    ("arctopk", "group_topk_no_reshape", "m002"),
)


def cells():
    number = 20
    for model, config in MODELS:
        for arm, compressor, method in ARMS:
            run_id = (
                f"CM{number:03d}-rerun1-{method}-{arm}-muon-llama{model}-c4-table-v-"
                "fp32-ws4-s1243"
            )
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
            yield {"run_id": run_id, "model": model, "arm": arm, "command": command}
            number += 1


def timestamp():
    return datetime.now().astimezone().isoformat()


def result_is_accepted(exit_code, result):
    return (
        exit_code == 0
        and result is not None
        and result.get("status") == "completed"
        and result.get("measured_steps") == 50
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
    data_files = sorted(DATASET.joinpath("en").glob("c4-train*.json*"))
    (ARTIFACTS / "data_inventory.json").write_text(
        json.dumps(
            [
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns,
                }
                for path in data_files
            ],
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

    def wait_for_idle():
        waiting_logged = False
        while True:
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
            if len(selected) == 4 and all(
                uuid.strip() not in occupied and int(free.strip()) >= 23000
                for _, uuid, free in selected
            ):
                return
            if not waiting_logged:
                record_status("queue", "waiting_for_idle", args.gpus)
                waiting_logged = True
            time.sleep(60)

    record_status("queue", "started", "serial; 16 cells; warmup=100; measured=50")
    for cell in matrix:
        wait_for_idle()
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
        result = (
            json.loads(result_path.read_text(encoding="utf-8"))
            if result_path.exists()
            else None
        )
        row = {
            "run_id": run_id,
            "model": cell["model"],
            "arm": cell["arm"],
            "exit_code": completed.returncode,
        }
        if result_is_accepted(completed.returncode, result):
            row.update(
                status="completed",
                mean_iteration_seconds=result["mean_iteration_seconds"],
                parameter_count=result["parameter_count"],
                peak_allocated_gib=max(result["rank_peak_allocated_bytes"]) / 2**30,
            )
        else:
            row.update(
                status="failed",
                mean_iteration_seconds=None,
                parameter_count=None,
                peak_allocated_gib=None,
            )
        rows.append(row)
        record_status(
            run_id,
            row["status"],
            f'exit_code={completed.returncode}; seconds={row["mean_iteration_seconds"]}',
        )
        (ARTIFACTS / "summary.json").write_text(
            json.dumps(rows, indent=2) + "\n", encoding="utf-8"
        )

    fieldnames = [
        "run_id",
        "model",
        "arm",
        "exit_code",
        "status",
        "mean_iteration_seconds",
        "parameter_count",
        "peak_allocated_gib",
    ]
    with (ARTIFACTS / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# ARC-TopK Table V：Muon 版本测速",
        "",
        "100 次更新热身后连续测量 50 次更新；主指标为最慢 rank 总时间除以 50。",
        "",
        "| 模型 | Dense Muon | Top-K + Muon | Rand-K + Muon | ARC-TopK + Muon | ARC 相对 Dense |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, _ in MODELS:
        values = {
            row["arm"]: row["mean_iteration_seconds"]
            for row in rows
            if row["model"] == model
        }
        display = [
            f'{values[arm]:.6f}' if values[arm] is not None else "failed"
            for arm, _, _ in ARMS
        ]
        speedup = (
            f'{values["dense"] / values["arctopk"]:.4f}x'
            if values["dense"] and values["arctopk"]
            else "—"
        )
        report.append(f'| {model} | {" | ".join(display)} | {speedup} |')
    (ARTIFACTS / "results.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    record_status(
        "queue",
        "finished",
        f'completed={sum(row["status"] == "completed" for row in rows)}/16',
    )


if __name__ == "__main__":
    main()
