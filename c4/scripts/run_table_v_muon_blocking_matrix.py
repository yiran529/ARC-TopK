"""Run the 48-cell Muon blocking/socket Table V timing matrix serially."""

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
ARTIFACTS = ROOT / "output/CM058-CM105-table-v-muon-blocking-matrix"
CONTROLLER_PATH = Path(__file__)
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
START_NUMBER = 58
DTYPE = "float32"
MAX_LENGTH = 256
WARMUP_ITERATIONS = 100
MEASURED_ITERATIONS = 50
MODELS = (
    ("60m", "llama_60m.json"),
    ("130m", "llama_130m.json"),
    ("350m", "llama_350m.json"),
)
ARMS = (
    ("dense", "none", "m001"),
    ("topk", "topk_sync", "m003"),
    ("randk", "randk_sync", "m003"),
    ("arctopk", "group_topk_no_reshape", "m002"),
)
GROUPS = (
    ("ws4-bucket1024-shm", 4, 1024, False, "shm"),
    ("ws4-blocking-shm", 4, None, True, "shm"),
    ("ws8-blocking-shm", 8, None, True, "shm"),
    ("ws8-blocking-socket", 8, None, True, "socket"),
)
CONTROLLED_NCCL_KEYS = (
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_CUMEM_HOST_ENABLE",
    "NCCL_MAX_CTAS",
    "NCCL_MIN_CTAS",
    "NCCL_MAX_NCHANNELS",
    "NCCL_MIN_NCHANNELS",
    "NCCL_NET",
    "NCCL_SOCKET_IFNAME",
    "NCCL_SOCKET_NTHREADS",
    "NCCL_NSOCKS_PERTHREAD",
    "NCCL_DEBUG",
)


def timestamp():
    return datetime.now().astimezone().isoformat()


def cells():
    number = START_NUMBER
    for group, world_size, bucket_cap_mb, blocking, transport in GROUPS:
        for model, config in MODELS:
            for arm, compressor, method in ARMS:
                run_id = (
                    f"CM{number:03d}-{method}-{arm}-muon-llama{model}-c4-table-v-"
                    f"{DTYPE}-{group}-s1243"
                )
                command = [
                    str(ROOT / ".venv/bin/torchrun"),
                    "--standalone",
                    f"--nproc_per_node={world_size}",
                    "--",
                    "c4/table_v_timing.py",
                    "--model_config",
                    f"c4/configs/{config}",
                    "--dataset_path",
                    str(DATASET),
                    "--batch_size",
                    "1",
                    "--max_length",
                    str(MAX_LENGTH),
                    "--dtype",
                    DTYPE,
                    "--warmup_iterations",
                    str(WARMUP_ITERATIONS),
                    "--measured_iterations",
                    str(MEASURED_ITERATIONS),
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
                yield {
                    "number": number,
                    "run_id": run_id,
                    "group": group,
                    "world_size": world_size,
                    "bucket_cap_mb": bucket_cap_mb,
                    "blocking_communication": blocking,
                    "transport": transport,
                    "model": model,
                    "arm": arm,
                    "compressor": compressor,
                    "command": command,
                }
                number += 1


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


def environment_for(cell, all_gpus):
    environment = dict(os.environ)
    for key in CONTROLLED_NCCL_KEYS:
        environment.pop(key, None)
    selected_gpus = all_gpus[: cell["world_size"]]
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(selected_gpus),
        NCCL_P2P_DISABLE="1",
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
    if cell["transport"] == "socket":
        environment.update(
            NCCL_SHM_DISABLE="1",
            NCCL_NET="Socket",
            NCCL_SOCKET_IFNAME="lo",
            NCCL_SOCKET_NTHREADS="1",
            NCCL_NSOCKS_PERTHREAD="1",
        )
    else:
        environment["NCCL_SHM_DISABLE"] = "0"
    return environment


def write_comparisons(rows):
    completed = {
        (row["group"], row["model"], row["arm"]): row
        for row in rows
        if row["status"] == "completed"
    }
    comparisons = []
    for group, *_ in GROUPS:
        for model, _ in MODELS:
            dense = completed.get((group, model, "dense"))
            if dense is None:
                continue
            dense_seconds = dense["mean_iteration_seconds"]
            for arm, _, _ in ARMS[1:]:
                method = completed.get((group, model, arm))
                if method is None:
                    continue
                method_seconds = method["mean_iteration_seconds"]
                comparisons.append(
                    {
                        "group": group,
                        "model": model,
                        "arm": arm,
                        "dense_seconds": dense_seconds,
                        "method_seconds": method_seconds,
                        "speedup": dense_seconds / method_seconds,
                        "time_reduction_percent": (
                            (dense_seconds - method_seconds) / dense_seconds * 100
                        ),
                    }
                )
    (ARTIFACTS / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    all_gpus = args.gpus.split(",")
    if (
        len(all_gpus) != 8
        or len(set(all_gpus)) != 8
        or any(not gpu.isdigit() for gpu in all_gpus)
    ):
        parser.error("--gpus must contain eight distinct GPU indices")

    matrix = list(cells())
    if args.dry_run:
        print(f"cells={len(matrix)} ids=CM{matrix[0]['number']:03d}-CM{matrix[-1]['number']:03d}")
        for cell in matrix:
            print(shlex.join(cell["command"]))
        return

    os.chdir(ROOT)
    if not DATASET.joinpath("en").is_dir():
        raise FileNotFoundError(f"C4 data directory does not exist: {DATASET / 'en'}")
    if not selected_gpus_are_idle(all_gpus):
        raise RuntimeError("Selected GPUs are not idle")
    ARTIFACTS.mkdir(parents=True, exist_ok=False)
    (ARTIFACTS / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": timestamp(),
                "gpus": all_gpus,
                "dtype": DTYPE,
                "max_length": MAX_LENGTH,
                "cell_count": len(matrix),
                "cells": matrix,
                "controller_sha256": hashlib.sha256(CONTROLLER_PATH.read_bytes()).hexdigest(),
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

    record_status("queue", "started", f"serial; {len(matrix)} cells; no polling")
    for cell in matrix:
        run_id = cell["run_id"]
        environment = environment_for(cell, all_gpus)
        record_status(run_id, "started", cell["group"])
        (ARTIFACTS / f"{run_id}.command.txt").write_text(
            shlex.join(cell["command"]) + "\n", encoding="utf-8"
        )
        (ARTIFACTS / f"{run_id}.environment.json").write_text(
            json.dumps(
                {
                    key: environment[key]
                    for key in ("CUDA_VISIBLE_DEVICES", *CONTROLLED_NCCL_KEYS)
                    if key in environment
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
            and result.get("measured_steps") == MEASURED_ITERATIONS
            and result.get("optimizer") == "muon"
            and result.get("compressor") == cell["compressor"]
        )
        row = {
            "run_id": run_id,
            "group": cell["group"],
            "world_size": cell["world_size"],
            "transport": cell["transport"],
            "model": cell["model"],
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
    write_comparisons(rows)
    record_status(
        "queue",
        "finished",
        f'completed={sum(row["status"] == "completed" for row in rows)}/{len(rows)}',
    )


if __name__ == "__main__":
    main()
