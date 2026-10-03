"""Serial FP32 A-D communication-path measurements, 50 warmup + 100 updates."""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import subprocess

from c4.scripts.run_table_v_muon_blocking_matrix import (
    CONTROLLED_NCCL_KEYS, selected_gpus_are_idle,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "output/CM122-CM265-strict-hook-abcd-fp32"
DATASET = Path("/home/wyr/greedy_lore/c4/c4_en")
GROUPS = (("A", 4, "default"), ("B", 4, "one"),
          ("C", 8, "default"), ("D", 8, "one"))
ARMS = (("dense", "none", "m001"), ("topk", "topk_sync", "m003"),
        ("randk", "randk_sync", "m003"),
        ("arctopk", "group_topk_no_reshape", "m002"))
SOURCES = ("c4/table_v_timing.py", "c4/scripts/run_strict_hook_abcd.py",
           "comm_hooks/utils.py", "comm_hooks/default_hooks.py",
           "comm_hooks/group_topk_hook_no_reshape.py", "comm_hooks/sparse_hook_c4.py",
           "optimizers/muon.py", "optimizers/utils.py")


def cells(artifacts=ARTIFACTS, smoke=False):
    number = 122
    rng = random.Random(20261003)
    groups = GROUPS[:1] if smoke else GROUPS
    models = ("60m",) if smoke else ("60m", "130m", "350m")
    for repeat in range(1, 2 if smoke else 4):
        for group, world_size, channels in groups:
            for model in models:
                arms = list(ARMS)
                if not smoke:
                    rng.shuffle(arms)
                for arm, compressor, method in arms:
                    run_id = (f"smoke-{arm}" if smoke else
                              f"CM{number:03d}-{method}-{arm}-llama{model}-"
                              f"{group}-fp32-rep{repeat}-s1243")
                    warmup, measured, start = (2, 2, 1) if smoke else (50, 100, 40)
                    command = [str(ROOT / ".venv/bin/torchrun"), "--standalone",
                               f"--nproc_per_node={world_size}", "--", "c4/table_v_timing.py"]
                    values = {
                        "model_config": f"c4/configs/llama_{model}.json",
                        "dataset_path": str(DATASET), "tokenizer_path": "t5-base",
                        "batch_size": 1, "max_length": 256, "dtype": "float32",
                        "warmup_iterations": warmup, "measured_iterations": measured,
                        "workers": 4, "ddp_bucket_cap_mb": 8192, "seed": 1243,
                        "optimizer": "muon", "lr": 0.01, "muon_scalar_lr": 0.001,
                        "muon_mu": 0.95, "muon_adjust_lr": "spectral_norm",
                        "weight_decay": 0, "muon_scalar_weight_decay": 0,
                        "grad_clipping": 1, "compressor": compressor,
                        "start_compress_iter": 0 if arm == "dense" else start,
                        "use_error_feedback": "ef14", "compress_ratio": 0.2,
                        "sparse_type": "tensor", "r": 4,
                        "output_dir": str(artifacts / run_id),
                    }
                    for key, value in values.items():
                        command.extend((f"--{key}", str(value)))
                    command.extend(("--disable_compression_warmup", "--strict_blocking_communication"))
                    yield dict(number=number, run_id=run_id, repeat=repeat, group=group,
                               world_size=world_size, channels=channels, model=model,
                               arm=arm, compressor=compressor, warmup=warmup,
                               measured=measured, command=command)
                    number += 1


def environment_for(cell, gpus):
    environment = dict(os.environ)
    for key in CONTROLLED_NCCL_KEYS:
        environment.pop(key, None)
    environment.update(
        CUDA_VISIBLE_DEVICES=",".join(gpus[:cell["world_size"]]),
        NCCL_P2P_DISABLE="1", NCCL_SHM_DISABLE="0", NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_DEBUG="INFO", HF_HOME="/home/wyr/.cache/huggingface",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="4", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
    )
    if cell["channels"] == "one":
        environment.update(NCCL_MIN_NCHANNELS="1", NCCL_MAX_NCHANNELS="1",
                           NCCL_MIN_CTAS="1", NCCL_MAX_CTAS="1")
    return environment


def accepted_result(result, cell, exit_code):
    return bool(exit_code == 0 and result and result.get("status") == "completed"
                and result.get("optimizer") == "muon"
                and result.get("compressor") == cell["compressor"]
                and result.get("strict_blocking_communication") is True
                and result.get("measured_steps") == cell["measured"]
                and result.get("first_measured_update") == cell["warmup"] + 1
                and result.get("last_measured_update") == cell["warmup"] + cell["measured"]
                and result.get("observed_bucket_counts") == [1]
                and result.get("mean_blocking_hook_seconds", 0) > 0
                and result.get("mean_iteration_seconds", 0) > 0)


def read_result(path):
    try:
        result = json.loads(path.read_text())
        if not isinstance(result, dict):
            return None, "Result must be a JSON object"
        return result, None
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{type(error).__name__}: {error}"


def comparisons(rows):
    dense = {(r["repeat"], r["group"], r["model"]): r for r in rows
             if r["status"] == "completed" and r["arm"] == "dense"}
    pairs = []
    for row in rows:
        base = dense.get((row["repeat"], row["group"], row["model"]))
        if row["status"] != "completed" or row["arm"] == "dense" or base is None:
            continue
        pair = {key: row[key] for key in ("repeat", "group", "model", "arm", "run_id")}
        pair["dense_run_id"] = base["run_id"]
        for label, metric in (("iter", "mean_iteration_seconds"),
                              ("hook", "mean_blocking_hook_seconds")):
            pair[f"{label}_speedup"] = base[metric] / row[metric]
            pair[f"{label}_time_reduction_percent"] = (1 - row[metric] / base[metric]) * 100
        pairs.append(pair)
    return pairs


def write_summaries(artifacts, rows):
    def save(name, value):
        (artifacts / name).write_text(json.dumps(value, indent=2) + "\n")
    save("summary.json", rows)
    with (artifacts / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    pairs = comparisons(rows)
    save("comparisons.json", pairs)
    aggregate = []
    for group, _, _ in GROUPS:
        for model in ("60m", "130m", "350m"):
            for arm, _, _ in ARMS:
                selected = [r for r in rows if (r["group"], r["model"], r["arm"])
                            == (group, model, arm) and r["status"] == "completed"]
                if not selected:
                    continue
                record = dict(group=group, model=model, arm=arm, n=len(selected))
                for metric in ("mean_iteration_seconds", "mean_blocking_hook_seconds"):
                    values = [r[metric] for r in selected]
                    record[metric] = statistics.mean(values)
                    record[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
                matching = [p for p in pairs if (p["group"], p["model"], p["arm"])
                            == (group, model, arm)]
                record["paired_n"] = len(matching)
                for metric in ("iter_speedup", "hook_speedup"):
                    values = [p[metric] for p in matching]
                    record[metric + "_mean"] = statistics.mean(values) if values else None
                    record[metric + "_std"] = statistics.stdev(values) if len(values) > 1 else None
                aggregate.append(record)
    save("aggregate.json", aggregate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    expected = 4 if args.smoke else 8
    if len(gpus) != expected or len(set(gpus)) != expected or not all(g.isdigit() for g in gpus):
        parser.error(f"--gpus requires {expected} distinct GPU indices")
    if args.smoke and args.output_root is None:
        parser.error("--smoke requires a separate --output-root")
    artifacts = (args.output_root or ARTIFACTS).resolve()
    matrix = list(cells(artifacts, args.smoke))
    if args.dry_run:
        print(json.dumps(matrix, indent=2))
        return
    os.chdir(ROOT)
    if not selected_gpus_are_idle(gpus):
        raise RuntimeError("Selected GPUs are occupied or have insufficient free memory")
    from c4.pept_utils.c4_data import _local_split_files
    shards = _local_split_files(str(DATASET), "train")
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES}
    artifacts.mkdir(parents=True, exist_ok=False)
    manifest = dict(started_at=datetime.now().astimezone().isoformat(), gpus=gpus,
                    cell_count=len(matrix), order_seed=20261003, cells=matrix,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    code_sha256=hashes,
                    data_files=[dict(path=str(p), bytes=p.stat().st_size) for p in shards])
    (artifacts / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    inventory = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True)
    (artifacts / "gpu-topology.txt").write_text(inventory)
    rows = []

    def status(run_id, state, detail=""):
        line = f"{datetime.now().astimezone().isoformat()}\t{run_id}\t{state}\t{detail}"
        with (artifacts / "status.tsv").open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    status("queue", "started", f"serial {len(matrix)} runs; no polling")
    for cell in matrix:
        if any(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError("Experiment source changed after manifest creation")
        run_id = cell["run_id"]
        environment = environment_for(cell, gpus)
        status(run_id, "started", f"group={cell['group']} repeat={cell['repeat']}")
        (artifacts / f"{run_id}.command.txt").write_text(shlex.join(cell["command"]) + "\n")
        (artifacts / f"{run_id}.environment.json").write_text(json.dumps(
            {k: environment[k] for k in ("CUDA_VISIBLE_DEVICES", *CONTROLLED_NCCL_KEYS)
             if k in environment}, indent=2) + "\n")
        with (artifacts / f"{run_id}.log").open("x") as log:
            run = subprocess.run(["timeout", "--signal=TERM", "--kill-after=60s", "2h",
                                  *cell["command"]], env=environment,
                                 stdout=log, stderr=subprocess.STDOUT)
        result_path = artifacts / run_id / "all_results.json"
        result, result_error = read_result(result_path)
        accepted = accepted_result(result, cell, run.returncode)
        row = {k: cell[k] for k in ("run_id", "repeat", "group", "world_size", "channels", "model", "arm")}
        row.update(exit_code=run.returncode, status="completed" if accepted else "failed",
                   result_error=result_error,
                   mean_iteration_seconds=result["mean_iteration_seconds"] if accepted else None,
                   mean_blocking_hook_seconds=result["mean_blocking_hook_seconds"] if accepted else None,
                   observed_bucket_counts=result.get("observed_bucket_counts") if result else None)
        rows.append(row)
        status(run_id, row["status"], f"exit={run.returncode}")
        write_summaries(artifacts, rows)
    failures = sum(row["status"] != "completed" for row in rows)
    status("queue", "finished", f"completed={len(rows)-failures}/{len(rows)} failures={failures}")
    if args.smoke and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
