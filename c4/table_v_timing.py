"""End-to-end Table V timing using C4, DDP communication hooks, and Muon."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time


def accumulate_hook_communication_bits(hook_state, total_bits):
    if hook_state is None or not hasattr(hook_state, "comm_bits_this_round"):
        return total_bits
    total_bits += hook_state.comm_bits_this_round
    hook_state.comm_bits_this_round = 0
    return total_bits


def summarize_timing(rank_elapsed, rank_steps, expected_steps):
    if expected_steps <= 0 or not rank_elapsed or len(rank_elapsed) != len(rank_steps):
        raise ValueError("Invalid timing window")
    if any(len(steps) != expected_steps for steps in rank_steps):
        raise ValueError("Incomplete timing window")
    if any(not math.isfinite(value) or value <= 0 for value in rank_elapsed):
        raise ValueError("Nonfinite or nonpositive elapsed time")
    if any(
        not math.isfinite(value) or value <= 0
        for steps in rank_steps
        for value in steps
    ):
        raise ValueError("Nonfinite or nonpositive iteration time")
    slowest_rank = max(range(len(rank_elapsed)), key=rank_elapsed.__getitem__)
    return {
        "measured_steps": expected_steps,
        "mean_iteration_seconds": rank_elapsed[slowest_rank] / expected_steps,
        "rank_elapsed_seconds": rank_elapsed,
        "slowest_rank": slowest_rank,
        "slowest_rank_iteration_median_seconds": statistics.median(
            rank_steps[slowest_rank]
        ),
        "rank_iteration_seconds": rank_steps,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--tokenizer_path", default="t5-base")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--warmup_iterations", type=int, default=100)
    parser.add_argument("--measured_iterations", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ddp_bucket_cap_mb", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1243)
    parser.add_argument("--optimizer", choices=("muon", "adamw"), default="muon")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_clipping", type=float, default=1.0)
    parser.add_argument("--output_dir", required=True)
    from comm_hooks.utils import add_comm_hook_args
    from optimizers import add_muon_args

    add_comm_hook_args(parser)
    add_muon_args(parser, scalar_lr_default=0.001, scalar_weight_decay_default=0.0)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.measured_iterations <= 0 or args.warmup_iterations < 0:
        parser.error("Batch size and timing window must be positive")
    if args.compressor not in (
        "none",
        "topk_sync",
        "randk_sync",
        "group_topk_no_reshape",
    ):
        parser.error("Unsupported Table V compressor")
    if args.compressor != "none" and args.start_compress_iter > args.warmup_iterations:
        parser.error("Compression must be active throughout the measured window")
    return args


def main():
    args = parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    import datasets.distributed
    from transformers import AutoConfig, AutoTokenizer

    from c4.pept_utils.c4_data import load_c4_split
    from c4.pept_utils.dataloader import PreprocessedIterableDataset
    from c4.pept_utils.modeling_llama import LlamaForCausalLM
    from c4.pept_utils.training_utils import get_scheduler
    from comm_hooks.utils import register_comm_hook_for_ddp_model
    from optimizers import build_muon_optimizer

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.max_length,
        local_files_only=True,
    )
    data = load_c4_split(args.dataset_path, "train", repeat_local=True).shuffle(seed=42)
    data = datasets.distributed.split_dataset_by_node(data, rank=rank, world_size=world_size)
    dataset = PreprocessedIterableDataset(data, tokenizer, args.batch_size, args.max_length)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers,
        pin_memory=True,
    )
    batches = iter(loader)

    model_config = AutoConfig.from_pretrained(args.model_config)
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.use_cache = False
    model = LlamaForCausalLM(model_config)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model.to(device=device, dtype=dtype)
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if args.optimizer == "muon":
        optimizer = build_muon_optimizer(
            model,
            lr=args.lr,
            scalar_lr=args.muon_scalar_lr,
            mu=args.muon_mu,
            weight_decay=args.weight_decay,
            scalar_weight_decay=args.muon_scalar_weight_decay,
            scalar_betas=(args.muon_scalar_beta1, args.muon_scalar_beta2),
            scalar_epsilon=args.muon_scalar_eps,
            muon_epsilon=args.muon_epsilon,
            adjust_lr=None if args.muon_adjust_lr == "none" else args.muon_adjust_lr,
            compile_orthogonalization=args.muon_compile,
            distributed_orthogonalization=not args.muon_local_orthogonalization,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    scheduler = get_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_training_steps=args.warmup_iterations + args.measured_iterations,
        warmup_steps=min(10, args.warmup_iterations),
        min_lr_ratio=0.1,
    )
    ddp_options = {
        "device_ids": [local_rank],
        "output_device": local_rank,
        "broadcast_buffers": False,
    }
    if args.ddp_bucket_cap_mb is not None:
        ddp_options["bucket_cap_mb"] = args.ddp_bucket_cap_mb
    model = torch.nn.parallel.DistributedDataParallel(model, **ddp_options)
    hook_state = register_comm_hook_for_ddp_model(model, dist.group.WORLD, args)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if rank == 0:
        source_paths = [
            Path(__file__),
            Path(args.model_config),
            Path("optimizers/muon.py"),
            Path("optimizers/utils.py"),
            Path("comm_hooks/utils.py"),
            Path("comm_hooks/sparse_hook_c4.py"),
            Path("comm_hooks/group_topk_hook_no_reshape.py"),
        ]
        config = vars(args).copy()
        config.update(
            status="running",
            world_size=world_size,
            total_batch_size=args.batch_size * world_size,
            gradient_accumulation=1,
            parameter_count=parameter_count,
            model=model_config.to_dict(),
            gpu=torch.cuda.get_device_name(),
            torch_version=torch.__version__,
            git_head=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            measurement=(
                f"max_rank(continuous_{args.measured_iterations}_update_wall_time)"
                f"/{args.measured_iterations}"
            ),
            nccl_environment={
                key: os.environ.get(key)
                for key in (
                    "NCCL_P2P_DISABLE",
                    "NCCL_SHM_DISABLE",
                    "NCCL_CUMEM_HOST_ENABLE",
                    "NCCL_DEBUG",
                )
            },
            code_sha256={
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in source_paths
            },
        )
        (output_dir / "config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"INITIALIZED params={parameter_count} compressor={args.compressor}",
            flush=True,
        )

    step_seconds = []
    losses = []
    ddp_communication_bits = 0
    window_start = None
    total_iterations = args.warmup_iterations + args.measured_iterations
    for step in range(total_iterations):
        if step == args.warmup_iterations:
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            window_start = time.perf_counter()
            if rank == 0:
                print(f"MEASUREMENT_START update={step + 1}", flush=True)
        iteration_start = time.perf_counter()
        batch = next(batches)
        if batch["input_ids"].shape[0] != args.batch_size:
            raise RuntimeError("Partial batch during timing")
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        labels = batch["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100
        loss = model(**batch, labels=labels).loss
        loss.backward()
        if args.grad_clipping:
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clipping)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        ddp_communication_bits = accumulate_hook_communication_bits(
            hook_state, ddp_communication_bits
        )
        torch.cuda.synchronize()
        iteration_end = time.perf_counter()
        if step >= args.warmup_iterations:
            step_seconds.append(iteration_end - iteration_start)
            losses.append(loss.detach().float().cpu())
        elif rank == 0 and (step == 0 or (step + 1) % 25 == 0):
            print(f"WARMUP update={step + 1} loss={loss.item():.6f}", flush=True)

    elapsed = iteration_end - window_start
    local_losses = torch.stack(losses).tolist()
    if not all(math.isfinite(value) for value in local_losses):
        raise RuntimeError("Nonfinite loss during measurement")
    local_result = {
        "elapsed": elapsed,
        "steps": step_seconds,
        "losses": local_losses,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "ddp_communication_bits": ddp_communication_bits,
    }
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_result)
    if rank == 0:
        result = summarize_timing(
            [item["elapsed"] for item in gathered],
            [item["steps"] for item in gathered],
            args.measured_iterations,
        )
        result.update(
            status="completed",
            optimizer=args.optimizer,
            compressor=args.compressor,
            parameter_count=parameter_count,
            warmup_iterations=args.warmup_iterations,
            first_measured_update=args.warmup_iterations + 1,
            last_measured_update=total_iterations,
            rank_losses=[item["losses"] for item in gathered],
            rank_peak_allocated_bytes=[
                item["peak_allocated_bytes"] for item in gathered
            ],
            rank_peak_reserved_bytes=[item["peak_reserved_bytes"] for item in gathered],
            rank_ddp_communication_bits=[
                item["ddp_communication_bits"] for item in gathered
            ],
            muon_communication_bits=(
                optimizer.communication_bits_stats()
                if args.optimizer == "muon"
                else None
            ),
        )
        (output_dir / "all_results.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f'TIMING_COMPLETED mean_iteration_seconds={result["mean_iteration_seconds"]:.6f}',
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
