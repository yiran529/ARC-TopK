# Sol mid review brief: integrated Muon sweep and Table III training

## Required behavior

Review `glue_fine-tuning/scripts/glue_muon_table3_sweep_then_formal.sh` as one
workflow. A normal invocation must:

1. Wait until all requested physical GPUs are below both the memory and
   utilization threshold.
2. Run one SST-2 Muon+dense sweep with exactly three paired settings:
   `0.02+0.001`, `0.002+0.0001`, and `0.0002+0.00005`.
3. Read `eval_accuracy` from the three completed `all_results.json` files.
4. Select the pair maximizing dense validation accuracy.
5. Run the formal eight-task, two-arm Table III experiment exactly once using
   that selected pair. It must not launch a second sweep or use a hard-coded
   formal LR pair.

`DRY_RUN=1` must print three sweep commands and sixteen formal commands without
creating outputs or contacting Hugging Face/W&B.

`SKIP_SWEEP=1` may reuse an existing `selected_config.json` to recover the
formal phase after an infrastructure or data-cache failure without repeating
the completed sweep.

## OOM and scheduling safeguards

- Default requested GPUs are physical `2,3,4,5`.
- The default wait condition is memory usage <20% of each card and utilization
  <20% on each card; the threshold, polling interval, timeout, GPU list, and
  wait bypass are configurable.
- SST-2 sweep uses local batch 8, gradient accumulation 2, and 5 epochs by
  default, preserving the
  paper's four-GPU global batch of 64 while reducing per-process peak memory.
- Formal runs use 10 epochs and retain the paper local batch sizes: 32 for
  CoLA/MRPC and 16 for other tasks by default. `FORMAL_BATCH_DIVISOR=2`
  together with the default accumulation setting gives a lower-memory,
  same-global-batch fallback; evaluation batch is explicitly bounded by 8.
- Compression warmup is computed dynamically as 1/10 of the 10-epoch formal
  optimizer-step count for each task; it is converted to communication-hook
  iterations when gradient accumulation is enabled. The 5-epoch sweep uses
  the same formal reference.
- Formal GLUE configs are prefetched in one process before distributed runs to
  avoid concurrent cache initialization failures.
- Runs retain logs, metrics, and selection manifests but do not save final
  model weights, checkpoints, or adapters unless explicitly requested.
- A failed sweep result or missing/non-numeric `eval_accuracy` must stop the
  workflow before formal training.
- `GPU_IDS` and `CUDA_VISIBLE_DEVICES` must agree exactly, so the waited cards
  are the cards used by Accelerate.

## Review questions

- Is dense SST-2 validation accuracy an appropriate selection metric for choosing
  one Muon pair for both formal arms?
- Are the three pairs kept paired rather than expanded into a Cartesian product?
- Does the selected pair flow from `selected_config.json` into every formal
  command without rerunning sweep?
- Are the paper's optimizer, batch, scheduler, compression, and one-seed
  settings preserved apart from the intended Muon LR sweep?
- Does the formal phase default to W&B online tracking while the short sweep
  remains untracked by default?
- Does the GPU wait happen before any model/data process starts, and do the
  defaults prevent an avoidable OOM?
- Are `DRY_RUN` and `.venv/bin/accelerate` behavior correct?
- Should formal runs also use accumulation to preserve global batch if the
  paper-local batch causes OOM?
