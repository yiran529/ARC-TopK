# Sol mid review brief: integrated Muon sweep and Table III training

## Required behavior

Review `glue_fine-tuning/scripts/glue_muon_table3_sweep_then_formal.sh` as one
workflow. A normal invocation must:

1. Wait until all requested physical GPUs are below both the memory and
   utilization threshold.
2. Run one SST-2 sweep with exactly three paired settings:
   `0.02+0.001`, `0.002+0.0001`, and `0.0002+0.00005`.
3. Run both Muon+dense and Muon+ARC-TopK for each pair.
4. Read `eval_accuracy` from the six completed `all_results.json` files.
5. Select the pair maximizing the mean of dense and ARC-TopK accuracy, with ARC
   accuracy and then dense accuracy as tie-breakers.
6. Run the formal eight-task, two-arm Table III experiment exactly once using
   that selected pair. It must not launch a second sweep or use a hard-coded
   formal LR pair.

`DRY_RUN=1` must print six sweep commands and sixteen formal commands without
creating outputs or contacting Hugging Face/W&B.

## OOM and scheduling safeguards

- Default requested GPUs are physical `2,3,4,5`.
- The default wait condition is memory usage <20% of each card and utilization
  <20% on each card; the threshold, polling interval, timeout, GPU list, and
  wait bypass are configurable.
- SST-2 sweep uses local batch 8 and gradient accumulation 2, preserving the
  paper's four-GPU global batch of 64 while reducing per-process peak memory.
- Formal runs retain the paper local batch sizes: 32 for CoLA/MRPC and 16 for
  other tasks by default. `FORMAL_BATCH_DIVISOR=2` together with the default
  accumulation setting gives a lower-memory, same-global-batch fallback;
  evaluation batch is explicitly bounded by 8.
- A failed sweep result or missing/non-numeric `eval_accuracy` must stop the
  workflow before formal training.
- `GPU_IDS` and `CUDA_VISIBLE_DEVICES` must agree exactly, so the waited cards
  are the cards used by Accelerate.

## Review questions

- Is the selection metric (mean of dense and ARC-TopK SST-2 accuracy) appropriate
  for choosing one Muon pair for both formal arms?
- Are the three pairs kept paired rather than expanded into a Cartesian product?
- Does the selected pair flow from `selected_config.json` into every formal
  command without rerunning sweep?
- Are the paper's optimizer, batch, scheduler, compression, and one-seed
  settings preserved apart from the intended Muon LR sweep?
- Does the GPU wait happen before any model/data process starts, and do the
  defaults prevent an avoidable OOM?
- Are `DRY_RUN` and `.venv/bin/accelerate` behavior correct?
- Should formal runs also use accumulation to preserve global batch if the
  paper-local batch causes OOM?
