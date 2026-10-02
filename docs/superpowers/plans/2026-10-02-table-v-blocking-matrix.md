# Table V Blocking Communication Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate strict non-overlapped DDP communication and run 48 Muon timing cells spanning four communication settings, three model sizes, and four communication methods.

**Architecture:** Keep `table_v_timing.py` as the single measurement entry point. Centralize the optional device synchronization used by all DDP hooks, then use one serial controller to preserve commands, environments, failures, and results for every cell.

**Tech Stack:** Python 3.11, PyTorch 2.7.1 DDP/NCCL, Muon, C4, tmux.

**Spec:** `docs/table-v-muon-blocking-matrix.md`

## Global Constraints

- Models are LLaMA 60M, 130M, and 350M; methods are Dense, TopK, RandK, and ARC-TopK.
- Use FP32, Muon, batch size 1 per GPU, sequence length 256, 100 warmup steps, and 50 measured steps.
- Run the 48 cells serially and retain failed cells; do not poll GPU availability.
- Compression starts at step 100 with ratio 0.2, EF14, tensor sparsity, no gradual compression, and ARC rank 4.
- Socket modes use loopback NCCL Socket with one thread/socket and disable SHM and P2P.
- SHM modes enable SHM, disable P2P, and use one NCCL channel/CTA.
- Do not run the repository test suite; use syntax, dry-run matrix, and diff checks only.

---

### Task 1: Review and complete strict blocking semantics

**Files:**
- Modify: `comm_hooks/utils.py`
- Modify: `comm_hooks/default_hooks.py`
- Modify: `comm_hooks/sparse_hook_c4.py`
- Modify: `comm_hooks/group_topk_hook_no_reshape.py`

**Interfaces:**
- Consumes: `HookState.blocking_communication: bool`.
- Produces: `synchronize_blocking_communication(state, tensor) -> None`, called immediately before every completed hook Future is returned.

- [ ] Confirm `dist.all_reduce(async_op=False)` calls `Work.wait()` but does not replace an explicit device-wide completion fence for this experiment.
- [ ] Move the CUDA synchronization helper into `comm_hooks/utils.py`.
- [ ] Apply the helper to Dense, TopK, RandK, and ARC-TopK completed-Future paths.
- [ ] Confirm pre-compression warmup routes through the same blocking Dense hook.
- [ ] Run `python -m py_compile` on the modified modules and restore generated tracked bytecode.

### Task 2: Define the experiment contract and 48-cell controller

**Files:**
- Create: `docs/table-v-muon-blocking-matrix.md`
- Create: `c4/scripts/run_table_v_muon_blocking_matrix.py`

**Interfaces:**
- Consumes: `c4/table_v_timing.py --blocking_communication` and `--ddp_bucket_cap_mb`.
- Produces: CM058–CM105 artifacts, `manifest.json`, per-cell commands/environments/logs, `status.tsv`, and JSON/CSV summaries.

- [ ] Encode the four groups: ws4 bucket1024+SHM, ws4 blocking+SHM, ws8 blocking+SHM, ws8 blocking+Socket.
- [ ] Cross each group with 60M/130M/350M and Dense/TopK/RandK/ARC-TopK.
- [ ] Validate exact GPU count and perform one initial idle check without polling.
- [ ] Run every cell serially with a two-hour timeout and continue after failures.
- [ ] Run controller `--dry-run` and verify 48 unique run IDs and expected flags.

### Task 3: Launch and record the experiment

**Files:**
- Runtime artifacts: `output/CM058-CM105-table-v-muon-blocking-matrix/`

**Interfaces:**
- Consumes: the validated controller and idle GPUs 0–7.
- Produces: an asynchronous tmux run that requires no polling.

- [ ] Run `git diff --check` and inspect the final diff.
- [ ] Check GPUs 0–7 once for active compute processes and available memory.
- [ ] Start the controller in a detached tmux session.
- [ ] Read the initial status once to verify CM058 was accepted by the controller.
