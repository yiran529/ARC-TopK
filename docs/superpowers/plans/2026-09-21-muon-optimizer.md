# Muon Optimizer Implementation Plan

> **For agentic workers:** Implement the tasks in order with a failing test before each production change.

**Goal:** Make Muon selectable beside Adam in all ARC-TopK-release training entries.

**Architecture:** A local PyTorch optimizer provides Muon matrix updates and AdamW scalar updates in one `Optimizer` instance. A small builder selects model parameters. Existing DDP hooks continue to own gradient synchronization.

**Tech Stack:** Python 3, PyTorch 2.7+; tests use pytest and CPU tensors.

**Spec:** `docs/superpowers/specs/2026-09-21-muon-optimizer-design.md`

## Global Constraints

- Preserve the current Adam and SGD branches.
- Do not change PyTorch, CUDA, NCCL, Triton, or virtual environments.
- Work in the current checkout because the user asked for the implementation here; preserve unrelated local changes.

---

### Task 1: Optimizer core

**Files:** Create `optimizers/muon.py`; test in `tests/test_muon.py`.

- [x] Test the momentum and Nesterov equation against hand calculated tensors; verify that the test fails before implementation.
- [x] Implement the Muon parameter group and Polar Express orthogonalization.
- [x] Test and implement AdamW fallback, convolution flattening, same-shape batching, distributed result AllGather, and state roundtrip.
- [x] Run `PYTHONPATH=. /home/wyr/dion/.venv/bin/python -m pytest tests/test_muon.py -q`.

### Task 2: Model parameter builder

**Files:** Create `optimizers/__init__.py`, `optimizers/utils.py`; test in `tests/test_muon.py`.

- [x] Test selection for linear, convolution, embedding, normalization, and output heads; verify failure first.
- [x] Implement `build_muon_optimizer(model, *, lr, scalar_lr, mu, weight_decay, scalar_weight_decay, adjust_lr, compile_orthogonalization)`.
- [x] Test that every trainable parameter appears once and schedules update each group.

### Task 3: Training entrypoints

**Files:** Modify `c4/run_llama_pretraining.py`, `glue_fine-tuning/run_glue_no_trainer_new.py`, `cifar10/run_cifar10.py`, `cifar10/run_cifar10_resnet50.py`, and `comm_hooks/utils.py`.

- [x] Add `muon` to optimizer validation and construction, with optional Muon flags and optimizer-aware logging.
- [x] Keep the compressor choice independent of the optimizer.
- [x] Run `python3 -m py_compile` for all modified entrypoints and the optimizer package.

### Task 4: Documentation and verification

**Files:** Update `README.md`; create `docs/worklog/M001-muon-baseline.md`.

- [x] Document a selectable Muon command and the gradient compression semantics.
- [x] Record implemented behavior, tests, and current limitations.
- [x] Run the focused tests and review the final diff for unrelated changes.
