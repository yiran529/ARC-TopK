# Local C4 Raw Shards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download a fixed, revision-pinned subset of raw `allenai/c4` English JSON.GZ shards once and let every ARC-TopK C4 run read those local shards deterministically while continuing to tokenize with `t5-base` at runtime.

**Architecture:** Add a focused C4 data-source module that resolves either the existing Hub streaming source or a directory of local raw shards into a Hugging Face `IterableDataset`. Add a resumable downloader that selects the first 30 of the 1024 English training shards plus one validation shard, checks free space, downloads directly into a gitignored dataset directory, and records an auditable manifest. Local training repeats the finite shard stream when necessary so the existing `--max_train_tokens 3B` step budget cannot terminate early; all comparison runs use the same manifest and record order.

**Tech Stack:** Python 3.11, `datasets==3.6.0`, `huggingface-hub==0.33.0`, pytest supplied ephemerally with `uv run --with pytest`.

**Spec:** Approved inline requirement in the 2026-09-21 conversation, anchored by `AGENTS.md`: store a fixed local subset of raw C4 `.json.gz` shards, do not pre-tokenize, dynamically apply the existing T5 tokenizer during training, avoid repeat Hub downloads, and preserve existing baselines and user changes.

## Global Constraints

- Preserve all pre-existing modified and untracked files; do not revert, overwrite, stage, or commit the user's work.
- Keep the existing `--dataset_path allenai/c4` Hub-streaming behavior working.
- Local data files must be ignored by Git and must not be committed.
- Pin downloads to dataset revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`.
- Default selection is train shards `00000` through `00029` and validation shard `00000`.
- Download raw `.json.gz` only; do not tokenize or preprocess while downloading.
- Tests must not download C4 and must follow red-green TDD.
- Do not start GPU training.
- Do not delete caches or unrelated files to reclaim space.
- Do not create Git commits in this dirty worktree.

---

### Task 1: Resolve Hub and local C4 sources

**Files:**
- Create: `c4/pept_utils/c4_data.py`
- Modify: `c4/run_llama_pretraining.py`
- Create: `tests/test_c4_data.py`

**Interfaces:**
- Produces: `load_c4_split(dataset_path: str, split: str, *, repeat_local: bool = False) -> datasets.IterableDataset`.
- Local layout consumed: `<dataset_path>/en/c4-train.NNNNN-of-01024.json.gz` and `<dataset_path>/en/c4-validation.NNNNN-of-00008.json.gz`.
- The training entry calls this function with `repeat_local=True` for train and `False` for validation; Hub sources remain non-repeating.

- [ ] **Step 1: Write failing local-loader tests**

  Create real temporary gzip JSONL shards containing `text`, `timestamp`, and `url`. Assert that local train files are discovered in lexical order, examples are readable without network access, validation uses only validation files, a missing split raises `FileNotFoundError` naming the expected glob, and `repeat_local=True` cycles deterministically.

- [ ] **Step 2: Run the focused tests and confirm RED**

  Run: `uv run --with pytest pytest -q tests/test_c4_data.py`

  Expected: collection/import failure because `c4.pept_utils.c4_data` does not exist.

- [ ] **Step 3: Implement the minimal source resolver**

  For `dataset_path == "allenai/c4"`, call `datasets.load_dataset(dataset_path, "en", split=split, streaming=True)`. For an existing directory, sort the exact split glob, call `datasets.load_dataset("json", data_files={split: files}, split=split, streaming=True)`, and apply `.repeat()` only when requested. For any other value, preserve the legacy generic Hub/dataset-script behavior with `datasets.load_dataset(dataset_path, split=split, streaming=True)`.

- [ ] **Step 4: Integrate without changing tokenization**

  Remove the inline `load_c4_split` from `c4/run_llama_pretraining.py`, import the new function, call it with `repeat_local=True` for training, and leave the existing `t5-base` runtime tokenizer and `PreprocessedIterableDataset` unchanged.

- [ ] **Step 5: Verify GREEN and regression coverage**

  Run: `uv run --with pytest pytest -q tests/test_c4_data.py tests/test_c4_args.py`

---

### Task 2: Add a revision-pinned raw-shard downloader

**Files:**
- Create: `c4/scripts/download_c4_raw_shards.py`
- Create: `tests/test_download_c4_raw_shards.py`
- Modify: `.gitignore`

**Interfaces:**
- CLI: `python c4/scripts/download_c4_raw_shards.py --output-dir PATH [--num-train-shards 30] [--num-validation-shards 1] [--revision SHA] [--minimum-free-gib 4]`.
- Produces: raw files under `PATH/en/` and `PATH/manifest.json`.
- Manifest fields: repository, revision, config, selected train/validation filenames, requested counts, successful file sizes, total bytes, and completion status.

- [ ] **Step 1: Write failing pure-function and CLI validation tests**

  Test filename generation and bounds (`1..1024` train, `0..8` validation), disk-space rejection, resumability when an expected-size destination already exists, and manifest serialization. Use temporary local files and an injected download callable; never contact Hugging Face in tests.

- [ ] **Step 2: Run the focused tests and confirm RED**

  Run: `uv run --with pytest pytest -q tests/test_download_c4_raw_shards.py`

  Expected: collection/import failure because the downloader module does not exist.

- [ ] **Step 3: Implement the downloader**

  Use `huggingface_hub.hf_hub_download` with `repo_id="allenai/c4"`, `repo_type="dataset"`, the pinned revision, filenames such as `en/c4-train.00000-of-01024.json.gz`, and `local_dir=output_dir`. Validate requested counts before network access. Query remote file metadata to calculate required bytes, require that free space after the planned download remains at least `minimum_free_gib`, skip complete existing files, and write the manifest atomically only after all requested files succeed. A failed run must leave downloaded shards reusable and must not claim `complete: true`.

- [ ] **Step 4: Ignore only the default local dataset output**

  Add `/data/c4/` to `.gitignore`; do not broaden the rule to unrelated `data` directories.

- [ ] **Step 5: Verify GREEN**

  Run: `uv run --with pytest pytest -q tests/test_download_c4_raw_shards.py tests/test_c4_data.py tests/test_c4_args.py`

---

### Task 3: Document and launch the one-time download

**Files:**
- Modify: `README.md`
- Runtime output only: `data/c4/en-30-shards/download.log`
- Runtime output only: `data/c4/en-30-shards/manifest.json`

**Interfaces:**
- Documented training argument: `--dataset_path data/c4/en-30-shards --max_train_tokens 3B`.
- Download command writes raw compressed shards only and is safe to rerun.

- [ ] **Step 1: Add concise documentation**

  Explain the Hub-streaming option versus the local-raw option, the pinned revision, dynamic T5 tokenization, deterministic shared shard order, approximate 9.6 GB download size, and the fact that 30 shards may repeat if 3B nominal token slots outlast the finite local examples.

- [ ] **Step 2: Run non-network verification**

  Run:

  ```bash
  uv lock --check
  uv pip check --python .venv/bin/python
  uv run --with pytest pytest -q tests/test_c4_args.py tests/test_c4_data.py tests/test_download_c4_raw_shards.py
  .venv/bin/python -m py_compile c4/pept_utils/c4_data.py c4/scripts/download_c4_raw_shards.py c4/run_llama_pretraining.py
  ```

- [ ] **Step 3: Launch the download without polling**

  Create `data/c4/en-30-shards/`, then start one detached `tmux` session named `arc-c4-download` that executes the downloader with `--output-dir data/c4/en-30-shards`, redirecting stdout and stderr to `data/c4/en-30-shards/download.log`. If that exact tmux session already exists, do not start a duplicate. Report the session name and log path; do not wait for or poll it.

- [ ] **Step 4: Report workspace state**

  Show `git diff --stat` and `git status --short`, distinguish pre-existing changes from newly implemented files, and do not stage or commit anything.
