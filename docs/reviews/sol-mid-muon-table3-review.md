# Sol mid review brief: Muon reproduction of ARC-TopK Table III

## Purpose

This experiment imitates the RoBERTa-base GLUE experiment in Table III of
[ARC-Top-K, arXiv:2510.26709](https://arxiv.org/pdf/2510.26709v3), while replacing
the paper's Adam optimizer with the repository's Muon implementation.

Because of resource limits, the experiment uses one seed and two arms for each
of the eight GLUE tasks:

1. Muon + dense gradient communication;
2. Muon + ARC-TopK gradient communication.

The script is
`glue_fine-tuning/scripts/glue_muon_table3.sh`. It supports `DRY_RUN=1`, which
prints all 16 commands without starting training.

## Intended configuration

| Item | Value |
|---|---|
| Model | RoBERTa-base |
| Tasks | CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE |
| Max sequence length | 512 |
| Epochs | 30 |
| LR schedule | Linear |
| Weight decay | 0 |
| Seed | `1240` by default |
| Matrix Muon LR | `0.02` |
| Scalar AdamW fallback LR | `0.001` |
| Muon momentum | `0.95` |
| Muon LR adjustment | `spectral_norm` |
| ARC-TopK ratio | `0.2` |
| ARC-TopK projection rank | `r=4` |
| ARC error feedback | `ef21` |
| Compression start | iteration `1000` |
| Local batch size | 32 for CoLA/MRPC; 16 otherwise |

## Points for review

- Confirm that `--learning_rate=0.02` is the matrix-parameter LR and
  `--muon_scalar_lr=0.001` is the scalar AdamW-fallback LR from the supplied
  Muon configuration.
- Confirm that the per-device batch sizes match the paper's stated local batch
  sizes under four processes.
- Confirm that the dense arm uses `compressor=none` and `noef`, while the ARC arm
  uses `group_topk_no_reshape`, `ef21`, `compress_ratio=0.2`, and `r=4`.
- Confirm that `start_compress_iter=1000` is the intended interpretation of the
  paper's statement that transformer compression begins after 1000 iterations.
- Confirm that one seed is reported as a single-run result, not as a mean or
  standard deviation.
- Confirm that the image-provided Muon settings are appropriate for this
  fine-tuning run: matrix LR `0.02`, scalar LR `0.001`, momentum `0.95`, and
  epsilon `1e-8`.
- Note that the repository's Muon is a hybrid optimizer: embeddings, heads,
  normalization parameters, and biases use AdamW fallback.
- Note that ARC-TopK is applied before Muon's nonlinear orthogonalization, so
  ARC-TopK + Muon is an approximate Muon variant. Muon's distributed
  orthogonalization can also add AllGather traffic beyond the compressor's
  gradient communication.

## Non-training checks

Before launching any run, execute:

```bash
bash -n glue_fine-tuning/scripts/glue_muon_table3.sh
DRY_RUN=1 bash glue_fine-tuning/scripts/glue_muon_table3.sh
```

The dry run should print exactly 16 training commands and must not create model
outputs or contact the Hugging Face/W&B services.
