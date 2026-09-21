# M002：ARC-TopK 与 Muon 组合

## 2026-09-22：正式实验队列

### 目的与实现口径

验证 ARC-TopK 能否降低 Muon 预训练中的梯度通信量，同时保持接近 Dense Muon
的 C4 validation perplexity。ARC-TopK 在 DDP gradient hook 中压缩并同步梯度，
Muon 随后对压缩后的全局梯度执行正交化；由于正交化是非线性的，该组合属于近似
Muon，而不是 Dense Muon 的通信等价实现。

### 配置

- `CM003-m002-arctopk-muon-llama60m-c4-1p1b-ws4-s1243`：60M，8,393
  optimizer updates，microbatch 32，gradient accumulation 4。
- `CM005-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243`：130M，16,785
  optimizer updates，microbatch 16，gradient accumulation 8。
- 两个实验均使用 global batch 512、sequence length 256、seed 1243、FP32、
  Muon matrix LR 0.02、scalar AdamW LR 0.001、weight decay 0、warmup 1,000。
- ARC-TopK 从 update 1,000 开始，`compress_ratio=0.2`、`r=4`、error feedback
  使用 `ef14`；每个 optimizer update 只触发一次 DDP gradient hook。
- 使用固定本地 C4 数据和 8 个 validation shards，最终评估约 10M effective
  prediction tokens；默认不保存 checkpoint。

### 执行

由 `c4/scripts/run_cm002_cm005_serial.sh` 在后台等待 4 张利用率不超过 50%、
空闲显存不少于 14 GiB 的 GPU，并在选定 GPU 上依次执行 CM002 至 CM005。
任何单项实验失败都不阻止后续实验运行。状态写入
`output/CM002-CM005-serial/status.tsv`，各实验保留独立日志和 W&B run。
