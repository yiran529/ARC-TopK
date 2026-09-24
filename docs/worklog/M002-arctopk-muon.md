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

## 2026-09-22：CM003、CM005 启动失败后的重跑

- 首次串行队列中，CM003 和 CM005 均在约 2 秒内以退出码 2 结束，未进入训练。
  原因是 `torchrun` 将训练脚本参数 `--r 4` 误判为自身选项缩写，报
  `ambiguous option: --r`。原始失败日志保留在对应实验目录。
- 在 `torchrun` 选项与训练脚本路径之间增加 `--` 参数分隔符；不修改训练算法或
  超参数。使用 `arc_rerun` 模式仅重跑 CM003 和 CM005，并为新运行目录追加
  `-rerun1`，避免覆盖原始失败证据。
- 新队列状态记录在 `output/CM003-CM005-arc-rerun/status.tsv`。继续使用后台
  GPU 低占用等待与无 gate 串行执行，单项失败不阻止另一项启动。

## 2026-09-24：CM008 GLUE Table III 风格比较

- 目的：在 RoBERTa-base 的 GLUE 八个任务上比较 Dense Muon 与 ARC-TopK + Muon；
  seed 1240，每个完整 run 计划 10 epochs。
- SST-2 Dense Muon 5-epoch validation sweep 在三组学习率中选出 matrix LR
  `0.0002`、scalar LR `0.00005`，并用于正式任务。正式设置包含 Muon momentum
  `0.95`、spectral-norm scaling；ARC 使用 `group_topk_no_reshape`、EF21、
  `compress_ratio=0.2`、`r=4`、start step 0、compression warmup fraction `0.1`。
- 结果目录：`outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/`；
  汇总见 `docs/results.md` 的 CM008 部分。15/16 runs 完成；SST-2 ARC-TopK 在
  epoch 8 验证后、epoch 9 训练期间被外部中止，保留为部分结果。
- 批大小口径：CoLA/MRPC 日志记录 effective global batch 128；SST-2、STS-B、QQP
  为 64；MNLI、QNLI、RTE 按后续修正使用 GA=1、每卡 batch 16，effective global
  batch 64。结果为单 seed，且学习率由 SST-2 validation 指标选择。
- 观察：在完整结果中 ARC-TopK + Muon 于 MRPC 两项指标较高、QNLI accuracy 相同，
  其余任务低于 Dense Muon；SST-2 ARC 结果不作为同预算结论。模型 checkpoint 未保存。
