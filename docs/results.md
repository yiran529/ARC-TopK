# C4 LLaMA 预训练结果

## 1. 实验矩阵

使用 C4 英文数据，比较 Dense Muon、Dense AdamW 和 ARC-TopK + Muon。所有实验
使用 seed 1243，4 张 RTX 4090。

| 实验 | 模型 | 优化器/通信 | 更新步数 |
| --- | --- | --- | ---: |
| CM001 | LLaMA 60M | Dense Muon | 8,393 |
| CM002 | LLaMA 60M | Dense AdamW | 8,393 |
| CM003-rerun1 | LLaMA 60M | ARC-TopK + Muon | 8,393 |
| CM004 | LLaMA 130M | Dense Muon | 16,785 |
| CM005-rerun1 | LLaMA 130M | ARC-TopK + Muon | 16,785 |

CM003 和 CM005 的首次启动因 `torchrun` 参数解析问题失败，修复后重跑成功。

## 2. 核心设置

- 数据集：C4 英文数据，本地分片目录为 `/dev/shm/wyr_tmp/c4/en-30-shards`。
- 序列长度：256；全局 batch size：512；训练 dtype：FP32；weight decay：0；
  gradient clipping：1.0；cosine schedule；warmup 1,000 步。
- 60M：microbatch 32、gradient accumulation 4。
- 130M：microbatch 16、gradient accumulation 8。
- Muon：matrix LR `0.02`、scalar AdamW LR `0.001`、momentum `0.95`、
  spectral-norm scaling。
- AdamW：LR `0.002`，betas `(0.9, 0.999)`。
- ARC-TopK：`compress_ratio=0.2`、`r=4`、`ef14`；完成前 1,000 个 update 后开始压缩。
- 运行脚本：
  - `c4/scripts/run_cm001_muon_60m_c4.sh`
  - `c4/scripts/run_cm002_cm005_serial.sh`

## 3. 最终验证结果

| 实验 | Eval loss | PPL |
| --- | ---: | ---: |
| CM001 60M Dense Muon | 3.419215 | 30.545439 |
| CM002 60M Dense AdamW | 3.401769 | 30.017143 |
| CM003-rerun1 60M ARC+Muon | 3.424462 | 30.706118 |
| CM004 130M Dense Muon | 3.144573 | 23.209753 |
| CM005-rerun1 130M ARC+Muon | 3.427855 | 30.810485 |

结论：

- 60M ARC+Muon 与 Dense Muon 接近，PPL 高 `0.160679`（约 `0.53%`）。
- 130M ARC+Muon 明显落后于 Dense Muon，PPL 高 `7.600732`（约 `32.75%`）。
- 当前 ARC 配置在 60M 上基本可用，但不能直接扩展到 130M；130M 需要进一步调节
  压缩比例、投影 rank 或压缩起始步数。

以上结果均为单 seed，属于初步比较。

# CIFAR-10 ResNet-18/50 实验结果

## 1. 实验矩阵

复现论文 Table II 的 CIFAR-10 设置；每个配置训练 200 个 epoch，seed 为 1410。

| 实验 | 模型 | 优化器/通信 | 训练预算 |
| --- | --- | --- | ---: |
| CM006 | ResNet-18 | Dense Adam（脚本选项 `adamw`，实际调用 `torch.optim.Adam`） | 200 epochs |
| CM006 | ResNet-18 | Dense Muon | 200 epochs |
| CM006 | ResNet-18 | ARC-TopK + EF14 + Muon | 200 epochs |
| CM007 | ResNet-50 | Dense Adam（脚本选项 `adamw`，实际调用 `torch.optim.Adam`） | 200 epochs |
| CM007 | ResNet-50 | Dense Muon | 200 epochs |
| CM007 | ResNet-50 | ARC-TopK + EF14 + Muon | 200 epochs |

## 2. 核心设置

- 数据集：CIFAR-10；训练集用于训练，`train=False` 的官方 10,000 张 test set 用于评估。
- 硬件：4 张 NVIDIA RTX 4090（GPU 2–5）。
- 共同设置：per-device batch size `16`、weight decay `5e-4`、seed `1410`、学习率 warmup 为总 epoch 数的 10%、ARC 压缩起始迭代 `1000`、`compress_ratio=0.2`、投影 rank `4`。
- Dense Adam：LR `1e-3`。
- Muon：matrix LR `0.02`、momentum `0.95`、scalar LR `1e-3`、spectral-norm scaling。
- ARC-TopK + Muon：在上述 Muon 设置上使用 `group_topk_no_reshape` 和 `ef14`。
- 运行脚本：
  - [cm001_table2_r18_pilot.sh](../cifar10/scripts/cm001_table2_r18_pilot.sh)
  - [cm007_table2_r50_gpu2_5.sh](../cifar10/scripts/cm007_table2_r50_gpu2_5.sh)

## 3. 结果

主报告指标为 200 个 epoch 内最高 test accuracy；同时列出最后一轮 test accuracy。未保存 checkpoint，因此 best 仅表示评估指标峰值，不对应可恢复的 best checkpoint。

| 实验 | 方法 | Best test acc | Best epoch | Final test acc | split | 原始日志 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| CM006 | ResNet-18 Dense Adam | 93.530% | 193 | 93.340% | test | [log](../output/CM006-table2-r18-gpu2-5/01-dense-adam.log) |
| CM006 | ResNet-18 Dense Muon | 95.100% | 191 | 95.050% | test | [log](../output/CM006-table2-r18-gpu2-5/02-dense-muon.log) |
| CM006 | ResNet-18 ARC-TopK + EF14 + Muon | 95.020% | 192 | 94.930% | test | [log](../output/CM006-table2-r18-gpu2-5/03-arctopk-ef14-muon.log) |
| CM007 | ResNet-50 Dense Adam | 93.800% | 198 | 93.590% | test | [log](../output/CM007-table2-r50-gpu2-5/01-dense-adam.log) |
| CM007 | ResNet-50 Dense Muon | 96.180% | 196 | 96.060% | test | [log](../output/CM007-table2-r50-gpu2-5/02-dense-muon.log) |
| CM007 | ResNet-50 ARC-TopK + EF14 + Muon | 95.880% | 188 | 95.790% | test | [log](../output/CM007-table2-r50-gpu2-5/03-arctopk-ef14-muon.log) |

## 4. 结论

- 在 ResNet-18 上，ARC-TopK + EF14 + Muon 比 Dense Muon 低 `0.080` 个百分点，比 Dense Adam 高 `1.490` 个百分点。
- 在 ResNet-50 上，ARC-TopK + EF14 + Muon 比 Dense Muon 低 `0.300` 个百分点，比 Dense Adam 高 `2.080` 个百分点。
- 所有 CIFAR-10 结果均为单 seed（`n=1`），属于初步比较；当前未报告多 seed 离散度。
