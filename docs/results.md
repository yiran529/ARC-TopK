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

# GLUE Table III 风格的 Muon 实验结果

## 1. 实验矩阵

实验编号：`CM008-muon-glue-table3-seed1240`。以 RoBERTa-base 在 GLUE 八个任务上比较 Dense Muon 与 ARC-TopK + Muon；每个完整 run 计划训练 10 epochs。SST-2 ARC-TopK 在第 9 个 epoch 尚未完成时被外部中止。

| 数据集 | Dense Muon | ARC-TopK + Muon | 训练状态 |
| --- | --- | --- | --- |
| CoLA | 10 epochs | 10 epochs | 两组完成 |
| SST-2 | 10 epochs | 中止于 epoch 9 训练期间 | ARC 仅有 epoch 8 最后一次完整验证 |
| MRPC | 10 epochs | 10 epochs | 两组完成 |
| STS-B | 10 epochs | 10 epochs | 两组完成 |
| QQP | 10 epochs | 10 epochs | 两组完成 |
| MNLI | 10 epochs | 10 epochs | 两组完成 |
| QNLI | 10 epochs | 10 epochs | 两组完成 |
| RTE | 10 epochs | 10 epochs | 两组完成 |

## 2. 核心设置

- 模型与数据：`FacebookAI/roberta-base`，GLUE 官方 validation；`max_length=512`，seed `1240`，FP32（mixed precision 关闭），线性学习率调度。
- Muon：matrix LR `0.0002`、scalar LR `0.00005`、momentum `0.95`、spectral-norm scaling；weight decay 为 0。
- LR 选择：先在 SST-2 上对 Dense Muon 做 5-epoch sweep，以 validation accuracy 选择配置。三组候选 `(matrix LR, scalar LR)` 与最终 validation accuracy 为 `(0.02, 0.001) → 50.917%`、`(0.002, 0.0001) → 90.252%`、`(0.0002, 0.00005) → 95.183%`。选中配置复用于正式 runs；SST-2 validation 因此参与了超参数选择。
- ARC-TopK：`group_topk_no_reshape`，EF21，`compress_ratio=0.2`、`r=4`、从 step 0 开始压缩，compression warmup fraction `0.1`；按本实验设置压缩全部梯度。
- 硬件：4 张 NVIDIA RTX 4090，物理 GPU 4–7。有效 global batch 的历史设置并非所有任务完全一致：CoLA/MRPC 为 `16×4×2=128`；SST-2/STS-B/QQP 为 `8×4×2=64`；MNLI/QNLI/RTE 为 `16×4×1=64`。MNLI 起已按 GA=1 且 global batch=64 执行。
- 运行脚本：[glue_muon_table3_sweep_then_formal.sh](../glue_fine-tuning/scripts/glue_muon_table3_sweep_then_formal.sh)。未保存模型 checkpoint 或 adapter。

## 3. 最终验证结果

指标以百分制分数列示；差值为 ARC-TopK + Muon 减 Dense Muon，单位为百分点。表中完整 run 的数值来自各自 `all_results.json`，对应最后一次验证（epoch 9，即第 10 个 epoch 结束后）；SST-2 ARC 使用最后一个已完成验证的 epoch 8，单独标为部分结果。MNLI 指标使用 `validation_mismatched`，其他任务使用 `validation`。

| 数据集 | 指标 | Dense Muon | ARC-TopK + Muon | 差值（百分点） | split / 状态 |
| --- | --- | ---: | ---: | ---: | --- |
| CoLA | Matthews corr. | [57.820](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/cola/muon_dense/all_results.json) | [57.064](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/cola/muon_arctopk/all_results.json) | -0.757 | validation，完成 |
| SST-2 | Accuracy | [94.839](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/sst2/muon_dense/all_results.json) | [94.495](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/sst2/muon_arctopk/all_results.json) | — | validation；ARC 部分结果，epoch 8 |
| MRPC | Accuracy / F1 | [84.069 / 88.656](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/mrpc/muon_dense/all_results.json) | [84.804 / 89.199](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/mrpc/muon_arctopk/all_results.json) | +0.735 / +0.542 | validation，完成 |
| STS-B | Pearson / Spearman | [89.823 / 89.590](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/stsb/muon_dense/all_results.json) | [89.582 / 89.435](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/stsb/muon_arctopk/all_results.json) | -0.241 / -0.155 | validation，完成 |
| QQP | Accuracy / F1 | [91.848 / 89.261](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/qqp/muon_dense/all_results.json) | [91.684 / 89.054](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/qqp/muon_arctopk/all_results.json) | -0.163 / -0.207 | validation，完成 |
| MNLI | Accuracy | [87.215](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/mnli/muon_dense/all_results.json) | [86.890](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/mnli/muon_arctopk/all_results.json) | -0.325 | validation_mismatched，完成 |
| QNLI | Accuracy | [92.532](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/qnli/muon_dense/all_results.json) | [92.532](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/qnli/muon_arctopk/all_results.json) | +0.000 | validation，完成 |
| RTE | Accuracy | [68.953](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/rte/muon_dense/all_results.json) | [67.870](../outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/rte/muon_arctopk/all_results.json) | -1.083 | validation，完成 |

## 4. 结论

- 七个任务的两种方法均有完整结果；其中 MRPC 的 ARC-TopK + Muon 分数较高，QNLI 相同，其余五个完整任务的 ARC 分数较低。SST-2 ARC 只完成到 epoch 8，不能作为同预算正式对照。
- 这是单 seed（`n=1`）的初步复现。CoLA/MRPC 的 effective global batch 为 128，而其他任务为 64；结果不应被解读为所有任务严格使用同一 batch 口径的成套比较。
- 学习率由 SST-2 validation sweep 选出，且该配置随后也用于 SST-2 正式结果；结果带有 validation 调参影响。模型 checkpoint 未保存，无法从本次产物恢复最优 epoch。
- 这里微调的结果比论文中的adam实验差可能是因为：
  - 用muon微调adam预训练的模型效果会比较差
  - 没有仔细sweep lr等超参数