# C4 LLaMA 预训练结果

记录更新：2026-10-02（北京时间）。数值为本仓库实际运行结果，论文结果不纳入此表。

## 1. 实验矩阵

使用 C4 英文数据，比较 Dense Muon、Dense AdamW 和压缩梯度后的 Muon。
所有已完成实验使用 seed 1243，4 张 RTX 4090；有损压缩后再正交化属于近似 Muon。

| 实验 | 模型 | 优化器/通信 | 训练预算（updates） | 状态 |
| --- | --- | --- | ---: | --- |
| CM001 | LLaMA 60M | Dense Muon，matrix LR 0.02 | 8,393 | 完成 |
| CM002 | LLaMA 60M | Dense AdamW | 8,393 | 完成 |
| CM003-rerun1 | LLaMA 60M | ARC-TopK + Muon，matrix LR 0.02 | 8,393 | 完成 |
| CM004 | LLaMA 130M | Dense Muon，matrix LR 0.02 | 16,785 | 完成 |
| CM005-rerun1 | LLaMA 130M | ARC-TopK + Muon，matrix LR 0.02 | 16,785 | 完成 |
| CM014 | LLaMA 130M | Dense Muon，matrix LR 0.01 | 20,000 | 完成 |
| CM015 | LLaMA 130M | ARC-TopK + Muon，matrix LR 0.01 | 20,000 | 完成 |
| CM016 | LLaMA 60M | Top-K + EF14 + Muon，matrix LR 0.02 | 8,393 | 完成 |
| CM017 | LLaMA 60M | Rand-K + EF14 + Muon，matrix LR 0.02 | 8,393 | 完成 |
| CM018 | LLaMA 130M | Top-K + EF14 + Muon，matrix LR 0.01 | 20,000 | 完成 |
| CM019 | LLaMA 130M | Rand-K + EF14 + Muon，matrix LR 0.01 | 20,000 | 完成 |

CM003/CM005 首次启动因 `torchrun` 参数解析问题失败，修复后重跑成功。
CM013 以及原 16,785 步的 CM014/CM015 队列均在训练前取消并被替代，不计为完成结果。
CM014–CM019 均有成功完成标记和 `exit_code=0`；CM016–CM019 队列于
2026-10-02 11:49（北京时间）结束，`finished_failures=0`。
这里的 Rand-K 是随机稀疏压缩，不是低秩 Rank-k。

## 2. 核心设置

- 数据集：C4 英文；旧实验使用 `/dev/shm/wyr_tmp/c4/en-30-shards`。
  CM014–CM019 从 `/home/wyr/greedy_lore/c4/c4_en/en` 引用前 30 个训练分片及全部
  8 个验证分片。旧文件已不可用，只能确认分片编号范围一致，无法核验文件内容相同。
- 序列长度 256；global batch 512；FP32；weight decay 0；gradient clipping 1.0；
  cosine schedule，warmup 1,000 步。60M 每卡 batch 32、GA 4；130M 每卡 batch 16、GA 8。
- Muon：momentum `0.95`、spectral-norm scaling、scalar AdamW LR `0.001`；matrix LR
  如矩阵所示。Dense AdamW：LR `0.002`，betas `(0.9, 0.999)`。
- ARC-TopK：`compress_ratio=0.2`、`r=4`、EF14，从梯度同步迭代 1,000 开始压缩。
  Top-K/Rand-K：tensor 级、比例 `0.2`、EF14，同样从迭代 1,000 开始；
  `--disable_compression_warmup` 关闭渐进压缩。
- CM014–CM019 实际使用 GPU 0–3。最终评估为 validation 上约 10M 有效预测 token，
  loss 按有效预测 token 加权，PPL 使用入口保存的 `exp(loss)`；未保存模型 checkpoint。
- 运行脚本：[CM001](../c4/scripts/run_cm001_muon_60m_c4.sh)、
  [CM002–CM005](../c4/scripts/run_cm002_cm005_serial.sh)、
  [CM014–CM015](../c4/scripts/run_cm014_cm015_muon_130m_lr001_c4.sh)、
  [CM016–CM019](../c4/scripts/run_cm016_cm019_sparse_muon_c4.sh)。

## 3. 最终验证结果

来源为各运行的 `all_results.json`，原始键为 `final_eval_loss` 和
`final_eval_perplexity`；均为 validation、final。最终步数与完成状态由同目录训练日志
的 `Reached max number of update steps`、`Script finished successfully` 和队列状态核对。
60M 最终评估为 10,020,666 个有效预测 token，130M 为 10,008,907 个；
同规模表内运行的评估 token 数一致。

| 实验 | 方法 | Eval loss | PPL | final update | 结果来源 |
| --- | --- | ---: | ---: | ---: | --- |
| CM001 | 60M Dense Muon | 3.419215 | 30.545439 | 8,393 | [JSON](../output/CM001-muon-dense-llama60m-c4-1p1b-ws4-s1243/all_results.json) |
| CM002 | 60M Dense AdamW | 3.401769 | 30.017143 | 8,393 | [JSON](../output/CM002-dense-adam-llama60m-c4-1p1b-ws4-s1243/all_results.json) |
| CM003-rerun1 | 60M ARC+Muon | 3.424462 | 30.706118 | 8,393 | [JSON](../output/CM003-m002-arctopk-muon-llama60m-c4-1p1b-ws4-s1243-rerun1/all_results.json) |
| CM004 | 130M Dense Muon | 3.144573 | 23.209753 | 16,785 | [JSON](../output/CM004-m001-dense-muon-llama130m-c4-2p2b-ws4-s1243/all_results.json) |
| CM005-rerun1 | 130M ARC+Muon | 3.427855 | 30.810485 | 16,785 | [JSON](../output/CM005-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243-rerun1/all_results.json) |
| CM014 | 130M Dense Muon，LR 0.01 | 3.076967 | 21.692504 | 20,000 | [JSON](../output/CM014-m001-dense-muon-llama130m-c4-2p62b-ws4-s1243-lr001/all_results.json) |
| CM015 | 130M ARC+Muon，LR 0.01 | 3.088616 | 21.946693 | 20,000 | [JSON](../output/CM015-m002-arctopk-muon-llama130m-c4-2p62b-ws4-s1243-lr001/all_results.json) |
| CM016 | 60M Top-K+Muon | 3.415821 | 30.441934 | 8,393 | [JSON](../output/CM016-m003-topk-muon-llama60m-c4-1p1b-ws4-s1243-lr002/all_results.json) |
| CM017 | 60M Rand-K+Muon | 3.643432 | 38.222805 | 8,393 | [JSON](../output/CM017-m003-randk-muon-llama60m-c4-1p1b-ws4-s1243-lr002/all_results.json) |
| CM018 | 130M Top-K+Muon，LR 0.01 | 3.082575 | 21.814495 | 20,000 | [JSON](../output/CM018-m003-topk-muon-llama130m-c4-2p62b-ws4-s1243-lr001/all_results.json) |
| CM019 | 130M Rand-K+Muon，LR 0.01 | 3.155567 | 23.466328 | 20,000 | [JSON](../output/CM019-m003-randk-muon-llama130m-c4-2p62b-ws4-s1243-lr001/all_results.json) |

## 4. 结论

- 60M、LR 0.02、8,393 步：ARC+Muon 相对 Dense Muon 的 PPL 高 `0.160679`
  （约 `0.53%`）。
- 130M、LR 0.02、16,785 步：ARC+Muon 的 PPL 高 `7.600732`（约 `32.75%`）。
- 新增 130M、LR 0.01、20,000 步对照：ARC+Muon 的 loss 高 `0.011650`，
  PPL 高 `0.254189`（约 `1.17%`），在本次配置下与 Dense Muon 接近。
- 同预算 130M 压缩器对照：Top-K 相对 CM014 Dense 的 loss 高 `0.005608`、
  PPL 高 `0.121990`（`0.56%`）；Rand-K 的 loss 高 `0.078600`、PPL 高
  `1.773823`（`8.18%`）。Top-K 的 PPL 比 CM015 ARC 低 `0.132199`，
  本次运行排序为 Dense、Top-K、ARC-TopK、Rand-K。
- 60M Top-K 的 PPL 比同数据、同预算 Rand-K 低 `7.780871`。相对历史 CM001
  Dense，Top-K 的 PPL 低 `0.103505`（`0.34%`），Rand-K 高 `7.677366`
  （`25.13%`）；与历史 Dense/ARC 的比较因旧数据内容无法核验，只作描述性参考。
- 新旧 130M 实验同时改变 matrix LR 和训练预算，且数据文件内容无法跨目录核验；
  不能把改善单独归因于 LR，也不能据此断言所有 130M ARC 配置均能保持质量。
- 均为单 seed（`n=1`）的初步结果，不据此判断稳定优劣。相同保留比例不代表相同
  实际通信量，本表仅比较固定训练预算下的验证质量。下一步补独立 seed，必要时在
  新数据路径重跑 60M Dense/ARC；分离 LR 与训练预算影响仍需独立对照。

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
| CM008-CIFAR | ResNet-18 | Top-K / Rand-K + EF14 + Muon | 各 200 epochs |
| CM008-CIFAR | ResNet-50 | Top-K / Rand-K + EF14 + Muon | 各 200 epochs |

新增运行的目录名为 `CM008-table2-randk-topk-muon-gpu0-1-2-7`；原有 GLUE 实验也使用 `CM008` 编号。这里以“CM008-CIFAR”区分两组已有运行，编号冲突待统一。

## 2. 核心设置

- 数据集：CIFAR-10；训练集用于训练，`train=False` 的官方 10,000 张 test set 用于评估。
- 硬件：每次运行使用 4 张 NVIDIA RTX 4090；CM006/CM007 的启动脚本指定 GPU 2–5。新增运行目录标识为 GPU 0、1、2、7。
- 共同设置：per-device batch size `16`、weight decay `5e-4`、seed `1410`、学习率 warmup 为总 epoch 数的 10%、压缩起始 hook 迭代 `1000`、`compress_ratio=0.2`。ARC-TopK 另使用投影 rank `4`。
- Dense Adam：LR `1e-3`。
- Muon：matrix LR `0.02`、momentum `0.95`、scalar LR `1e-3`、spectral-norm scaling。
- ARC-TopK + Muon：在上述 Muon 设置上使用 `group_topk_no_reshape` 和 `ef14`。
- 新增 Top-K/Rand-K + Muon：使用 `topk_sync` / `randk_sync`、`ef14` 和 tensor 级稀疏化；其余 Muon、batch、seed、训练轮数及压缩起始迭代与上表相同。四个运行启用 `--disable_compression_warmup`，从第 1,000 次 hook 迭代开始直接使用目标压缩比例；ARC hook 同样没有渐进压缩阶段。新增运行参数以各自本地 W&B `wandb-metadata.json` 的 `args` 字段为准，结果以训练日志为准。
- 运行脚本：
  - [cm001_table2_r18_pilot.sh](../cifar10/scripts/cm001_table2_r18_pilot.sh)
  - [cm007_table2_r50_gpu2_5.sh](../cifar10/scripts/cm007_table2_r50_gpu2_5.sh)
  - CM008-CIFAR 的独立启动脚本未在仓库中找到；四个实际运行的命令行参数保存在本地 W&B 元数据中。

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
| CM008-CIFAR | ResNet-18 Top-K + EF14 + Muon | 95.170% | 175 | 95.100% | test | [log](../output/CM008-table2-randk-topk-muon-gpu0-1-2-7/01-resnet18-topk-ef14-muon.log) |
| CM008-CIFAR | ResNet-18 Rand-K + EF14 + Muon | 95.130% | 180 | 95.070% | test | [log](../output/CM008-table2-randk-topk-muon-gpu0-1-2-7/02-resnet18-randk-ef14-muon.log) |
| CM008-CIFAR | ResNet-50 Top-K + EF14 + Muon | 95.590% | 199 | 95.550% | test | [log](../output/CM008-table2-randk-topk-muon-gpu0-1-2-7/03-resnet50-topk-ef14-muon.log) |
| CM008-CIFAR | ResNet-50 Rand-K + EF14 + Muon | 95.670% | 193 | 95.620% | test | [log](../output/CM008-table2-randk-topk-muon-gpu0-1-2-7/04-resnet50-randk-ef14-muon.log) |

## 4. 结论

- 在 ResNet-18 上，ARC-TopK + EF14 + Muon 比 Dense Muon 低 `0.080` 个百分点，比 Dense Adam 高 `1.490` 个百分点。
- 在 ResNet-50 上，ARC-TopK + EF14 + Muon 比 Dense Muon 低 `0.300` 个百分点，比 Dense Adam 高 `2.080` 个百分点。
- 新增 Top-K/Rand-K 两组在 ResNet-18 的 best test accuracy 分别为 `95.170%` / `95.130%`，在 ResNet-50 分别为 `95.590%` / `95.670%`。ResNet-50 两组均低于 Dense Muon 的 `96.180%`；不同压缩器的结果仅作为单 seed 描述性比较。
- 所有 CIFAR-10 结果均为单 seed（`n=1`），属于初步比较；当前未报告多 seed 离散度。
- “Best” 是同一 test set 上 200 次评估中的峰值，可能高估独立测试表现；最后一轮数值更适合作为固定训练预算的对照。

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

# GLUE Top-K / Rand-K + Muon 新增结果

## 1. 实验矩阵

2026-10-02 核验：以下 16 个 RoBERTa-base runs 均完成 10 epochs。
每个任务分别使用 Top-K + EF21 + Muon 和 Rand-K + EF21 + Muon；seed 1240。

| 运行组 | 任务 | 每种方法训练预算 | 结果状态 |
| --- | --- | --- | --- |
| 原 Top-K/Rand-K 组（目录无 CM 前缀） | CoLA、SST-2、MRPC、STS-B | 10 epochs | 8/8 完成 |
| CM009 | QQP | 10 epochs | 2/2 完成 |
| CM012 | MNLI、QNLI、RTE | 10 epochs | 6/6 完成 |

失败与中断记录：原 QQP Top-K 在 GA=2 时被停止；CM009、CM011 的 MNLI Top-K
日志均有 CUDA OOM。CM010 使用了随后被用户否决的 GPU 组合并被停止，CM011
MNLI Rand-K 随主机重启中断。以上无完整结果的目录不计入下表；CM012 从头重跑
MNLI/QNLI/RTE，编号中的 `resume` 不表示从 checkpoint 续训。

## 2. 核心设置

- 数据与模型：`FacebookAI/roberta-base`、GLUE，max length 512，FP32；线性 LR
  调度、LR warmup fraction 0、weight decay 0。学习率沿用此前 SST-2 Dense Muon
  5-epoch sweep 的选中配置：matrix LR `0.0002`、scalar LR `0.00005`，不属于独立确认调参。
- Muon momentum `0.95`、spectral-norm scaling；压缩器 `topk_sync` / `randk_sync`、
  tensor 级、比例 `0.2`、EF21。配置 `start_compress_iter=0`，但入口按
  `compression_warmup_fraction=0.1` 和 10 epochs 计算初始 dense 阶段；实际日志显示
  该阶段后才开始压缩。`--disable_compression_warmup` 只关闭后续渐进压缩，
  不表示从第一步开始压缩。
- CoLA/MRPC 每卡 batch 16、GA 2、global batch 128；SST-2/STS-B 每卡 batch 8、
  GA 2、global batch 64；QQP/MNLI/QNLI/RTE 每卡 batch 16、GA 1、global batch 64。
- 每次运行使用 4 张 RTX 4090；原组/CM009 使用 GPU 0、1、2、7，CM012 使用 GPU 0–3。
  硬件型号由本地 W&B 元数据核对，使用卡数由启动参数及训练日志核对；元数据列出的
  主机总 GPU 数不作为本次使用卡数。未保存模型 checkpoint。
- 运行脚本：[glue_muon_table3_randk_topk.sh](../glue_fine-tuning/scripts/glue_muon_table3_randk_topk.sh)。
  已完成原组的实际参数以各运行本地 W&B 元数据 `args` 及 `train.log` 为准，
  当前脚本默认仅覆盖后续 QQP/MNLI/QNLI/RTE。

## 3. 最终验证结果

下表为百分制分数，来源均为链接对应的 `all_results.json`，使用 `eval_*` 原始键。
全部为 final：epoch 9（第 10 个 epoch）；MNLI 使用训练后的 `validation_mismatched`
评估，其他任务使用 `validation`。已核对 JSON 与日志最后完整评估一致且达到目标更新步数。

| 数据集 | 指标（原始键去掉 `eval_`） | Top-K + Muon | Rand-K + Muon | split | final update |
| --- | --- | ---: | ---: | --- | ---: |
| CoLA | matthews_correlation | [57.320](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/cola/muon_topk/all_results.json) | [55.804](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/cola/muon_randk/all_results.json) | validation | 670 |
| SST-2 | accuracy | [94.610](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/sst2/muon_topk/all_results.json) | [94.839](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/sst2/muon_randk/all_results.json) | validation | 10,530 |
| MRPC | accuracy / f1 | [85.539 / 89.667](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/mrpc/muon_topk/all_results.json) | [81.373 / 87.459](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/mrpc/muon_randk/all_results.json) | validation | 290 |
| STS-B | pearson / spearmanr | [89.840 / 89.701](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/stsb/muon_topk/all_results.json) | [88.926 / 88.785](../outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/stsb/muon_randk/all_results.json) | validation | 900 |
| QQP | accuracy / f1 | [91.890 / 89.076](../outputs/CM009-m003-glue-topk-randk-ga1-seed1240-gpu0127/qqp/muon_topk/all_results.json) | [91.870 / 89.139](../outputs/CM009-m003-glue-topk-randk-ga1-seed1240-gpu0127/qqp/muon_randk/all_results.json) | validation | 56,860 |
| MNLI | accuracy | [86.656](../outputs/CM012-m003-glue-ga1-gpu0123-resume/mnli/muon_topk/all_results.json) | [86.758](../outputs/CM012-m003-glue-ga1-gpu0123-resume/mnli/muon_randk/all_results.json) | validation_mismatched | 61,360 |
| QNLI | accuracy | [92.623](../outputs/CM012-m003-glue-ga1-gpu0123-resume/qnli/muon_topk/all_results.json) | [92.385](../outputs/CM012-m003-glue-ga1-gpu0123-resume/qnli/muon_randk/all_results.json) | validation | 16,370 |
| RTE | accuracy | [67.870](../outputs/CM012-m003-glue-ga1-gpu0123-resume/rte/muon_topk/all_results.json) | [66.426](../outputs/CM012-m003-glue-ga1-gpu0123-resume/rte/muon_randk/all_results.json) | validation | 390 |

## 4. 结论

- 同任务 Top-K 与 Rand-K 的预算及配置匹配，仅作单 seed（`n=1`）描述性比较。
  Top-K 在 CoLA、MRPC、STS-B、QNLI、RTE 的报告指标较高；Rand-K 在 SST-2、MNLI
  accuracy 较高；QQP 的 accuracy 与 F1 排序不同，未呈现一种方法全面占优。
- 可与上文 Dense/ARC 的同任务 final validation 指标对照，但 QQP 新组使用 GA=1，
  旧组使用 GA=2；SST-2 ARC 只完成 epoch 8，不能纳入同预算 final 比较。
  各任务 global batch 也不完全相同，不计算跨任务总平均作为统一排名。
- 学习率由 SST-2 validation 选出；当前没有多 seed 离散度，无法据此判断差异是否稳定。
  下一步优先补全 SST-2 ARC，并在统一 batch/GA 的条件下重复比较。
