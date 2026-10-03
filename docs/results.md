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

# C4 Table V 风格的 Muon 通信压力测速

## 1. 实验矩阵

CM058–CM105 与 CM106–CM121 比较 Dense、Top-K、Rand-K 和 ARC-TopK 四种梯度
通信方法。两组共 64 个 cell 均达到 100 步 warmup 后的 50 步计时窗口并以退出码 0
完成。主指标是各 rank 连续 50 个完整训练 update 的 wall time 中，最慢 rank 的总时间
除以 50；下表单位均为 ms/update。

| 实验编号 | 模型 | dtype / seq | cell 数 | 状态 | canonical source |
| --- | --- | --- | ---: | --- | --- |
| CM058–CM105 | 60M、130M、350M | FP32 / 256 | 48 | 48/48 完成 | [summary](../output/CM058-CM105-table-v-muon-blocking-matrix/summary.csv) / [comparisons](../output/CM058-CM105-table-v-muon-blocking-matrix/comparisons.json) |
| CM106–CM121 | 1B（实际 1,339,082,752 参数） | BF16 / 64 | 16 | 16/16 完成 | [summary](../output/CM106-CM121-table-v-muon-1b-bf16-blocking-matrix/summary.csv) / [comparisons](../output/CM106-CM121-table-v-muon-1b-bf16-blocking-matrix/comparisons.json) |

## 2. 核心设置

- 数据与训练：本地 C4，每卡 batch 1、GA1、seed 1243；Muon matrix LR 0.01、
  scalar LR 0.001、momentum 0.95、spectral-norm scaling，两条路径 weight decay 0。
- 压缩：梯度 DDP hook 上的 tensor 级稀疏化，保留比例 0.2；ARC-TopK 使用 rank 4
  和 EF14。压缩臂前 100 步走 Dense，从第 101 步起直接使用目标比例；关闭渐进压缩。
- 硬件：单机 RTX 4090；禁用 NCCL P2P，并限制为单 NCCL channel/CTA。
- 四种 setting：

| setting | GPU | DDP bucket | hook blocking | NCCL transport |
| --- | ---: | ---: | --- | --- |
| A | 4 | 1024 MiB | 否 | SHM |
| B | 4 | PyTorch 默认 | 是 | SHM |
| C | 8 | PyTorch 默认 | 是 | SHM |
| D | 8 | PyTorch 默认 | 是 | Socket/loopback，禁用 SHM |

严格阻塞表示每个 DDP hook 的同步 collective 后、返回已完成 Future 前执行设备同步，
从而不允许该梯度通信与后续 backward 重叠。Socket setting 是人为削弱的单机 loopback
链路，只用于通信压力诊断。

## 3. 结果

### Setting A：4 卡、1024 MiB bucket、非阻塞、SHM

| 模型 | Dense | Top-K | Rand-K | ARC-TopK | ARC 相对 Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60M | 60.12 | 115.12 | 73.50 | 73.83 | 慢 22.80% |
| 130M | 121.02 | 280.86 | 130.91 | 123.77 | 慢 2.27% |
| 350M | 305.92 | 726.99 | 310.58 | 287.41 | 快 6.05% |
| 1B† | 667.56 | 1433.71 | 904.65 | 576.26 | 快 13.68% |

### Setting B：4 卡、严格阻塞、SHM

| 模型 | Dense | Top-K | Rand-K | ARC-TopK | ARC 相对 Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60M | 62.19 | 103.91 | 77.04 | 76.67 | 慢 23.29% |
| 130M | 125.46 | 225.24 | 133.54 | 129.11 | 慢 2.91% |
| 350M | 324.99 | 495.94 | 321.97 | 296.69 | 快 8.71% |
| 1B† | 702.06 | 1630.27 | 936.96 | 601.15 | 快 14.37% |

### Setting C：8 卡、严格阻塞、SHM

| 模型 | Dense | Top-K | Rand-K | ARC-TopK | ARC 相对 Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60M | 69.12 | 225.41 | 77.64 | 77.31 | 慢 11.84% |
| 130M | 139.77 | 615.15 | 139.44 | 138.51 | 快 0.90% |
| 350M | 356.33 | 1202.17 | 329.58 | 308.13 | 快 13.53% |
| 1B† | 723.16 | 5546.88 | 920.87 | 583.88 | 快 19.26% |

### Setting D：8 卡、严格阻塞、Socket

| 模型 | Dense | Top-K | Rand-K | ARC-TopK | ARC 相对 Dense |
| --- | ---: | ---: | ---: | ---: | ---: |
| 60M | 244.75 | 392.85 | 135.21 | 155.89 | 快 36.31% |
| 130M | 558.77 | 899.76 | 289.22 | 320.97 | 快 42.56% |
| 350M | 1560.76 | 2414.47 | 747.79 | 818.54 | 快 47.55% |
| 1B† | 3351.05 | 6715.47 | 2124.38 | 1888.22 | 快 43.65% |

† 1B 使用 BF16、seq64；其余三种模型使用 FP32、seq256。因此 1B 只可在自身同一行内
比较方法，不能把跨模型变化完全归因于参数规模。

## 4. 结论与限制

- ARC-TopK 的收益随模型和通信压力增大而出现：setting A/B 的 60M、130M 仍慢于
  Dense，350M 和 1B 则更快；8 卡严格阻塞 SHM 下从 130M 起不慢于 Dense，350M
  和 1B 分别快 13.53% 和 19.26%。
- 人为 Socket 压力下，ARC-TopK 四个规模均快于 Dense 36.31%–47.55%，说明通信进入
  关键路径后压缩流量可以转化为端到端收益；该 setting 不代表真实多节点网络。
- Top-K 在全部 16 个同组比较中都慢于 Dense，尤其 8 卡 SHM 的大模型开销异常高；
  需要通过 collective 数量和通信/索引聚合 profiler 进一步定位，不能把它解释为
  稀疏压缩本身必然无效。
- Socket 下 Rand-K 在 60M、130M、350M 比 ARC-TopK 更快，1B 则由 ARC-TopK 更快；
  这反映 ARC 投影/选行开销与通信节省之间存在规模相关的折衷。
- Setting A 与 B 同时改变 bucket cap 和 blocking，不能用两者差值单独估计关闭 overlap
  的因果效应。所有 cell 均为单次进程轨迹、单 seed（`n=1`）；50 个 measured steps
  不是 50 个独立重复，当前结果属于探索性 timing。

# C4 / Muon：A–D FP32 严格通信路径测速（CM122–CM265）

## 1. 实验矩阵与完成状态

2026-10-03 20:41–22:36（北京时间）完成。4种设置 × 3种模型 × 4种方法 ×
3次独立进程重复，共144次运行，**144/144成功，0失败**。本轮没有1B、BF16或Socket。
以下为本仓库实测，不是论文报告值。

本节A–D是卡数/channel矩阵，与上一节CM058–CM121的同名字母含义不同。

| 设置 | GPU数 | dtype | NCCL channel | transport |
| --- | ---: | --- | --- | --- |
| A | 4 | FP32 | 默认（日志确认2个coll channel） | SHM |
| B | 4 | FP32 | 1 | SHM |
| C | 8 | FP32 | 默认（日志确认2个coll channel） | SHM |
| D | 8 | FP32 | 1 | SHM |

模型为LLaMA 60M/130M/350M，实际参数量分别为58,073,600、134,105,856、367,969,280。
方法为Dense Muon、Top-K + Muon、Rand-K + Muon及ARC-TopK + Muon。

## 2. 配置与计时口径

- 本地英文C4，50个训练分片，t5-base tokenizer，sequence256，每卡batch1、GA1，
  关闭activation checkpointing；单机RTX4090，4卡臂GPU0–3，8卡臂GPU0–7。
- Muon matrix/scalar LR 0.01/0.001、momentum0.95、spectral-norm scaling、
  weight decay0、gradient clipping1。Cosine schedule总步数150，LR warmup10步。
- 压缩保留比例0.2、EF14、关闭渐进压缩、tensor sparsity；ARC projection rank4。
  压缩起点40：更新1–40使用Dense，41–50预热压缩路径；各臂预热50步，
  连续测量更新51–150共100步。压缩发生在Muon动量及正交化之前，属于近似Muon。
- DDP bucket cap8192 MiB，144个原始结果均确认测量窗口所有rank实际单bucket。
  统一关闭P2P、开启SHM、关闭cuMem host路径；单channel组限制channel/CTA为1。
- 严格包装：hook入口CUDA同步，开始计时，调用原hook并等待Future，再CUDA同步后
  停止计时。`hook`包括EF、投影/选择、压缩、collective、解压及出口同步，
  **不是纯NCCL网络传输时间**；入口同步耗时不计入hook。
- `iter`是完整更新窗口墙钟时间，包含数据读取、forward/backward、hook、clipping、
  Muon step、scheduler、zero_grad、逐步CUDA同步和计时记录开销。Muon正交化结果
  AllGather计入iter，不计入梯度hook。窗口内不评估、保存checkpoint或连接tracker。
- 每次运行：`iter=max_rank(连续100步窗口墙钟时间)/100`；
  `hook=max_rank(100个逐步hook总时间之和)/100`；两指标独立选取最慢rank。
- 每个配置3次独立进程重复，训练seed均1243；100个更新步不是100次独立实验。
  每轮同模型/设置内随机排列四种方法，排序seed20261003，全部串行执行。

## 3. 完整训练步时间

单位ms/update；数值为3次独立运行的均值 ± 样本标准差（n=3）。

| 设置 | 模型 | Dense | Top-K | Rand-K | ARC-TopK |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 60M | 57.41 ± 0.21 | 105.25 ± 1.79 | 73.85 ± 1.18 | 73.99 ± 0.86 |
| A | 130M | 116.31 ± 1.52 | 248.75 ± 3.16 | 129.66 ± 0.49 | 123.98 ± 2.33 |
| A | 350M | 302.71 ± 0.98 | 394.64 ± 3.33 | 310.20 ± 2.12 | 287.80 ± 1.10 |
| B | 60M | 61.72 ± 0.88 | 118.22 ± 1.37 | 74.41 ± 1.29 | 73.57 ± 1.04 |
| B | 130M | 121.23 ± 0.77 | 283.19 ± 3.41 | 129.32 ± 2.23 | 127.35 ± 0.86 |
| B | 350M | 314.81 ± 4.15 | 409.33 ± 2.28 | 313.81 ± 0.86 | 293.25 ± 2.56 |
| C | 60M | 66.06 ± 0.38 | 274.93 ± 2.01 | 75.10 ± 1.14 | 77.52 ± 1.30 |
| C | 130M | 134.20 ± 1.18 | 789.14 ± 2.30 | 133.73 ± 0.59 | 131.21 ± 1.60 |
| C | 350M | 335.17 ± 0.40 | 612.75 ± 1.71 | 314.22 ± 1.78 | 292.41 ± 1.03 |
| D | 60M | 70.64 ± 2.17 | 306.79 ± 3.00 | 77.95 ± 2.36 | 77.65 ± 0.39 |
| D | 130M | 139.90 ± 1.29 | 933.63 ± 8.15 | 134.79 ± 0.75 | 134.78 ± 0.84 |
| D | 350M | 351.10 ± 0.82 | 637.85 ± 4.48 | 320.14 ± 2.86 | 298.98 ± 0.99 |

## 4. 梯度通信路径时间

单位ms/update；均值 ± 样本标准差（n=3），包含压缩及解压开销。

| 设置 | 模型 | Dense | Top-K | Rand-K | ARC-TopK |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 60M | 23.55 ± 0.11 | 69.01 ± 0.34 | 38.07 ± 0.65 | 38.46 ± 0.51 |
| A | 130M | 52.87 ± 0.69 | 183.40 ± 1.62 | 64.74 ± 0.38 | 60.77 ± 0.78 |
| A | 350M | 141.29 ± 2.09 | 234.46 ± 1.79 | 150.61 ± 1.45 | 127.97 ± 1.03 |
| B | 60M | 26.20 ± 0.94 | 81.38 ± 0.92 | 38.85 ± 0.63 | 38.45 ± 0.61 |
| B | 130M | 57.49 ± 0.88 | 216.50 ± 2.99 | 64.79 ± 0.20 | 63.18 ± 0.88 |
| B | 350M | 151.27 ± 2.78 | 245.78 ± 2.07 | 152.00 ± 1.57 | 130.99 ± 1.19 |
| C | 60M | 30.64 ± 0.25 | 236.90 ± 1.57 | 40.10 ± 0.77 | 41.92 ± 0.60 |
| C | 130M | 69.57 ± 1.58 | 721.28 ± 1.03 | 68.98 ± 0.57 | 66.96 ± 1.42 |
| C | 350M | 179.24 ± 0.14 | 454.13 ± 1.19 | 159.99 ± 1.86 | 140.11 ± 0.80 |
| D | 60M | 34.48 ± 1.30 | 267.96 ± 2.62 | 41.85 ± 1.67 | 41.89 ± 0.36 |
| D | 130M | 73.73 ± 0.76 | 865.34 ± 7.27 | 69.68 ± 0.50 | 68.80 ± 1.03 |
| D | 350M | 193.21 ± 0.78 | 477.74 ± 4.77 | 163.86 ± 2.68 | 142.47 ± 0.59 |

## 5. ARC-TopK 相对 Dense 的配对加速

每轮先计算同设置、同模型的 `speedup=Dense/ARC`，再汇总3轮均值 ± 样本标准差。
大于1表示ARC更快；本表不是两个耗时均值直接相除。

| 设置 | 模型 | iter speedup | hook speedup |
| --- | --- | ---: | ---: |
| A | 60M | 0.7760 ± 0.0097 | 0.6125 ± 0.0095 |
| A | 130M | 0.9385 ± 0.0293 | 0.8701 ± 0.0194 |
| A | 350M | 1.0518 ± 0.0074 | 1.1041 ± 0.0180 |
| B | 60M | 0.8390 ± 0.0063 | 0.6816 ± 0.0268 |
| B | 130M | 0.9520 ± 0.0116 | 0.9102 ± 0.0261 |
| B | 350M | 1.0735 ± 0.0071 | 1.1547 ± 0.0133 |
| C | 60M | 0.8523 ± 0.0184 | 0.7311 ± 0.0158 |
| C | 130M | 1.0229 ± 0.0193 | 1.0396 ± 0.0425 |
| C | 350M | 1.1462 ± 0.0034 | 1.2793 ± 0.0083 |
| D | 60M | 0.9098 ± 0.0317 | 0.8231 ± 0.0338 |
| D | 130M | 1.0380 ± 0.0101 | 1.0718 ± 0.0220 |
| D | 350M | 1.1743 ± 0.0016 | 1.3562 ± 0.0022 |

## 6. 观察与限制

- 350M的A–D四组中ARC在3轮里均降低iter和hook耗时。最大相对收益为D组：
  iter从351.10降至298.98 ms，配对耗时降低均值14.85%；hook从193.21降至
  142.47 ms，配对耗时降低均值26.26%。三轮iter speedup为1.1762/1.1730/1.1738。
- 60M的四组ARC均慢于Dense；130M在4卡组仍慢，在8卡组iter分别降低2.22%（C）
  和3.66%（D）。C组130M hook仅2/3轮更快，小幅差异仍需谨慎解释。
- Top-K的12个模型/设置组合均比Dense慢；Rand-K在350M的8卡组有收益，低于ARC。
  当前结果不能将Top-K的额外开销完全归因于索引通信，尚无逐collective profiler分解。
- 单channel增加ARC的相对speedup，但没有降低350M的绝对时间：4卡ARC的A/B分别
  为287.80/293.25 ms，8卡C/D为292.41/298.98 ms。不能将相对加速更大解读为
  更高绝对吞吐。
- 这是严格阻塞、单bucket的系统性能实验，取消了外部backward/通信重叠；不能直接
  外推为正常异步DDP吞吐。本轮不评价收敛或PPL，也不是GreedyLoRE论文数值复现。
- 3次重复固定同一训练seed，刻画系统运行波动，不是3个独立训练seed的质量比较。
  未做显著性检验；同保留比例不代表同通信量。既有hook bits为公式估算，不是wire流量。

## 7. 结果来源与核验

- [实验协议](strict-hook-abcd-fp32.md)；控制器
  [run_strict_hook_abcd.py](../c4/scripts/run_strict_hook_abcd.py)。
- [manifest](../output/CM122-CM265-strict-hook-abcd-fp32/manifest.json)保存完整编号、
  配置、运行顺序及源码SHA256；Git基线为 `de12e2c75314f60554c1277a20a61cd7f8843b1e`，
  实际运行包括尚未提交的计时扩展，以manifest和逐臂config的源码SHA256为准。
- [status](../output/CM122-CM265-strict-hook-abcd-fp32/status.tsv)、
  [逐臂汇总](../output/CM122-CM265-strict-hook-abcd-fp32/summary.csv)、
  [重复均值/标准差](../output/CM122-CM265-strict-hook-abcd-fp32/aggregate.json)、
  [逐轮配对](../output/CM122-CM265-strict-hook-abcd-fp32/comparisons.json)。
- 各运行目录的 `all_results.json` 保存逐rank时间、bucket布局、loss及显存；同级
  `<run_id>.log` 保存NCCL日志。已逐一核对144个结果的更新窗口、单bucket以及
  iter/hook的独立max-rank计算；同模型各臂源码哈希一致。
