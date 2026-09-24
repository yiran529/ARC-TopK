---
name: arc-topk-experiment-workflow
description: Use when planning, configuring, launching, resuming, recording, reviewing, or comparing experiments in the ARC-TopK-release research repository, including Muon communication comparisons on GLUE, CIFAR-10, or C4.
---

# ARC-TopK 研究实验工作流

本项目的核心目的是检验梯度压缩方法在 Muon 优化器下是否有效，重点观察相同 Muon 配置下 Dense 与 ARC-TopK 的模型指标差距及通信代价。论文实验提供任务与训练协议的参照：除 Muon 特有超参数需要适配外，其余设置尽可能沿用论文。此 skill 将该研究目标贯穿实验设计、代码配置、运行记录和结果汇报；具体参数仍以当前实验 brief 和运行产物为准。

## 选择工作模式

- **设计方案**：明确研究问题、对照组、训练预算、指标和配置选择规则。用户未要求时，不创建正式 run 或启动训练。
- **配置或实现**：检查相关训练入口和脚本，再做范围最小的改动。基线与新方法应能分别选择。
- **启动或续跑**：准备并验证准确的运行配置，再使用仓库 launcher。仅当用户要求运行、启动或继续训练时才启动。续跑前检查该训练入口确实支持恢复，并确认 checkpoint 存在且包含恢复所需状态；否则将其视为新 run 重跑，不能记录为续跑。
- **结果审查或整理**：审查、比较、提取或撰写结果时，使用[实验结果汇报 skill](../experiment-results-reporting/SKILL.md)。本 skill 提供项目上下文，不重复那份结果汇报流程。

## 建立实验约定

提出或调整正式实验前：

1. 阅读 `AGENTS.md`、`README.md`、对应训练入口及相关脚本。如果 `docs/worklog/` 或 `docs/results.md` 已有同一方法或任务的记录，也一并查看。
2. 将“Muon + Dense”与“Muon + ARC-TopK”设为主要对照，目标是测量压缩方法接入 Muon 后的影响。原论文的 Adam 结果用于实验协议/参考，不替代这组 Muon 内部对照。
3. 尽可能沿用论文中的数据、模型、任务、训练预算、batch、scheduler、评价 split/指标及压缩设置。两组使用相同的 Muon 参数分组、Muon 学习率、训练预算和评价口径；除压缩/通信设置外不引入额外差异。
4. 仅对 Muon 特有超参数做必要适配或调优，尤其是矩阵参数 LR，以及实现中由 AdamW 更新的 scalar 参数 LR。将调优范围、选择规则和 validation 使用情况写清楚；其他超参数的偏离应有理由并单独记录。
5. 区分论文报告值、仓库默认值、计划覆盖值和实际运行值。实际命令/配置及完成的运行产物用于确认跑了什么；worklog 或脚本可能记录的是后来被改动的计划。
6. 方法进入正式代码、配置或比较实验时，分配 `M###` 方法编号。正式对照分配 `CM###-...` 实验编号，并为每个方法臂/run 建立唯一身份。输出目录和主日志应包含该身份；启动脚本支持时，让 tracker run name 也包含它。若入口生成的名称不可配置，则在 worklog 或 run manifest 中记录 run ID、方法臂、本地路径和 tracker 标识的一一映射，不要假定入口已自动关联。
7. 单 seed 结果标为初步结果。若用 validation 指标挑选 Muon 超参数，披露选择过程，不把参与选择的同一 validation 结果写成独立确认结果。

复现论文时，核对引用的论文版本及具体表格/章节。尽可能保留原设置，并标记每项偏离、近似和推断；不要从其他表格或仓库默认值推断论文没有说明的设置。

当前 GLUE 记录中的 Muon matrix/scalar LR 是通过 SST-2 Dense Muon 的 5-epoch validation sweep 选出，再用于其他 GLUE 任务。若沿用这套配置，须注明这是 SST-2 validation 选参后的共享 LR 比较：SST-2 validation 不是独立确认集，跨任务结果也不代表各任务分别调到最优 LR。以后若 sweep 规则或 LR 配置变化，以当次脚本和运行记录为准。

## 保留方法语义

每种通信方法都要说明传输了什么，以及压缩位于更新链的哪一步。Muon 实验要区分压缩原始梯度、动量、正交化输入、正交化后的更新或参数更新。正交化是非线性操作；若在 Muon 正交化前压缩，通常属于近似 Muon，应如实描述，不要声称与 Dense Muon 通信等价。

研究问题需要时，保留 dense/no-compression、Top-K 和 Rand-K 对照。新方法不应顺手改动基线实现。确认训练入口实际注册的是哪个通信 hook；不能因为文件名相似就假定实现相同。

检查 hook 的参数覆盖范围。当前 GLUE 的 `group_topk_no_reshape` 注册在整个 DDP 模型上，对每个 bucket 中的梯度张量分别压缩，因此也会压缩后续由 AdamW fallback 更新的 embedding、classifier、normalization 和 bias 梯度。若使用这一路径，结论应归于“梯度压缩 + Muon/AdamW 混合优化器”，不能表述为只压缩了 Muon 矩阵参数；若入口采用 selective compression，则按实际参数名单说明。

## 配置与启动

正式启动前：

1. 确认模型/数据路径、任务、seed、优化器与参数分组、学习率、scheduler、epoch 或更新步数、本地与全局 batch、压缩比例/秩、error feedback、压缩起点/warmup、评估方式、输出路径和 tracking 模式。
2. 在对应入口确认 CLI 参数确实存在，并检查 launcher 本身；不要未经核实地从另一训练任务照抄参数。
3. 检查输入与环境、GPU 可见性和占用、已有进程及输出盘空间。不得干扰无关任务，也不要因为旧脚本列出了某张 GPU 就认为它当前可用。
4. 先做低成本验证，例如 shell 语法和 launcher dry-run。适用且已获授权时再做短 smoke test。dry-run 不得下载数据/模型或启动训练。
5. 长任务使用仓库既有后台/session 方式，保留主日志，并确保运行能关联配置与编号。不要泄露脚本或环境文件中的 API key、token。
6. 每次实际尝试后，按时间追加到对应方法 worklog：日期、目的、方法/实验编号、实际命令与配置、验证、结果、产物路径及下一步。失败或中断也要记；同一方法的重试不要另建 worklog。

正式训练默认按项目约定启用 W&B；用户明确要求离线或关闭 tracking 时，遵从该设置并记录偏离。若当前入口不能让 W&B run name 携带 run ID，应使用上述映射记录保持可追溯性。

启动训练、下载大型资源、连接外部 tracker 或续跑，都必须在用户当前请求授权范围内。用户只要求设计、审查、准备命令或 dry-run 时，不代表授权启动训练。

## 记录与汇报结果

原始日志、checkpoint 和机器可读输出保存在运行配置指定的目录。`docs/worklog/<M###-...>.md` 按时间记录方法尝试，`docs/results.md` 汇总跨 run 结果；链接到原始产物，不将大段日志复制进这两类文档。

更新结果报告时，遵循现有 `experiment-results-reporting` skill 和简洁模板。核实 run 完成状态、实际预算、指标与 split、final/best 选择规则，以及每个报告数字的来源定位。来源冲突时保留冲突并标为未解决；不要把最后一条日志自动当成最终结果。

报告通信代价时，区分 DDP/压缩 hook 的梯度通信、Muon 优化器内部的 collective（如梯度布局检查和正交化结果 AllGather），并在计数口径一致时给出总通信量。标明统计的 collective、计数范围和估算/实测性质；某一路径没有可用计数时注明未统计，不要把 hook 字节数当作端到端总通信量。

## 仓库路径速查

- GLUE：`glue_fine-tuning/run_glue_no_trainer_new.py`、`glue_fine-tuning/scripts/`
- CIFAR-10：`cifar10/run_cifar10.py`、`cifar10/run_cifar10_resnet50.py`、`cifar10/scripts/`
- C4/LLaMA：`c4/run_llama_pretraining.py`、`c4/configs/`、`c4/scripts/`
- 通信 hook：`comm_hooks/`；实际注册实现应从训练入口确认。
- Muon 优化器与参数分组：`optimizers/muon.py`、`optimizers/utils.py`
- 方法记录：`docs/worklog/`；跨 run 结果汇报：`docs/results.md`
