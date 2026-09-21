# Agent 工作说明

本文件只保留执行任何任务前都需要知道的规则。开展新方法、正式实验或结果整理前，先阅读 [README](README.md)，再查看对应实验入口及 `scripts/` 中的运行示例。

## 研究目标

本项目研究 Muon 优化器的分布式通信优化，目标是参考团队前期的 [ARC-TopK](https://arxiv.org/abs/2510.26709)，将 Adam/分布式梯度压缩中的 Top-K、低秩压缩、误差反馈和 All-Reduce 兼容设计迁移到 Muon，并形成可发表的研究成果。

Muon 的正交化是非线性操作，通常有：

```text
Ortho(Average(G)) != Average(Ortho(G))
```

因此，研究方案必须说明压缩发生在梯度、动量、正交化输入、正交化结果还是参数更新阶段，并区分它是原始 Muon 的通信实现、近似实现，还是新的优化器。

## 必须遵守

- 保留用户已有修改，不覆盖或回退无关内容。
- 保留当前 Muon 作为 baseline；新方法应尽量能够通过配置、独立函数或独立代码路径切换。
- 避免与研究目标无关的大范围重构，重要实现应便于解释和比较。
- 未经允许，不修改虚拟环境，不升级 PyTorch、CUDA、NCCL、Triton 等关键依赖。
- 不停止、修改或干扰未经授权的进程，绝不干扰其他用户的进程。
- 不在代码、配置或日志中记录密钥、Token 等敏感信息。

## 主要代码位置

- 通信 hook 注册、参数和状态：`comm_hooks/utils.py`
- 当前使用的 ARC-TopK 投影、位置选择与聚合：`comm_hooks/group_topk_hook_no_reshape.py`
- ARC-TopK 的另一版实现：`comm_hooks/group_topk_hook_no_reshape_c4.py`；修改前先确认入口实际注册的版本。
- TopK/RandK 稀疏通信：`comm_hooks/sparse_hook_c4.py`；相关实现另见 `comm_hooks/sparse_hook.py`。
- 无压缩 All-Reduce 基线：`comm_hooks/default_hooks.py`
- 合成实验：`synthetic_release/main.py`
- GLUE 微调：`glue_fine-tuning/run_glue_no_trainer_new.py`
- CIFAR-10 训练：`cifar10/run_cifar10.py`、`cifar10/run_cifar10_resnet50.py`；模型定义在 `cifar10/resnet.py`。
- C4/LLaMA 预训练：`c4/run_llama_pretraining.py`；模型配置在 `c4/configs/`，训练辅助代码在 `c4/pept_utils/`。

## 文件放置

沿用本仓库现有布局：

- 通信算法与辅助函数放在 `comm_hooks/`。
- LLaMA 模型配置放在 `c4/configs/`；各任务的复现脚本分别放在 `c4/scripts/`、`cifar10/scripts/`、`glue_fine-tuning/scripts/`。
- 合成实验代码放在 `synthetic_release/`。该脚本会将 CSV 和 PNG 写入当前工作目录，运行前选择实验输出目录。
- 训练日志、checkpoint 和结果放在运行参数指定的输出目录，记录对应配置；不要散落在仓库根目录。
- 本仓库目前没有独立的 `tests/` 或 `benchmark/` 目录；新增验证代码时按涉及的模块或实验选择位置。

已有运行命令和参数示例见 README 与各任务的 `scripts/`。

## 修改与验证

- 先做与改动规模相称的低成本检查，再考虑完整训练。
- 涉及 collective 时，条件允许应进行多 GPU smoke test，并注意各 rank 的调用顺序和张量大小一致。
- 修改 ARC-TopK 时，保留 `none`、`topk_sync`、`randk_sync` 等可比较的基线；有损压缩方法不要求与无压缩结果逐元素一致，但应关注通信行为、稳定性和训练表现。
- 原型阶段允许只覆盖主要路径；尚未覆盖的场景应在汇报或方法记录中说明。

## 方法与实验编号

- 当一个方案开始产生正式代码、配置或比较实验时，为其分配 `M001`、`M002` 等方法编号。
- 用于正式比较、图表、论文判断或长期训练的运行，使用 `CM001-...` 格式的实验编号。
- 单元测试、短 smoke test 和临时调试不强制编号。
- 本地实验目录、主要日志和 W&B run name 应使用同一实验编号。

编号及其对应配置、结果在方法记录中保持一致。

## Worklog

- 每次实际尝试一个方法，或在已有方法上完成一次实验后，都应在 `docs/worklog/` 中记录；该目录尚未建立，首次需要时创建。失败或结论不明确的尝试也应记录。
- 同一方法的不同尝试和实验必须按时间追加到同一个 worklog 文件，不得为每次运行分别新建文件。
- 文件名使用 `<方法编号>-<简短名称>.md`；每条记录简要说明日期、目的与假设、修改或实验配置、验证、结果与观察、结论和下一步。
- 注明关联的代码、配置、实验编号和产物路径。大段日志和原始结果放在实验输出目录，worklog 只保留摘要和路径。
- 除非特别要求，用中文写

## GPU 与长任务

- 启动 GPU 或分布式任务前检查 GPU 可用情况和已有进程。
- 长时间任务应在 `tmux` 中运行并保留日志。
- 正式训练默认启用 W&B，并保持 W&B 名称与本地实验编号可对应。
- 注意磁盘空间，按需保存 checkpoint 和 profiler trace。

## 何时阅读项目资料

以下任务开始前，必须阅读 `README.md`、相关训练入口及对应的 `scripts/` 示例：

- 提出或实现新的压缩、通信或 Muon 变体；
- 创建正式配置、方法编号或实验编号；
- 设计或启动用于比较的实验；
- 整理研究结果、图表或论文内容；
- 新增或调整研究目录与长期文档。

普通 bug 修复、代码阅读和小范围维护不要求重复阅读全部示例。

## 任务汇报

任务结束时简要说明：

- 完成了什么；
- 修改了哪些关键文件；
- 做了哪些验证；
- 当前结果或观察；
- 已知问题和建议的下一步。

## 长任务与额度控制

* 对训练、benchmark、profiler 等长任务，避免使用 Agent/LLM 循环等待或轮询状态。
* 完成必要的启动检查后，优先将任务一次性放到 `tmux` 或后台运行并记录日志；不要反复执行 `tail`、`ps`、`nvidia-smi` 等命令等待结束。
* 只有当任务状态会影响当前决策时才主动检查；普通等待应交给进程、脚本或调度系统，而不是通过重复 Agent 回合实现。
* 若任务之间存在依赖，优先由脚本串行编排，例如 `task_a && task_b`，而不是让 Agent 持续观察 `task_a`，结束后再手动启动 `task_b`。
