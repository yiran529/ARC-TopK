# M001：可切换的 Muon 基线

## 2026-09-21：Muon 优化器与 DDP 正交化调度

### 目的与假设

在 C4、GLUE 和 CIFAR 入口中，将 Muon 作为与 Adam/SGD 并列的
`--optimizer` 选项。压缩器仍由 `--compressor` 单独选择，以便比较
Adam/Muon 与 dense/ARC-TopK 的组合。

### 实现

- `optimizers/muon.py`：Nesterov 动量、五步 Polar Express、矩阵尺度
  学习率、解耦权重衰减，以及非矩阵参数的 AdamW 后备更新。
- `optimizers/utils.py`：按模块类型选择 Muon 矩阵参数；embedding、输出
  head、norm 和 bias 进入 AdamW 参数组。
- 同形状矩阵合并为一个正交化 batch。DDP 下将 batch 均分给各 rank，
  通过 AllGather 收集更新；不足 world size 的部分补零。
- C4、GLUE、CIFAR-10 ResNet-18/50 增加 `muon` 分支和相关命令行参数。

### 验证

- CPU 单元测试覆盖动量公式、Nesterov、学习率缩放、卷积 flatten、
  AdamW 后备、checkpoint 恢复和参数分组。
- 两进程 Gloo 测试覆盖矩阵跨 rank 分配、padding 和结果 AllGather。
- 两张 RTX 4090 上的 NCCL smoke test 完成 DDP 梯度同步与分布式
  正交化，step 无死锁，各 rank 参数最大差异为 `0.000e+00`。
- 对修改后的 Python 文件执行语法检查和 `git diff --check`。

### 当前限制

- 当前是 DDP 非分片实现，不包含 dion 的 FSDP2/DTensor AllToAll 路径。
- 当前使用 PyTorch Polar Express，可选 `torch.compile`；尚未移植 dion
  的专用 Triton 对称矩阵内核。
- 端到端质量和速度需要在目标 GPU、模型与压缩配置上测量。跨 GPU
  调度减少重复正交化计算，但新增一次结果 AllGather，实际收益取决于
  矩阵形状、world size 和互联带宽。

### 下一步

先运行小模型 dense Muon smoke test，确认 loss 与各 rank 参数一致；再用
相同训练配置比较 local/distributed orthogonalization，以及 dense/ARC-TopK
通信组合。若正交化仍是主要瓶颈，再移植 Triton Polar Express 内核。

## 2026-09-21：恢复精度与实验口径修正

- 修复 BF16/FP16 参数加载 checkpoint 时 AdamW FP32 moments 被父类先降精度
  转换的问题；现在加载前保留 checkpoint 中的 FP32 moments，参数映射完成后
  原值恢复。
- AdamW fallback 的默认 beta 改为 `(0.9, 0.999)`。C4 默认采用 dion 正式
  语言模型实验的 scalar LR `0.001`、scalar weight decay `0`；GLUE/CIFAR
  继续继承各自任务 LR 和 decay。
- 修复 GLUE 在第一个 microbatch 提前更新的问题，并将 step checkpoint 移入
  optimizer 更新分支，避免同一步重复保存。
- Muon 独立累计 gradient-presence 与正交化结果 AllGather 的全网 bit 数；该值
  与 DDP hook 通信量分开报告。
- ARC hook 增加 rank-local error feedback、迭代计数和 RNG 状态保存恢复。
  GLUE checkpoint 已接入；C4 同时恢复模型、optimizer/scheduler、训练计数、
  rank-local hook/RNG，并按 `global_step` 跳过 streaming microbatches。严格续训
  要求数据源版本、worker 数与 shuffle 配置保持一致。
- 回归结果：`15 passed`；所有修改脚本通过 `py_compile` 和 `git diff --check`。

## 2026-09-21：正式 C4 训练入口边界修复

### 目的

为 60M Muon 正式预训练准备可审计的训练步数、评估和运行指标；默认不保存
checkpoint，只有显式设置 `--save_every > 0` 时才启用周期保存。

### 修改

- 修复训练循环边界：`num_training_steps=N` 现在恰好执行 N 次 optimizer update，
  不再多执行一次；`--save_every` 默认改为 `0`，关闭时不自动生成 `save_dir`。
  显式启用周期保存时仍保留原有保存和续训路径。
- 修正评估为按 causal LM 的有效预测 token（`labels[..., 1:] != -100`）对 batch
  mean loss 加权；各 rank 读取约 `10M / world_size` 本地 token，最后用一次
  `all_reduce` 聚合 loss numerator 与 token count，在达到全局 10M effective
  prediction tokens 后停止（最后一批可略超）。最终记录并写入 `all_results.json`
  的 `final_eval_perplexity=exp(final_eval_loss)`。
- 每个 update 的 W&B 指标增加明确的 `step_time_s`、
  `throughput_tokens_per_s`、显存 allocated/reserved，以及可用的 DDP 和 Muon
  通信 bits（step/total 与 Muon 分类别统计）；保留旧吞吐和显存键兼容已有面板。
- 正常结束时调用 `wandb.finish()` 和 `dist.destroy_process_group()`，移除 `exit()`。

### 验证

- 新增 CPU 单测覆盖训练步数边界、checkpoint 默认行为、评估 token 边界与批次计数、
  perplexity、W&B 指标命名及 DDP/Muon 通信统计。
- 未启动训练、未下载数据、未读取 `/home/wyr/.netrc`。
