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
