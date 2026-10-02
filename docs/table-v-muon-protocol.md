# ARC-TopK Table V：Muon 版本测速约定（2026-10-02）

## 研究问题与模式

本实验比较同模型、同数据、同 Muon 设置下 Dense、Top-K、Rand-K 和 ARC-TopK
每次完整训练更新的墙钟时间。它是将论文 Adam 替换为 Muon 后的探索性系统实验，
不是论文 Adam 数值复现，也不评价 150 次更新内的收敛质量。

论文目标为 [arXiv:2510.26709](https://arxiv.org/pdf/2510.26709) Table V。
论文明确使用 C4、LLaMA-60M/130M/350M/1B、4 张 A100 40GB、每卡 batch 1、
50 次计时迭代、稀疏率 0.2、NCCL SHM，并显式禁用 NVLink P2P。

## 正式配置

- 模型：`c4/configs/llama_{60m,130m,350m,1b}.json`；序列长度 256。
- 数据：`/home/wyr/greedy_lore/c4/c4_en` 的本地英文 C4；T5 tokenizer；
  shuffle seed 42。控制器保存训练分片路径、大小和 mtime。
- 4 GPU DDP；每卡 batch 1，无梯度累积，global batch 4；模型与梯度使用 FP32，
  与仓库正式 C4 训练脚本保持一致；Muon Polar Express 内部仍按实现使用 BF16。
- 每臂 seed 1243。先执行 100 次不计时真实更新，再连续统计 update 101–150。
- Muon：matrix LR 0.01、scalar AdamW LR 0.001、momentum 0.95、Nesterov、
  spectral-norm scaling、五步 Polar Express；两条参数路径 weight decay 均为 0；
  不启用 `torch.compile`。学习率用于保持方法臂一致，没有为本测速调优。
- 压缩臂在 hook iter 100 开始立即使用目标压缩率，不做渐进压缩；
  `compress_ratio=0.2`、tensor-wise、EF14；ARC-TopK 使用 `r=4`。
- 按用户决定，沿用仓库正常训练的压缩覆盖范围，不增加仅压缩二维张量的路径。
  三种有损压缩发生在 Muon 正交化前，属于近似 Muon。
- `NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=0`、
  `NCCL_CUMEM_HOST_ENABLE=0`；smoke 和正式日志使用 `NCCL_DEBUG=INFO` 核验传输。
- 关闭 W&B、评估和 checkpoint。测速包含数据等待、H2D、forward/backward、
  DDP hook、裁剪、Muon step及其 collectives、scheduler、zero_grad 和逐步 CUDA 同步。

论文未说明 dtype、计时同步、error-feedback 具体实现、ARC rank、Muon 参数及100步
热身；上述值分别取仓库正式 C4 脚本/同节 C4 设置、仓库正式 C4 路径和用户决定。
实际硬件若不是 A100 40GB，绝对秒数不能与论文表格直接比较。

## 计时、身份与接受规则

每个 rank 保存完整50步的逐步时间和连续窗口总时间，主指标为：

```text
mean_iteration_seconds = max(rank_elapsed_seconds) / 50
```

50步是同一训练轨迹的重复测量，不是50个独立实验。单 seed 结果只作描述性比较。
同时保存 loss、显存峰值、DDP hook通信统计与Muon内部通信统计。

| 模型 | Dense | Top-K | Rand-K | ARC-TopK |
| --- | --- | --- | --- | --- |
| 60M | CM020 / M001 | CM021 / M003 | CM022 / M003 | CM023 / M002 |
| 130M | CM024 / M001 | CM025 / M003 | CM026 / M003 | CM027 / M002 |
| 350M | CM028 / M001 | CM029 / M003 | CM030 / M003 | CM031 / M002 |
| 1B | CM032 / M001 | CM033 / M003 | CM034 / M003 | CM035 / M002 |

入口为 `c4/table_v_timing.py`，串行控制器为
`c4/scripts/run_table_v_muon.py`，正式产物根目录为
`output/CM020-CM035-table-v-muon-fp32-rerun1/`。只有 exit code 0、状态 `completed` 且
`measured_steps=50` 的单元进入汇总；失败日志保留，后续单元继续运行。

首次 BF16 队列的 CM020–CM031 已完成，CM032 启动后因 dtype 口径修正而中止；
旧产物完整保存在 `output/CM020-CM035-table-v-muon/`，不纳入 FP32 正式汇总。
1B Dense FP32 的1+1步启动前smoke在第一次Muon step确定性OOM：每卡约已占用
23.02 GiB，批量更新堆叠还需1.50 GiB。正式矩阵仍保留1B四臂；若失败则记录为
failed并继续，不自动降低dtype、batch、序列长度或改变Muon实现。
