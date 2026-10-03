# ARC-TopK / Muon 通信路径测速：A–D（2026-10-03）

## 目的与计时口径

比较 Dense、Top-K、Rand-K、ARC-TopK 的梯度通信路径耗时与完整更新耗时。
参考 `greedy_lore/docs/results.md` 中 CM188–CM243 的严格阻塞实验。
压缩发生在梯度同步阶段、Muon 动量及非线性正交化之前，因此压缩臂属于近似 Muon。

四种方法统一使用入口 CUDA 同步、开始墙钟计时、原 hook Future.wait、出口 CUDA
同步、停止计时的包装。入口同步耗时不计入 hook；出口同步计入。原算法与 collective
顺序不变，旧 `--blocking_communication` 关闭，避免重复栅栏。

- `mean_blocking_hook_seconds`：各 rank 的100个逐步 hook 总耗时之和的最大值 / 100。
  包含 EF、投影/选择、压缩、collective、解压及出口同步，不是纯网络传输时间。
- `mean_iteration_seconds`：各 rank 连续100步完整更新窗口总墙钟时间的最大值 / 100。
  包含数据读取、forward/backward、hook、clipping、Muon step、scheduler、zero_grad、
  逐步 CUDA 同步及窗口内计时记录开销。
- 两个指标分别选取各自最慢 rank，保存所有 rank 的逐步数据与 bucket 布局。
- Muon 自身的正交化结果 AllGather 包含在完整步时间中，不包含在梯度 hook 时间中。
  `rank_measured_muon_communication_bits` 保存测量窗口增量；旧
  `muon_communication_bits` 字段保持整个运行累计口径。
- DDP 通信 bits 为现有 hook 的公式估计，并非真实 wire traffic；本轮只计测量窗口。
  主实验不额外做逐 collective profiler 或异步 bucket sweep。

## 固定设置

- 本地 C4：`/home/wyr/greedy_lore/c4/c4_en`；50个训练分片，保存路径及文件大小。
  tokenizer 为本地缓存 `t5-base`，离线运行；数据 shuffle seed42，训练 seed1243。
- LLaMA 60M/130M/350M，使用本仓库 `c4/configs/llama_*.json`；FP32，序列256，
  每卡 batch1、GA1、不启用 activation checkpointing。
- Muon matrix/scalar LR 0.01/0.001，momentum0.95，spectral-norm scaling，
  weight decay0，gradient clipping1；cosine scheduler总步数150，LR warmup10步。
- 压缩保留比例0.2，EF14，tensor sparsity，关闭渐进压缩；ARC projection rank4。
  同比例不等于同通信字节数；ARC有投影 AllReduce，Top-K需要索引 AllGather。
- 50步预热：前40步 Dense，更新41–50使用压缩；连续测量更新51–150共100步。
  Dense臂同样预热50步。首次压缩初始化排除在正式窗口外。
- DDP bucket cap8192 MiB。接受条件要求测量窗口所有 rank 的实际 bucket 数为1。
- 不评估、不存 checkpoint、不连接 tracker；窗口内不逐步打印或上传指标。
  loss留在GPU，窗口结束后统一转CPU。

| 组 | GPU数 | dtype | channel | transport |
| --- | ---: | --- | --- | --- |
| A | 4 | FP32 | 默认 | SHM |
| B | 4 | FP32 | 1 | SHM |
| C | 8 | FP32 | 默认 | SHM |
| D | 8 | FP32 | 1 | SHM |

所有组关闭P2P、开启SHM、关闭cuMem host路径。默认channel组清除继承的channel/CTA
限制，单channel组将 NCCL_MIN/MAX_NCHANNELS 与 NCCL_MIN/MAX_CTAS 设为1。
保存NCCL环境、INFO日志与GPU拓扑，确认实际使用路径。没有BF16或Socket组。

## 重复、调度与产物

48个配置各做3次独立进程重复，共144次，编号CM122–CM265。
每轮依次覆盖A–D及60M/130M/350M，同一配置内四方法随机排列，排序seed20261003。
训练seed固定1243；100步是一次运行内测量，不是100次独立重复。
4卡组使用物理GPU0–3，8卡组使用0–7；全部串行、启动前检查空闲、不循环轮询。

控制器：`c4/scripts/run_strict_hook_abcd.py`。
产物：`output/CM122-CM265-strict-hook-abcd-fp32/`，包含manifest、命令、环境、日志、
状态、逐rank原始JSON、summary.csv/json、同轮配对comparisons.json、aggregate.json。
分别汇总iter/hook均值、样本标准差、每轮 Dense/method speedup；失败臂不参与配对，
保留全部失败日志，不自动改变精度、batch、模型或重跑。每臂超时2小时。
开始记录源码SHA256，后续臂若源码变化则停止队列，防止混合代码版本。

```bash
PYTHONPATH=. .venv/bin/python -B c4/scripts/run_strict_hook_abcd.py --dry-run
PYTHONPATH=. .venv/bin/python -B c4/scripts/run_strict_hook_abcd.py --gpus 0,1,2,3,4,5,6,7
```

这是取消backward/通信重叠的系统诊断，不能直接外推为正常异步DDP吞吐或收敛优劣。
