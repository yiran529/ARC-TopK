# ARC-TopK 实现优化：4 卡前后对照

每个配置3次独立进程重复，固定seed1243；下表为均值 ± 样本标准差，单位ms/update。降低比例为同轮配对百分比的均值。

| 模型 | 指标 | 原实现 | 优化实现 | 配对耗时降低 |
| --- | --- | ---: | ---: | ---: |
| 60m | NCCL AllReduce kernel | 36.549 ± 4.050 | 39.506 ± 1.353 | -8.78% |
| 60m | 完整梯度 hook | 38.172 ± 0.883 | 28.131 ± 0.387 | 26.29% |
| 60m | 完整训练单步 | 72.646 ± 1.459 | 63.267 ± 1.047 | 12.88% |
| 130m | NCCL AllReduce kernel | 47.883 ± 1.672 | 54.575 ± 2.160 | -14.09% |
| 130m | 完整梯度 hook | 61.150 ± 0.725 | 43.752 ± 0.464 | 28.44% |
| 130m | 完整训练单步 | 125.803 ± 1.483 | 108.921 ± 1.256 | 13.41% |
| 350m | NCCL AllReduce kernel | 115.391 ± 6.389 | 105.996 ± 3.650 | 8.07% |
| 350m | 完整梯度 hook | 131.077 ± 0.358 | 93.583 ± 0.584 | 28.60% |
| 350m | 完整训练单步 | 289.984 ± 1.458 | 254.668 ± 1.746 | 12.18% |

所有配对测量窗口的逐rank loss序列及DDP通信bits逐项完全一致；profiler捕获的AllReduce kernel数量、逐次payload尺寸与dtype也一致，barrier依据NCCL元数据排除。

实现：以紧凑行索引替代逐元素索引展开，去除重复gather和独立projection的clone；保留reference路径，随机投影、TopK、EF和collective顺序/形状不变。

设置：4张空闲RTX4090，FP32，seq256，每卡batch1，C4/Muon，EF14，保留0.2、r4，SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞。每臂启动前检查空闲，运行中检查其他GPU进程。

无profiler单步/hook：50步预热+100步测量。独立profiler：50步预热+10步测量；每rank累计NCCL AllReduce GPU kernel时长/10，再取最慢rank，包含投影和稀疏值AllReduce，排除Muon AllGather。非wire time；与无profiler单步不可直接相减。

纯通信数据量和collective布局未变，因此kernel时间变化不能解释为流量减少，可能包含协议等待、kernel调度和测量波动。压缩计算的节省主要反映在hook和单步。

这是严格阻塞诊断，不能外推异步DDP吞吐或收敛。未做显著性检验。此前误判自身worker的首次试运行保存在rejected-monitor-attempt1，不纳入结果。

实验编号CM266–CM301，方法M002；源码哈希、逐臂命令/配置、原始逐rank结果与CUDA trace保存在同目录。

存储更新（2026-11-12）：依用户要求已删除本批原始大trace，保留汇总、配置、日志及源码；删除清单与SHA256见同目录removed-large-traces.json。
