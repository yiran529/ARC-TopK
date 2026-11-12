# ARC-TopK 投影流水线：原 setting 对照

4张空闲RTX4090；FP32、seq256、每卡batch1、Muon、EF14、保留0.2、r4、SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞；每版本一次，seed1243。

| 模型 | 实现 | NCCL kernel ms/步 | 完整hook ms/步 | 完整单步 ms/步 |
| --- | --- | ---: | ---: | ---: |
| 60m | dense | 25.643 | 26.332 | 61.832 |
| 60m | fused | 33.378 | 25.288 | 60.204 |
| 60m | pipeline | 34.739 | 23.760 | 58.202 |
| 130m | dense | 52.202 | 54.852 | 118.171 |
| 130m | fused | 54.272 | 41.943 | 105.879 |
| 130m | pipeline | 43.896 | 42.202 | 106.152 |
| 350m | dense | 140.337 | 140.219 | 300.052 |
| 350m | fused | 88.367 | 90.087 | 252.542 |
| 350m | pipeline | 76.861 | 81.611 | 241.188 |

| 模型 | pipeline对比基线 | hook耗时降低 | 完整单步耗时降低 |
| --- | --- | ---: | ---: |
| 60m | fused | 6.04% | 3.33% |
| 60m | dense | 9.77% | 5.87% |
| 130m | fused | -0.62% | -0.26% |
| 130m | dense | 23.06% | 10.17% |
| 350m | fused | 9.41% | 4.50% |
| 350m | dense | 41.80% | 19.62% |

fused/pipeline在两种窗口中的逐rank测量loss序列、DDP bits及实际collective入队顺序、payload尺寸/dtype/world-size哈希完全一致。四卡180组/rank逐位smoke通过（FP32/BF16、noef/EF14/EF21、多比例、多bucket），含CUDA RNG与误差状态。EF21和CPU保留行索引路径。

流水线先按原顺序生成投影并异步提交AllReduce，再按原顺序wait、平均、TopK和打包；不合并collective、不改浮点归约、随机数调用形状或顺序。峰值显存记录在comparison.json，投影临时缓冲仅保留到当前hook结束。

主timing无profiler：50步预热+100步测量。独立诊断50+10，在主线程中从更新50完成后捕获到更新60完成，校验并剔除起点barrier；聚合NCCL AllReduce CUDA kernel时长，排除Muon AllGather，按最慢rank累计/10。kernel包含协议等待与profiling扰动，非wire time，不能与无profiler hook直接相减。

不导出或保留原始trace，不保存checkpoint；只保留精简JSON、日志、manifest、源码snapshot。按用户要求已清理本会话此前156份大trace（约1.58GiB），保留旧汇总和删除哈希清单。

正式臂启动前等待GPU0–3空闲，运行中监测其他GPU进程，干扰结果拒绝接受。每版本仅一次，未评估统计显著性；严格阻塞结果不能外推异步DDP吞吐或收敛。

首轮独立诊断因DDP hook线程与profiler启停线程不一致而失败；该轮日志/源码存放rejected-profile-thread-attempt1。修复仅涉及诊断器，将启停移至主线程；9次无profiler timing保留，算法代码未改，诊断重新运行。

实验CM344–CM361，沿用方法M002（实现调度优化），Dense为M001；代码未提交。

最终接入：保留fused默认，pipeline作为可选实现；通用CLI默认/覆盖和CPU noef/EF14/EF21回退逐位检查通过。新产物目录含失败日志和源码快照共约1.40MiB，无trace/checkpoint。18/18正式臂接受，36份通信聚合JSON保留，git diff --check及语法检查通过。未提交代码。
