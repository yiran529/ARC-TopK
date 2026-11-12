# 60M / 130M：融合 ARC-TopK 原 setting 补测

4张空闲RTX4090，FP32，seq256，每卡batch1，Muon，EF14，保留比例0.2、r4，SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞。各版本一次，seed1243。

| 模型 | 实现 | NCCL AllReduce kernel ms/步 | 完整 hook ms/步 | 完整单步 ms/步 |
| --- | --- | ---: | ---: | ---: |
| 60m | dense | 22.742 | 24.636 | 59.362 |
| 60m | reference | 40.211 | 38.952 | 76.610 |
| 60m | optimized | 38.531 | 27.467 | 64.486 |
| 60m | fused | 32.378 | 25.546 | 62.065 |
| 130m | dense | 52.321 | 52.766 | 118.204 |
| 130m | reference | 43.229 | 59.660 | 121.947 |
| 130m | optimized | 56.270 | 42.831 | 105.731 |
| 130m | fused | 49.279 | 40.347 | 104.243 |

| 模型 | fused对比基线 | hook耗时降低 | 完整单步耗时降低 |
| --- | --- | ---: | ---: |
| 60m | dense | -3.69% | -4.55% |
| 60m | reference | 34.42% | 18.99% |
| 60m | optimized | 6.99% | 3.75% |
| 130m | dense | 23.54% | 11.81% |
| 130m | reference | 32.37% | 14.52% |
| 130m | optimized | 5.80% | 1.41% |

每个模型中reference/optimized/fused在两种窗口的逐rank loss序列、DDP bits和逐次AllReduce payload尺寸/dtype布局完全一致。Dense与压缩Muon不是同一算法路径，不要求loss相同。

单步与hook：无profiler，50步预热+100步测量，第41步起启用压缩。纯通信：独立profiler，50步预热+10步测量；最慢rank的NCCL AllReduce kernel累计时长/10，排除barrier和Muon AllGather。含协议等待与profiling扰动，非wire time，不能与无profiler hook直接相减。

全部正式臂启动前检查4卡无GPU compute进程且空闲显存>=23000MiB，空闲不足由脚本等待；运行中监测外部GPU进程，干扰结果拒绝接受。原始trace无损gzip保留，实际源码snapshot与哈希保留。

本轮每项一次，没有估计独立重复波动或统计显著性；严格阻塞结果不外推正常异步DDP吞吐或收敛。模型、seq、精度、压缩比例、NCCL设置均未为追求收益而改变。

首轮在进程退出阶段触发rank3归属误报，NCCL日志证实是自有rank；该次试运行完整保存在output/CM328-CM343-fused-small-models-rejected-monitor-attempt1，不纳入数字。监测器已缓存运行中确认的子进程身份，正式重跑前重新检查四卡空闲。

实验CM328–CM343；未修改算法实现，仅新增补测和汇总脚本；代码未提交。

存储更新（2026-11-12）：依用户要求已删除本批原始大trace，保留汇总、配置、日志及源码；删除清单与SHA256见同目录removed-large-traces.json。
