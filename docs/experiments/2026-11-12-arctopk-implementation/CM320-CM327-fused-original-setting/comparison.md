# 原 setting：融合 ARC-TopK 对照

4张空闲RTX4090；350M，FP32，seq256，每卡batch1，Muon，EF14，保留0.2，r4，SHM默认channel、禁用P2P，单bucket8192MiB，严格阻塞。每版本一次，seed1243。

| 实现 | NCCL AllReduce kernel ms/步 | 完整 hook ms/步 | 完整单步 ms/步 |
| --- | ---: | ---: | ---: |
| dense | 137.408 | 141.163 | 300.389 |
| reference | 104.839 | 129.994 | 290.208 |
| optimized | 101.692 | 91.504 | 252.037 |
| fused | 92.759 | 87.095 | 249.712 |

| fused 对比基线 | hook 耗时降低 | 完整单步耗时降低 |
| --- | ---: | ---: |
| dense | 38.30% | 16.87% |
| reference | 33.00% | 13.95% |
| optimized | 4.82% | 0.92% |

reference、optimized、fused两种计时窗口的逐rank loss序列和DDP bits完全相等；逐次AllReduce payload尺寸/dtype布局也相等。4卡smoke每rank180组逐位一致性通过（FP32/BF16、noef/ef14/ef21、多比例、多bucket）。EF21沿用已验证的行索引路径。

无profiler单步/hook：50步预热+100步测量。独立通信profiler：50步预热+10步测量，排除barrier与Muon AllGather；NCCL kernel含协议等待，不能与无profiler hook相减，非wire time。

本轮每版本仅一次，未评估独立重复波动或显著性，不外推正常异步DDP吞吐或收敛。所有正式臂启动前检查4卡空闲；脚本等待空闲卡并监测运行中外部GPU进程，干扰结果拒绝接受。

实现：融合行提取/写通信缓冲/EF14清零，避免中间gather张量与索引复制；投影randn、GEMM、TopK、平均与collective顺序/形状不变。Triton只做无算术数据复制/清零，不改变浮点归约。

实验CM320–CM327，原始trace无损gzip保存；源码哈希和逐臂配置保留。

已将融合实现设为默认。通用参数`--arc_topk_implementation reference|optimized|fused`可选择原版、上一版和融合版。EF21及CPU保留行索引路径；通用CLI默认/覆盖和CPU noef/EF14/EF21回退对照已通过。正式测量使用显式版本参数，测试源码保存在source-snapshot；测量完成后只做默认开关接入和CPU分派保护，不改变已测CUDA融合kernel。

累计相对原版ARC的hook降低为33.00%；本轮相对上一版优化的hook增量为4.82%，完整单步增量为0.92%。未调整seq256、FP32、压缩率或NCCL setting。全部改动尚未提交。

存储更新（2026-11-12）：依用户要求已删除本批原始大trace，保留汇总、配置、日志及源码；删除清单与SHA256见同目录removed-large-traces.json。
