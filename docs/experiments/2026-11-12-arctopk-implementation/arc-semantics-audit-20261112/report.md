# ARC-TopK 优化语义审计

结论：在当前已测 DDP 梯度 bucket 路径中，reference、optimized、fused、pipeline 保持位级一致；未发现压缩算法、选择结果、误差反馈或随机数状态变化。执行调度变化限于异步提交投影AllReduce及无算术的行复制/清零。

| 检查项 | 结果 |
| --- | --- |
| 原始投影/压缩/解压函数、cal_k、状态保存/恢复方法与Git HEAD的AST比较 | 未变 |
| 随机投影矩阵V的实际数值及随机数调用顺序 | 逐位一致 |
| 每次collective的入队顺序、形状、dtype、group大小、SUM操作、归约前后张量 | 逐位一致；warmup异步Future路径另由最终输出验证 |
| 归约后norm、Top-K值与索引顺序 | 逐位一致 |
| 输出、局部/全局误差、CPU/CUDA RNG及hook checkpoint状态 | 逐位一致，使用uint8字节比较，包含正/负零 |
| 从reference checkpoint恢复到各优化实现后继续执行 | 一致 |
| 子通信组和非默认CUDA stream | 一致 |
| 60M/130M/350M已有完整模型测量窗口的loss和DDP bits | 一致，且算法源码哈希与本次审计匹配 |

主体测试：4张RTX4090，PyTorch 2.7.1+cu126；FP32/BF16/FP16/FP64；noef/EF14/EF21；r=1/4/8；不同保留率（含最少保留及全保留）；1D/2D/4D、奇数宽度513和真实LLaMA矩阵尺寸；随机、全零、相同值并列和带符号零；预热、多bucket及阻塞开关。子组/非默认stream补测使用FP32/BF16，涵盖world size 1/2/3/4。该覆盖不是所有参数的笛卡尔积。

每rank 158组配置、2844次bucket候选对照、9516次collective候选对照、632次checkpoint恢复；四rank合计11376次bucket对照、38064次collective对照和2528次恢复。

额外核对已有完整模型数据：每模型100步无profiler窗口和10步独立诊断窗口、各4个rank，fused/pipeline的loss及bits相同；对应核心算法文件SHA256与当前工作区一致。此前已有reference/fused的完整模型对照，本次又直接增加reference对所有实现的中间张量审计。

代码层面的依据：梯度视图互不重叠，后续投影不依赖前面张量的EF清零；随机数draw和GEMM仍按原顺序调用；projection AllReduce仍按原顺序入队，读取前逐项wait；值buffer的最终AllReduce、除world_size、EF更新与反稀疏化顺序保留。Triton kernel只做按索引复制/清零，不融合或重排浮点归约。EF21和CPU继续使用已验证行索引路径。

结论范围：这是对当前DDP路径、有限输入和上述环境的差分验证，不是对任意外部别名张量、并发随机数调用或所有硬件的形式化证明。ARC与Dense本来就是不同算法路径，本审计检查原版ARC与优化ARC是否等价。

本次未修改算法实现，仅新增审计脚本；未生成trace、checkpoint或张量dump，只保存小型JSON/日志。审计脚本为c4/scripts/audit_arc_semantics.py；结果与源码哈希见summary.json。代码未提交。
