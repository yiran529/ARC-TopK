# M001：可切换的 Muon 基线

## 2026-10-02：准备 ARC-TopK Table V Muon 测速

- 计划用 CM020、CM024、CM028、CM032 测量 60M、130M、350M、1B Dense Muon。
- 论文 Table V 的 Adam 替换为统一 Muon；每卡 batch 1、序列长度256、100步热身后
  连续统计50步。详细协议见 `docs/table-v-muon-protocol.md`。
- Dense 是同模型压缩臂的主要墙钟对照；Muon内部AllGather保留并计入端到端时间。
- 当前条目记录实现与启动准备，结果须在各单元完整结束后另行追加。
- 4张RTX 4090真实C4短smoke使用GPU 4–7，Dense完成1步热身和2步测量；
  NCCL日志确认 `SHM/direct/direct`。短窗口结果不进入正式表格。
- 16:57（北京时间）正式串行队列已在tmux会话 `arctopk_table_v_muon` 启动，
  固定使用GPU 4–7；CM020已进入运行。产物根目录为
  `output/CM020-CM035-table-v-muon/`，控制器日志为
  `logs/CM020-CM035-table-v-muon-controller.log`。
- 用户随后要求使用仓库正式C4训练脚本的FP32口径。BF16队列在CM020–CM031完成、
  CM032运行时停止；全部旧产物保留但不纳入正式汇总。FP32重跑使用 `rerun1`
  身份和独立目录 `output/CM020-CM035-table-v-muon-fp32-rerun1/`。
- 1B Dense FP32的1+1步smoke在首次Muon step OOM；24GB卡每rank约已用
  23.02 GiB，`torch.stack` 尚需1.50 GiB。按预注册失败规则保留1B正式单元，
  不自动改变dtype、batch或实现，失败后继续队列。
- 17:09（北京时间）FP32 `rerun1` 正式串行队列已在tmux会话
  `arctopk_table_v_muon_fp32` 启动，固定使用GPU 4–7；控制器日志为
  `logs/CM020-CM035-table-v-muon-fp32-rerun1-controller.log`。

## 2026-10-01：CM014，130M matrix LR 0.01 对照

- 用户要求与 ARC-TopK + Muon 先后串行比较。CM014 在 CM004 的 130M Dense Muon
  配置基础上仅将 matrix LR 从 `0.02` 改为 `0.01`；保留 seed 1243、16,785
  update、全局 batch 512、FP32、scalar LR 0.001 等设置。
- 新 C4 路径为 `/home/wyr/greedy_lore/c4/c4_en/en`，串行脚本引用前 30 个训练
  分片及全部 8 个验证分片，以匹配旧运行的数据范围；原数据文件已不可用，无法
  逐文件核验内容一致性。
- 实验编号：`CM014-m001-dense-muon-llama130m-c4-2p2b-ws4-s1243-lr001`。
  脚本：`c4/scripts/run_cm014_cm015_muon_130m_lr001_c4.sh`，状态记录在
  `output/CM014-CM015-muon-llama130m-c4-lr001-serial/status.tsv`。
- 用户在开始训练前将 CM014 和 CM015 的训练步数统一改为 20,000；CM014 的
  最终运行编号改为 `CM014-m001-dense-muon-llama130m-c4-2p62b-ws4-s1243-lr001`。
  W&B project 为 `ARC-TopK-LLaMA-130M`，`WANDB_MODE=online`。

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
- 梯度累积时，非最后一个 microbatch 在 DDP `no_sync()` 中执行 forward/backward，
  只有完成一个完整 accumulation group 才同步梯度；W&B 的 `loss` 改为该 update
  内所有 microbatch 原始 loss 的算术平均，并在 update 后清零。W&B 的 step 统一使用
  optimizer `update_step`，新增 `--eval_tokens`（默认 10M）可缩短 smoke evaluation。
- 正常结束时调用 `wandb.finish()` 和 `dist.destroy_process_group()`，移除 `exit()`。

### 验证

- 新增 CPU 单测覆盖训练步数边界、checkpoint 默认行为、评估 token 边界与批次计数、
  perplexity、W&B 指标命名及 DDP/Muon 通信统计。
- 未启动训练、未下载数据、未读取 `/home/wyr/.netrc`。

## 2026-09-21：CM001 正式训练启动

- 实验编号：`CM001-muon-dense-llama60m-c4-1p1b-ws4-s1243`。
- 使用 GPU 2、3、6、7，LLaMA-60M、C4、sequence length 256、FP32，
  per-device microbatch 32、gradient accumulation 4、global batch 512，训练
  8,393 optimizer updates（1,100,087,296 个名义 token slots）。
- 优化器为 dense Muon：matrix LR 0.02、momentum 0.95、spectral-norm scaling；
  scalar AdamW LR 0.001；两条路径 weight decay 均为 0。cosine scheduler、
  warmup 1,000、minimum LR ratio 0.1、gradient clipping 1.0。
- 默认不保存 checkpoint；W&B project 为 `ARC-TopK-LLaMA-60M`，最终在本地
  8 个 validation shards 上评估约 10M effective prediction tokens。
- 启动时 GPU 2、3、6 空闲；GPU 7 有另一个用户进程占用约 8.0 GiB，利用率约
  36%。用户明确要求不做 GPU smoke、直接启动，因此 CM001 的 wall-time/吞吐可能
  受共享 GPU 7 干扰，质量指标仍可使用。
- 可复现启动脚本：`c4/scripts/run_cm001_muon_60m_c4.sh`；tmux 会话：
  `cm001-muon60m`；主日志：`output/CM001-muon-dense-llama60m-c4-1p1b-ws4-s1243/train.log`。

## 2026-09-22：CM002--CM005 串行实验队列

- `CM002` 使用 Dense AdamW 训练 60M，LR 为 0.002，作为论文 Table IV 基线。
- `CM004-m001-dense-muon-llama130m-c4-2p2b-ws4-s1243` 使用与 CM001 相同的
  Muon 配置训练 130M，名义 token slots 为 2.2B。
- 二者与 M002 的 60M/130M ARC-TopK + Muon 实验由
  `c4/scripts/run_cm002_cm005_serial.sh` 串行编排；不使用结果 gate，单项失败后
  仍继续下一项。控制脚本等待 4 张低占用 GPU 后启动，允许与已有进程共享。

## 2026-10-02：CM014 最终结果核验

- 目的与配置：记录 130M Dense Muon 的 matrix LR `0.01`、20,000 update 对照，
  seed 1243；运行编号 `CM014-m001-dense-muon-llama130m-c4-2p62b-ws4-s1243-lr001`。
- 来源：同名 `output/` 目录的 `all_results.json`、`train.log`，以及
  `output/CM014-CM015-muon-llama130m-c4-lr001-serial/status.tsv`。
  训练日志达到 20,000 步并标记成功，状态退出码 0，完成于 2026-10-01 19:01（北京时间）。
- validation final loss `3.0769667814366852`，PPL `21.692504398180365`，
  有效预测 token `10008907`。实际 GPU 0–3、4 张 RTX 4090；未保存 checkpoint。
- 结果及对照汇总见 [results](../results.md)。相对旧 CM004 同时改变了 LR、训练预算
  和数据路径，不能据此做单因素归因；仍为单 seed 初步结果。下一步核验同预算压缩对照。

## 2026-10-03：CM058–CM121 Table V 风格 Dense timing 记录

- Dense Muon 作为四种通信 setting 的同组基线；CM058–CM105 覆盖 FP32、seq256 的
  60M/130M/350M，CM106–CM121 补充 BF16、seq64 的 1B（实际 1,339,082,752 参数）。
- 四组 Dense 步时（A/B/C/D，单位 ms）：60M 为 `60.12/62.19/69.12/244.75`，
  130M 为 `121.02/125.46/139.77/558.77`，350M 为
  `305.92/324.99/356.33/1560.76`，1B 为 `667.56/702.06/723.16/3351.05`。
- 全部相关 cell 退出码为 0；每项为 100 步 warmup 后连续 50 步的最慢 rank
  wall-time 均值。完整设置、四方法对照和来源链接见 [results](../results.md)。
- 结果为单次、单 seed 的探索性 timing；1B 与前三种模型 dtype/seq 不同，不能直接
  用跨模型斜率归因于参数规模。

## 2026-11-12：CM302–CM319 Dense同条件补测启动

新增Dense基线18臂：60M/130M/350M，各3次独立进程重复，50+100无profiler timing与50+10独立profiling。GPU0–3空闲门禁和运行中进程树检查沿用CM266–CM301；每臂完成即gzip压缩trace。其余配置对齐本轮ARC，最终汇总核对配置与源码哈希。队列tmux dense_arc_control，产物output/CM302-CM319-dense-arc-control。参考与优化ARC复用紧接此前完成的数据，不宣称批次间随机配对。

## 2026-11-12：CM302–CM319 Dense同条件补测完成

# Dense 与原版/优化版 ARC-TopK 对照

4张空闲RTX4090，FP32，seq256，每卡batch1，Muon，SHM默认channel，禁用P2P，单bucket8192MiB，严格阻塞。各配置3次独立进程重复，固定seed1243。

单位ms/update；均值 ± 样本标准差。降低比例使用均值之比；正值表示比Dense更快。

| 模型 | 指标 | Dense | ARC 原版 | ARC 优化版 | 原版相对Dense降低 | 优化版相对Dense降低 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 60m | NCCL AllReduce kernel | 23.635 ± 0.408 | 36.549 ± 4.050 | 39.506 ± 1.353 | -54.64% | -67.15% |
| 60m | 完整梯度hook | 24.897 ± 1.212 | 38.172 ± 0.883 | 28.131 ± 0.387 | -53.32% | -12.99% |
| 60m | 完整训练单步 | 60.039 ± 0.966 | 72.646 ± 1.459 | 63.267 ± 1.047 | -21.00% | -5.38% |
| 130m | NCCL AllReduce kernel | 52.228 ± 0.839 | 47.883 ± 1.672 | 54.575 ± 2.160 | 8.32% | -4.49% |
| 130m | 完整梯度hook | 52.689 ± 0.431 | 61.150 ± 0.725 | 43.752 ± 0.464 | -16.06% | 16.96% |
| 130m | 完整训练单步 | 116.132 ± 0.582 | 125.803 ± 1.483 | 108.921 ± 1.256 | -8.33% | 6.21% |
| 350m | NCCL AllReduce kernel | 136.668 ± 2.293 | 115.391 ± 6.389 | 105.996 ± 3.650 | 15.57% | 22.44% |
| 350m | 完整梯度hook | 139.909 ± 0.691 | 131.077 ± 0.358 | 93.583 ± 0.584 | 6.31% | 33.11% |
| 350m | 完整训练单步 | 299.544 ± 2.236 | 289.984 ± 1.458 | 254.668 ± 1.746 | 3.19% | 14.98% |

Dense新测CM302–CM319；ARC使用紧接此前完成的CM266–CM301。已逐臂核对模型、精度、batch、窗口、GPU数、优化器、NCCL环境和关键源码哈希一致。两个实验批次没有交错随机化，比例是独立批次均值比较，未做显著性检验。

单步与hook为无profiler的50步预热+100步测量。纯通信来自独立profiler的50步预热+10步测量：各rank NCCL AllReduce kernel总时长/10，取最慢rank，排除barrier和Muon AllGather；包含NCCL协议等待，非物理wire time。不能与无profiler的hook直接相减。

Dense与ARC是不同训练算法路径，不要求loss相等；ARC在Muon正交化前压缩梯度，属于近似Muon。原版与优化版ARC的loss、bits、通信payload布局已确认完全一致。

这些结果属于严格阻塞系统诊断，不能直接外推正常异步DDP吞吐或收敛质量。代码尚未提交；原始trace均无损gzip保存。

## 2026-11-12：CM328–CM343 60M/130M融合版补测启动

沿用原4卡FP32/seq256/default-channel SHM严格阻塞setting，60M和130M各对比Dense/reference/optimized/fused，每臂一次无profiler timing（50+100）与独立通信profiling（50+10），seed1243。无算法改动，沿用空闲卡等待和外部GPU进程干扰拒绝机制。队列tmux arc_fused_small；产物output/CM328-CM343-fused-small-models，含实际源码snapshot和hash，原始trace逐臂gzip压缩。

## 2026-11-12：CM328–CM343 60M/130M融合版补测完成

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
