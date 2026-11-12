# M002：ARC-TopK 与 Muon 组合

## 2026-10-03：CM122–CM265 A–D FP32 计时完成

- 22:36（北京时间）队列完成，144/144成功、0失败；每配置3次独立进程运行，
  固定seed1243，预热50步、测量更新51–150。144个原始结果均实际单bucket，
  同模型源码哈希一致；NCCL日志确认SHM，默认/单channel分别为2/1个coll channel。
- 完整四方法iter/hook均值、样本标准差及ARC配对speedup记录于 `docs/results.md`
  的“C4 / Muon：A–D FP32 严格通信路径测速（CM122–CM265）”章节。
  原始结果与aggregate/comparisons位于 `output/CM122-CM265-strict-hook-abcd-fp32/`。
- 350M四组ARC均在3轮降低iter/hook；D组iter 351.10→298.98 ms、hook
  193.21→142.47 ms，配对耗时降低均值14.85%/26.26%。60M四组慢于Dense，
  130M仅8卡出现小幅iter收益；Top-K全部组合慢于Dense。
- 单channel增加相对speedup但350M绝对耗时高于默认channel。严格阻塞结果不能
  外推为正常异步DDP吞吐；3次系统重复不是多训练seed质量证据。
- 用户取消1B追加，未修改矩阵或启动1B。下一步若研究正常吞吐，需要单独做异步
  bucket实验；本条仅归档现有结果，不启动新任务。

## 2026-10-03：A–D FP32 严格通信路径计时准备

- 用户批准4/8卡 × 默认/单channel SHM，只测FP32；50步预热、100步测量。
  60M/130M/350M × Dense/Top-K/Rand-K/ARC，3次独立重复，共144次，CM122–CM265。
- 在 `c4/table_v_timing.py` 增加统一严格hook包装、逐rank hook耗时及bucket布局；
  `comm_hooks/utils.py` 仅增加可选hook包装接口，原算法不改。
  压缩起点40，更新41–50预热压缩，窗口51–150；DDP cap8192，接受实际单bucket。
- 控制器 `c4/scripts/run_strict_hook_abcd.py`；协议 `docs/strict-hook-abcd-fp32.md`。
  结果目录 `output/CM122-CM265-strict-hook-abcd-fp32/`；同配置四方法随机顺序，
  排序seed20261003，训练seed1243，串行运行，失败保留。
- CPU计时汇总、无效窗口拒绝、单bucket接受及同轮配对检查通过；144个编号唯一。
  本地C4 50个分片与t5-base缓存可用；8张RTX4090检查时均空闲。
  四种方法4卡smoke位于 `/tmp/arctopk-strict-hook-smoke-20261003/`，4/4完成，
  均退出0且实际单bucket。只读审查发现部分JSON可能中断队列，已增加错误记录并
  继续下一臂的处理及回归验证；没有改变实验配置或算法。
- 未产生正式结果。计时口径为梯度通信路径时间及完整更新墙钟时间，不能称为纯
  wire time；Muon内部AllGather单独记录窗口bits增量，仍计入完整步耗时。
- 20:41（北京时间）队列已在tmux会话 `arc_strict_abcd_fp32` 启动。
  `status.tsv`确认144次串行运行，首臂
  `CM122-m003-topk-llama60m-A-fp32-rep1-s1243` 已开始；会话pane存活。
  launcher日志为 `output/CM122-CM265-strict-hook-abcd-fp32-launcher.log`。
  5项CPU检查、四种方法4卡smoke、语法检查及diff检查通过；正式结果待队列完成。

## 2026-10-02：准备 ARC-TopK Table V Muon 测速

- 计划用 CM023、CM027、CM031、CM035 测量 ARC-TopK + Muon 的四种 LLaMA 规模。
- 比例0.2、rank 4、EF14，在100步热身后立即以目标比例压缩并连续统计50步。
- 按用户决定沿用正常训练的压缩覆盖，不增加二维参数筛选；压缩梯度后再执行
  Muon正交化，属于近似Muon。协议与接受规则见 `docs/table-v-muon-protocol.md`。
- 当前条目记录实现与启动准备，结果须在各单元完整结束后另行追加。
- 4张RTX 4090真实C4短smoke使用GPU 4–7，ARC-TopK完成1步热身和2步测量；
  NCCL日志确认 `SHM/direct/direct`。短窗口结果不进入正式表格。
- 16:57（北京时间）CM020–CM035正式串行队列已在tmux会话
  `arctopk_table_v_muon` 启动，固定使用GPU 4–7；ARC单元将在各同规模前三个
  对照之后运行。状态与原始产物保存在 `output/CM020-CM035-table-v-muon/`。
- 用户随后将正式dtype修正为训练脚本使用的FP32。已完成的BF16 ARC单元仅保留为
  非正式产物；FP32单元使用 `CM023/027/031/035-rerun1` 身份和独立目录重跑。
- 17:09（北京时间）FP32队列已在 `arctopk_table_v_muon_fp32` 会话启动，
  固定使用GPU 4–7；不对预计无法装入24GB显存的1B单元自动降级。

## 2026-10-01：CM013，130M matrix LR 0.01 对照

- 目的：在 CM005-rerun1 的 130M C4 ARC-TopK + Muon 配置上仅将 matrix LR 从
  `0.02` 改为 `0.01`，观察 validation loss 和 perplexity 是否改善。
- 配置：16,785 update、seed 1243、4 GPU、全局 batch 512、FP32、序列长度 256；
  scalar LR `0.001`，ARC 比例 `0.2`、rank `4`、EF14、压缩起始步 1,000，均保持原值。
- 新 C4 源路径为 `/home/wyr/greedy_lore/c4/c4_en/en`，包含 50 个训练分片和
  8 个验证分片；脚本引用前 30 个训练分片及全部 8 个验证分片，以匹配原实验的
  分片范围。旧文件已不在，尚不能核验两个目录的文件哈希是否相同。
- 实验编号：`CM013-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243-lr001`。
  脚本：`c4/scripts/run_cm013_arctopk_muon_130m_lr001_c4.sh`；状态及产物写入同名
  `output/` 目录。等待 4 张空闲 GPU 后由 tmux 启动训练。
- 已通过 `bash -n` 和 `git diff --check`，确认目标 30+8 个分片存在且非空。
  2026-10-01 13:09（北京时间）已启动 tmux 会话 `CM013_m002_130m_lr001`；
  `status.tsv` 当前记录为 `waiting_for_4_idle_gpus`。GPU 空闲判据为剩余显存
  不少于 23,000 MiB、利用率不高于 5%，且无 compute 进程；每 60 秒检查一次。
- 用户随后要求先运行 Dense Muon，再运行 ARC-TopK + Muon，二者 matrix LR 均为
  `0.01`。CM013 在进入训练前取消，状态保留为 `superseded_before_training`。
  新 ARC 实验编号为 `CM015-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243-lr001`，
  在 CM014 Dense 成功结束后由串行脚本自动启动；其余训练和压缩参数延续 CM005。
  串行脚本：`c4/scripts/run_cm014_cm015_muon_130m_lr001_c4.sh`。
- 2026-10-01 13:14（北京时间）启动 tmux 会话 `CM014_CM015_130m_lr001`。
  语法、关键参数、Dense→ARC 顺序和 30+8 个源分片已验证；状态文件当前为
  `waiting_for_4_idle_gpus`，尚未进入训练。
- 用户在开始训练前将两组步数改为 20,000，并确认 W&B project
  `ARC-TopK-LLaMA-130M`、`WANDB_MODE=online`。旧等待队列于 13:26 取消并归档到
  `output/CM014-CM015-muon-llama130m-c4-lr001-serial-superseded-16785/`；
  新 ARC 运行编号为 `CM015-m002-arctopk-muon-llama130m-c4-2p62b-ws4-s1243-lr001`。
- 13:26 重启同名 tmux 会话；新的状态文件显示 `waiting_for_4_idle_gpus`，
  尚未开始训练。已验证两个运行均为 20,000 步、Dense→ARC 顺序、W&B 项目与
  online 模式；38 个分片链接已建立。

## 2026-09-22：正式实验队列

### 目的与实现口径

验证 ARC-TopK 能否降低 Muon 预训练中的梯度通信量，同时保持接近 Dense Muon
的 C4 validation perplexity。ARC-TopK 在 DDP gradient hook 中压缩并同步梯度，
Muon 随后对压缩后的全局梯度执行正交化；由于正交化是非线性的，该组合属于近似
Muon，而不是 Dense Muon 的通信等价实现。

### 配置

- `CM003-m002-arctopk-muon-llama60m-c4-1p1b-ws4-s1243`：60M，8,393
  optimizer updates，microbatch 32，gradient accumulation 4。
- `CM005-m002-arctopk-muon-llama130m-c4-2p2b-ws4-s1243`：130M，16,785
  optimizer updates，microbatch 16，gradient accumulation 8。
- 两个实验均使用 global batch 512、sequence length 256、seed 1243、FP32、
  Muon matrix LR 0.02、scalar AdamW LR 0.001、weight decay 0、warmup 1,000。
- ARC-TopK 从 update 1,000 开始，`compress_ratio=0.2`、`r=4`、error feedback
  使用 `ef14`；每个 optimizer update 只触发一次 DDP gradient hook。
- 使用固定本地 C4 数据和 8 个 validation shards，最终评估约 10M effective
  prediction tokens；默认不保存 checkpoint。

### 执行

由 `c4/scripts/run_cm002_cm005_serial.sh` 在后台等待 4 张利用率不超过 50%、
空闲显存不少于 14 GiB 的 GPU，并在选定 GPU 上依次执行 CM002 至 CM005。
任何单项实验失败都不阻止后续实验运行。状态写入
`output/CM002-CM005-serial/status.tsv`，各实验保留独立日志和 W&B run。

## 2026-09-22：CM003、CM005 启动失败后的重跑

- 首次串行队列中，CM003 和 CM005 均在约 2 秒内以退出码 2 结束，未进入训练。
  原因是 `torchrun` 将训练脚本参数 `--r 4` 误判为自身选项缩写，报
  `ambiguous option: --r`。原始失败日志保留在对应实验目录。
- 在 `torchrun` 选项与训练脚本路径之间增加 `--` 参数分隔符；不修改训练算法或
  超参数。使用 `arc_rerun` 模式仅重跑 CM003 和 CM005，并为新运行目录追加
  `-rerun1`，避免覆盖原始失败证据。
- 新队列状态记录在 `output/CM003-CM005-arc-rerun/status.tsv`。继续使用后台
  GPU 低占用等待与无 gate 串行执行，单项失败不阻止另一项启动。

## 2026-09-24：CM008 GLUE Table III 风格比较

- 目的：在 RoBERTa-base 的 GLUE 八个任务上比较 Dense Muon 与 ARC-TopK + Muon；
  seed 1240，每个完整 run 计划 10 epochs。
- SST-2 Dense Muon 5-epoch validation sweep 在三组学习率中选出 matrix LR
  `0.0002`、scalar LR `0.00005`，并用于正式任务。正式设置包含 Muon momentum
  `0.95`、spectral-norm scaling；ARC 使用 `group_topk_no_reshape`、EF21、
  `compress_ratio=0.2`、`r=4`、start step 0、compression warmup fraction `0.1`。
- 结果目录：`outputs/glue_muon_table3_formal_seed1240_dense5_formal10_retry3/`；
  汇总见 `docs/results.md` 的 CM008 部分。15/16 runs 完成；SST-2 ARC-TopK 在
  epoch 8 验证后、epoch 9 训练期间被外部中止，保留为部分结果。
- 批大小口径：CoLA/MRPC 日志记录 effective global batch 128；SST-2、STS-B、QQP
  为 64；MNLI、QNLI、RTE 按后续修正使用 GA=1、每卡 batch 16，effective global
  batch 64。结果为单 seed，且学习率由 SST-2 validation 指标选择。
- 观察：在完整结果中 ARC-TopK + Muon 于 MRPC 两项指标较高、QNLI accuracy 相同，
  其余任务低于 Dense Muon；SST-2 ARC 结果不作为同预算结论。模型 checkpoint 未保存。

## 2026-10-02：CM015 最终结果与 Dense 对照

- 目的与配置：记录 130M ARC-TopK + Muon 的 matrix LR `0.01`、20,000 update
  结果，seed 1243；编号 `CM015-m002-arctopk-muon-llama130m-c4-2p62b-ws4-s1243-lr001`。
  脚本为 `c4/scripts/run_cm014_cm015_muon_130m_lr001_c4.sh`，压缩梯度后执行 Muon
  正交化，比例 0.2、rank 4、EF14、压缩起始迭代 1,000。
- 来源：同名 `output/` 目录的 `all_results.json`、`train.log` 和串行队列 `status.tsv`。
  日志达到 20,000 步并标记成功，退出码 0；2026-10-01 23:50（北京时间）完成。
- validation final loss `3.0886164719212412`，PPL `21.94669309530929`，
  有效预测 token `10008907`。与 CM014 同数据、同预算；loss 高 `0.011650`，
  PPL 高 `0.254189`（`1.17%`），本次配置下接近 Dense Muon。
- 相较 CM005-rerun1，LR 和步数均改变，旧数据无法做文件内容核验；不能将改善
  单独归因于 LR。结果为单 seed、未保存 checkpoint。汇总及原始来源链接见
  [results](../results.md)。下一步比较完成后的 CM018/CM019，并补独立重复。

## 2026-10-03：CM058–CM121 ARC-TopK timing 矩阵

- 在梯度 DDP hook 上运行 ARC-TopK + EF14，比例 0.2、rank 4；前 100 步 Dense，
  第 101 步起压缩。四种 setting 依次为 4 卡/1024 MiB/非阻塞/SHM、4 卡/默认
  bucket/严格阻塞/SHM、8 卡/严格阻塞/SHM、8 卡/严格阻塞/Socket。
- ARC 相对同组 Dense 的耗时变化（A/B/C/D）：60M 为
  `慢22.80%/慢23.29%/慢11.84%/快36.31%`，130M 为
  `慢2.27%/慢2.91%/快0.90%/快42.56%`，350M 为
  `快6.05%/快8.71%/快13.53%/快47.55%`，1B 为
  `快13.68%/快14.37%/快19.26%/快43.65%`。
- 64/64 总矩阵 cell 均完成；原始 summary/comparisons 与完整四方法表见
  [results](../results.md)。结果为单次、单 seed；1B 使用 BF16/seq64，其余使用
  FP32/seq256，Socket 为人为通信压力，A/B 还同时改变 bucket 与 blocking。

## 2026-11-12：ARC-TopK 行索引实现优化与前后对照（进行中）

- 目的：不改变 ARC-TopK 数学/随机数/通信语义，减少重复 gather、clone 和逐元素索引展开。
- 实现：保留 reference 路径；optimized 使用紧凑 int64 行索引，index_select/index_fill_/index_copy_ 完成提取、误差反馈和恢复。投影 randn 的形状与顺序、除法及平方求和顺序、TopK 参数、collective 形状和顺序均保留。
- 修改：`comm_hooks/group_topk_hook_no_reshape.py`；计时入口 `--arc_topk_implementation reference|optimized`，默认 optimized。
- 一致性 smoke：`c4/scripts/check_arc_rows.py`，4 GPU，每 rank 180 次逐位比较；覆盖 FP32/BF16、noef/ef14/ef21、比例0.01/0.2/1.0、1D/2D/>2D、两个 bucket、5次更新及 warmup。输出、局部/全局误差、CUDA RNG、iter、bits、collective 布局全部相等。日志 `/tmp/arc-rows-check.log`。
- 正式编号 CM266–CM301，沿用 M002（仅实现优化，无新算法）。4卡 GPU0–3、FP32、SHM 默认 channel、关闭P2P、单bucket8192MiB、C4、每卡batch1、seq256、Muon、EF14、保留0.2、r4、seed1243；设置同 CM122–CM265 的 A 组。
- 模型60M/130M/350M，每版本3次独立进程配对重复，版本顺序固定随机化。无 profiler 的正式 timing 为50步预热+100步测量，压缩起点40；独立 profiler 为50步预热+10步测量（scheduler horizon60，故单独报告，不能用其 iter 作为正常 timing）。
- 纯通信口径：CUDA profiler 中 NCCL AllReduce kernel 时长之和（projection + packed values），每 rank 除以10，再取最慢 rank；排除 Muon AllGather。它包含 NCCL 协议/等待，非物理链路 wire time。严格 hook 时间包括投影、选行、EF、解压与同步；完整单步另行报告。
- 正式启动前每臂检查4张卡无 compute 进程且每卡空闲显存>=23000MiB；运行中每5秒检查所选卡其他进程组，若有干扰则拒绝接受结果。其他用户进程不受影响。空闲检查与启动之间非原子，不是排他调度锁。
- 队列：tmux `arc_rows_ab`，控制器 `c4/scripts/run_arc_rows_ab.py`，产物 `output/CM266-CM301-arc-rows-ab/`；源码SHA256、命令、逐臂配置、逐rank结果、profiler trace和接受状态均保留。
- 验证：Python编译和 git diff --check 通过。正式结果待追加；不评估PPL或收敛，不外推正常异步DDP吞吐。

### CM266–CM301 完成与结果

- 36/36正式运行完成并通过接受条件；每臂启动时GPU0–3空闲，检测未发现其他GPU进程干扰。首轮进程组误判的试运行已单独归档，不纳入汇总。
- 60M/130M/350M：完整单步均值(ms)分别为72.646→63.267、125.803→108.921、289.984→254.668；同轮配对耗时降低均值12.88%/13.41%/12.18%。完整hook分别38.172→28.131、61.150→43.752、131.077→93.583 ms，降低26.29%/28.44%/28.60%。
- 独立profile的纯NCCL AllReduce kernel(ms/update)分别36.549→39.506、47.883→54.575、115.391→105.996；60M/130M变慢、350M变快，没有统一通信收益。collective字节数、次数、顺序未改，不能解释为通信流量减少。kernel包含协议等待与profiling/调度影响，不能与无profiler的hook直接相减。
- 全部配对运行的逐rank测量loss序列和DDP通信bits完全一致。profiler kernel数量、逐次payload尺寸/dtype布局也完全一致；barrier和Muon AllGather已排除。一条kernel缺少External id及collective标签，但原始时长存在，通过CUDA correlation连接cuLaunchKernelEx及唯一包含它的同线程record_param_comms恢复归属；未估算或补造耗时。
- 结果：`output/CM266-CM301-arc-rows-ab/comparison.md` / `comparison.json`。源码和analysis脚本哈希保留；trace进行gzip无损压缩，不删除原始数据内容。
- 原参考投影/压缩/解压/cal_k函数AST与HEAD一致；4卡180组/rank的逐位hook对照通过。最终全部脚本AST语法、git diff --check通过。
- 默认optimized，计时入口可用`--arc_topk_implementation reference`回到原实现。全部改动未提交。下一步若要判断真实吞吐，另行测试正常异步DDP；本轮只完成严格阻塞前后对照。

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

## 2026-11-12：融合行kernel实现及原setting单次迭代

保持原350M/FP32/seq256/4卡默认channel SHM严格阻塞setting为主目标，不用缩短序列或BF16替代原setting。新增fused候选：Triton融合行提取、写入通信buffer及EF14清零，局部行索引用Python列表保留，不拷贝到索引缓冲；恢复使用无算术行scatter。随机投影、TopK、平均和collective布局不变，EF21沿用上一版已验证路径。四卡180次/rank逐位smoke通过（含FP32/BF16、全部EF模式），日志/tmp/arc-fused-check.log。正式CM320–CM327对比Dense/reference/optimized/fused，各一次无profiler timing及一次独立profile，50+100和50+10窗口。脚本等待4张空闲卡并拒绝运行期间的外部GPU干扰；逐臂压缩trace。未调整环境或依赖，使用已有Triton3.3.1。

## 2026-11-12：CM320–CM327 融合行kernel正式结果

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

### 默认接入与最终检查

融合实现已设为默认，公共add_comm_hook_args提供`--arc_topk_implementation reference|optimized|fused`；EF21和CPU走上一版行索引路径。公共CLI默认/覆盖及单进程Gloo CPU noef/ef14/ef21逐位回退检查通过。正式测量全用显式参数且保存source-snapshot；完成后仅调整默认值、公共参数位置与CPU保护，CUDA融合kernel未改动。原setting hook累计相对reference降低33.00%，相对Dense降低38.30%，相对上一版新增4.82%；完整单步相对Dense16.87%、原版13.95%、上一版0.92%，未达到30%。单次探索性结果，未做显著性检验。未用短seq或其他精度替代原setting目标，未提交改动。

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

## 2026-11-12：CM344–CM361 投影流水线启动

新增pipeline可选实现，默认仍为已验证的fused。保持randn/GEMM和collective入队顺序，投影AllReduce异步提交后逐项wait、平均、TopK及融合打包；EF21/CPU沿用原行索引路径。4卡180组/rank逐位smoke通过，日志/tmp/arc-pipeline-check.log。新增compact内存聚合profiler，不导出trace，四卡小型统计smoke通过；只记录kernel计数/时长及payload签名。原350M/60M/130M FP32 seq256 setting，各比较Dense/fused/pipeline一次无profiler timing（50+100）和独立诊断（50+10），等待4卡空闲、拒绝外部GPU干扰。用户要求不留大产物，已清理会话产生的156份旧trace，释放1620.3MiB，旧结果/源码/日志保留。新队列tmux arc_pipeline；目录output/CM344-CM361-pipeline。

## 2026-11-12：CM344–CM361 投影流水线结果

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

## 2026-11-12：优化语义增强审计

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

## 2026-11-12：持久报告归档与提交

依用户要求，仅远端保留资料。六批报告、JSON汇总、manifest和源码哈希已校验复制到docs/experiments/2026-11-12-arctopk-implementation/，README提供统一索引。本地导出的报告/补丁/审计副本将删除；补丁不重复归档，代码由Git提交保存。原有output目录只作为远端精简运行记录，大trace已清理。提交涵盖实现优化、测量/验证脚本、报告文档及此前授权的缓存取消跟踪；不推送。
