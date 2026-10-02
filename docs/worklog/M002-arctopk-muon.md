# M002：ARC-TopK 与 Muon 组合

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
