# M003：GLUE Top-K/Rand-K 与 Muon 组合

## 2026-10-02：准备 ARC-TopK Table V Muon 测速

- 计划用 CM021/CM025/CM029/CM033 测量 Top-K + Muon，用
  CM022/CM026/CM030/CM034 测量 Rand-K + Muon。
- 两组均使用 tensor-wise 比例0.2、EF14，在100步热身后立即以目标比例压缩，
  连续统计50步；其余数据、模型与Muon设置和Dense/ARC臂一致。
- 按用户决定沿用正常训练的压缩覆盖，不增加二维参数筛选。详细协议见
  `docs/table-v-muon-protocol.md`；结果须在各单元完整结束后另行追加。
- 4张RTX 4090真实C4短smoke使用GPU 4–7，Top-K、Rand-K均完成1步热身和2步
  测量；NCCL日志确认 `SHM/direct/direct`。修正测速入口的逐步通信量累计后，
  Top-K复测四个rank均记录非零DDP通信量。短窗口结果不进入正式表格。
- 16:57（北京时间）CM020–CM035正式串行队列已在tmux会话
  `arctopk_table_v_muon` 启动，固定使用GPU 4–7；Top-K/Rand-K单元按模型规模
  串行执行。状态与原始产物保存在 `output/CM020-CM035-table-v-muon/`。
- 用户随后将正式dtype修正为训练脚本使用的FP32。已完成的BF16 Top-K/Rand-K
  单元仅保留为非正式产物；FP32单元使用原实验号的 `rerun1` 身份独立重跑。
- 17:09（北京时间）FP32队列已在 `arctopk_table_v_muon_fp32` 会话启动，
  固定使用GPU 4–7；失败单元保留证据并继续后续单元。

## 2026-09-29：EF21 正式对照启动

- 目的：以此前 GLUE Dense Muon／ARC-TopK + Muon 正式运行的训练设置，对照 tensor 级 Top-K、Rand-K 与 Muon 的组合。梯度在 DDP 通信 hook 中压缩，同步后再由 Muon 更新。
- 启动器：`glue_fine-tuning/scripts/glue_muon_table3_randk_topk.sh`；8 个 GLUE 任务各运行 Top-K + EF21、Rand-K + EF21，seed `1240`，各计划 10 epochs。
- 模型为 `FacebookAI/roberta-base`；Muon matrix LR `0.0002`、scalar LR `0.00005`、momentum `0.95`，线性学习率调度；`compress_ratio=0.2`，前 10% optimizer steps 不压缩，压缩开始后直接使用目标比例。
- 与此前正式运行相同的 batch/累积口径：CoLA/MRPC 每卡 `16×2`、SST-2/STS-B/QQP 每卡 `8×2`、MNLI/QNLI/RTE 每卡 `16×1`；4 张 GPU，对应 global batch 分别为 128、64、64。
- `bash -n` 和 16 条 dry-run 命令检查通过。使用 GPU `0,1,2,7` 直接启动，tmux 会话 `glue_topk_randk_ef21`；启动日志为 `outputs/glue_muon_table3_randk_topk_ef21_launcher.log`，运行目录为 `outputs/glue_muon_table3_randk_topk_seed1240-formal10-gpu0127/`。
- 启动检查时 CoLA Top-K 进程与训练日志已出现；其余运行由启动器串行执行，结果尚待完成后核验。GPU 0、1 同时存在其他用户的显存占用，本实验未停止或修改其进程。

## 2026-09-29：未完成任务改为 GA=1

- 按用户要求，保留已经完成的 CoLA、SST-2、MRPC、STS-B 各两组运行及原始结果目录，不重跑。
- 旧启动器在 QQP Top-K 尚未完成时停止；旧的 QQP Top-K 训练日志保留在原目录，作为中止记录，不作为完成结果。
- 新实验编号 `CM009-m003-glue-topk-randk-ga1-seed1240-gpu0127`，仅从头运行 QQP、MNLI、QNLI、RTE 的 Top-K 与 Rand-K，共 8 组。QQP 改为每卡 batch `16`、GA `1`，保持全局 batch `64`；MNLI/QNLI/RTE 原本就是每卡 `16`、GA `1`，保持不变。均使用 4 张 GPU。
- 运行脚本：`glue_fine-tuning/scripts/glue_muon_table3_randk_topk.sh`；新结果目录：`outputs/CM009-m003-glue-topk-randk-ga1-seed1240-gpu0127/`。原目录及旧日志不覆盖。
- `bash -n` 和 dry-run 检查通过：恰好生成 8 条命令，全部为每卡 batch `16`、GA `1`，顺序为 QQP/MNLI/QNLI/RTE 每任务 Top-K、Rand-K。
- 新启动器已在 tmux 会话 `CM009_m003_glue_ga1` 启动，启动日志为 `outputs/CM009-m003-glue-topk-randk-ga1-launcher.log`；第一条运行是 QQP Top-K。后续任务由脚本串行执行。

## 2026-09-29：MNLI 显存不足后重启未完成任务

- CM009 的 QQP Top-K 与 Rand-K 均已完成，各有 `all_results.json`；MNLI Top-K 在约 7,304/61,360 步时因 GPU 1 显存不足而中止，随后串行启动器退出。失败日志保留在 `outputs/CM009-m003-glue-topk-randk-ga1-seed1240-gpu0127/mnli/muon_topk/train.log`。
- 报错显示另一个进程占用 GPU 1 约 16.13 GiB，MNLI rank 1 本身约占 7.36 GiB，申请额外 20 MiB 时仅余 13.31 MiB。未修改或停止该进程。
- 新重试编号 `CM010-m003-glue-ga1-mnli-retry-gpu0237`：从 MNLI Top-K 开始，之后顺序执行 MNLI Rand-K、QNLI Top-K/Rand-K、RTE Top-K/Rand-K。仍为每卡 batch `16`、GA `1`、四卡、seed `1240`、10 epochs 和其余相同训练参数；物理 GPU 改为 `0,2,3,7`，输出放在 `outputs/CM010-m003-glue-ga1-mnli-retry-gpu0237/`，避免复用失败目录。
- 启动器新增 `FORMAL_START_TASK` 以跳过已完成的 QQP，并可用 `GPU_MIN_FREE_MIB` 在每组运行前检查显存余量。`bash -n` 与 dry-run 检查通过，生成 6 条 MNLI/QNLI/RTE 命令。
- 重试已在 tmux 会话 `CM010_m003_glue_retry` 启动，日志为 `outputs/CM010-m003-glue-ga1-mnli-retry-launcher.log`。启动时 GPU 0/2/3/7 的空闲显存分别为 9605/24561/10913/24561 MiB，均满足 8500 MiB 门槛；MNLI Top-K 四个 rank 进程已启动。GPU 3 正有其他任务运行，因此本次训练的墙钟速度不宜与此前 GPU 组合直接比较。

## 2026-09-29：固定 GPU 0/1/2/7，OOM 后继续

- 用户要求不使用 GPU 0/2/3/7。已停止 CM010，会话及其 MNLI Top-K 训练进程退出；保留本地训练日志以记录错跑原因。
- 启动器新增 OOM 处理：训练子进程失败后，仅当本次训练日志包含 CUDA OOM 时，追加 `oom_skipped.tsv` 并继续下一组；其他错误仍停止。用替身训练命令验证了 OOM 后 Rand-K 继续、非 OOM 报错停止、失败目录不复用。
- 新运行编号 `CM011-m003-glue-ga1-gpu0127-oomskip`，从 MNLI 开始，顺序执行 MNLI/QNLI/RTE 的 Top-K 与 Rand-K 共 6 组；物理 GPU 固定 `0,1,2,7`，每卡 batch `16`、GA `1`。QQP 两组已完成结果保持在 CM009 中，不重跑。tmux 会话 `CM011_m003_glue_oomskip`，启动日志 `outputs/CM011-m003-glue-ga1-gpu0127-oomskip-launcher.log`，输出目录 `outputs/CM011-m003-glue-ga1-gpu0127-oomskip/`。
- 已从 W&B 云端删除三条未完成 run：`kyrhbvyj`（MNLI OOM）、`wcj65zp7`（错误 GPU 组合中止）、`bx6gqyts`（旧 QQP GA=2 中止）；对应本地 `wandb/run-*` 目录也已删除。训练日志与输出目录保留，已完成的 W&B run 未删除。

## 2026-09-30：主机重启后改用 GPU 0/1/2/3 恢复

- 主机于 2026-09-30 13:57 重启，CM011 的 tmux 会话消失。重启前 MNLI Top-K 再次因 GPU 1 显存不足于约 7,304/61,360 步失败并被跳过；MNLI Rand-K 在约 3,975/61,360 步时随主机重启中断。两组均无完整结果。
- 新运行编号 `CM012-m003-glue-ga1-gpu0123-resume`，从 MNLI Top-K/Rand-K 开始，继续 QNLI、RTE 两种压缩，共 6 组。物理 GPU 为 `0,1,2,3`；每卡 batch `16`、GA `1`，其余配置延续 CM011。原失败和中断日志保留，新输出目录 `outputs/CM012-m003-glue-ga1-gpu0123-resume/`。
- 启动前四卡均空闲；`bash -n` 和 dry-run 验证了 6 条命令。tmux 会话 `CM012_m003_glue_gpu0123`、启动日志 `outputs/CM012-m003-glue-ga1-gpu0123-resume-launcher.log`；启动检查时 MNLI Top-K 的四个进程均已出现。
- 按此前清理 OOM W&B 记录的要求，删除了 CM011 的 MNLI Top-K OOM run `i46ecyrt`（云端及本地缓存）。MNLI Rand-K run `ga9p3s61` 因主机重启中断，非 OOM，保留供追溯。

## 2026-10-01：CM016–CM019，C4 LLaMA Top-K/Rand-K + Muon

- 目的：分别在 60M 和 130M 规模下比较 Top-K 与 Rand-K；同一规模只改变压缩器。
  60M 参照 CM003-rerun1，使用 8,393 更新步、每卡 batch 32、GA 4、Muon matrix LR
  `0.02`；130M 参照 CM015，使用 20,000 更新步、每卡 batch 16、GA 8、matrix LR
  `0.01`。两种规模均为 4 GPU、全局 batch 512、seed 1243、FP32、序列长度 256、
  scalar LR `0.001`、cosine 调度、warmup 1,000 步。
- 压缩器为 `topk_sync`/`randk_sync`，tensor 级、`compress_ratio=0.2`、EF14，
  从第 1,000 次梯度同步起直接压缩，关闭稀疏 hook 的渐进压缩。压缩后梯度进入
  Muon 正交化，因此与 Dense Muon 相比为近似实现。
- 数据源为 `/home/wyr/greedy_lore/c4/c4_en/en`，引用前 30 个训练分片和全部
  8 个验证分片；最终评估目标为 10M 有效 token。正式运行启用 W&B online，
  60M/130M 分别用项目 `ARC-TopK-LLaMA-60M`/`ARC-TopK-LLaMA-130M`。
  明确设置 `--save_every 0`，不保存模型 checkpoint。
- 实验编号按串行顺序：CM016 60M Top-K、CM017 60M Rand-K、CM018 130M Top-K、
  CM019 130M Rand-K。启动器 `c4/scripts/run_cm016_cm019_sparse_muon_c4.sh` 在每组
  开始前等待 4 张空闲 GPU，状态文件位于
  `output/CM016-CM019-sparse-muon-c4-serial/status.tsv`，各运行保留独立训练日志。
- `bash -n`、`git diff --check`、四组顺序与配置断言已通过，目标 38 个数据分片
  链接已建立。2026-10-01 23:33（北京时间）启动 tmux 会话
  `CM016_CM019_sparse_c4`；状态记录 CM016 正在等待 4 张空闲 GPU，尚未开始训练。

## 2026-10-02：GLUE 完成结果及 C4 队列快照

- 目的：按 `experiment-results-reporting` 核验并记录已有产物，不启动或改变训练。
- GLUE Top-K/Rand-K 共 16 runs 已完成：原目录 CoLA/SST-2/MRPC/STS-B 八组，
  CM009 QQP 两组，CM012 MNLI/QNLI/RTE 六组。各 `all_results.json` 与 epoch 9
  最后验证（MNLI 为 `mnli-mm`）一致，训练进度达到各自目标步数。来源映射和逐任务
  final 指标见 [results](../results.md) 的新增 GLUE 部分。
- 配置核验：seed 1240、10 epochs、tensor 级、比例 0.2、EF21；虽配置 start iter 0，
  入口仍计算约 10% 训练预算的初始 dense 阶段。日志 `Starting compression at iteration`
  显示实际压缩在该阶段后开始，`--disable_compression_warmup` 仅关闭渐进压缩。
- Top-K/Rand-K 在不同任务各有高低，QQP 两项指标排序不一致；均为单 seed，
  不作稳定优劣判断。保留原 QQP 中止、MNLI OOM、CM010 错用 GPU 组合中止与
  CM011 重启中断记录；成功结果采用从头重跑后的 CM012，失败目录不作 final。
- C4：`output/CM016-CM019-sparse-muon-c4-serial/status.tsv` 显示 CM016 于
  2026-10-01 23:51（北京时间）在 GPU 0–3 启动。核验时训练日志未有成功完成标记，
  无 `all_results.json`；CM017–CM019 尚无 started 记录，不填最终指标。
- 下一步：C4 队列完成后核验其最终 validation loss/PPL；GLUE 补全 SST-2 ARC，
  并统一 batch/GA 后做多 seed 对照。

## 2026-10-02：CM016–CM019 最终 C4 结果核验

- 按 `experiment-results-reporting` 更新结果记录。四组均达到目标更新步数，
  `train.log` 有 `Reached max number of update steps` 和 `Script finished successfully`，
  队列状态均为 `exit_code=0`；最后一组于北京时间 11:49 完成，队列失败数为 0。
- 来源：各 `output/CM016-…` 至 `output/CM019-…` 目录的 `all_results.json`，
  使用 `final_eval_loss`、`final_eval_perplexity`；对应 validation、final，
  非训练中最后一条 loss 或 best。完整运行标识和来源链接见 [results](../results.md)。
- 实际启动参数与本地 W&B 元数据一致：60M 为 8,393 步、matrix LR 0.02、
  每卡 batch 32、GA 4；130M 为 20,000 步、matrix LR 0.01、每卡 batch 16、GA 8。
  均为 GPU 0–3 的 4 张 RTX 4090、seed 1243、FP32、全局 batch 512；
  tensor 级稀疏化、比例 0.2、EF14、第 1,000 次梯度同步起压缩，关闭渐进压缩。
- 最终指标：

| 实验 | 方法 | final update | validation loss | PPL | 有效预测 token |
| --- | --- | ---: | ---: | ---: | ---: |
| CM016 | 60M Top-K | 8,393 | 3.415821 | 30.441934 | 10,020,666 |
| CM017 | 60M Rand-K | 8,393 | 3.643432 | 38.222805 | 10,020,666 |
| CM018 | 130M Top-K | 20,000 | 3.082575 | 21.814495 | 10,008,907 |
| CM019 | 130M Rand-K | 20,000 | 3.155567 | 23.466328 | 10,008,907 |

- 130M 与同数据、同预算 CM014 Dense 对照：Top-K 的 PPL 高 0.121990（0.56%），
  Rand-K 高 1.773823（8.18%）；Top-K 比同预算 CM015 ARC 低 0.132199。
  60M 同数据 Top-K 比 Rand-K 低 7.780871；相对历史 CM001 Dense 分别为
  −0.103505（−0.34%）、+7.677366（+25.13%），但旧数据文件内容无法核验，
  与历史 Dense/ARC 的差值仅作描述性参考。
- 本次实际运行为 Rand-K，而非低秩 Rank-k；每组仅一个 seed，未保存 checkpoint，
  不能给出稳定优劣结论，也不将相同保留比例解释为相同通信量。下一步补独立 seed，
  必要时在相同数据目录重跑 60M Dense/ARC。
