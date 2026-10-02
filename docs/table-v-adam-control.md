# ARC-TopK Table V：Dense AdamW 配对测速

- question：Muon 是否是本地 Dense 明显快于论文 Table V Dense 的主要原因。
- mode：exploratory paired control。
- target：论文 Table V 的 Dense Adam；论文值为 60M/130M/350M/1B：
  0.1469/0.3243/0.8775/3.0854 秒每迭代。
- changed variable：仅将本地 Dense Muon 换为 AdamW。
- controlled variables：同一模型配置、本地 C4、FP32、4×RTX 4090、每卡 batch 1、
  sequence 256、GA1、NCCL SHM、禁用 P2P、100步热身及连续50步计时。
- AdamW：LR 0.002、betas (0.9, 0.999)、eps 1e-8、weight decay 0；LR不作为测速变量。
- replication：每个模型单次运行；50步是同一轨迹内的计时样本，不是独立重复。
- decision rule：比较 AdamW 与既有配对 Muon Dense 的每迭代时间。如果 AdamW 仍显著
  快于论文，则优化器不是论文/本地绝对时间差异的主要解释。
- failures：保留全部失败和 OOM；不调整 batch、dtype、模型或计时窗口。
- artifacts：`output/CM036-CM039-table-v-adam-fp32/`。
