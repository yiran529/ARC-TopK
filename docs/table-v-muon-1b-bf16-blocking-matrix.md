# ARC-TopK Table V：1B BF16 严格阻塞与 Socket 补充矩阵

- question：在 1B 模型和 BF16 下，四种既有通信 setting 中 Dense、TopK、RandK 与
  ARC-TopK 的端到端迭代耗时如何变化。
- mode：exploratory communication-stress ablation，不作为论文 Table V 复现。
- experiment IDs：CM106–CM121。
- model：`llama_1b.json`（实际参数量以运行产物为准）。
- arms：Dense、TopK、RandK、ARC-TopK。
- common controls：本地 C4、BF16、每卡 batch 1、seq64、GA1、Muon、seed 1243、
  100 步热身、50 步计时、压缩从第 101 步开始、ratio 0.2、EF14、tensor sparsity、
  禁用渐进压缩、ARC rank 4、禁用 P2P、单 NCCL channel/CTA。

| IDs | GPUs | DDP bucket | Hook blocking | NCCL transport |
|---|---:|---:|---|---|
| CM106–CM109 | 4 | 1024 MB | 否 | SHM |
| CM110–CM113 | 4 | PyTorch 默认 | 是 | SHM |
| CM114–CM117 | 8 | PyTorch 默认 | 是 | SHM |
| CM118–CM121 | 8 | PyTorch 默认 | 是 | Socket/loopback，禁用 SHM |

- primary outcome：最慢 rank 的连续 50 步 wall time除以 50。
- decision rule：在每个 group 内计算 `Dense time / method time`；大于 1 表示压缩方法
  更快，同时报告相对 Dense 的耗时变化百分比。
- replication：每个 cell 单次轨迹，50 步不是独立重复；保留所有失败，不选择性重跑。
- execution：16 个 cell 串行运行，每个 cell 超时 2 小时；启动前只检查一次 GPU 空闲状态，
  不轮询。
- artifacts：`output/CM106-CM121-table-v-muon-1b-bf16-blocking-matrix/`。
