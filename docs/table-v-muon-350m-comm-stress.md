# ARC-TopK Table V：350M Muon 通信占比消融

- question：Socket、短序列和大DDP bucket能否提高通信占比并扩大ARC-TopK相对Dense的优势。
- mode：exploratory communication-stress ablation，不作为论文Table V复现。
- common controls：350M、本地C4、8×RTX 4090、FP32、每卡batch 1、GA1、禁用P2P、
  Muon配置、100步热身、50步计时、ARC ratio 0.2/EF14/r=4。
- priority 1：seq256，禁用SHM，强制单线程/单socket NCCL Socket和单CTA。
- priority 2：seq64，SHM单channel；其余沿用既有8卡FP32设置。
- priority 3：seq256，SHM单channel，DDP `bucket_cap_mb=1024`。
- arms：每个优先级各运行Dense Muon和ARC-TopK + Muon。
- decision rule：每对计算 `Dense time / ARC time`；大于1表示ARC更快，并与既有
  8卡FP32单channel基线1.1104×比较。
- replication：每臂单次轨迹，50步不是独立实验；保留所有失败，不调整配置。
- artifacts：`output/CM048-CM053-table-v-muon-350m-comm-stress/`。
