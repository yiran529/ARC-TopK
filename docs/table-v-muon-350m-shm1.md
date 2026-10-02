# ARC-TopK Table V：350M Muon 单通道 SHM 消融

- question：降低 SHM collective 并行度后，ARC-TopK 是否相对 Dense Muon 获得墙钟优势。
- mode：exploratory communication-stressed ablation，不作为论文 Table V 复现。
- baseline：既有350M FP32结果，Dense Muon 0.262689秒，ARC-TopK + Muon
  0.273476秒。
- changed variable：两臂统一设置 `NCCL_MAX_NCHANNELS=1` 和
  `NCCL_MIN_NCHANNELS=1`。
- controls：350M模型、本地C4、FP32、4×RTX 4090、每卡batch 1、sequence 256、
  GA1、SHM、禁用P2P、Muon配置、100步热身和50步计时均保持不变。
- arms：Dense Muon与ARC-TopK + Muon；ARC使用ratio 0.2、EF14、r=4。
- decision rule：主指标为 `Dense time / ARC time`；大于1表示该受限设置下ARC更快。
- replication：每臂单次轨迹，50步不是独立实验；保留失败，不调整配置。
- artifacts：`output/CM040-CM041-table-v-muon-350m-shm1/`。
