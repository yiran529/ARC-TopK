# ARC-TopK Table V：350M Muon 八卡单通道 SHM 消融

- question：将world size从4增至8后，增加的通信压力是否使ARC-TopK快于Dense Muon。
- mode：exploratory communication-stressed ablation，不作为论文Table V复现。
- paired baseline：四卡单通道SHM下Dense为0.272782秒，ARC-TopK为0.279739秒。
- changed variable：world size从4变为8，因此每卡batch保持1而global batch从4变为8。
- controls：350M、本地C4、FP32、sequence 256、GA1、SHM、禁用P2P、单NCCL
  channel、Muon配置、100步热身和50步计时均保持不变。
- arms：Dense Muon与ARC-TopK + Muon；ARC使用ratio 0.2、EF14、r=4。
- decision rule：主指标为 `Dense time / ARC time`；大于1表示八卡受限设置下ARC更快。
- replication：每臂单次轨迹，50步不是独立实验；保留失败，不调整配置。
- artifacts：`output/CM042-CM043-table-v-muon-350m-shm1-ws8/`。
