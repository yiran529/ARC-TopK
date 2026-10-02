# ARC-TopK Table V：350M Muon BF16 四卡/八卡单通道 SHM 消融

- question：BF16梯度将Dense通信量减半后，ARC-TopK在四卡和八卡下是否仍有墙钟优势。
- mode：exploratory communication-stressed ablation，不作为论文Table V复现。
- changed variables：相对对应FP32实验将模型/梯度dtype改为BF16；分别使用world size 4和8。
- controls：350M、本地C4、每卡batch 1、sequence 256、GA1、SHM、禁用P2P、
  单NCCL channel、Muon配置、100步热身和50步计时均保持不变。
- arms：每个world size下配对Dense Muon与ARC-TopK + Muon；ARC使用ratio 0.2、
  EF14、r=4。
- decision rule：分别计算四卡和八卡的 `Dense time / ARC time`；大于1表示ARC更快。
- replication：每臂单次轨迹，50步不是独立实验；保留失败，不调整配置。
- artifacts：`output/CM044-CM047-table-v-muon-350m-bf16-shm1-ws4-ws8/`。
