# ARC-TopK Table V：350M Muon 无反向传播通信重叠消融

- question：移除梯度通信与backward的重叠，是否会扩大ARC-TopK相对Dense的优势。
- mode：exploratory communication-overlap ablation，不作为论文Table V复现。
- common controls：350M、本地C4、8×RTX 4090、FP32、每卡batch 1、seq256、GA1、
  SHM单channel、禁用P2P、Muon配置、100步热身、50步计时、ARC ratio 0.2/EF14/r=4。
- implementation 1（CM054–CM055）：设置DDP `bucket_cap_mb=2048`，使350M模型尽量形成
  一个大梯度桶，从而把梯度同步推迟到backward末端；这是近单桶实现，不预先声称实际桶数恰好为1。
- implementation 2（CM056–CM057）：保持默认DDP bucket大小；每个通信hook使用阻塞collective，
  并在返回已完成Future之前执行显式CUDA同步，严格阻止该hook的通信与后续backward重叠。
- arms：每种实现各运行Dense Muon和ARC-TopK + Muon。
- decision rule：每对计算 `Dense time / ARC time`；大于1表示ARC更快。与既有8卡FP32
  SHM单channel基线1.1104×、`bucket_cap_mb=1024`结果1.1852×比较。
- interpretation：blocking实现包含逐桶host等待开销，衡量的是“严格串行通信”这一人为setting，
  不能单独用来代表正常DDP训练吞吐。
- replication：每臂单次轨迹，50步不是独立实验；保留所有失败，不调整配置。
- artifacts：`output/CM054-CM057-table-v-muon-350m-no-overlap/`。
