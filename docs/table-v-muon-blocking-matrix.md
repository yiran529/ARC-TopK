# ARC-TopK Table V：严格阻塞与Socket通信完整矩阵

- question：在不同world size与人为削弱的通信路径下，完全移除backward/梯度通信重叠后，
  TopK、RandK和ARC-TopK相对Dense的端到端迭代耗时如何变化。
- mode：exploratory communication-stress ablation，不作为论文Table V复现。
- models：LLaMA 60M、130M、350M。
- arms：Dense、TopK、RandK、ARC-TopK。
- common controls：本地C4、FP32、每卡batch 1、seq256、GA1、Muon、seed 1243、
  100步热身、50步计时、压缩从第101步开始、ratio 0.2、EF14、tensor sparsity、
  禁用渐进压缩、ARC rank 4、禁用P2P、单NCCL channel/CTA。

| IDs | GPUs | DDP bucket | Hook blocking | NCCL transport |
|---|---:|---:|---|---|
| CM058–CM069 | 4 | 1024 MB | 否 | SHM |
| CM070–CM081 | 4 | PyTorch默认 | 是 | SHM |
| CM082–CM093 | 8 | PyTorch默认 | 是 | SHM |
| CM094–CM105 | 8 | PyTorch默认 | 是 | Socket/loopback，禁用SHM |

- strict-blocking semantics：每个DDP通信hook使用同步collective；在返回已完成Future前执行
  `torch.cuda.synchronize(tensor.device)`。这会阻止autograd继续计算后续梯度，因此梯度同步不能与
  后续backward重叠。该栅栏对Dense、TopK、RandK、ARC-TopK及其前100步Dense warmup一致生效。
- Socket controls：`NCCL_NET=Socket`、`NCCL_SOCKET_IFNAME=lo`、1个socket thread、
  每线程1个socket、`NCCL_SHM_DISABLE=1`。
- primary outcome：最慢rank的连续50步wall time除以50。
- decision rule：在每个group/model内计算 `Dense time / method time`；大于1表示压缩方法更快，
  同时报告相对Dense的耗时变化百分比。
- replication：每个cell单次轨迹，50步不是独立重复；保留所有失败，不进行选择性重跑。
- execution：48个cell串行运行，每个cell超时2小时；启动前只检查一次GPU空闲状态，不轮询。
- artifacts：`output/CM058-CM105-table-v-muon-blocking-matrix/`。
