# ARC-TopK 实现优化、测量与语义审计归档

这些是本次工作的持久报告。运行配置、精简逐步结果和日志仍在服务器output/对应目录；这里的报告和汇总随Git提交，供独立于运行目录查阅。

| 批次 | 内容 | 报告 |
| --- | --- | --- |
| CM266-CM301-arc-rows-ab | 首轮行索引优化 | [报告](CM266-CM301-arc-rows-ab/comparison.md) |
| CM302-CM319-dense-arc-control | Dense补测 | [报告](CM302-CM319-dense-arc-control/comparison.md) |
| CM320-CM327-fused-original-setting | 350M融合版 | [报告](CM320-CM327-fused-original-setting/comparison.md) |
| CM328-CM343-fused-small-models | 60M/130M融合版 | [报告](CM328-CM343-fused-small-models/comparison.md) |
| CM344-CM361-pipeline | 流水线实现 | [报告](CM344-CM361-pipeline/comparison.md) |
| arc-semantics-audit-20261112 | 增强语义审计 | [报告](arc-semantics-audit-20261112/report.md) |

实现入口：comm_hooks/group_topk_hook_no_reshape.py、comm_hooks/arc_row_kernels.py；公共参数--arc_topk_implementation reference|optimized|fused|pipeline，默认fused，pipeline可选。

复现脚本位于c4/scripts/：run_arc_pipeline.py与profile_arc_compact.py用于当前小体积诊断；audit_arc_semantics.py用于增强语义审计。旧profile_arc_collectives.py及早期控制器保留用于历史复现，会产生较大trace，日常优先使用compact版本。

结果口径：四卡FP32/seq256、每卡batch1、Muon、EF14、保留0.2/r4、SHM默认channel、严格阻塞。早期批次各3次重复，后续探索批次各1次，具体见各报告。通信kernel来自独立instrumented run，含协议等待和profiling扰动，不是wire time，不与无profiler的hook相减。

已删除会话生成的156份原始大trace（约1.58GiB）；删除清单和SHA256保留。新诊断不保存trace/checkpoint。

本地导出的报告、JSON、补丁和审计脚本副本不再保留。补丁无需重复归档，完整实现和脚本直接保存在Git提交中。
