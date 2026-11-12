"""Compact aggregate-only pipeline comparison; no raw traces retained."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM344-CM361-pipeline'

def main():
    jobs=json.loads((OUT/'manifest.json').read_text())
    assert len(jobs)==18 and all((OUT/j['id']/'accepted.json').exists() for j in jobs)
    summary={};text=['# ARC-TopK 投影流水线：原 setting 对照','',
        '4张空闲RTX4090；FP32、seq256、每卡batch1、Muon、EF14、保留0.2、r4、SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞；每版本一次，seed1243。',
        '', '| 模型 | 实现 | NCCL kernel ms/步 | 完整hook ms/步 | 完整单步 ms/步 |',
        '| --- | --- | ---: | ---: | ---: |']
    reductions=[]
    for model in ('60m','130m','350m'):
        results={};summary[model]={}
        for arm in ('dense','fused','pipeline'):
            result={}
            for profile in (False,True):
                j=next(j for j in jobs if j['model']==model and j['arm']==arm and j['profile']==profile)
                p=OUT/j['id'];result['profile' if profile else 'timing']=json.loads((p/'all_results.json').read_text())
                if profile:
                    ss=[json.loads((p/f'collective-summary-rank{r}.json').read_text()) for r in range(4)]
                    assert all(not s['raw_trace_saved'] for s in ss)
                    result['collective_ms']=max(s['allreduce_kernel_ms_per_update'] for s in ss)
                    result['layouts']=[s['collective_layout_sha256'] for s in ss]
            results[arm]=result
            m=summary[model][arm]={'collective_ms':result['collective_ms'],
                'hook_ms':result['timing']['mean_blocking_hook_seconds']*1000,
                'iteration_ms':result['timing']['mean_iteration_seconds']*1000,
                'peak_allocated_mib':max(result['timing']['rank_peak_allocated_bytes'])/1024**2}
            text.append(f"| {model} | {arm} | {m['collective_ms']:.3f} | {m['hook_ms']:.3f} | {m['iteration_ms']:.3f} |")
        for mode in ('timing','profile'):
            assert results['fused'][mode]['rank_losses']==results['pipeline'][mode]['rank_losses'],(model,mode,'loss')
            assert results['fused'][mode]['rank_ddp_communication_bits']==results['pipeline'][mode]['rank_ddp_communication_bits'],(model,mode,'bits')
        assert results['fused']['layouts']==results['pipeline']['layouts'],(model,'payload order')
        for arm in ('fused','dense'):
            for metric in ('hook_ms','iteration_ms'):
                summary[model][arm]['pipeline_reduction_'+metric]=(1-summary[model]['pipeline'][metric]/summary[model][arm][metric])*100
            reductions.append(f"| {model} | {arm} | {summary[model][arm]['pipeline_reduction_hook_ms']:.2f}% | {summary[model][arm]['pipeline_reduction_iteration_ms']:.2f}% |")
    text+=['','| 模型 | pipeline对比基线 | hook耗时降低 | 完整单步耗时降低 |','| --- | --- | ---: | ---: |',*reductions,
        '', 'fused/pipeline在两种窗口中的逐rank测量loss序列、DDP bits及实际collective入队顺序、payload尺寸/dtype/world-size哈希完全一致。四卡180组/rank逐位smoke通过（FP32/BF16、noef/EF14/EF21、多比例、多bucket），含CUDA RNG与误差状态。EF21和CPU保留行索引路径。',
        '', '流水线先按原顺序生成投影并异步提交AllReduce，再按原顺序wait、平均、TopK和打包；不合并collective、不改浮点归约、随机数调用形状或顺序。峰值显存记录在comparison.json，投影临时缓冲仅保留到当前hook结束。',
        '', '主timing无profiler：50步预热+100步测量。独立诊断50+10，在主线程中从更新50完成后捕获到更新60完成，校验并剔除起点barrier；聚合NCCL AllReduce CUDA kernel时长，排除Muon AllGather，按最慢rank累计/10。kernel包含协议等待与profiling扰动，非wire time，不能与无profiler hook直接相减。',
        '', '不导出或保留原始trace，不保存checkpoint；只保留精简JSON、日志、manifest、源码snapshot。按用户要求已清理本会话此前156份大trace（约1.58GiB），保留旧汇总和删除哈希清单。',
        '', '正式臂启动前等待GPU0–3空闲，运行中监测其他GPU进程，干扰结果拒绝接受。每版本仅一次，未评估统计显著性；严格阻塞结果不能外推异步DDP吞吐或收敛。',
        '', '首轮独立诊断因DDP hook线程与profiler启停线程不一致而失败；该轮日志/源码存放rejected-profile-thread-attempt1。修复仅涉及诊断器，将启停移至主线程；9次无profiler timing保留，算法代码未改，诊断重新运行。',
        '', '实验CM344–CM361，沿用方法M002（实现调度优化），Dense为M001；代码未提交。']
    report='\n'.join(text)+'\n';(OUT/'comparison.json').write_text(json.dumps(summary,indent=2));(OUT/'comparison.md').write_text(report)
    p=ROOT/'docs/worklog/M002-arctopk-muon.md';p.write_text(p.read_text().rstrip()+'\n\n## 2026-11-12：CM344–CM361 投影流水线结果\n\n'+report)
    print(report,flush=True)
if __name__=='__main__':main()
