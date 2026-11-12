"""Summarize paired uninstrumented timing and separate NCCL profiling."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM266-CM301-arc-rows-ab'

def main():
    jobs=json.loads((OUT/'manifest.json').read_text())
    assert all((OUT/j['id']/'accepted.json').exists() for j in jobs)
    from c4.scripts.extract_arc_collectives import extract
    extract(OUT,jobs)
    rows=[]
    for model in ('60m','130m','350m'):
        selected=[j for j in jobs if j['model']==model]
        metrics={}
        for metric,profile,key in [('iteration',False,'mean_iteration_seconds'),('hook',False,'mean_blocking_hook_seconds'),('collective',True,None)]:
            values={arm:[] for arm in ('reference','optimized')}
            for repeat in range(1,4):
                results={}
                for arm in values:
                    job=next(j for j in selected if j['repeat']==repeat and j['profile']==profile and j['arm']==arm)
                    path=OUT/job['id']
                    result=json.loads((path/'all_results.json').read_text())
                    results[arm]=result
                    if profile:
                        summaries=[json.loads((path/f'collective-summary-rank{r}.json').read_text()) for r in range(4)]
                        value=max(s['allreduce_kernel_ms_per_update'] for s in summaries)
                    else: value=result[key]*1000
                    values[arm].append(value)
                assert results['reference']['rank_losses']==results['optimized']['rank_losses'],(model,repeat,metric,'loss mismatch')
                assert results['reference']['rank_ddp_communication_bits']==results['optimized']['rank_ddp_communication_bits']
                if profile:
                    counts=[]
                    for arm in values:
                        job=next(j for j in selected if j['repeat']==repeat and j['profile'] and j['arm']==arm)
                        counts.append([json.loads((OUT/job['id']/f'collective-summary-rank{r}.json').read_text())['allreduce_kernel_count'] for r in range(4)])
                    assert counts[0]==counts[1]
            metrics[metric]={arm:{'mean_ms':statistics.mean(v),'stdev_ms':statistics.stdev(v),'runs_ms':v} for arm,v in values.items()}
            metrics[metric]['paired_reduction_percent']=statistics.mean((a-b)/a*100 for a,b in zip(values['reference'],values['optimized']))
        rows.append({'model':model,'metrics':metrics})
    (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
    text=['# ARC-TopK 实现优化：4 卡前后对照','',
          '每个配置3次独立进程重复，固定seed1243；下表为均值 ± 样本标准差，单位ms/update。降低比例为同轮配对百分比的均值。',
          '', '| 模型 | 指标 | 原实现 | 优化实现 | 配对耗时降低 |', '| --- | --- | ---: | ---: | ---: |']
    for row in rows:
        for name,label in [('collective','NCCL AllReduce kernel'),('hook','完整梯度 hook'),('iteration','完整训练单步')]:
            m=row['metrics'][name];a=m['reference'];b=m['optimized']
            text.append(f"| {row['model']} | {label} | {a['mean_ms']:.3f} ± {a['stdev_ms']:.3f} | {b['mean_ms']:.3f} ± {b['stdev_ms']:.3f} | {m['paired_reduction_percent']:.2f}% |")
    text+=['','所有配对测量窗口的逐rank loss序列及DDP通信bits逐项完全一致；profiler捕获的AllReduce kernel数量、逐次payload尺寸与dtype也一致，barrier依据NCCL元数据排除。',
           '', '实现：以紧凑行索引替代逐元素索引展开，去除重复gather和独立projection的clone；保留reference路径，随机投影、TopK、EF和collective顺序/形状不变。',
           '', '设置：4张空闲RTX4090，FP32，seq256，每卡batch1，C4/Muon，EF14，保留0.2、r4，SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞。每臂启动前检查空闲，运行中检查其他GPU进程。',
           '', '无profiler单步/hook：50步预热+100步测量。独立profiler：50步预热+10步测量；每rank累计NCCL AllReduce GPU kernel时长/10，再取最慢rank，包含投影和稀疏值AllReduce，排除Muon AllGather。非wire time；与无profiler单步不可直接相减。',
           '', '纯通信数据量和collective布局未变，因此kernel时间变化不能解释为流量减少，可能包含协议等待、kernel调度和测量波动。压缩计算的节省主要反映在hook和单步。',
           '', '这是严格阻塞诊断，不能外推异步DDP吞吐或收敛。未做显著性检验。此前误判自身worker的首次试运行保存在rejected-monitor-attempt1，不纳入结果。',
           '', '实验编号CM266–CM301，方法M002；源码哈希、逐臂命令/配置、原始逐rank结果与CUDA trace保存在同目录。']
    report='\n'.join(text)+'\n'
    (OUT/'comparison.md').write_text(report)
    print(report)
if __name__=='__main__':main()
