"""Compare fresh Dense with the immediately preceding ARC implementation runs."""
import hashlib,json,statistics
from pathlib import Path
from c4.scripts.extract_arc_collectives import extract
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM302-CM319-dense-arc-control'
ARC=ROOT/'output/CM266-CM301-arc-rows-ab'

def main():
    jobs=json.loads((OUT/'manifest.json').read_text())
    assert len(jobs)==18 and all((OUT/j['id']/'accepted.json').exists() for j in jobs)
    extract(OUT,jobs,verify_pairs=False)
    arc=json.loads((ARC/'comparison.json').read_text())
    arcjobs=json.loads((ARC/'manifest.json').read_text())
    rows=[]
    for model in ('60m','130m','350m'):
        metrics={}
        for metric,profile,key in [('iteration',False,'mean_iteration_seconds'),('hook',False,'mean_blocking_hook_seconds'),('collective',True,None)]:
            selected=[j for j in jobs if j['model']==model and j['profile']==profile]
            assert len(selected)==3
            values=[]
            for job in selected:
                path=OUT/job['id'];result=json.loads((path/'all_results.json').read_text())
                oldjob=next(j for j in arcjobs if j['model']==model and j['profile']==profile and j['repeat']==job['repeat'] and j['arm']=='optimized')
                config=json.loads((path/'config.json').read_text());oldconfig=json.loads((ARC/oldjob['id']/'config.json').read_text())
                for field in ('dtype','max_length','batch_size','seed','optimizer','warmup_iterations','measured_iterations','ddp_bucket_cap_mb','world_size','lr','muon_scalar_lr','strict_blocking_communication','nccl_environment'):
                    assert config[field]==oldconfig[field],(field,job['id'])
                for source in ('comm_hooks/group_topk_hook_no_reshape.py','comm_hooks/default_hooks.py','comm_hooks/utils.py','optimizers/muon.py','optimizers/utils.py','c4/table_v_timing.py'):
                    def source_hash(cfg):
                        matches=[value for path,value in cfg['code_sha256'].items() if path==source or path.endswith('/'+source)]
                        assert len(matches)==1,(source,matches)
                        return matches[0]
                    assert source_hash(config)==source_hash(oldconfig),source
                if profile:
                    summaries=[json.loads((path/f'collective-summary-rank{r}.json').read_text()) for r in range(4)]
                    assert all(s['allreduce_kernel_count']==10 for s in summaries)
                    value=max(s['allreduce_kernel_ms_per_update'] for s in summaries)
                else:value=result[key]*1000
                values.append(value)
            previous=next(r for r in arc if r['model']==model)['metrics'][metric]
            metrics[metric]={'dense':{'mean_ms':statistics.mean(values),'stdev_ms':statistics.stdev(values),'runs_ms':values},
                             'reference':previous['reference'],'optimized':previous['optimized']}
            for arm in ('reference','optimized'):
                metrics[metric][arm+'_reduction_vs_dense_percent']=(1-metrics[metric][arm]['mean_ms']/statistics.mean(values))*100
        rows.append({'model':model,'metrics':metrics})
    (OUT/'comparison.json').write_text(json.dumps(rows,indent=2))
    text=['# Dense 与原版/优化版 ARC-TopK 对照','',
          '4张空闲RTX4090，FP32，seq256，每卡batch1，Muon，SHM默认channel，禁用P2P，单bucket8192MiB，严格阻塞。各配置3次独立进程重复，固定seed1243。',
          '', '单位ms/update；均值 ± 样本标准差。降低比例使用均值之比；正值表示比Dense更快。',
          '', '| 模型 | 指标 | Dense | ARC 原版 | ARC 优化版 | 原版相对Dense降低 | 优化版相对Dense降低 |',
          '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        for metric,label in [('collective','NCCL AllReduce kernel'),('hook','完整梯度hook'),('iteration','完整训练单步')]:
            m=row['metrics'][metric]
            formatted=[f"{m[a]['mean_ms']:.3f} ± {m[a]['stdev_ms']:.3f}" for a in ('dense','reference','optimized')]
            text.append(f"| {row['model']} | {label} | "+' | '.join(formatted)+f" | {m['reference_reduction_vs_dense_percent']:.2f}% | {m['optimized_reduction_vs_dense_percent']:.2f}% |")
    text+=['', 'Dense新测CM302–CM319；ARC使用紧接此前完成的CM266–CM301。已逐臂核对模型、精度、batch、窗口、GPU数、优化器、NCCL环境和关键源码哈希一致。两个实验批次没有交错随机化，比例是独立批次均值比较，未做显著性检验。',
           '', '单步与hook为无profiler的50步预热+100步测量。纯通信来自独立profiler的50步预热+10步测量：各rank NCCL AllReduce kernel总时长/10，取最慢rank，排除barrier和Muon AllGather；包含NCCL协议等待，非物理wire time。不能与无profiler的hook直接相减。',
           '', 'Dense与ARC是不同训练算法路径，不要求loss相等；ARC在Muon正交化前压缩梯度，属于近似Muon。原版与优化版ARC的loss、bits、通信payload布局已确认完全一致。',
           '', '这些结果属于严格阻塞系统诊断，不能直接外推正常异步DDP吞吐或收敛质量。代码尚未提交；原始trace均无损gzip保存。']
    report='\n'.join(text)+'\n';(OUT/'comparison.md').write_text(report);print(report,flush=True)
    for name in ('M001-muon-baseline.md','M002-arctopk-muon.md'):
        p=ROOT/'docs/worklog'/name
        with p.open('a') as f:f.write('\n## 2026-11-12：CM302–CM319 Dense同条件补测完成\n\n'+report+'\n')
    (OUT/'analysis-sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'c4/scripts/extract_arc_collectives.py',Path(__file__))},indent=2))
if __name__=='__main__':main()
