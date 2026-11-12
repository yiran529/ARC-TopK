"""Four-way 60M/130M original-setting fused ARC comparison, single runs."""
import hashlib,json
from pathlib import Path
from c4.scripts.extract_arc_collectives import extract
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM328-CM343-fused-small-models'

def main():
    jobs=json.loads((OUT/'manifest.json').read_text())
    assert len(jobs)==16 and all((OUT/j['id']/'accepted.json').exists() for j in jobs)
    extract(OUT,jobs,verify_pairs=False)
    all_results={}
    text=['# 60M / 130M：融合 ARC-TopK 原 setting 补测','',
          '4张空闲RTX4090，FP32，seq256，每卡batch1，Muon，EF14，保留比例0.2、r4，SHM默认channel、禁用P2P、单bucket8192MiB、严格阻塞。各版本一次，seed1243。',
          '', '| 模型 | 实现 | NCCL AllReduce kernel ms/步 | 完整 hook ms/步 | 完整单步 ms/步 |',
          '| --- | --- | ---: | ---: | ---: |']
    comparisons=[]
    for model in ('60m','130m'):
        data={};summary={}
        for arm in ('dense','reference','optimized','fused'):
            runs={}
            for profile in (False,True):
                j=next(j for j in jobs if j['model']==model and j['arm']==arm and j['profile']==profile)
                p=OUT/j['id'];runs['profile' if profile else 'timing']=json.loads((p/'all_results.json').read_text())
                if profile:
                    ss=[json.loads((p/f'collective-summary-rank{r}.json').read_text()) for r in range(4)]
                    runs['collective_ms']=max(s['allreduce_kernel_ms_per_update'] for s in ss)
                    runs['layouts']=[s['collective_layout_sha256'] for s in ss]
            data[arm]=runs
            summary[arm]={'collective_ms':runs['collective_ms'],
                          'hook_ms':runs['timing']['mean_blocking_hook_seconds']*1000,
                          'iteration_ms':runs['timing']['mean_iteration_seconds']*1000}
            m=summary[arm];text.append(f"| {model} | {arm} | {m['collective_ms']:.3f} | {m['hook_ms']:.3f} | {m['iteration_ms']:.3f} |")
        for arm in ('optimized','fused'):
            for mode in ('timing','profile'):
                assert data['reference'][mode]['rank_losses']==data[arm][mode]['rank_losses'],(model,arm,mode,'loss')
                assert data['reference'][mode]['rank_ddp_communication_bits']==data[arm][mode]['rank_ddp_communication_bits'],(model,arm,mode,'bits')
            assert data['reference']['layouts']==data[arm]['layouts'],(model,arm,'layout')
        for arm in ('dense','reference','optimized'):
            for metric in ('collective_ms','hook_ms','iteration_ms'):
                summary[arm]['fused_reduction_'+metric]=(1-summary['fused'][metric]/summary[arm][metric])*100
            comparisons.append(f"| {model} | {arm} | {summary[arm]['fused_reduction_hook_ms']:.2f}% | {summary[arm]['fused_reduction_iteration_ms']:.2f}% |")
        all_results[model]=summary
    text+=['','| 模型 | fused对比基线 | hook耗时降低 | 完整单步耗时降低 |','| --- | --- | ---: | ---: |',*comparisons,
           '', '每个模型中reference/optimized/fused在两种窗口的逐rank loss序列、DDP bits和逐次AllReduce payload尺寸/dtype布局完全一致。Dense与压缩Muon不是同一算法路径，不要求loss相同。',
           '', '单步与hook：无profiler，50步预热+100步测量，第41步起启用压缩。纯通信：独立profiler，50步预热+10步测量；最慢rank的NCCL AllReduce kernel累计时长/10，排除barrier和Muon AllGather。含协议等待与profiling扰动，非wire time，不能与无profiler hook直接相减。',
           '', '全部正式臂启动前检查4卡无GPU compute进程且空闲显存>=23000MiB，空闲不足由脚本等待；运行中监测外部GPU进程，干扰结果拒绝接受。原始trace无损gzip保留，实际源码snapshot与哈希保留。',
           '', '本轮每项一次，没有估计独立重复波动或统计显著性；严格阻塞结果不外推正常异步DDP吞吐或收敛。模型、seq、精度、压缩比例、NCCL设置均未为追求收益而改变。',
           '', '首轮在进程退出阶段触发rank3归属误报，NCCL日志证实是自有rank；该次试运行完整保存在output/CM328-CM343-fused-small-models-rejected-monitor-attempt1，不纳入数字。监测器已缓存运行中确认的子进程身份，正式重跑前重新检查四卡空闲。',
           '', '实验CM328–CM343；未修改算法实现，仅新增补测和汇总脚本；代码未提交。']
    report='\n'.join(text)+'\n';(OUT/'comparison.json').write_text(json.dumps(all_results,indent=2));(OUT/'comparison.md').write_text(report)
    for name in ('M001-muon-baseline.md','M002-arctopk-muon.md'):
        p=ROOT/'docs/worklog'/name;p.write_text(p.read_text().rstrip()+'\n\n## 2026-11-12：CM328–CM343 60M/130M融合版补测完成\n\n'+report)
    (OUT/'analysis-sources.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'c4/scripts/extract_arc_collectives.py')},indent=2))
    print(report,flush=True)
if __name__=='__main__':main()
