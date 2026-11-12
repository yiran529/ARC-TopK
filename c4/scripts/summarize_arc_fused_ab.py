"""Same-setting four-way, single-run fused ARC comparison."""
import json
from pathlib import Path
from c4.scripts.extract_arc_collectives import extract
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM320-CM327-fused-original-setting'

def main():
    jobs=json.loads((OUT/'manifest.json').read_text())
    assert len(jobs)==8 and all((OUT/j['id']/'accepted.json').exists() for j in jobs)
    extract(OUT,jobs,verify_pairs=False)
    output={}
    for arm in ('dense','reference','optimized','fused'):
        runs={}
        for profile in (False,True):
            j=next(j for j in jobs if j['arm']==arm and j['profile']==profile)
            p=OUT/j['id'];r=json.loads((p/'all_results.json').read_text())
            runs['profile' if profile else 'timing']=r
            if profile:
                summaries=[json.loads((p/f'collective-summary-rank{rank}.json').read_text()) for rank in range(4)]
                runs['collective_ms']=max(s['allreduce_kernel_ms_per_update'] for s in summaries)
                runs['layouts']=[s['collective_layout_sha256'] for s in summaries]
        output[arm]=runs
    for arm in ('optimized','fused'):
        for mode in ('timing','profile'):
            assert output['reference'][mode]['rank_losses']==output[arm][mode]['rank_losses'],(arm,mode,'loss')
            assert output['reference'][mode]['rank_ddp_communication_bits']==output[arm][mode]['rank_ddp_communication_bits'],(arm,mode,'bits')
        assert output['reference']['layouts']==output[arm]['layouts'],(arm,'layout')
    summary={}
    text=['# 原 setting：融合 ARC-TopK 对照','',
          '4张空闲RTX4090；350M，FP32，seq256，每卡batch1，Muon，EF14，保留0.2，r4，SHM默认channel、禁用P2P，单bucket8192MiB，严格阻塞。每版本一次，seed1243。',
          '', '| 实现 | NCCL AllReduce kernel ms/步 | 完整 hook ms/步 | 完整单步 ms/步 |',
          '| --- | ---: | ---: | ---: |']
    for arm,r in output.items():
        summary[arm]={'collective_ms':r['collective_ms'],
                      'hook_ms':r['timing']['mean_blocking_hook_seconds']*1000,
                      'iteration_ms':r['timing']['mean_iteration_seconds']*1000}
        m=summary[arm];text.append(f"| {arm} | {m['collective_ms']:.3f} | {m['hook_ms']:.3f} | {m['iteration_ms']:.3f} |")
    text+=['','| fused 对比基线 | hook 耗时降低 | 完整单步耗时降低 |','| --- | ---: | ---: |']
    for arm in ('dense','reference','optimized'):
        for metric in ('hook_ms','iteration_ms'):
            summary[arm]['fused_reduction_'+metric]=(1-summary['fused'][metric]/summary[arm][metric])*100
        text.append(f"| {arm} | {summary[arm]['fused_reduction_hook_ms']:.2f}% | {summary[arm]['fused_reduction_iteration_ms']:.2f}% |")
    summary['target_met_vs_reference']=max(summary['reference']['fused_reduction_hook_ms'],summary['reference']['fused_reduction_iteration_ms'])>=30
    summary['target_met_vs_dense']=max(summary['dense']['fused_reduction_hook_ms'],summary['dense']['fused_reduction_iteration_ms'])>=30
    text+=['', 'reference、optimized、fused两种计时窗口的逐rank loss序列和DDP bits完全相等；逐次AllReduce payload尺寸/dtype布局也相等。4卡smoke每rank180组逐位一致性通过（FP32/BF16、noef/ef14/ef21、多比例、多bucket）。EF21沿用已验证的行索引路径。',
           '', '无profiler单步/hook：50步预热+100步测量。独立通信profiler：50步预热+10步测量，排除barrier与Muon AllGather；NCCL kernel含协议等待，不能与无profiler hook相减，非wire time。',
           '', '本轮每版本仅一次，未评估独立重复波动或显著性，不外推正常异步DDP吞吐或收敛。所有正式臂启动前检查4卡空闲；脚本等待空闲卡并监测运行中外部GPU进程，干扰结果拒绝接受。',
           '', '实现：融合行提取/写通信缓冲/EF14清零，避免中间gather张量与索引复制；投影randn、GEMM、TopK、平均与collective顺序/形状不变。Triton只做无算术数据复制/清零，不改变浮点归约。',
           '', '实验CM320–CM327，原始trace无损gzip保存；源码哈希和逐臂配置保留。']
    report='\n'.join(text)+'\n';(OUT/'comparison.md').write_text(report)
    (OUT/'comparison.json').write_text(json.dumps(summary,indent=2));print(report,flush=True)
    p=ROOT/'docs/worklog/M002-arctopk-muon.md'
    p.write_text(p.read_text().rstrip()+'\n\n## 2026-11-12：CM320–CM327 融合行kernel正式结果\n\n'+report)
if __name__=='__main__':main()
