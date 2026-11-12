"""Use NCCL metadata to exclude barrier kernels and verify collective payload layouts."""
import gzip,hashlib,json
from pathlib import Path

def extract(root,jobs,verify_pairs=True):
    for job in jobs:
        if not job['profile']: continue
        for rank in range(4):
            path=root/job['id']
            target=path/f'collective-summary-rank{rank}.json'
            raw=json.loads(target.read_text())
            if raw.get('barrier_excluded'): continue
            trace=path/f'collectives-rank{rank}.json'
            try:
                events=json.loads(trace.read_text())['traceEvents']
            except FileNotFoundError:
                with gzip.open(str(trace)+'.gz','rt') as source:
                    events=json.load(source)['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel' and 'nccl' in e.get('name','').lower()]
            # Some cuLaunchKernelEx events lack External id. Recover attribution
            # from the exact CUDA correlation and enclosing CPU comm scope.
            launches={e.get('args',{}).get('correlation'):e for e in events
                      if e.get('cat') in ('cuda_runtime','cuda_driver') and 'Launch' in e.get('name','')}
            scopes=[e for e in events if e.get('cat')=='cpu_op' and e.get('name')=='record_param_comms']
            recovered=0
            for kernel in kernels:
                if kernel.get('args',{}).get('Collective name'): continue
                launch=launches[kernel['args']['correlation']]
                matches=[e for e in scopes if e['tid']==launch['tid'] and e['ts']<=launch['ts']<=e['ts']+e['dur']]
                assert len(matches)==1,(job['id'],rank,'ambiguous correlation')
                for field in ('Collective name','In msg nelems','Out msg nelems','dtype'):
                    kernel['args'][field]=matches[0]['args'][field]
                recovered+=1
            reduced=sorted([e for e in kernels if e.get('args',{}).get('Collective name')=='allreduce'],key=lambda e:e['ts'])
            count=sum(e.get('name')=='c10d::allreduce_' for e in events)
            assert len(reduced)==count>0,(job['id'],rank,count,len(reduced))
            assert count%10==0
            layout=[(e['args']['In msg nelems'],e['args']['Out msg nelems'],e['args']['dtype']) for e in reduced]
            assert all(layout[:count//10]==layout[i:i+count//10] for i in range(0,count,count//10))
            raw.update(recovered_kernel_metadata_count=recovered,allreduce_kernel_ms_per_update=sum(e['dur'] for e in reduced)/1000/10,
                allreduce_kernel_count=count,barrier_excluded=True,
                collective_layout_sha256=hashlib.sha256(json.dumps(layout).encode()).hexdigest(),
                excluded_barrier_kernel_count=sum(e.get('args',{}).get('Collective name')=='barrier' for e in kernels),
                scope='NCCL CUDA kernel metadata Collective name=allreduce only; excludes barrier and Muon AllGather')
            target.write_text(json.dumps(raw,indent=2))
        print('EXTRACTED',job['id'],flush=True)
    if not verify_pairs: return
    for model in ('60m','130m','350m'):
        for repeat in (1,2,3):
            pair=[j for j in jobs if j['model']==model and j['repeat']==repeat and j['profile']]
            for rank in range(4):
                a,b=[json.loads((root/j['id']/f'collective-summary-rank{rank}.json').read_text()) for j in pair]
                assert a['collective_layout_sha256']==b['collective_layout_sha256'],(model,repeat,rank)
