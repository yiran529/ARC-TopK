"""CM328-CM343: four idle GPUs, paired ARC reference/row-index implementation."""
import gzip,hashlib,json,os,random,shutil,signal,subprocess,time
from pathlib import Path
from c4.scripts.run_strict_hook_abcd import cells, environment_for
from c4.scripts.run_table_v_muon_blocking_matrix import selected_gpus_are_idle
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/CM328-CM343-fused-small-models'
GPUS=['0','1','2','3']

def main():
    OUT.mkdir(exist_ok=True)
    sources=['comm_hooks/group_topk_hook_no_reshape.py','comm_hooks/utils.py','c4/table_v_timing.py',
             'c4/scripts/profile_arc_collectives.py','c4/scripts/run_arc_fused_small.py','comm_hooks/default_hooks.py','comm_hooks/arc_row_kernels.py']
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources}
    (OUT/'sources.json').write_text(json.dumps(hashes,indent=2))
    for name in sources:
        target=OUT/'source-snapshot'/name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,target)
    jobs=[];number=328;rng=random.Random(20261112)
    template=next(c for c in cells() if c['arm']=='dense')
    for profile in (False,True):
        for model in ('60m','130m'):
            for repeat in (1,):
                arms=['dense','reference','optimized','fused'];rng.shuffle(arms)
                for arm in arms:
                    template=next(c for c in cells() if c['arm']==('dense' if arm=='dense' else 'arctopk'))
                    name=f'CM{number}-'+('m001' if arm=='dense' else 'm002')+f'-{model}-{arm}-rep{repeat}'+('-profile' if profile else '')
                    cmd=list(template['command'])
                    def setarg(key,value): cmd[cmd.index('--'+key)+1]=str(value)
                    setarg('model_config',f'c4/configs/llama_{model}.json')
                    setarg('output_dir',OUT/name)
                    setarg('measured_iterations',10 if profile else 100)

                    if arm!='dense': cmd+=['--arc_topk_implementation',arm]
                    if profile: cmd[cmd.index('c4/table_v_timing.py')]='c4/scripts/profile_arc_collectives.py'
                    jobs.append(dict(id=name,model=model,repeat=repeat,arm=arm,profile=profile,command=cmd))
                    number+=1
    (OUT/'manifest.json').write_text(json.dumps(jobs,indent=2))
    env=environment_for(dict(world_size=4,channels='default'),GPUS)
    for job in jobs:
        target=OUT/job['id']
        if (target/'accepted.json').exists(): continue
        if target.exists(): raise RuntimeError(f'Existing incomplete run: {target}')
        if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h for p,h in hashes.items()):
            raise RuntimeError('Source changed')
        while not selected_gpus_are_idle(GPUS):
            (OUT/'state.json').write_text(json.dumps({'status':'waiting_for_idle_gpus','next':job['id']}))
            time.sleep(15)
        (OUT/'state.json').write_text(json.dumps({'status':'running','current':job['id']}))
        with (OUT/(job['id']+'.log')).open('w') as log:
            proc=subprocess.Popen(job['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            interference=[]
            known_run_pids={proc.pid}
            while True:
                try: code=proc.wait(timeout=5);break
                except subprocess.TimeoutExpired:
                    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
                    uuids={line.split(',')[1].strip() for line in inventory.splitlines() if line.split(',')[0].strip() in GPUS}
                    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                    parents = {int(parts[0]): int(parts[1]) for row in subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True).splitlines() if len(parts := row.split()) == 2}
                    def belongs_to_run(pid):
                        while pid > 1:
                            if pid == proc.pid: return True
                            pid = parents.get(pid, 0)
                        return False
                    known_run_pids.update(pid for pid in parents if belongs_to_run(pid))
                    for line in apps.splitlines():
                        uuid,pid=[v.strip() for v in line.split(',')]
                        if uuid in uuids:
                            try:
                                if int(pid) not in known_run_pids: interference.append(line)
                            except ProcessLookupError: pass
        if code or interference: raise RuntimeError(f'{job["id"]}: code={code}, interference={interference}')
        result=json.loads((target/'all_results.json').read_text())
        assert result['status']=='completed' and result['observed_bucket_counts']==[1]
        if job['profile']:
            summaries=[json.loads((target/f'collective-summary-rank{r}.json').read_text()) for r in range(4)]
            assert all(s['allreduce_kernel_count']>0 for s in summaries)
        (target/'accepted.json').write_text(json.dumps({'accepted':True,'interference':interference}))
        if job['profile']:
            for trace in target.glob('collectives-rank*.json'):
                temp=trace.with_suffix('.json.gz.tmp')
                with trace.open('rb') as src,gzip.open(temp,'wb',compresslevel=1) as out: shutil.copyfileobj(src,out)
                temp.rename(trace.with_suffix('.json.gz'));trace.unlink()
        print('ACCEPTED',job['id'],flush=True)
    (OUT/'state.json').write_text(json.dumps({'status':'completed','runs':len(jobs)}))
    from c4.scripts.summarize_arc_fused_small import main as summarize
    summarize()
if __name__=='__main__': main()
