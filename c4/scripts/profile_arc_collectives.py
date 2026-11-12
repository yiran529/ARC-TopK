"""Separate instrumented run: export CUDA NCCL kernel timings, not iteration claims."""
import json
import os
from pathlib import Path
import torch
import c4.table_v_timing as timing

Base=timing.StrictHookRecorder
class Recorder(Base):
    def wrap(self,state,hook):
        self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
             torch.profiler.ProfilerActivity.CUDA],
             schedule=torch.profiler.schedule(wait=49,warmup=1,active=10,repeat=1))
        self.prof.start()
        global recorder
        recorder=self
        return super().wrap(state,hook)
    def finish_step(self):
        result=super().finish_step()
        self.prof.step()
        return result

timing.StrictHookRecorder=Recorder
if __name__=='__main__':
    import sys
    output=Path(sys.argv[sys.argv.index('--output_dir')+1])
    timing.main()
    recorder.prof.stop()
    trace=output/f'collectives-rank{os.environ["RANK"]}.json'
    recorder.prof.export_chrome_trace(str(trace))
    events=json.loads(trace.read_text())['traceEvents']
    kernels=[e for e in events if e.get('cat')=='kernel' and 'nccl' in e.get('name','').lower()]
    reductions=[e for e in kernels if 'allreduce' in e['name'].lower().replace('_','')]
    summary={'rank':int(os.environ['RANK']), 'profiled_updates':10,
             'allreduce_kernel_ms_per_update':sum(e['dur'] for e in reductions)/1000/10,
             'allreduce_kernel_count':len(reductions),
             'nccl_kernel_names':sorted(set(e['name'] for e in kernels)),
             'scope':'CUDA NCCL AllReduce kernel duration sum; projection + packed values; excludes Muon AllGather; separate profiled run'}
    (output/f'collective-summary-rank{os.environ["RANK"]}.json').write_text(json.dumps(summary,indent=2))
