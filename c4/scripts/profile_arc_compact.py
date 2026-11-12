"""Main-thread in-memory NCCL aggregates; never export a raw profiler trace."""
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import c4.table_v_timing as timing

BaseRecorder=timing.StrictHookRecorder
OUTPUT=None

class Recorder(BaseRecorder):
    def __init__(self):
        super().__init__()
        self.step=0
        self.prof=None
    def finish_step(self):
        result=super().finish_step()
        self.step+=1
        # finish_step is called from the training main thread, unlike DDP hooks.
        if self.step==50:
            self.layout=[]
            self.original_all_reduce=dist.all_reduce
            def record_reduce(tensor,*args,**kwargs):
                self.layout.append((tuple(tensor.shape),str(tensor.dtype),dist.get_world_size(kwargs.get('group'))))
                return self.original_all_reduce(tensor,*args,**kwargs)
            dist.all_reduce=record_reduce
            self.prof=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
            self.prof.start()
        elif self.step==60:
            dist.all_reduce=self.original_all_reduce
            self.prof.stop()
            events=self.prof.events()
            kernels=sorted([e for e in events if e.device_type==torch.autograd.DeviceType.CUDA
                and e.name.startswith(('ncclDevKernel','ncclKernel'))
                and 'allreduce' in e.name.lower().replace('_','')],key=lambda e:e.time_range.start)
            calls=sum(e.name=='c10d::allreduce_' for e in events)
            barriers=sum(e.name=='c10d::barrier' for e in events)
            # Capture starts after update50's device fence. The very next NCCL
            # operation is table_v_timing's single measurement-start barrier.
            assert barriers==1 and len(kernels)==calls+1,(barriers,len(kernels),calls)
            kernels=kernels[1:]
            assert calls==len(self.layout)>0 and calls%10==0
            per_step=calls//10
            assert all(self.layout[:per_step]==self.layout[i:i+per_step] for i in range(0,calls,per_step))
            elapsed=sum(e.device_time_total for e in kernels)/1000/10
            assert elapsed>0
            summary={'rank':int(os.environ['RANK']),'profiled_updates':10,
                'allreduce_kernel_ms_per_update':elapsed,'allreduce_kernel_count':len(kernels),
                'allreduce_cpu_calls':calls,'excluded_initial_barrier_kernels':1,
                'nccl_kernel_names':sorted(set(e.name for e in kernels)),
                'scope':'NCCL AllReduce CUDA kernels only, updates51-60; initial barrier verified and excluded; Muon AllGather excluded; includes protocol waits',
                'raw_trace_saved':False,
                'collective_layout_sha256':hashlib.sha256(json.dumps(self.layout).encode()).hexdigest(),
                'per_step_layout':self.layout[:per_step]}
            (OUTPUT/f'collective-summary-rank{os.environ["RANK"]}.json').write_text(json.dumps(summary,indent=2))
            self.prof=None
        return result

timing.StrictHookRecorder=Recorder

if __name__=='__main__':
    import sys
    if '--self-test' in sys.argv:
        import tempfile
        from c4.scripts.check_arc_rows import Bucket
        from comm_hooks.group_topk_hook_no_reshape import GroupTopKState,group_topk_hook
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
        with tempfile.TemporaryDirectory() as folder:
            OUTPUT=Path(folder)
            state=Recorder();state.step=49
            state.wrap(GroupTopKState(dist.group.WORLD,start_compress_iter=0,use_error_feedback='ef14',implementation='pipeline'),group_topk_hook)
            state.times.append(1e-9);state.bucket_bytes.append(4);state.finish_step()
            dist.barrier();torch.cuda.synchronize()
            for i in range(10):
                torch.manual_seed(100+i+dist.get_rank())
                bucket=Bucket([torch.randn(17,device='cuda'),torch.randn(13,16,device='cuda')],0)
                timing.strict_blocking_hook(state,bucket).wait();state.finish_step()
            summary=json.loads((OUTPUT/f'collective-summary-rank{os.environ["RANK"]}.json').read_text())
            assert summary['allreduce_kernel_count']==30,summary
            if dist.get_rank()==0:print('PASS main-thread compact profiler',summary,flush=True)
        dist.destroy_process_group()
    else:
        OUTPUT=Path(sys.argv[sys.argv.index('--output_dir')+1]);timing.main()
