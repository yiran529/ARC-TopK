"""Distributed bitwise equivalence check for the ARC row-index optimization."""
import os
import torch
import torch.distributed as dist
from comm_hooks.group_topk_hook_no_reshape import GroupTopKState, group_topk_hook

class Bucket:
    def __init__(self, values, index):
        self.flat = torch.cat([x.flatten() for x in values])
        self.views=[]
        offset=0
        for x in values:
            self.views.append(self.flat[offset:offset+x.numel()].view_as(x))
            offset+=x.numel()
        self.number=index
    def buffer(self): return self.flat
    def gradients(self): return self.views
    def index(self): return self.number
    def is_last(self): return self.number == 0

def main():
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    checks=0
    original=dist.all_reduce
    calls=[]
    def traced(tensor, *a, **kw):
        calls.append((tuple(tensor.shape), str(tensor.dtype)))
        return original(tensor,*a,**kw)
    dist.all_reduce=traced
    for dtype in (torch.float32,torch.bfloat16):
        for mode in ('noef','ef14','ef21'):
            for ratio in (0.01,0.2,1.0):
                states=[GroupTopKState(dist.group.WORLD, compress_ratio=ratio,
                        start_compress_iter=1,use_error_feedback=mode,seed=1243,
                        implementation=impl) for impl in ('reference',os.environ.get('ARC_CHECK_IMPLEMENTATION','optimized'))]
                for step in range(5):
                    for index in (1,0):
                        torch.manual_seed(100+step+dist.get_rank())
                        values=[torch.randn(shape,device='cuda',dtype=dtype) for shape in
                                ((17,), (13,16), (8,8), (4,2,2,2))]
                        buckets=[Bucket(values,index),Bucket(values,index)]
                        traces=[];rngs=[]
                        for state,bucket in zip(states,buckets):
                            calls.clear()
                            group_topk_hook(state,bucket).wait()
                            traces.append(list(calls));rngs.append(torch.cuda.get_rng_state())
                        assert torch.equal(buckets[0].flat,buckets[1].flat),(dtype,mode,ratio,step,index)
                        assert traces[0]==traces[1]
                        assert torch.equal(rngs[0],rngs[1])
                        assert states[0].comm_bits_this_round==states[1].comm_bits_this_round
                        assert states[0].iter==states[1].iter
                        for attr in ('error_dict','global_error_dict'):
                            a,b=[getattr(s,attr) for s in states]
                            assert a.keys()==b.keys()
                            assert all(torch.equal(a[k],b[k]) for k in a)
                        checks+=1
    dist.barrier()
    if dist.get_rank()==0: print(f'PASS {checks} bitwise hook comparisons per rank, including RNG, EF and collective layouts',flush=True)
    dist.destroy_process_group()
if __name__=='__main__': main()
