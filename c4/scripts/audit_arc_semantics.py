"""Bit-level differential audit; save hashes/counters only, never tensor dumps."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
import torch.distributed as dist
from c4.scripts.check_arc_rows import Bucket
from comm_hooks.group_topk_hook_no_reshape import GroupTopKState, group_topk_hook

OUT=Path('output/arc-semantics-audit-20261112')
RANDN=torch.randn
TOPK=torch.topk
REDUCE=dist.all_reduce
ACTIVE=None

def bits_equal(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.detach().contiguous().view(torch.uint8).cpu(),b.detach().contiguous().view(torch.uint8).cpu())

def same(a,b,where):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and bits_equal(a,b),where
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),where
        for k in a:same(a[k],b[k],where+f'/{k}')
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b),where
        for i,(x,y) in enumerate(zip(a,b)):same(x,y,where+f'/{i}')
    else:assert a==b,(where,a,b)

class WorkAudit:
    def __init__(self,work,tensor,record):self.work,self.tensor,self.record=work,tensor,record
    def wait(self,*args,**kwargs):
        result=self.work.wait(*args,**kwargs)
        self.record['after']=self.tensor.clone()
        return result
    def __getattr__(self,name):return getattr(self.work,name)

def record_reduce(tensor,*args,**kwargs):
    if ACTIVE is None:return REDUCE(tensor,*args,**kwargs)
    item={'shape':tuple(tensor.shape),'dtype':str(tensor.dtype),
          'world_size':dist.get_world_size(kwargs.get('group')),
          'op':str(kwargs.get('op',dist.ReduceOp.SUM)),
          'before':tensor.clone(),'after':None}
    ACTIVE['collectives'].append(item)
    result=REDUCE(tensor,*args,**kwargs)
    if kwargs.get('async_op',False):return WorkAudit(result,tensor,item)
    item['after']=tensor.clone()
    return result

def record_randn(*args,**kwargs):
    result=RANDN(*args,**kwargs)
    if ACTIVE is not None:ACTIVE['random_projections'].append(result.clone())
    return result

def record_topk(tensor,*args,**kwargs):
    result=TOPK(tensor,*args,**kwargs)
    if ACTIVE is not None:ACTIVE['selection'].append({'norms':tensor.clone(),'values':result.values.clone(),'indices':result.indices.clone()})
    return result

def sample(shapes,dtype,pattern,step):
    torch.manual_seed(20000+step+dist.get_rank()*101)
    values=[]
    for shape in shapes:
        if pattern=='random':value=RANDN(shape,device='cuda',dtype=dtype)
        elif pattern=='zeros':value=torch.zeros(shape,device='cuda',dtype=dtype)
        elif pattern=='ties':value=torch.full(shape,float(dist.get_rank()+1),device='cuda',dtype=dtype)
        else:
            value=torch.zeros(shape,device='cuda',dtype=dtype)
            value.flatten()[::2]=-0.0
        values.append(value)
    return values

COUNTS={'configurations':0,'bucket_comparisons':0,'checkpoint_restores':0,'collective_comparisons':0}

def audit_case(group,dtype,mode,r,ratio,pattern,shapes,steps=3,warmup=1,checkpoint=True,blocking=False):
    global ACTIVE
    impls=('reference','optimized','fused','pipeline')
    def state(impl):
        result=GroupTopKState(group,r=r,compress_ratio=ratio,start_compress_iter=warmup,use_error_feedback=mode,seed=1243,implementation=impl)
        result.blocking_communication=blocking
        return result
    states=[state(impl) for impl in impls]
    for step in range(steps):
        for index in (1,0):
            values=sample(shapes,dtype,pattern,step)
            buckets=[Bucket(values,index) for _ in states]
            observed=[]
            for current,bucket in zip(states,buckets):
                ACTIVE={'collectives':[],'random_projections':[],'selection':[]}
                group_topk_hook(current,bucket).wait()
                torch.cuda.synchronize()
                observed.append({'output':bucket.flat.clone(),'state':current.state_dict(),
                    'cpu_rng':torch.get_rng_state().clone(),'cuda_rng':torch.cuda.get_rng_state().clone(),
                    'operations':ACTIVE})
                ACTIVE=None
            for i in range(1,len(observed)):
                same(observed[0],observed[i],f'{dtype}/{mode}/r{r}/p{ratio}/{pattern}/step{step}/bucket{index}/{impls[i]}')
                COUNTS['bucket_comparisons']+=1
                COUNTS['collective_comparisons']+=len(observed[0]['operations']['collectives'])
        if checkpoint and step==1:
            # Cross-implementation continuation from the reference checkpoint.
            checkpoint_state=states[0].state_dict()
            for i,impl in enumerate(impls):
                states[i]=state(impl)
                states[i].load_state_dict(checkpoint_state,device=torch.device('cuda',int(os.environ['LOCAL_RANK'])))
                COUNTS['checkpoint_restores']+=1
    COUNTS['configurations']+=1


def static_audit():
    p=Path('comm_hooks/group_topk_hook_no_reshape.py')
    old=ast.parse(subprocess.check_output(['git','show','HEAD:'+str(p)],text=True));new=ast.parse(p.read_text())
    names=('group_topk_project_and_select','compress_tensor_to_memory','decompress_memory_to_tensor_and_aggregate','cal_k')
    for name in names:
        a=next(n for n in old.body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in new.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    p=Path('comm_hooks/utils.py')
    old=ast.parse(subprocess.check_output(['git','show','HEAD:'+str(p)],text=True));new=ast.parse(p.read_text())
    for name in ('state_dict','load_state_dict'):
        a=next(n for n in next(n for n in old.body if isinstance(n,ast.ClassDef) and n.name=='HookState').body if isinstance(n,ast.FunctionDef) and n.name==name)
        b=next(n for n in next(n for n in new.body if isinstance(n,ast.ClassDef) and n.name=='HookState').body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(a)==ast.dump(b),name
    return list(names)+['HookState.state_dict','HookState.load_state_dict']


def main():
    global ACTIVE
    started=time.monotonic();torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    static=static_audit()
    torch.randn=record_randn;torch.topk=record_topk;dist.all_reduce=record_reduce
    shapes=((17,),(13,16),(4,2,2,2),(7,513))
    case=0
    for dtype in (torch.float32,torch.bfloat16,torch.float16,torch.float64):
        for mode in ('noef','ef14','ef21'):
            for r in (1,4,8):
                for pattern in ('random','zeros','ties','signedzero'):
                    ratio=(0.0,0.2,0.55,1.0)[case%4]
                    audit_case(dist.group.WORLD,dtype,mode,r,ratio,pattern,shapes,blocking=case%2==0)
                    case+=1
        if dist.get_rank()==0:print('PASS world4 dtype',dtype,'configs',COUNTS['configurations'],flush=True)
    # Actual LLaMA-sized layout, including embedding and odd FFN row widths.
    for dtype in (torch.float32,torch.bfloat16):
        audit_case(dist.group.WORLD,dtype,'ef14',4,0.2,'random',((32000,1024),(2736,1024),(1024,2736),(1024,)),warmup=0)
    # Non-default CUDA stream, two-rank groups and non-power-of-two groups.
    groups=[dist.new_group([0,1]),dist.new_group([2,3]),dist.new_group([0,1,2]),dist.new_group([3])]
    for group in (groups[0] if dist.get_rank()<2 else groups[1],groups[2] if dist.get_rank()<3 else groups[3]):
        stream=torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for dtype in (torch.float32,torch.bfloat16):
                for mode in ('noef','ef14','ef21'):
                    audit_case(group,dtype,mode,4,0.2,'random',shapes,warmup=0)
        stream.synchronize();dist.barrier()
    ACTIVE=None;torch.randn=RANDN;torch.topk=TOPK;dist.all_reduce=REDUCE
    OUT.mkdir(parents=True,exist_ok=True)
    files=('comm_hooks/group_topk_hook_no_reshape.py','comm_hooks/arc_row_kernels.py','comm_hooks/utils.py','c4/scripts/audit_arc_semantics.py')
    result={'status':'passed','rank':dist.get_rank(),'elapsed_seconds':round(time.monotonic()-started,3),
        **COUNTS,'reference_ast_unchanged':static,
        'comparisons':'uint8 bitwise output/state/RNG/projection draws/projection pre-post AllReduce/norms/TopK values and indices/packed payloads',
        'dtypes':['float32','bfloat16','float16','float64'],'world_sizes':[4,2,3 if dist.get_rank()<3 else 1],
        'patterns':['random','zeros','ties','signedzero'],'checkpoint_cross_implementation':True,
        'nondefault_stream':True,'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    (OUT/f'rank{dist.get_rank()}.json').write_text(json.dumps(result,indent=2))
    dist.barrier()
    if dist.get_rank()==0:print('PASS FINAL',result,flush=True)
    dist.destroy_process_group()
if __name__=='__main__':main()
