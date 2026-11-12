# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist

import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import (
    HookState,
    _get_allgather_out_list,
    dtype_bits,
    synchronize_blocking_communication,
    tensor_bits,
)

logger = logging.getLogger(__name__)


def group_topk_project_and_select(tensor, r, compress_ratio, group):

    d = tensor.numel()
    if tensor.dim()==1:  
        
        k = max(1, int(d * compress_ratio))

        P_local = tensor

        # AllReduce to get global projection P
        P = P_local.clone()
        P_bits = tensor_bits(P)
        dist.all_reduce(P, group=group)
        P /= dist.get_world_size(group)

        # Compute squared norm per row
        norms = P ** 2 
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)

        comm_bits = tensor_bits(tensor[topk_indices])
        return tensor[topk_indices].clone(), topk_indices, P_bits, comm_bits


    elif tensor.dim()==2:
        n = tensor.shape[0]
        m = tensor.shape[1]
        k = max(1, int(n * compress_ratio))
    
        V = torch.randn(m, r, device=tensor.device, dtype=tensor.dtype)
        # print(f"tensor.shape = {tensor.shape}, V.shape = {V.shape}")
        if V.dtype != tensor.dtype or V.device != tensor.device:
            V = V.to(device=tensor.device, dtype=tensor.dtype)
        P_local = tensor @ V     # shape: [n, r]
    
        # AllReduce to get global projection P
        P = P_local.clone()
        P_bits = tensor_bits(P)
        dist.all_reduce(P, group=group)
        P /= dist.get_world_size(group)

        # Compute squared norm per row
        norms = torch.sum(P ** 2, dim=1)  # [n]
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        row_offset = topk_indices.view(-1, 1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device).view(1, -1)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
    
    
        # print(f"indices={indices}")
        comm_bits = tensor_bits(tensor[topk_indices])
        return tensor[topk_indices].flatten().clone(), indices , P_bits, comm_bits 
    else:
        t = tensor.shape[-1]
        m = 2 * t * t
        n = d // m
        tensor_2D = tensor.reshape(n, m)
        k = max(1, int(n * compress_ratio))
    
        V = torch.randn(m, r, device=tensor.device, dtype=tensor.dtype)
        # print(f"tensor.shape = {tensor.shape}, V.shape = {V.shape}")
        if V.dtype != tensor.dtype or V.device != tensor.device:
            V = V.to(device=tensor.device, dtype=tensor.dtype)
        P_local = tensor_2D @ V     # shape: [n, r]
    
        # AllReduce to get global projection P
        P = P_local.clone()
        P_bits = tensor_bits(P)
        dist.all_reduce(P, group=group)
        P /= dist.get_world_size(group)

        # Compute squared norm per row
        norms = torch.sum(P ** 2, dim=1)  # [n]
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        row_offset = topk_indices.view(-1, 1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device).view(1, -1)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
    
        # print(f"indices.dtype={indices.dtype}")
        # print(f"values.dtype={tensor_2D[topk_indices].dtype}")
        # print(f"indices={indices}")
        comm_bits = tensor_bits(tensor_2D[topk_indices])
        return tensor_2D[topk_indices].flatten().clone(), indices, P_bits, comm_bits 







    
def compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, compress_ratio, r, group, use_error_feedback):
    offset = 0
    bits_sum = 0
    for tensor, k in zip(tensors, k_list):  
        values, indices, P_bits, comm_bits= group_topk_project_and_select(tensor=tensor, r=r, compress_ratio=compress_ratio, group=group)
        values_memory[offset:offset+k] = values
        indices_memory[offset:offset+k] = indices.to(torch.int) 
        offset += k
        bits_sum += P_bits + comm_bits 


        if use_error_feedback == "ef14":
            # input_tensor = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
            tensor.view(-1)[indices] = 0  
        elif use_error_feedback == "ef21":
            tensor.zero_()
            # input_tensor = C[\nabla F_i - E_{i-1}]
            tensor.view(-1)[indices] = values
    return values_memory, indices_memory, bits_sum

def decompress_memory_to_tensor_and_aggregate(tensors, k_list, values_memory, indices_memory):
    # add the decompressed values to the tensors
    offset = 0
    for tensor, k in zip(tensors, k_list):
        values = values_memory[offset:offset+k]
        indices = indices_memory[offset:offset+k]
        # avoid creating a new tensor for the view
        flattened_tensor = tensor.view(-1)
        flattened_tensor[indices.to(torch.int64)] = values
        offset += k
    return None


def _row_view(tensor):
    # Preserve the original dimensional layout, including its >2D reshape rule.
    if tensor.dim() == 1:
        return tensor.view(-1, 1)
    if tensor.dim() == 2:
        return tensor
    m = 2 * tensor.shape[-1] ** 2
    return tensor.reshape(tensor.numel() // m, m)


def compress_rows(tensors, k_list, values_memory, indices_memory,
                  compress_ratio, r, group, use_error_feedback):
    value_offset = row_offset = bits_sum = 0
    world_size = dist.get_world_size(group)
    for tensor, k in zip(tensors, k_list):
        rows = _row_view(tensor)
        width = rows.shape[1]
        if tensor.dim() == 1:
            projection = tensor.clone()
        else:
            # Same random draw shape/order and GEMM as the reference.
            v = torch.randn(width, r, device=tensor.device, dtype=tensor.dtype)
            projection = rows @ v
        bits_sum += tensor_bits(projection)
        dist.all_reduce(projection, group=group)
        projection /= world_size
        norms = projection ** 2
        if tensor.dim() != 1:
            norms = torch.sum(norms, dim=1)
        selected = torch.topk(norms, k=k // width, largest=True, sorted=False).indices
        values = rows.index_select(0, selected)
        values_memory[value_offset:value_offset+k].copy_(values.flatten())
        indices_memory[row_offset:row_offset+selected.numel()].copy_(selected)
        bits_sum += tensor_bits(values)
        if use_error_feedback == "ef14":
            rows.index_fill_(0, selected, 0)
        elif use_error_feedback == "ef21":
            rows.zero_()
            rows.index_copy_(0, selected, values)
        value_offset += k
        row_offset += selected.numel()
    return values_memory, indices_memory, bits_sum


def decompress_rows(tensors, k_list, values_memory, indices_memory):
    value_offset = row_offset = 0
    for tensor, k in zip(tensors, k_list):
        rows = _row_view(tensor)
        count = k // rows.shape[1]
        selected = indices_memory[row_offset:row_offset+count]
        rows.index_copy_(0, selected, values_memory[value_offset:value_offset+k].view(count, -1))
        value_offset += k
        row_offset += count


def compress_fused_rows(tensors, k_list, values_memory, indices_memory,
                        compress_ratio, r, group, use_error_feedback):
    from comm_hooks.arc_row_kernels import gather_rows_and_erase
    selected_rows = []
    offset = bits_sum = 0
    world_size = dist.get_world_size(group)
    for tensor, k in zip(tensors, k_list):
        rows = _row_view(tensor)
        width = rows.shape[1]
        if tensor.dim() == 1:
            projection = tensor.clone()
        else:
            v = torch.randn(width, r, device=tensor.device, dtype=tensor.dtype)
            projection = rows @ v
        bits_sum += tensor_bits(projection)
        dist.all_reduce(projection, group=group)
        projection /= world_size
        norms = projection ** 2
        if tensor.dim() != 1:
            norms = torch.sum(norms, dim=1)
        selected = torch.topk(norms, k=k // width, largest=True, sorted=False).indices
        gather_rows_and_erase[((k + 255) // 256,)](
            rows, selected, values_memory, k, offset, WIDTH=width,
            ERASE=use_error_feedback == "ef14", BLOCK=256)
        selected_rows.append(selected)
        bits_sum += k * tensor.element_size() * 8
        offset += k
    return values_memory, selected_rows, bits_sum



def compress_pipeline_rows(tensors, k_list, values_memory, indices_memory,
                           compress_ratio, r, group, use_error_feedback):
    from comm_hooks.arc_row_kernels import gather_rows_and_erase
    pending = []
    bits_sum = 0
    # Keep RNG/GEMM and collective enqueue order unchanged. Disjoint gradient
    # views allow later projections to run before earlier rows are cleared.
    for tensor, k in zip(tensors, k_list):
        rows = _row_view(tensor)
        if tensor.dim() == 1:
            projection = tensor.clone()
        else:
            v = torch.randn(rows.shape[1], r, device=tensor.device, dtype=tensor.dtype)
            projection = rows @ v
        bits_sum += tensor_bits(projection) + k * tensor.element_size() * 8
        work = dist.all_reduce(projection, group=group, async_op=True)
        pending.append((tensor.dim(), rows, k, projection, work))
    selected_rows = []
    offset = 0
    world_size = dist.get_world_size(group)
    for ndim, rows, k, projection, work in pending:
        work.wait()
        projection /= world_size
        norms = projection ** 2
        if ndim != 1:
            norms = torch.sum(norms, dim=1)
        width = rows.shape[1]
        selected = torch.topk(norms, k=k // width, largest=True, sorted=False).indices
        gather_rows_and_erase[((k + 255) // 256,)](
            rows, selected, values_memory, k, offset, WIDTH=width,
            ERASE=use_error_feedback == "ef14", BLOCK=256)
        selected_rows.append(selected)
        offset += k
    return values_memory, selected_rows, bits_sum

def decompress_fused_rows(tensors, k_list, values_memory, indices_memory):
    from comm_hooks.arc_row_kernels import scatter_rows
    offset = 0
    for tensor, k, selected in zip(tensors, k_list, indices_memory):
        rows = _row_view(tensor)
        scatter_rows[((k + 255) // 256,)](
            rows, selected, values_memory, k, offset, WIDTH=rows.shape[1], BLOCK=256)
        offset += k

class GroupTopKState(HookState):
    def __init__(self, 
        process_group: dist.ProcessGroup, 
        r: int = 4, 
        compress_ratio: float = 0.08, 
        start_compress_iter: int = 2, 
        use_error_feedback="noef", 
        seed=0, 
        error_decay=1.0,
        implementation="fused",
    ):
        super().__init__(process_group)
        if implementation not in ("reference", "optimized", "fused", "pipeline"):
            raise ValueError(implementation)
        self.implementation = implementation
        self.r = r
        self.compress_ratio = compress_ratio

        self.iter = 0
        self.start_compress_iter = start_compress_iter

        self.seed = seed

        # error feedback
        self.use_error_feedback = use_error_feedback 
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.global_error_dict: Dict[int, torch.Tensor] = {}
        self.error_decay = error_decay  

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

def cal_k(state, tensor):
    if tensor.dim() == 1:
        # k = tensor.numel()
        d = tensor.numel()
        k = max(1, int(d * state.compress_ratio))
    elif tensor.dim()==2:
        k = max(1, int(tensor.shape[0] * state.compress_ratio)) * tensor.shape[1]
    else:
        t = tensor.shape[-1]
        m = 2 * t * t
        d = tensor.numel()
        n = d // m
        k = max(1, int(n * state.compress_ratio)) * m
        
    return k


def group_topk_hook(
    state: GroupTopKState, bucket: dist.GradBucket
) -> torch.futures.Future[torch.Tensor]:
    # bucket.buffer()  Gradient: a one-dimensional tensor
    # bucket.gradients()  Gradient list layer by layer
    # bucket.parameters() Parameters
    # bucket.is_last()  Is this the last buffer
    # bucket.index()    Index of the buffer

    # check if the key is ready in precond adam
    state.maybe_accumulate_momentum_on_bucket(bucket)

    process_group = state.process_group
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    rank = dist.get_rank(group=group_to_use)

    # The input tensor is a flattened 1D tensor.
    input_tensor = bucket.buffer()  
    tensors = bucket.gradients() 

    # Run vanilla allreduce in the first `start_compress_iter` iterations.
    if state.iter < state.start_compress_iter:
        state.maybe_increase_iter(bucket)
        return default_hooks._allreduce_fut(group_to_use, input_tensor, state)
    
    device = tensors[0].device  
    dtype = tensors[0].dtype

    # Incorporate the error from the previous state into the gradients.
    bucket_index = bucket.index() 
    total_length = input_tensor.shape[0]
    if state.use_error_feedback == "ef14":
        if bucket_index in state.error_dict:
            # input_tensor = \nabla F_i + E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index], alpha=1.0)
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    elif state.use_error_feedback == "ef21":
        if bucket_index in state.error_dict:
            # input_tensor = \nabla F_i - E_{i-1}
            input_tensor.add_(state.error_dict[bucket_index], alpha=-1.0)
            # logger.info(f"\nabla F_i - E_i-1:{input_tensor}")
        else:
            # E_0 = \nabla F_0 
            logger.info("A tensor of length %s that represents local/global error is created.", total_length)
            state.error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # allreduce \nabla F_0
            state.comm_bits_this_round += tensor_bits(input_tensor)
            dist.all_reduce(input_tensor, group=group_to_use, async_op=False) 
            input_tensor.div_(world_size)
            # \overline{E_0} = \overline{\nabla F_0}
            state.global_error_dict[bucket_index] = torch.clone(input_tensor).detach()
            # reset the full input tensor
            state.maybe_increase_iter(bucket)
            synchronize_blocking_communication(state, input_tensor)
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(input_tensor)
            return fut


    seed = torch.randint(0, 1_000_000_000, (1,), generator=state.rng).item()
    torch.manual_seed(seed)
  
    

    k_list = [cal_k(state, tensor) for tensor in tensors]

    sum_k = sum(k_list) 

    values_memory = torch.empty(sum_k, dtype=dtype, device=device) 
    optimized = state.implementation != "reference"
    # EF21 keeps the already verified row path; its local reconstruction differs.
    fused = state.implementation in ("fused", "pipeline") and state.use_error_feedback != "ef21" and input_tensor.is_cuda
    index_count = sum(k // _row_view(t).shape[1] for t, k in zip(tensors, k_list)) if optimized else sum_k
    indices_memory = None if fused else torch.empty(index_count, dtype=torch.int64 if optimized else torch.int, device=device)
    compress = (compress_pipeline_rows if state.implementation == "pipeline" else compress_fused_rows) if fused else (compress_rows if optimized else compress_tensor_to_memory)
    _, indices_memory, bits_sum = compress(tensors=tensors, k_list=k_list, values_memory=values_memory, indices_memory=indices_memory, compress_ratio=state.compress_ratio, r=state.r, group=group_to_use, use_error_feedback=state.use_error_feedback)

   

    if state.use_error_feedback == "ef14":
        # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]
        state.error_dict[bucket_index].copy_(input_tensor)              
    elif state.use_error_feedback == "ef21":
        # E_i = E_{i-1} + C[\nabla F_i - E_{i-1}]
        state.error_dict[bucket_index].add_(input_tensor)    
        
    # Allreduce
    state.comm_bits_this_round += 2 *(world_size -1) * bits_sum 
    dist.all_reduce(values_memory, group=group_to_use, async_op=False)
    values_memory.div_(world_size)

    # Zero the input tensor.
    input_tensor.zero_() 
    decompress = decompress_fused_rows if fused else (decompress_rows if optimized else decompress_memory_to_tensor_and_aggregate)
    decompress(tensors=tensors, k_list=k_list, values_memory=values_memory, indices_memory=indices_memory)

    if state.use_error_feedback == "ef21":
        state.global_error_dict[bucket_index].add_(input_tensor) # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
        input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)
    synchronize_blocking_communication(state, input_tensor)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut
