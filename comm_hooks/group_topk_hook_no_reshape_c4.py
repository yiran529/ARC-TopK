# mypy: allow-untyped-defs
import logging
from functools import partial
from typing import Dict

import torch
import torch.distributed as dist

import comm_hooks.default_hooks as default_hooks

from comm_hooks.utils import HookState, _get_allgather_out_list, dtype_bits, tensor_bits

logger = logging.getLogger(__name__)


def group_topk_project_and_select(tensor, r, compress_ratio, group, use_deterministic_projection=False, projection_seed=None):

    d = tensor.numel()
    if tensor.dim() == 1:  
        # 1D tensor: no compression, pass through directly
        P_bits = 0
        comm_bits = tensor_bits(tensor)
        indices = torch.arange(d, device=tensor.device, dtype=torch.int64)
        return tensor, indices, P_bits, comm_bits

    elif tensor.dim() == 2:
        n, m = tensor.shape
        k = max(1, int(n * compress_ratio))
    
        V = torch.empty(m, r, device=tensor.device, dtype=tensor.dtype)
        if use_deterministic_projection and projection_seed is not None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(projection_seed)
            V.normal_(generator=generator)
        else:
            V.normal_()
        
        P_local = tensor @ V     # shape: [n, r]
        P_bits = tensor_bits(P_local)
        
        # AllReduce operation
        dist.all_reduce(P_local, group=group)
        P_local.div_(dist.get_world_size(group))

        norms = torch.sum(P_local.square(), dim=1)
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        
        row_offset = topk_indices.unsqueeze(1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device, dtype=torch.int64).unsqueeze(0)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
        
        selected_rows = tensor[topk_indices]
        comm_bits = tensor_bits(selected_rows)
        return selected_rows.flatten(), indices, P_bits, comm_bits
        
    else:
        # Higher dimensional tensors
        t = tensor.shape[-1]
        m = 2 * t * t
        n = d // m
        tensor_2D = tensor.view(n, m) 
        k = max(1, int(n * compress_ratio))
    
        V = torch.empty(m, r, device=tensor.device, dtype=tensor.dtype)
        if use_deterministic_projection and projection_seed is not None:
            generator = torch.Generator(device=tensor.device)
            generator.manual_seed(projection_seed)
            V.normal_(generator=generator)
        else:
            V.normal_()
        
        P_local = tensor_2D @ V     # shape: [n, r]
        # P_local = torch.norm(tensor_2D, dim=1, keepdim=True) # shape: [n, 1]
        P_bits = tensor_bits(P_local)
        
        # AllReduce operation
        dist.all_reduce(P_local, group=group)
        P_local.div_(dist.get_world_size(group))

        norms = torch.sum(P_local.square(), dim=1)
        _, topk_indices = torch.topk(norms, k=k, largest=True, sorted=False)
        
        row_offset = topk_indices.unsqueeze(1) * m  # [k, 1]
        col_indices = torch.arange(m, device=tensor.device, dtype=torch.int64).unsqueeze(0)  # [1, m]
        indices = (row_offset + col_indices).flatten()  # shape: [k * m]
        
        selected_rows = tensor_2D[topk_indices]
        comm_bits = tensor_bits(selected_rows)
        return selected_rows.flatten(), indices, P_bits, comm_bits







    
def compress_tensor_to_memory(tensors, k_list, values_memory, indices_memory, compress_ratio, r, group, use_error_feedback, use_deterministic_projection=False, projection_seed=None):
    offset = 0
    bits_sum = 0
    for tensor, k in zip(tensors, k_list):  
        values, indices, P_bits, comm_bits= group_topk_project_and_select(
            tensor=tensor, 
            r=r, 
            compress_ratio=compress_ratio, 
            group=group,
            use_deterministic_projection=use_deterministic_projection,
            projection_seed=projection_seed
        )
        values_memory[offset:offset+k] = values
        indices_memory[offset:offset+k] = indices 
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
        flattened_tensor[indices] = values 
        offset += k
    return None

class GroupTopKState(HookState):
    def __init__(self, 
        process_group: dist.ProcessGroup, 
        r: int = 4, 
        compress_ratio: float = 0.08, 
        start_compress_iter: int = 2, 
        use_error_feedback="noef", 
        seed=0, 
        error_decay=1.0,
        gradual_compression=True,
        warmup_iters=100  
    ):
        super().__init__(process_group)
        self.r = r
        self.base_compress_ratio = compress_ratio 
        self.compress_ratio = compress_ratio
        self.gradual_compression = gradual_compression
        self.warmup_iters = start_compress_iter + warmup_iters

        self.iter = 0
        self.start_compress_iter = start_compress_iter

        self.seed = seed

        # error feedback
        self.use_error_feedback = use_error_feedback 
        self.error_dict: Dict[int, torch.Tensor] = {}
        self.global_error_dict: Dict[int, torch.Tensor] = {}
        self.error_decay = error_decay  

        self.compression_started = False

        self.rng = torch.Generator()
        self.rng.manual_seed(seed)
        self.generator: Dict[int, torch.Generator] = {}       # store 1D error tensor for EF14 and EF21

    
    def get_current_compress_ratio(self):
        if not self.gradual_compression or not self.compression_started:
            return self.base_compress_ratio
            
        compress_progress = self.iter - self.start_compress_iter
        if compress_progress < self.warmup_iters:
            start_ratio = 0.8
            progress = compress_progress / self.warmup_iters
            current_ratio = start_ratio - (start_ratio - self.base_compress_ratio) * progress
            return max(current_ratio, self.base_compress_ratio)
        else:
            return self.base_compress_ratio

def cal_k(state, tensor):
    current_compress_ratio = state.get_current_compress_ratio()
    
    if tensor.dim() == 1:
        k = tensor.numel()
        # d = tensor.numel()
        # k = max(1, int(d * current_compress_ratio))
    elif tensor.dim()==2:
        k = max(1, int(tensor.shape[0] * current_compress_ratio)) * tensor.shape[1]
    else:
        t = tensor.shape[-1]
        m = 2 * t * t
        d = tensor.numel()
        n = d // m
        k = max(1, int(n * current_compress_ratio)) * m
        
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
    
    if not state.compression_started:
        state.compression_started = True
        logger.info(f"Starting compression at iteration {state.iter} with gradual compression enabled: {state.gradual_compression}")
        if state.use_error_feedback in ["ef14", "ef21"]:
            logger.info("Initializing error feedback mechanism for smooth compression transition")
    
    device = tensors[0].device  
    dtype = tensors[0].dtype

    current_compress_ratio = state.get_current_compress_ratio()
    
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
            fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
            fut.set_result(input_tensor)
            return fut

    seed = torch.randint(0, 1_000_000_000, (1,), generator=state.rng).item()
    torch.manual_seed(seed)
  
    use_deterministic = state.compression_started and (state.iter - state.start_compress_iter < state.warmup_iters)
    projection_seed = seed if use_deterministic else None

    k_list = [cal_k(state, tensor) for tensor in tensors]

    sum_k = sum(k_list) 

    values_memory = torch.empty(sum_k, dtype=dtype, device=device)
    indices_memory = torch.empty(sum_k, dtype=torch.int64, device=device) 
    _, _, bits_sum = compress_tensor_to_memory(
        tensors=tensors, 
        k_list=k_list, 
        values_memory=values_memory, 
        indices_memory=indices_memory, 
        compress_ratio=current_compress_ratio, 
        r=state.r, 
        group=group_to_use, 
        use_error_feedback=state.use_error_feedback,
        use_deterministic_projection=use_deterministic,
        projection_seed=projection_seed
    )

   

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
    decompress_memory_to_tensor_and_aggregate(tensors=tensors, k_list=k_list, values_memory=values_memory, indices_memory=indices_memory)

    if state.use_error_feedback == "ef21":
        state.global_error_dict[bucket_index].add_(input_tensor) # \overline{E_i} = \overline{E_{i-1}} + \overline{C[\nabla F_i - E_{i-1}]}
        input_tensor.copy_(state.global_error_dict[bucket_index])

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut



def fake_group_topk_hook(
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
    
    if not state.compression_started:
        state.compression_started = True
        logger.info(f"Starting compression at iteration {state.iter} with gradual compression enabled: {state.gradual_compression}")
        if state.use_error_feedback in ["ef14", "ef21"]:
            logger.info("Initializing error feedback mechanism for smooth compression transition")
    
    device = tensors[0].device  
    dtype = tensors[0].dtype

    current_compress_ratio = state.get_current_compress_ratio()

    # allreduce
    dist.all_reduce(input_tensor, group=group_to_use, async_op=False)
    
    bucket_index = bucket.index() 
    total_length = input_tensor.shape[0]
    if state.use_error_feedback == "ef14":
        if bucket_index in state.error_dict:
            input_tensor.add_(state.error_dict[bucket_index], alpha=1.0) # input_tensor = \nabla F_i + E_{i-1}
        else:
            logger.info("A zero tensor of length %s that represents local error is created.", total_length)
            state.error_dict[bucket_index] = torch.zeros(total_length, device=device, dtype=dtype)
    
    state.error_dict[bucket_index].copy_(input_tensor)  # E_i = \nabla F_i + E_{i-1}

    if bucket_index not in state.generator:
        state.generator[bucket_index] = torch.Generator(device = device)
        state.generator[bucket_index].manual_seed(torch.randint(0, 100000, (1,), generator=state.rng).item())

    for tensor in tensors:
        if len(tensor.shape) == 2:
            m, n = tensor.shape # [m, n]
            k = max(1, int(m * current_compress_ratio))
            V = torch.empty(n, state.r, dtype=dtype, device=device) # [n, r]
            V.normal_(generator=state.generator[bucket_index])
            sigma = torch.norm(tensor @ V, dim=1).abs() # [m, 1]
            zero_indices = torch.argsort(sigma, descending=True)[k:]
            tensor[zero_indices] = 0.0 
        elif len(tensor.shape) > 2:
            raise NotImplementedError("Fake Group TopK compression only supports 2D tensors.")
    
    state.error_dict[bucket_index].add_(input_tensor, alpha=-1.0)  # E_i = \nabla F_i + E_{i-1} - C[\nabla F_i + E_{i-1}]

    state.maybe_increase_iter(bucket)

    fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
    fut.set_result(input_tensor)

    return fut