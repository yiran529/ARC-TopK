import torch
import torch.distributed as dist
import os

import logging
#from optimizer import PrecondAdam

logger = logging.getLogger(__name__)


def save_comm_hook_state(hook_state, checkpoint_dir):
    """Save one rank's communication-hook state beside the training checkpoint."""
    if hook_state is None:
        return
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"comm_hook_rank{rank}.pt")
    temporary_path = f"{path}.tmp"
    torch.save(hook_state.state_dict(), temporary_path)
    os.replace(temporary_path, path)


def load_comm_hook_state(hook_state, checkpoint_dir, *, device=None):
    """Load this rank's hook state; return False for legacy checkpoints."""
    if hook_state is None:
        return False
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    path = os.path.join(checkpoint_dir, f"comm_hook_rank{rank}.pt")
    if not os.path.exists(path):
        logger.warning("No rank-local communication-hook state found at %s", path)
        return False
    hook_state.load_state_dict(torch.load(path, map_location="cpu"), device=device)
    return True

def _get_allgather_out_list(all_gather_in_list, world_size): 
    out_list = [
        torch.zeros_like(
            all_gather_in_list,
            device=all_gather_in_list.device,
            dtype=all_gather_in_list.dtype,
        )
        for _ in range(world_size)
    ]
    return out_list

 
class HookState:
    
    def __init__(self, process_group: dist.ProcessGroup): 
        self.process_group = process_group 
        self.start_compress_iter = 0 # start_compress_iter
        self.iter = 0 

        self.total_bit_before_compression = 0 
        self.total_bit_after_compression = 0 
        self.compressor_name="none compressor" 

        self.compress_momentum = False 
        self.param_state = None 
        self.param_to_name = None 
        self.beta1 = None 
        self.adam_freeze_key = False 

        self.comm_bits_this_round = 0

    def init_momentum_field(self, param_state, beta1):
        self.param_state = param_state
        self.beta1 = beta1
        self.compress_momentum = True 

    def maybe_accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum: 
            return
        if self.iter >= self.start_compress_iter and not self.adam_freeze_key:
            self.adam_freeze_key = True 
            logger.info(f"Freeze the second momentum of Adam optimizer after {self.iter}(included) steps")
        if self.adam_freeze_key:
            self.accumulate_momentum_on_bucket(bucket)

    def accumulate_momentum_on_bucket(self, bucket: dist.GradBucket):
        if not self.compress_momentum:
            raise RuntimeError("Momentum compression is not enabled!")
        if self.param_state is None:
            raise RuntimeError("Parameter state is not initialized!")
        # accumulate momentum
        parameters, gradients = bucket.parameters(), bucket.gradients()
        assert len(parameters) == len(gradients), "The number of parameters and gradients should be the same."
        for tensor, grad in zip(parameters, gradients):
            state = self.param_state[tensor] 
            grad.mul_(1 - self.beta1).add_(state['exp_avg'], alpha=self.beta1)

    def maybe_increase_iter(self, bucket):
        """Track iterations and trigger log message at start of local SGD."""
        # Since bucket 0 is the last bucket to allreduce in an iteration.
        # Only increase `iter` when bucket 0 is processed.
        if bucket.is_last(): 
            self.iter += 1

            if self.iter == self.start_compress_iter:
                logger.info(f"Start to apply {self.compressor_name} hook after {self.start_compress_iter} iterations.")

    def compression_bits_stats(self):

        compress_rate = ( 
            self.total_bit_before_compression / self.total_bit_after_compression
            if self.total_bit_after_compression > 0
            else 0
        )
        return (
            compress_rate,
            self.total_bit_before_compression,
            self.total_bit_after_compression,
        )

    def state_dict(self):
        """Serialize rank-local state needed for an exact hook continuation."""
        scalar_names = (
            "iter", "total_bit_before_compression", "total_bit_after_compression",
            "comm_bits_this_round", "adam_freeze_key", "large_batch_init",
            "compression_started", "compress_ratio",
        )
        state = {
            "version": 1,
            "scalars": {
                name: getattr(self, name) for name in scalar_names if hasattr(self, name)
            },
        }
        for name in ("error_dict", "global_error_dict"):
            if hasattr(self, name):
                state[name] = {
                    key: value.detach().cpu().clone()
                    for key, value in getattr(self, name).items()
                }
        if hasattr(self, "rng"):
            state["rng_state"] = self.rng.get_state().clone()
        if hasattr(self, "generator"):
            state["generator_states"] = {
                key: generator.get_state().clone()
                for key, generator in self.generator.items()
            }
        return state

    def load_state_dict(self, state_dict, *, device=None):
        """Restore a state produced by :meth:`state_dict` onto this rank."""
        if state_dict.get("version") != 1:
            raise ValueError("Unsupported communication-hook checkpoint version")
        device = torch.device("cpu") if device is None else torch.device(device)
        for name, value in state_dict.get("scalars", {}).items():
            if hasattr(self, name):
                setattr(self, name, value)
        for name in ("error_dict", "global_error_dict"):
            if name in state_dict and hasattr(self, name):
                setattr(self, name, {
                    key: value.to(device=device).clone()
                    for key, value in state_dict[name].items()
                })
        if "rng_state" in state_dict and hasattr(self, "rng"):
            self.rng.set_state(state_dict["rng_state"].cpu())
        if "generator_states" in state_dict and hasattr(self, "generator"):
            restored = {}
            for key, generator_state in state_dict["generator_states"].items():
                generator = torch.Generator(device=device)
                generator.set_state(generator_state.cpu())
                restored[key] = generator
            self.generator = restored


def register_comm_hook_for_ddp_model(model, process_group, args, optimizer=None):
    hook_state = None
    if args.compressor == "topk_sync" or args.compressor == "randk_sync": 
        from comm_hooks.sparse_hook_c4 import SparseState, sparse_hook_sync
        random = 'randk' in args.compressor
        hook_state = SparseState(
            process_group=process_group,
            compress_ratio=args.compress_ratio, 
            sparse_type=args.sparse_type,
            use_error_feedback=args.use_error_feedback,
            random=random,
            start_compress_iter=args.start_compress_iter,
            random_seed=args.seed,
        )
        model.register_comm_hook(hook_state, sparse_hook_sync)
  

    elif args.compressor == "group_topk_no_reshape" :
        #from comm_hooks.group_topk_hook_no_reshape_c4 import group_topk_hook, GroupTopKState
        from comm_hooks.group_topk_hook_no_reshape import group_topk_hook, GroupTopKState
        hook_state = GroupTopKState(
            process_group=process_group, 
            r=args.r, 
            use_error_feedback=args.use_error_feedback, 
            seed=args.seed,
            start_compress_iter=args.start_compress_iter,
            compress_ratio=args.compress_ratio,
        )
        model.register_comm_hook(hook_state, group_topk_hook) 

    elif args.compressor == 'noop': 
        from comm_hooks.debugging_hooks import noop_hook
        model.register_comm_hook(None, noop_hook) 

    elif args.compressor == 'none' :
        from comm_hooks.default_hooks import my_allreduce_hook
        hook_state = HookState(process_group) 
        hook_state.start_compress_iter = args.start_compress_iter
        model.register_comm_hook(hook_state, my_allreduce_hook) 
    else:
        raise ValueError(f"Compressor {args.compressor} not supported.")
    
    # For selective compression
    if hasattr(hook_state, 'param_to_name'):
        hook_state.param_to_name = {param: name for name, param in model.named_parameters()}
    # for param, name in hook_state.param_to_name.items():
    #     if dist.get_rank() == 0:
    #         logger.info(f"Parameter name: {name}, shape: {param.shape}")
    
    return hook_state

def add_comm_hook_args(parser):
    ### Commpressor arguments
    parser.add_argument(
        "--compressor",
        type=str,
        default="none",
        help="Set the compressor to use.",
    )
    parser.add_argument(
        "--start_compress_iter",  
        type=int,
        default=10,
        help="Set the iteration to start compression.",
    )
    parser.add_argument( 
        "--use_error_feedback",
        type=str,
        default="noef", 
        choices=["noef", "ef14", "ef21"],
        help="Set the error feedback to use.",
    )

    parser.add_argument(
        "--sparse_type",
        type=str,
        default='tensor',
        choices=['row', 'column', 'tensor'], 
        help="Set the type of top-k sparsification to use.",
    )
    parser.add_argument(
        "--compress_ratio", 
        type=float,
        default=0.08,
        help="Set the ratio of the top-k elements to keep.",
    )
    
    parser.add_argument(
        "--r", 
        type=int,
        default=4,
        help="num of cols after projection.",
    )
    

    ### check whether the gradients are identical across all processes
    parser.add_argument(
        "--check_grad",
        action="store_true",
        default=False,
        help="Whether to check the identity of the gradients.",
    )
 

def dtype_bits(tensor): 
    dtype = tensor.dtype
    if dtype.is_floating_point: 
        return torch.finfo(dtype).bits
    elif dtype.is_complex:
        return torch.finfo(dtype).bits * 2  # Complex numbers have twice the bits
    elif dtype == torch.bool:
        return 1
    elif "int" in str(dtype): 
        return torch.iinfo(dtype).bits
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    
def tensor_bits(tensor):
    return tensor.numel() * dtype_bits(tensor)


def name_func_glue(args):
    if args.optimizer == "muon":
        program_name = f"muon_glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
        run_name = (
            f"muon_lr{args.learning_rate}_mu{args.muon_mu}_bs{args.per_device_train_batch_size}"
            f"_seed{args.seed}_{args.compressor}_{args.use_error_feedback}"
            f"_wd{args.weight_decay}_ratio{args.compress_ratio}"
        )
        if args.compressor == "group_topk_no_reshape":
            run_name += f"_r{args.r}"
        return program_name, run_name
    if args.optimizer=="adamw":
        if args.weight_decay==0:
            if args.compress_ratio==0.08:
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_r{args.r}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}"
                        
            else:   # compress_ratio=0.2
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_r{args.r}_ratio{args.compress_ratio}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_ratio{args.compress_ratio}"
                    
        else: # weight_decay
            if args.compress_ratio==0.08:
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name = f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}"
                    
            else:   # compress_ratio=0.2
                if args.compressor=="group_topk_no_reshape":
                    program_name = f"glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_r{args.r}_ratio{args.compress_ratio}"
                    
                else:  # args.compressor=topk
                    program_name = f"glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                    run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_ratio{args.compress_ratio}"
                    

    else:  # sgd
        if args.compress_ratio==0.08:
            if args.compressor=="group_topk_no_reshape":
                program_name = f"msgd_glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_r{args.r}"
                
            else:
                program_name = f"msgd_glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}"
                    
        else: 
            if args.compressor=="group_topk_no_reshape":
                program_name = f"msgd_glue_no_trainer_{args.task_name}_group_topk_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_r{args.r}_ratio{args.compress_ratio}"
                
            else:
                program_name = f"msgd_glue_no_trainer_{args.task_name}_{args.compressor}_{args.use_error_feedback}"
                run_name =f"lr{args.learning_rate}_bs{args.per_device_train_batch_size}_seed{args.seed}_{args.compressor}_{args.use_error_feedback}_wd{args.weight_decay}_mo{args.momentum}_ratio{args.compress_ratio}"
                
    return program_name, run_name
