import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def init_distributed() -> tuple[int, int, int]:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def build_hsdp_mesh(replicate: int, shard: int):
    world_size = dist.get_world_size()
    if replicate * shard != world_size:
        raise ValueError(
            f"parallel.replicate({replicate}) * parallel.shard({shard}) "
            f"must equal world_size({world_size})"
        )
    return init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("replicate", "shard"))


def _mixed_precision_policy(bf16: bool) -> MixedPrecisionPolicy:
    if not bf16:
        return MixedPrecisionPolicy()
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0):
        param_dtype = torch.bfloat16
    else:
        param_dtype = torch.float16
    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32)


def apply_hsdp(model: nn.Module, mesh, bf16: bool = True):
    mp_policy = _mixed_precision_policy(bf16)
    for layer in model.encoder.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    return model
