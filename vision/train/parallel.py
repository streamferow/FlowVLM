import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


def init_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def build_hsdp_mesh(replicate: int, shard: int):
    return init_device_mesh("cuda", (replicate, shard), mesh_dim_names=("replicate", "shard"))

def apply_hsdp(model: nn.Module, mesh, bf16: bool = True):
    mixed_precision = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_type=torch.float32) if bf16 else None
    for layer in model.encoder.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mixed_precision)
    fully_shard(model, mesh=mesh, mp_policy=mixed_precision)
    return model
