import math

import torch
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LambdaLR

from ..genlip.config import OptimizerConfig, SchedulerConfig


def build_optimizer(config: OptimizerConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    params = model.parameters()
    name = config.name.lower()
    if name == "adamw":
        return AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=tuple(config.betas),
            eps=config.eps,
        )
    if name == "adam":
        return Adam(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=tuple(config.betas),
            eps=config.eps,
        )
    if name == "sgd":
        return SGD(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=config.betas[0],
        )
    raise ValueError(f"Unknown optimizer.name={config.name!r}; expected adamw, adam, or sgd")


def build_scheduler(
    config: SchedulerConfig,
    optimizer: torch.optim.Optimizer,
    max_steps: int,
) -> LambdaLR:
    warmup = config.warmup_steps
    min_lr_ratio = config.min_lr_ratio
    name = config.name.lower()

    def warmup_factor(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step) / float(warmup))

    if name == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return warmup_factor(step)
            progress = (step - warmup) / max(1, max_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    elif name == "linear":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return warmup_factor(step)
            progress = (step - warmup) / max(1, max_steps - warmup)
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
    elif name == "constant":
        def lr_lambda(step: int) -> float:
            return warmup_factor(step)
    else:
        raise ValueError(f"Unknown scheduler.name={config.name!r}; expected cosine, linear, or constant")

    return LambdaLR(optimizer, lr_lambda)
