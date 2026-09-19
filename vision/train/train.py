import argparse

import torch
import torch.distributed as dist

from ..genlip.config import load_config
from ..genlip.model import GenLIP, build_tokenizer
from .data import build_dataloader
from .optim import build_optimizer, build_scheduler
from .parallel import apply_hsdp, build_hsdp_mesh, init_distributed, is_main_process
from .trainer import Trainer


def main(config_path: str):
    config = load_config(config_path)
    local_rank, rank, world_size = init_distributed()
    device = torch.device("cuda", local_rank)
    torch.manual_seed(config.seed + rank)

    mesh = build_hsdp_mesh(config.parallel.replicate, config.parallel.shard)

    tokenizer = build_tokenizer(config.data.tokenizer_name)
    model = GenLIP(config.model, config.dart).to(device)
    model.encoder.gradient_checkpointing = True
    model = apply_hsdp(model, mesh, bf16=config.parallel.bf16)

    loader = build_dataloader(
        config.data,
        tokenizer,
        split=config.data.train_split,
        rank=rank,
        world_size=world_size,
    )

    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.trainer.max_steps)

    if is_main_process():
        dp_size = world_size
        print(
            f"training on {world_size} GPU(s): "
            f"replicate={config.parallel.replicate}, shard={config.parallel.shard}, "
            f"batch_size={config.data.batch_size}/gpu, "
            f"global_batch={config.data.batch_size * dp_size * config.trainer.gradient_accumulation_steps}"
        )

    trainer = Trainer(
        config.trainer,
        model,
        optimizer,
        loader,
        scheduler,
        device=device,
    )
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vision/config.yaml")
    main(parser.parse_args().config)
