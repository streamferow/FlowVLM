import argparse

import torch

from ..genlip.config import load_config
from ..genlip.model import GenLIP, build_tokenizer
from .data import build_dataloader
from .optim import build_optimizer, build_scheduler
from .parallel import init_distributed, build_hsdp_mesh, apply_hsdp
from .trainer import Trainer


def main(config_path: str):
    config = load_config(config_path)
    torch.manual_seed(config.seed)

    init_distributed()
    mesh = build_hsdp_mesh(config.parallel.replicate, config.parallel.shard)

    tokenizer = build_tokenizer(config.data.tokenizer_name)
    model = GenLIP(config.model, config.dart).cuda()
    model = apply_hsdp(model, mesh, bf16=config.parallel.bf16)

    loader = build_dataloader(config.data, tokenizer, split=config.data.train_split)

    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.trainer.max_steps)

    trainer = Trainer(config.trainer, model, optimizer, loader, scheduler)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="vision/config.yaml")
    main(parser.parse_args().config)
