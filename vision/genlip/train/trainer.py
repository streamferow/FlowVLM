import torch
from config import load_config
from model import GenLIP


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        loader,
        scheduler,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.step = 0


    def train_step(self, batch):
        self.model.train()
        batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
        out = self.model(**batch)
        loss = out["loss"] / config.gradient_accumulation_steps
        loss.backward()
        return loss.detach()

    
    def train(self):
        iterator = iter(self.loader)
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.config.max_steps:
            for _ in range(self.config.gradient_accumulation_steps):

                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(self.loader)
                    batch = next(iterator)

                loss = self.train_step(batch)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1


if __name__ == "__main__":
    config = load_config("config.yaml")
    model = GenLIP(config.model)
                
