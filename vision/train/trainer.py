import torch
import torch.distributed as dist

from .parallel import is_main_process


class Trainer:
    def __init__(
        self,
        config,
        model,
        optimizer,
        loader,
        scheduler,
        *,
        device: torch.device | None = None,
        log_every: int = 10,
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.log_every = log_every
        self.step = 0
        self.epoch = 0

    def _next_batch(self, iterator):
        try:
            return iterator, next(iterator)
        except StopIteration:
            self.epoch += 1
            sampler = self.loader.sampler
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(self.epoch)
            iterator = iter(self.loader)
            return iterator, next(iterator)

    def _clip_grads(self):
        if hasattr(self.model, "clip_grad_norm_"):
            return self.model.clip_grad_norm_(self.config.max_grad_norm)
        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )

    def train_step(self, batch):
        self.model.train()
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        out = self.model(**batch)
        loss = out["loss"] / self.config.gradient_accumulation_steps
        loss.backward()
        return loss.detach()

    def train(self):
        iterator = iter(self.loader)
        self.optimizer.zero_grad(set_to_none=True)

        while self.step < self.config.max_steps:
            step_loss = torch.zeros((), device=self.device)
            for _ in range(self.config.gradient_accumulation_steps):
                iterator, batch = self._next_batch(iterator)
                step_loss = step_loss + self.train_step(batch)

            self._clip_grads()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            if is_main_process() and self.step % self.log_every == 0:
                avg_loss = step_loss.item() * self.config.gradient_accumulation_steps
                lr = self.scheduler.get_last_lr()[0]
                print(f"step={self.step} loss={avg_loss:.4f} lr={lr:.2e}")

        if dist.is_initialized():
            dist.barrier()
