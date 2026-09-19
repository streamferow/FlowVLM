import csv
from pathlib import Path

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
    ):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.scheduler = scheduler
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.step = 0
        self.epoch = 0
        self.loss_history: list[tuple[int, float]] = []
        self.log_dir = self._init_log_dir(config.log_dir)

    def _init_log_dir(self, log_dir: str) -> Path | None:
        if not is_main_process():
            return None

        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "loss.csv").open("w", newline="") as f:
            csv.writer(f).writerow(["step", "loss", "lr"])
        return path

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

    def _log_metrics(self, avg_loss: float, lr: float):
        self.loss_history.append((self.step, avg_loss))

        if self.log_dir is None:
            return

        with (self.log_dir / "loss.csv").open("a", newline="") as f:
            csv.writer(f).writerow([self.step, f"{avg_loss:.6f}", f"{lr:.2e}"])

        if self.step % self.config.log_every != 0:
            return

        print(f"step={self.step} loss={avg_loss:.4f} lr={lr:.2e}")
        self._save_loss_plot()

    def _save_loss_plot(self):
        if self.log_dir is None or not self.loss_history:
            return

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return

        steps, losses = zip(*self.loss_history)
        plt.figure(figsize=(8, 4))
        plt.plot(steps, losses, linewidth=1.5)
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.log_dir / "loss.png", dpi=120)
        plt.close()

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

            avg_loss = step_loss.item() * self.config.gradient_accumulation_steps
            lr = self.scheduler.get_last_lr()[0]
            self._log_metrics(avg_loss, lr)

        self._save_loss_plot()
        if self.log_dir is not None:
            print(f"loss log saved to {self.log_dir / 'loss.csv'}")
            print(f"loss plot saved to {self.log_dir / 'loss.png'}")

        if dist.is_initialized():
            dist.barrier()
