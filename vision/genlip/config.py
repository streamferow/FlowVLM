from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ModelConfig:
    num_channels: int
    patch_size: int
    spatial_merge_size: int

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    vocab_size: int
    layer_norm_eps: float

    use_swiglu_ffn: bool
    gated_attention: bool
    ls_init_value: float
    drop_path_rate: float

    mrope_sections: tuple[int, int, int]
    mrope_theta: float

    attention_dropout: float
    max_position_embeddings: int


@dataclass
class DARTConfig:
    image_size: tuple[int, int]
    patch_size: tuple[int, int]
    grid: tuple[int, int]
    num_patches: int
    stride: int
    in_channels: int
    embedding_dim: int

    input_dim: int
    hidden_size: int
    output_dim: int


@dataclass
class DataConfig:
    dataset_name: str
    cache_dir: str
    train_split: str
    eval_split: str
    caption_column: str
    image_size: int
    tokenizer_name: str
    max_text_length: int

    batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool


@dataclass
class OptimizerConfig:
    name: str
    learning_rate: float
    weight_decay: float
    betas: list[float]
    eps: float


@dataclass
class SchedulerConfig:
    name: str
    warmup_steps: int
    min_lr_ratio: float


@dataclass
class TrainerConfig:
    max_steps: int
    gradient_accumulation_steps: int
    max_grad_norm: float
    eval_steps: int
    save_steps: int
    bf16: bool
    log_dir: str
    log_every: int


@dataclass
class ParallelConfig:
    replicate: int
    shard: int
    bf16: bool


@dataclass
class Config:
    seed: int
    model: ModelConfig
    dart: DARTConfig
    data: DataConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    trainer: TrainerConfig
    parallel: ParallelConfig


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((REPO_ROOT / p).resolve())


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())

    model_raw = dict(raw["model"])
    model_raw["mrope_sections"] = tuple(model_raw["mrope_sections"])
    model_raw["layer_norm_eps"] = float(model_raw["layer_norm_eps"])
    model_raw["ls_init_value"] = float(model_raw["ls_init_value"])
    model_raw["drop_path_rate"] = float(model_raw["drop_path_rate"])
    model_raw["mrope_theta"] = float(model_raw["mrope_theta"])
    model_raw["attention_dropout"] = float(model_raw["attention_dropout"])

    data_raw = dict(raw["data"])
    data_raw["cache_dir"] = _resolve_path(data_raw["cache_dir"])

    dart_raw = dict(raw["dart"])
    dart_raw["image_size"] = tuple(dart_raw["image_size"])
    dart_raw["patch_size"] = tuple(dart_raw["patch_size"])
    dart_raw["grid"] = tuple(dart_raw["grid"])
    assert dart_raw["num_patches"] == dart_raw["grid"][0] * dart_raw["grid"][1]

    model = ModelConfig(**model_raw)
    dart = DARTConfig(**dart_raw)
    data = DataConfig(**data_raw)
    assert data.image_size == dart.image_size[0] == dart.image_size[1]
    assert model.patch_size == dart.patch_size[0] == dart.patch_size[1]
    assert model.hidden_size == dart.embedding_dim
    assert model.num_channels == dart.in_channels
    assert (data.image_size // model.patch_size) ** 2 == dart.num_patches
    assert dart.grid[0] % model.spatial_merge_size == 0
    assert dart.grid[1] % model.spatial_merge_size == 0
    merged_patches = (dart.grid[0] // model.spatial_merge_size) * (dart.grid[1] // model.spatial_merge_size)
    assert dart.num_patches // (model.spatial_merge_size ** 2) == merged_patches

    optimizer_raw = dict(raw["optimizer"])
    optimizer_raw["learning_rate"] = float(optimizer_raw["learning_rate"])
    optimizer_raw["weight_decay"] = float(optimizer_raw["weight_decay"])
    optimizer_raw["eps"] = float(optimizer_raw["eps"])
    optimizer_raw["betas"] = [float(b) for b in optimizer_raw["betas"]]

    scheduler_raw = dict(raw["scheduler"])
    scheduler_raw["warmup_steps"] = int(scheduler_raw["warmup_steps"])
    scheduler_raw["min_lr_ratio"] = float(scheduler_raw["min_lr_ratio"])

    trainer_raw = dict(raw["trainer"])
    trainer_raw["log_dir"] = _resolve_path(trainer_raw.get("log_dir", "runs"))

    return Config(
        seed=raw["seed"],
        model=model,
        dart=dart,
        data=data,
        optimizer=OptimizerConfig(**optimizer_raw),
        scheduler=SchedulerConfig(**scheduler_raw),
        trainer=TrainerConfig(**trainer_raw),
        parallel=ParallelConfig(**raw["parallel"]),
    )
