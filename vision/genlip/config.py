from dataclasses import dataclass
from pathlib import Path
import yaml


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
class DataConfig:
    dataset_name: str
    cache_dir: str
    train_split: str
    eval_split: str
    caption_column: str
    image_size: int
    tokenizer_name: str

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
class Config:
    seed: int
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig

def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        seed=raw["seed"],
        model=ModelConfig(**raw["model"]),
        data=DataConfig(**raw["data"]),
        optimizer=OptimizerConfig(**raw["optimizer"]),
    )