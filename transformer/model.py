from dataclasses import dataclass

import torch
import torch.nn as nn

from .interleaved_mrope import (
    get_mrope_index,
    build_mrope_frequencies,
    apply_rotary_embeddings,
)


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int | None = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5

    mrope_sections: tuple[int, int, int] = (24, 20, 20)
    mrope_theta: float = 1_000_000.0

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None



class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.vocab_size != -1, "Vocab_size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.token_embeddings = nn.Embedding(args.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.normalization = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)


    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.token_embeddings(tokens)

        position_ids, deltas = get_mrope_index(tokens)
        frequencies_complex = build_mrope_frequencies(position_ids, self.args.dim // self.args.n_heads, self.args.mrope_theta, self.args.mrope_sections)

        # Consecutively apply all the layers of the transformer
        for layer in self.layers:
            h = layer(h, start_pos, frequencies_complex)
        h = self.normalization(h)
        output = self.output(h).float()
        return output


