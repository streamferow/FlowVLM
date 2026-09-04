from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"

    # Build theta parameters according to the formula: theta_i = 10000 ^ (-2 * (i - 1) / dim) for i  = [1, 2, ... dim / 2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2)
    # Shape: (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # Construct the positions (the "m" parameter)
    # Shape: (Seq_len)
    m = torch.arange(seq_len, device=device)

    # Multiply each theta by each position using outer product
    # Shape: (Seq_len) outer_product (Head_Dim / 2) -> (Seq_len, Head_Dim / 2)
    frequencies = torch.outer(m, theta).float()

    # Compute complex numbers in the polar form c = R * exp(i * m * theta) , where R = 1 as follows:
    # Shape: (Seq_len, Head_Dim / 2) -> (Seq_len, Head_Dim / 2)
    frequencies_complex = torch.polar(torch.ones_like(frequencies), frequencies)
    return frequencies_complex

    



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

        self.frequencies_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.token_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        frequencies_complex = self.frequencies_complex[start_pos : start_pos + seq_len]

        # Consecutively apply all the layers of the transformer
        for layer in self.layers:
            h = layer(h, start_pos, frequencies_complex)
        h = self.normalization(h)
        output = self.output(h).float()
        return output


