from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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



class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_len, Dim) * (B, Seq_len, 1) -> (B, Seq_len, Dim)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (Dim) * (B, Seq_len, Dim) -> (B, Seq_len, Dim)
        return self.weight * self._norm(x.float()).type_as(x)



class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * args.ffn_dim_multiplier)

        rounded_hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, rounded_hidden_dim, bias=False)
        self.w2 = nn.Linear(rounded_hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, rounded_hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x



class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, frequencies_complex: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # (B, Seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), frequencies_complex, attention_mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out



def repeat_kv(x: torch.Tensor, n_rep: int):
    if n_rep == 1:
        return x
    B, S, n_kv_heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(B, S, n_kv_heads, n_rep, head_dim).reshape(B, S, n_kv_heads * n_rep, head_dim)
    return x



class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_kv_heads if args.n_kv_heads is not None else args.n_heads
        self.n_heads_q = args.n_heads

        self.n_rep = self.n_heads_q // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, frequencies_complex: torch.Tensor, attention_mask: torch.Tensor | None = None):
        B, S, _ = x.shape

        # (B, S, D) -> (B, S, N_heads_q, Head_dim)
        xq = self.wq(x).view(B, S, self.n_heads_q, self.head_dim)
        # (B, S, D) -> (B, S, N_kv_heads, Head_dim)
        xk = self.wk(x).view(B, S, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, S, self.n_kv_heads, self.head_dim)
        
        xq = apply_rotary_embeddings(xq, frequencies_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, frequencies_complex, device=x.device)

        # (B, S, N_kv_heads, Head_dim) -> (B, S, N_kv_heads * N_rep, Head_dim)
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # (B, S, N_heads, Head_dim) -> (B, N_heads, S, Head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # (B, N_heads, S, Head_dim) @ (B, N_heads, Head_dim, S) -> (B, N_heads, S, S)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / (self.head_dim ** 0.5)
        causal_mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device, dtype=scores.dtype), diagonal=1)
        scores = scores + causal_mask
        if attention_mask is not None:
            # (B, S) -> (B, 1, 1, S)
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float("-inf"))
        scores = torch.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, N_heads, S, S) @ (B, N_heads, S, Head_dim) -> (B, N_heads, S, Head_dim)
        out = torch.matmul(scores, xv)
        # (B, N_heads, S, Head_dim) -> (B, S, N_heads * Head_dim)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        # (B, S, N_heads * Head_dim) @ (N_heads * Head_dim, D) -> (B, S, D)
        out = self.wo(out)
        return out



class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert sum(args.mrope_sections) == (args.dim // args.n_heads) // 2
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

    def forward(
        self, 
        tokens: torch.Tensor, 
        attention_mask: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ):
        # (B, Seq_len)
        batch_size, seq_len = tokens.shape

        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.token_embeddings(tokens)

        if attention_mask is None:
            attention_mask = torch.ones_like(tokens, device=tokens.device)

        position_ids, deltas = get_mrope_index(
            tokens,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        frequencies_complex = build_mrope_frequencies(
            position_ids,
            head_dim=self.args.dim // self.args.n_heads,
            theta=self.args.mrope_theta,
            mrope_sections=self.args.mrope_sections,
        )

        # Consecutively apply all the layers of the transformer
        for layer in self.layers:
            h = layer(h, frequencies_complex, attention_mask)
        h = self.normalization(h)
        output = self.output(h).float()
        return output


