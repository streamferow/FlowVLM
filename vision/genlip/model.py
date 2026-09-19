from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

from timm.layers import DropPath

from transformer.interleaved_mrope import apply_rotary_embeddings, build_mrope_frequencies
from vision.tokenizer.dart import build_dart

from .config import DARTConfig, ModelConfig


class SpatialMerger(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.merge_size = config.spatial_merge_size
        if self.merge_size > 1:
            self.proj = nn.Linear(
                config.hidden_size * self.merge_size ** 2,
                config.hidden_size,
                bias=False,
            )

    def forward(self, tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        if self.merge_size == 1:
            return tokens
        batch_size, _, hidden_size = tokens.shape
        merge = self.merge_size
        tokens = tokens.view(batch_size, grid_h, grid_w, hidden_size)
        merged_h, merged_w = grid_h // merge, grid_w // merge
        tokens = tokens.view(batch_size, merged_h, merge, merged_w, merge, hidden_size)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(
            batch_size, merged_h * merged_w, merge * merge * hidden_size
        )
        return self.proj(tokens)


class GenLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embedding_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, pixel_values: torch.Tensor, grid_twh: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, C, H, W)
        """
        target_dtype = self.patch_embedding.weight.dtype
        # (B, C, H, W) -> (B, D, Hp, Wp)
        patch_embeddings = self.patch_embedding(pixel_values.to(target_dtype))

        # Fixed resolution 
        B, D, Hp, Wp = patch_embeddings.shape
        assert Hp * self.patch_size == pixel_values.shape[2] and Wp * self.patch_size == pixel_values.shape[3]
        # (B, D, Hp, Wp) -> (B, (Hp * Wp), D)
        embeddings = patch_embeddings.flatten(2).transpose(1, 2).contiguous()
        return embeddings



def build_tokenizer(tokenizer_name: str = "Qwen/Qwen3-0.6B"):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def encode_captions(tokenizer: AutoTokenizer, texts: list[str], max_len: int):
    encoded_texts = tokenizer(
        texts,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded_texts["input_ids"]
    labels = input_ids.clone()
    labels[encoded_texts["attention_mask"] == 0] = -100
    return input_ids, labels

class TextEmbedding(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(self, input_ids: torch.Tensor):
        return self.token_embedding(input_ids)



class EarlyFusion(nn.Module):
    def forward(self, vision: torch.Tensor, text: torch.Tensor):
        # vision: (B, N, D), text: (B, L, D) → (B, N+L, D)
        return torch.cat([vision, text], dim=1)  # (B, K+L, D)



class Gate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.projection = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.projection(x)) * attn_out



def calculate_pad_length(seq_len: int, block_size: int = 128) -> int:
    if seq_len % block_size == 0:
        return 0
    return block_size - (seq_len % block_size)



class GenLIPGatedAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embedding_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embedding_dim:
            raise ValueError(
                f"embedding_dim must be divisible by num_heads (got `embedding_dim`: {self.embedding_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.is_causal = False
        self.k_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.v_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.q_projection = nn.Linear(self.embedding_dim, self.embedding_dim * 2)
        self.out_projection = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


    def forward(
        self, 
        hidden_states: torch.Tensor, 
        frequencies_complex: torch.Tensor,
        flex_attention_args: dict | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, embedding_dim = hidden_states.shape

        # (B, S, D) -> (B, S, D * 2) -> q: (B, S, D) g: (B, S, D)
        q, gate_score = self.q_projection(hidden_states).chunk(2, dim=-1)
        # (B, S, D * 2) -> (B, S, H, Dh)
        gate_score = gate_score.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        # (B, S, D) -> (B, S, H, D)
        k = self.k_projection(hidden_states)
        v = self.v_projection(hidden_states)

        # (B, S, D) -> (B, S, H, Dh) 
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        q = apply_rotary_embeddings(q, frequencies_complex, device=hidden_states.device)
        k = apply_rotary_embeddings(k, frequencies_complex, device=hidden_states.device)
        
        # (B, S, H, Dh) -> (B, H, S, Dh) 
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        assert flex_attention_args is not None, "flex_attention_args must be provided"

        target_len = flex_attention_args["max_seqlen"]
        pad_length = target_len - q.shape[2]
        pad_head_dim = (64 - self.head_dim % 64) % 64

        # (B, H, S, Dh) -> (B, S, H, Dh)
        attention_output = self.flex_attention(
            F.pad(q, (0, pad_head_dim, 0, pad_length)), # (B, H, S + pad_S, Dh + pad_Dh)
            F.pad(k, (0, pad_head_dim, 0, pad_length)),
            F.pad(v, (0, pad_head_dim, 0, pad_length)),
            block_mask=flex_attention_args["block_mask"],
            scale=self.scale,
            kernel_options={"FORCE_USE_FLEX_ATTENTION": True},
        )[:, :, :q.shape[2], :self.head_dim].transpose(1, 2).contiguous()

        # (B, S, H, Dh) * (B, S, H, Dh) -> (B, S, H, Dh)
        attention_output = attention_output * torch.sigmoid(gate_score)
        # (B, S, H, Dh) -> (B, S, D)
        attention_output = attention_output.reshape(batch_size, seq_len, embedding_dim).contiguous()
        # (B, S, D) -> (B, S, D)
        attention_output = self.out_projection(attention_output)
        return attention_output



class GenLIPSwigluFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.activation_function = nn.SiLU()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.gate_fc = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated_hidden_states = self.activation_function(self.gate_fc(hidden_states))
        hidden_states = gated_hidden_states * self.fc1(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states



class GenLIPLayerScale(nn.Module):
    def __init__(self, config: ModelConfig, inplace: bool = False):
        super().__init__()
        self.lambda1 = nn.Parameter(config.ls_init_value * torch.ones(config.hidden_size))
        self.inplace = inplace

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states._mul_(self.lambda1) if self.inplace else hidden_states * self.lambda1



class GenLIPEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embedding_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embedding_dim, eps=config.layer_norm_eps)
        self.self_attention = GenLIPGatedAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embedding_dim, eps=config.layer_norm_eps)
        self.ffn = GenLIPSwigluFFN(config)

        self.layer_scale1 = GenLIPLayerScale(config, inplace=True) if config.ls_init_value > 1e-6 else nn.Identity()
        self.layer_scale2 = GenLIPLayerScale(config, inplace=True) if config.ls_init_value > 1e-6 else nn.Identity()

        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 1e-6 else nn.Identity()

    def forward(
        self,
        hidden_states: torch.Tensor,
        frequencies_complex: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        flex_attention_args: dict | None = None,
    ):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attention(hidden_states, frequencies_complex, flex_attention_args)

        hidden_states = self.drop_path(self.layer_scale1(hidden_states))
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.ffn(hidden_states)

        hidden_states = self.drop_path(self.layer_scale2(hidden_states))
        hidden_states = hidden_states + residual
        return hidden_states



class GenLIPEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.layers = nn.ModuleList([GenLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        input_embeddings: torch.Tensor,
        frequencies_complex: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        flex_attention_args: dict | None = None,
    ) -> torch.Tensor:
        hidden_states = input_embeddings
        for encoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    frequencies_complex,
                    attention_mask,
                    flex_attention_args,
                    use_reentrant=False,
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    frequencies_complex,
                    attention_mask,
                    flex_attention_args,
                )
        return hidden_states



class GenLIP(nn.Module):
    def __init__(self, config: ModelConfig, dart_config: DARTConfig | None = None):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.spatial_merge_size = config.spatial_merge_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings

        if dart_config is not None:
            self.vision_embeddings = build_dart(dart_config)
            self.use_dart = True
        else:
            self.vision_embeddings = GenLIPVisionEmbeddings(config)
            self.use_dart = False
        self.spatial_merger = SpatialMerger(config)
        self.text_embeddings = TextEmbedding(config)
        self.fusion = EarlyFusion()
        self.encoder = GenLIPEncoder(config)
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)


    def _vision_text_position_offset(self, height: int, width: int) -> int:
        hp = height // self.patch_size // self.spatial_merge_size
        wp = width // self.patch_size // self.spatial_merge_size
        return max(hp, wp)

    def _build_vision_position_ids_from_centers(
        self,
        patch_centers: torch.Tensor,
    ) -> torch.Tensor:
        """Build (t, h, w) mRoPE ids from DART patch centroids in patch units."""
        patch_size = float(self.patch_size)
        t = torch.zeros_like(patch_centers[..., 0])
        h = patch_centers[..., 0] / patch_size
        w = patch_centers[..., 1] / patch_size
        return torch.stack([t, h, w], dim=0)

    def _build_vision_position_ids(
        self, 
        batch_size: int, 
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        hp = height // self.patch_size // self.spatial_merge_size
        wp = width // self.patch_size // self.spatial_merge_size
        
        # (B, Hp, Wp)
        t = torch.zeros(batch_size, hp, wp, device=device, dtype=torch.long)
        h = torch.arange(hp, device=device).view(1, hp, 1).expand(batch_size, hp, wp)
        w = torch.arange(wp, device=device).view(1, 1, wp).expand(batch_size, hp, wp)
        
        # (3, B, N = Hp * Wp)
        position_ids = torch.stack([t.flatten(1), h.flatten(1), w.flatten(1)], dim=0)
        return position_ids


    def _build_fused_position_ids(
        self,
        batch_size: int,
        height: int,
        width: int,
        text_len: int,
        device: torch.device,
        text_attention_mask: torch.Tensor | None = None,
        patch_centers: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if patch_centers is not None:
            vision_position_ids = self._build_vision_position_ids_from_centers(patch_centers)
        else:
            vision_position_ids = self._build_vision_position_ids(batch_size, height, width, device)
        offset = self._vision_text_position_offset(height, width)

        if text_attention_mask is None:
            # (B, L)
            text_attention_mask = torch.ones(batch_size, text_len, device=device, dtype=torch.long)

        # (B, L)
        text_mask_cumsum = text_attention_mask.long().cumsum(dim=-1) - 1
        text_mask_cumsum = text_mask_cumsum.clamp(min=0) + offset
        text_mask_cumsum = text_mask_cumsum.masked_fill(text_attention_mask == 0, 0)

        # (3, B, L)
        text_position_ids = text_mask_cumsum.unsqueeze(0).expand(3, -1, -1)

        # (3, B, N+L)
        fused_position_ids = torch.cat([vision_position_ids, text_position_ids], dim=-1)
        return fused_position_ids


    def _build_prefix_lm_flex_args(
        self,
        vision_len: int,
        seq_len: int,
        device: torch.device,
        block_size: int = 128,
    ) -> dict:
        if seq_len > self.max_position_embeddings:
            raise ValueError(
                f"seq_len={seq_len} > max_position_embeddings={self.max_position_embeddings}"
            )

        pad_length = calculate_pad_length(seq_len, block_size)
        padded_seq_len = seq_len + pad_length
        if padded_seq_len > self.max_position_embeddings:
            raise ValueError(
                f"padded_seq_len={padded_seq_len} > max_position_embeddings={self.max_position_embeddings}"
            )

        def prefix_lm(batch_idx, head_idx, query_idx, key_idx):
             # real (not padded) query and key
            real_query = query_idx < seq_len
            real_key = key_idx < seq_len

            query_is_vision = query_idx < vision_len
            key_is_vision = key_idx < vision_len
            query_is_text = query_idx >= vision_len
            key_is_text = key_idx >= vision_len
            
            # vision to vision, full attention within vision tokens
            vision_attends_vision = query_is_vision & key_is_vision

            # text to vision, causal attention from text to vision tokens
            text_attends_vision = query_is_text & key_is_vision

            # text to text, causal attention within text tokens in causal order
            text_attends_text = query_is_text & key_is_text & (key_idx <= query_idx)

            allowed = vision_attends_vision | text_attends_vision | text_attends_text
            return real_query & real_key & allowed

        block_mask = create_block_mask(
            prefix_lm,
            B=None,
            H=None,
            Q_LEN=padded_seq_len,
            KV_LEN=padded_seq_len,
            device=device,
        )
        return {"block_mask": block_mask, "max_seqlen": padded_seq_len}


    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        _, _, height, width = pixel_values.shape
        grid_h = height // self.patch_size
        grid_w = width // self.patch_size

        # (B, N, D)
        if self.use_dart:
            vision, patch_centers = self.vision_embeddings(pixel_values)
        else:
            vision = self.vision_embeddings(pixel_values)
            patch_centers = None
        vision = self.spatial_merger(vision, grid_h, grid_w)
        # (B, L, D)
        text = self.text_embeddings(input_ids)
        # (B, N+L, D)
        hidden = self.fusion(vision, text)

        batch_size, seq_len, _ = hidden.shape
        vision_len = vision.shape[1]
        text_len = input_ids.shape[1]
        
        # (3, B, N+L)
        position_ids = self._build_fused_position_ids(
            batch_size,
            height,
            width,
            text_len,
            pixel_values.device,
            attention_mask,
            patch_centers=patch_centers,
        )
        # (3, B, N+L, Dh)
        frequencies_complex = build_mrope_frequencies(
            position_ids,
            head_dim=self.head_dim,
            theta=self.config.mrope_theta,
            mrope_sections=self.config.mrope_sections,
        )
        # dict with "block_mask" and "max_seqlen"
        flex_attention_args = self._build_prefix_lm_flex_args(
            vision_len, 
            seq_len, 
            pixel_values.device
        )
        # (B, N+L, D)
        hidden = self.encoder(
            input_embeddings=hidden,
            frequencies_complex=frequencies_complex,
            attention_mask=None,
            flex_attention_args=flex_attention_args,
        )
        
        # (B, N+L, D)
        hidden = self.ln_post(hidden)
        # (B, L, D) — text tokens follow the vision prefix
        text_hidden = hidden[:, vision_len:, :]
        # (B, L, V)
        logits = self.lm_head(text_hidden)
        
        out: dict[str, torch.Tensor] = {"logits": logits}
        if labels is not None:
            # next-token: logits[:-1] vs labels[1:]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = self.loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            out["loss"] = loss
            out["ce"] = loss
        return out




