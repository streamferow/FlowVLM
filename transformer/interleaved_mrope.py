import torch


def get_mrope_index(
    input_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None = None,
    video_grid_thw: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
    image_token_id: int = 151655,
    video_token_id: int = 151656,
    vision_start_token_id: int = 151652,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build position ids and deltas for interleaved MRoPE.

    Args:
        input_ids: (B, S)
        image_grid_thw: (N, 3)
        video_grid_thw: (M, 3)
        attention_mask: (B, S)
    """
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    B, S = input_ids.shape
    device = input_ids.device

    if attention_mask is None:
        # (B, S)
        attention_mask = torch.ones_like(input_ids)

    # pure text only: no image/video grids → each token gets (m, m, m)
    # so I-MRoPE reduces to vanilla 1D RoPE
    if image_grid_thw is None and video_grid_thw is None:
        # turn mask into running positions among *valid* tokens
        # tokens: [A, B, C, PAD, PAD], mask: [1, 1, 1, 0, 0] -> position_ids: [0, 1, 2, 2, 2]
        position_ids = attention_mask.long().cumsum(dim=-1) - 1

        # overwrite PAD so it does not keep a "real" position
        # position_ids: [0, 1, 2, 2, 2] -> [0, 1, 2, 1, 1]
        position_ids.masked_fill_(attention_mask == 0, 1)

        # (B, S) -> (3, B, S)
        position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        max_pos = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
        deltas = max_pos + 1 - attention_mask.shape[-1]
        return position_ids, deltas

    # Multimodal path: mix of text (m,m,m) and vision patches (t,h,w) with spatial-reset
    # (3, B, S) filled with 1
    position_ids = torch.ones(3, B, S, dtype=input_ids.dtype, device=device)
    mrope_position_deltas = []
    image_index, video_index = 0, 0

    for batch_idx, ids in enumerate(input_ids):
        # filter out PAD tokens
        ids = ids[attention_mask[batch_idx] == 1]

        # after <vision_start_token_id>, we have a sequence of vision tokens (<img>, <vid>)
        vision_starts = torch.argwhere(ids == vision_start_token_id).squeeze(1)
        vision_tokens = ids[vision_starts + 1] if vision_starts.numel() else ids.new_empty(0)
        remain_images = int((vision_tokens == image_token_id).sum())
        remain_videos = int((vision_tokens == video_token_id).sum())

        tokens = ids.tolist()
        llm_pos_ids_list = []
        st = 0

        # iterate over all vision tokens
        for _ in range(remain_images + remain_videos):
            # find the next image or video token index
            ed_image = tokens.index(image_token_id, st) if (image_token_id in tokens[st:] and remain_images > 0) else len(tokens) + 1
            ed_video = tokens.index(video_token_id, st) if (video_token_id in tokens[st:] and remain_videos > 0) else len(tokens) + 1

            if ed_image < ed_video:
                t, h, w = image_grid_thw[image_index].tolist()
                image_index += 1
                remain_images -= 1
                ed = ed_image
            
            else:
                t, h, w = video_grid_thw[video_index].tolist()
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            # convert vision patch dimensions to LLM grid dimensions
            llm_grid_t = int(t)
            llm_grid_h = int(h) // spatial_merge_size
            llm_grid_w = int(w) // spatial_merge_size

            # text before current vision block
            text_len = ed - st
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx)

            # vision spatial-reset: build index triplets (t,h,w) for each vision patch
            t_index = torch.arange(llm_grid_t, device=device).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h, device=device).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w, device=device).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            offset = text_len + st_idx
            llm_pos_ids_list.append(torch.stack([
                t_index + offset,
                h_index,
                w_index,
            ]))

            # update start index for next vision block
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        # text after last vision block
        if st < len(tokens):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = len(tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + st_idx
            )
        
        # concatenate all position ids for this batch
        # (3, B, S) -> (3, B*S)
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[:, batch_idx, attention_mask[batch_idx] == 1] = llm_positions
        mrope_position_deltas.append(llm_positions.max() + 1 - S)

    # (B) -> (B, 1)
    deltas = torch.tensor(mrope_position_deltas, device=device).unsqueeze(1)
    return position_ids, deltas


def apply_interleaved_mrope(
    frequencies: torch.Tensor,
    mrope_section: tuple[int, int, int],
) -> torch.Tensor:
    """Apply interleaved MRoPE to frequencies.

    Args:
        frequencies: (3, B, S, D/2)
        mrope_section: (t, h, w)
    """
    # (3, B, S, D/2) -> (B, S, D/2)
    frequencies_t = frequencies[0].clone()
    # round robin THWTHW...
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        frequencies_t[..., offset:length:3] = frequencies[dim, ..., offset:length:3]
    return frequencies_t


def build_mrope_frequencies(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    mrope_sections: tuple[int, int, int],
) -> torch.Tensor:
    """Build frequencies for interleaved MRoPE.

    Args:
        position_ids: (B, S)
        head_dim: int
        theta: float
        mrope_sections: (t, h, w)
    """
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    assert sum(mrope_sections) == head_dim // 2, "Sum of mrope sections must be equal to head dimension"
 
    device = position_ids.device
    # inv_freq[i] = theta ** (-(2i)/head_dim), i = 0..D/2-1
    # (D/2,)
    theta_numerator = torch.arange(0, head_dim, 2)
    inverse_frequencies = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)

    # (3, B, S, 1) * (D/2,) → (3, B, S, D/2)
    frequencies_3d = position_ids[..., None].float() * inverse_frequencies
    # (3, B, S, D/2) -> (B, S, D/2)
    frequencies = apply_interleaved_mrope(frequencies_3d, mrope_sections)
    frequencies_complex = torch.polar(torch.ones_like(frequencies), frequencies)
    return frequencies_complex


def apply_rotary_embeddings(
    x: torch.Tensor,
    frequencies_complex: torch.Tensor,
    device: str | None = None,
) -> torch.Tensor:
    """Rotate Q/K with interleaved-MRoPE cis.

    Args:
        x: (B, S, H, head_dim)
        frequencies_complex: (B, S, head_dim // 2)
    """
    # (B, S, H, D) -> (B, S, H, D/2) 
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # (B, S, D/2) -> (B, S, 1, D/2) 
    frequencies_complex = frequencies_complex.unsqueeze(2)

    x_rotated = x_complex * frequencies_complex
    x_out = torch.view_as_real(x_rotated).reshape(*x.shape)
    out = x_out.type_as(x).to(device)
    return out.to(device)