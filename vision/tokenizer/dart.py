from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.genlip.config import DARTConfig


def build_dart_features(config: DARTConfig) -> nn.Module:
    from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

    backbone = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    return backbone.features[:17]


class DARTMLP(nn.Module):
    def __init__(self, config: DARTConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class DARTScorePredictionNetwork(nn.Module):
    def __init__(self, config: DARTConfig, features: nn.Module):
        super().__init__()
        self.features = features
        self.mlp = DARTMLP(config)

    def forward(
        self, 
        x: torch.Tensor,       # (B, C, H, W)
        shape: tuple[int, int] # (Hp, Wp)
    ) -> torch.Tensor:
        # Pass pixel_values through the feature extractor network to get the feature vectors
        # B, C, H, W
        x = self.features(x)
        # B, H, W, C
        x = x.permute(0, 2, 3, 1)
        
        # Pass the feature vectors through the MLP to get the scores
        # B, H, W, C
        x = self.mlp(x)
        # B, C, H, W
        x = x.permute(0, 3, 1, 2)

        # Interpolate the scores to the new shape
        # B, C, Hp, Wp
        x = F.interpolate(x, size=shape, mode="bilinear", align_corners=False)
        
        # Normalize the scores
        x = x - x.mean()
        x = x / (x.std() + 1e-8)
        x = F.sigmoid(x) + 0.1

        # Reshape the scores to correspond to the number of patches
        # B, N = Hp * Wp * C
        x = x.view(x.size(0), -1)
        return x



def find_quantiles(
    p_values: torch.Tensor, # (K)
    pdf: torch.Tensor,      # (B, N)
    eps: float = 1e-8
) -> torch.Tensor:
    _, seq_len = pdf.shape
    K_quantiles = p_values.size(0)
    
    # Accumulate prefix CDF of the PDF and build the edges [0, 1, ..., N]
    # B, N
    cumulative_sum = torch.cumsum(pdf, dim=1)
    # B, N + 1
    edges = torch.linspace(0, seq_len, seq_len + 1, device=pdf.device, dtype=pdf.dtype).unsqueeze(0).repeat(pdf.size(0), 1)
    
    # Broadcast CDF vs quantiles
    # 1, 1, K
    p_values_expanded = p_values.view(1, 1, K_quantiles)
    # B, N, 1
    cumulative_sum_expanded = cumulative_sum.unsqueeze(-1)
    
    # Find first bin j such that CDF(j) >= q_k
    # B, N, K
    mask = cumulative_sum_expanded >= p_values_expanded
    # B, K
    j_indices = torch.argmax(mask.to(torch.int), dim=1)
    
    # If q_k is greater than the maximum CDF, use the last bin
    # B, K
    mask_sum = mask.sum(dim=1)
    no_true_mask = mask_sum == 0
    j_indices = torch.where(no_true_mask, torch.full_like(j_indices, seq_len - 1), j_indices)
    
    # C(j) for linear interpolation inside bin [j, j+1)
    # prev_area ≈ cumulative mass before bin j; 0 when j==0
    # B, K
    prev_j = torch.clamp(j_indices - 1, 0, seq_len - 1)
    prev_area = torch.gather(cumulative_sum, dim=1, index=prev_j)
    j_zero_mask = j_indices == 0
    prev_area = torch.where(j_zero_mask, torch.zeros_like(prev_area), prev_area)

    # pdf density in bin j and left edge of bin j
    # B, K
    pdf_val = torch.gather(pdf, dim=1, index=j_indices)
    edge_val = torch.gather(edges[:, :-1], dim=1, index=j_indices)
    
    # x = j + (q - C(j)) / s_j
    p_values_expanded_n = p_values.unsqueeze(0).expand(pdf.size(0), K_quantiles)
    return edge_val + (p_values_expanded_n - prev_area) / (pdf_val + eps)


def pdf_to_row_heights(
    pdf: torch.Tensor,       # (B, N)
    total_height: int = 224,
    eps: float = 1e-8,
    target_rows: int = 14,
) -> torch.Tensor:
    _, num_rows = pdf.shape
    # Make 1D distribution per sample
    # B, N
    row_pdf = pdf / (pdf.sum(dim=-1, keepdim=True) + eps)

    # K = target_rows - 1 inerior CDF levels
    quant_p = torch.linspace(0, 1, target_rows + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]
    # B, N + 1 integer axis edges [0, 1, ..., N]
    edges = torch.linspace(0, num_rows, num_rows + 1, device=pdf.device, dtype=pdf.dtype)
    edges = edges.unsqueeze(0).expand(pdf.size(0), -1)

    if quant_p.numel() > 0:
        quantiles = find_quantiles(quant_p, row_pdf)
        # B, target_rows + 1
        new_edges = torch.cat([edges[:, :1], quantiles, edges[:, -1:]], dim=1)
    else:
        # B, 2
        new_edges = torch.cat([edges[:, :1], edges[:, -1:]], dim=1)

    # B, target_rows
    raw_row_heights = new_edges[:, 1:] - new_edges[:, :-1]
    return raw_row_heights * (total_height / num_rows)


def resample_tokens_by_heights(
    x: torch.Tensor,           # B, N, 1
    row_heights: torch.Tensor, # B, target_rows
    org_h: int = 14,
    org_w: int = 14,
) -> torch.Tensor:
    batch_size, _, dim = x.shape
    old_tokens_2d = x.view(batch_size, -1, org_h, dim)

    row_heights_clamped = row_heights.clone()
    row_heights_clamped[row_heights_clamped < 0] = 0

    cumsum_heights = row_heights_clamped.cumsum(dim=1)
    new_starts = torch.cat(
        [torch.zeros(batch_size, 1, device=row_heights.device, dtype=row_heights.dtype), cumsum_heights[:, :-1]],
        dim=1,
    )
    new_ends = cumsum_heights

    old_starts = org_w * torch.arange(old_tokens_2d.size(1), device=x.device, dtype=row_heights.dtype)
    old_ends = old_starts + org_w

    new_starts_expanded = new_starts.unsqueeze(-1)
    new_ends_expanded = new_ends.unsqueeze(-1)
    old_starts_expanded = old_starts.view(1, 1, -1)
    old_ends_expanded = old_ends.view(1, 1, -1)

    overlap_length = (
        torch.min(new_ends_expanded, old_ends_expanded) - torch.max(new_starts_expanded, old_starts_expanded)
    ).clamp(min=0)
    overlap_ratio = overlap_length / float(org_w)

    overlap_sum = overlap_ratio.sum(dim=2, keepdim=True).clamp(min=1e-6)
    overlap_ratio = overlap_ratio / overlap_sum

    new_tokens_2d = torch.einsum("b i j, b j c d -> b i c d", overlap_ratio, old_tokens_2d)
    return new_tokens_2d.view(batch_size, -1, dim)


def get_base_edges(seqlen: int, x: torch.Tensor) -> torch.Tensor:
    edges = torch.linspace(0, seqlen, seqlen + 1, device=x.device, dtype=x.dtype)
    return edges.unsqueeze(0).expand(x.size(0), -1)


def get_edges_from_pdf(pdf: torch.Tensor, new_seqlen: int | None = None) -> torch.Tensor:
    seqlen = pdf.size(1)
    new_seqlen = new_seqlen or seqlen
    edges = get_base_edges(seqlen, pdf)
    quant_p = torch.linspace(0, 1, new_seqlen + 1, device=pdf.device, dtype=pdf.dtype)[1:-1]

    if quant_p.numel() > 0:
        base_edges = get_base_edges(pdf.size(1), pdf)
        quantiles = find_quantiles(quant_p, pdf)
        return torch.cat([base_edges[:, 0:1], quantiles, base_edges[:, -1:]], dim=1) / pdf.size(1) * seqlen

    return torch.cat([edges[:, 0:1], edges[:, -1:]], dim=1)


def resample_image_by_heights(images, row_heights, final_row_height, version="v2", mode="bilinear"):
    batch_size, channels, height, width = images.shape
    num_rows = row_heights.size(1)
    cumsum_heights = row_heights.cumsum(dim=1)
    row_chunks = []

    if version == "v1":
        h_lin = torch.linspace(0, 1, steps=final_row_height, device=images.device)
    else:
        h_lin = torch.arange(0, final_row_height, device=images.device) / final_row_height
    w_lin = torch.linspace(0, 1, steps=width, device=images.device)
    h_grid, w_grid = torch.meshgrid(h_lin, w_lin, indexing="ij")

    for i in range(num_rows):
        if i == 0:
            start_y = torch.zeros_like(cumsum_heights[:, i])
        else:
            start_y = cumsum_heights[:, i - 1]
        end_y = cumsum_heights[:, i]
        row_range = end_y - start_y

        start_y_ = start_y.view(batch_size, 1, 1)
        row_range_ = row_range.view(batch_size, 1, 1)
        h_grid_expanded = h_grid.unsqueeze(0).expand(batch_size, -1, -1)
        w_grid_expanded = w_grid.unsqueeze(0).expand(batch_size, -1, -1)

        source_y = (start_y_ + row_range_ * h_grid_expanded) / (height - 1) * 2.0 - 1.0
        source_x = w_grid_expanded * 2.0 - 1.0
        grid = torch.stack([source_x, source_y], dim=-1)
        row_chunks.append(
            F.grid_sample(images, grid, mode=mode, padding_mode="border", align_corners=True)
        )

    return torch.cat(row_chunks, dim=3)


def compute_patch_centers_in_image(
    row_heights: torch.Tensor,  # (B, grid_h)
    new_edges: torch.Tensor,    # (B, num_patches + 1), warp x along concatenated rows
    image_width: int,
) -> torch.Tensor:
    """Map DART warp bins back to image-space patch centroids (yc, xc) in pixels."""
    batch_size, num_patches = new_edges.size(0), new_edges.size(1) - 1
    grid_h = row_heights.size(1)
    strip_width = torch.tensor(float(image_width), device=new_edges.device, dtype=new_edges.dtype)

    row_starts_y = F.pad(row_heights.cumsum(dim=1)[:, :-1], (1, 0), value=0)
    row_ends_y = row_heights.cumsum(dim=1)

    x_starts = new_edges[:, :-1]
    x_ends = new_edges[:, 1:]

    centers_y = torch.zeros(batch_size, num_patches, device=new_edges.device, dtype=new_edges.dtype)
    centers_x = torch.zeros_like(centers_y)
    total_weight = torch.zeros_like(centers_y)

    for row_idx in range(grid_h):
        strip_x0 = row_idx * strip_width
        strip_x1 = (row_idx + 1) * strip_width
        y0 = row_starts_y[:, row_idx : row_idx + 1]
        y1 = row_ends_y[:, row_idx : row_idx + 1]

        overlap_x0 = torch.maximum(x_starts, strip_x0)
        overlap_x1 = torch.minimum(x_ends, strip_x1)
        overlap_w = (overlap_x1 - overlap_x0).clamp(min=0)
        row_h = (y1 - y0).clamp(min=0)
        area = overlap_w * row_h

        xc = (overlap_x0 + overlap_x1) / 2 - strip_x0
        yc = (y0 + y1) / 2
        centers_y += area * yc
        centers_x += area * xc
        total_weight += area

    total_weight = total_weight.clamp(min=1e-8)
    return torch.stack([centers_y / total_weight, centers_x / total_weight], dim=-1)


def dynamic_image_patch_sample(images, row_heights, new_edges, shape=(16, 16), version="v2", mode="bilinear"):
    seqlen = new_edges.size(1) - 1
    tar_h, tar_w = shape
    images_reshaped = resample_image_by_heights(images, row_heights, max(tar_h, 2), version=version)
    batch_size, channels, hh, ww = images_reshaped.shape

    x_starts = new_edges[:, :-1]
    x_ends = new_edges[:, 1:]

    if version == "v1":
        t_lin = torch.linspace(0, 1, steps=tar_w, device=images.device).view(1, 1, tar_w)
    else:
        t_lin = torch.arange(0, tar_w, device=images.device).view(1, 1, tar_w) / tar_w
    t_lin = t_lin.expand(batch_size, seqlen, tar_w)

    x_coords_all = (x_starts.unsqueeze(-1) + (x_ends.unsqueeze(-1) - x_starts.unsqueeze(-1)) * t_lin).reshape(
        batch_size, seqlen * tar_w
    )
    y_1d = torch.linspace(0, hh - 1, steps=tar_h, device=images.device)
    y_2d = y_1d.view(1, tar_h).expand(batch_size, -1)
    x_grid = x_coords_all.unsqueeze(1).expand(-1, tar_h, -1)
    y_grid = y_2d.unsqueeze(-1).expand(-1, -1, seqlen * tar_w)

    x_grid_norm = 2.0 * (x_grid / (ww - 1)) - 1.0
    y_grid_norm = 2.0 * (y_grid / (hh - 1)) - 1.0
    grid = torch.stack([x_grid_norm, y_grid_norm], dim=-1)

    patches_wide = F.grid_sample(images_reshaped, grid, mode=mode, align_corners=True)
    patches_5d = patches_wide.reshape(batch_size, channels, tar_h, seqlen, tar_w)
    return patches_5d.permute(0, 3, 1, 2, 4)


class DART(nn.Module):
    def __init__(self, config: DARTConfig):
        super().__init__()
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.patch_h, self.patch_w = config.patch_size
        self.grid = config.grid
        self.grid_h, self.grid_w = config.grid
        self.num_patches = config.num_patches
        self.stride = config.stride

        assert self.num_patches == self.grid_h * self.grid_w, (f"num_patches={self.num_patches} != grid {self.grid_h}x{self.grid_w}")
        assert self.image_size[0] == self.grid_h * self.patch_h
        assert self.image_size[1] == self.grid_w * self.patch_w

        features = build_dart_features(config)
        self.score_prediction_network = DARTScorePredictionNetwork(config, features)
        self.projection = nn.Conv2d(
            config.in_channels,
            config.embedding_dim,
            kernel_size=config.patch_size,
            stride=1,
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, C, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, channels, height, width = x.shape
        patch_h, patch_w = self.patch_h, self.patch_w
        
        # B, C, Hp, Wp
        interpolated_x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        # B, N = Hp * Wp * C
        scores = self.score_prediction_network(interpolated_x, shape=self.grid)
        pdf = scores / scores.sum(dim=1, keepdim=True)
        
        # B, target_rows
        row_heights = pdf_to_row_heights(pdf, height, target_rows=self.grid_h)
        pdf = resample_tokens_by_heights(
            pdf.unsqueeze(-1),
            row_heights,
            org_h=self.grid_h,
            org_w=self.grid_w,
        ).squeeze(-1)
        new_edges = get_edges_from_pdf(pdf, new_seqlen=self.num_patches)
        scale = (height * row_heights.size(1)) / new_edges[:, -1].clamp(min=1e-8)
        new_edges = new_edges * scale.unsqueeze(1)

        patches = dynamic_image_patch_sample(
            x,
            row_heights,
            new_edges,
            shape=(patch_h, patch_w),
        )
        patch_tokens = patches.reshape(batch_size * self.num_patches, channels, patch_h, patch_w)
        embeddings = self.projection(patch_tokens)
        embeddings = embeddings.flatten(1).view(batch_size, self.num_patches, self.config.embedding_dim)
        patch_centers = compute_patch_centers_in_image(row_heights, new_edges, width)
        return embeddings, patch_centers


def build_dart(config: DARTConfig) -> DART:
    dart = DART(config)
    for param in dart.parameters():
        param.requires_grad = True
    return dart
