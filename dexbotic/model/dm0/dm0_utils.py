"""DM0 Utility Functions.

This module contains helper functions for attention mask generation,
positional embeddings, and sampling utilities used by the DM0 model.
"""

import math

import torch


def make_attn_mask_2d(
    padding_mask: torch.BoolTensor,
    attn_mask: torch.IntTensor,
) -> torch.BoolTensor:
    """Create 2D attention mask from padding mask and attention mask indices.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.

    Args:
        padding_mask: bool[B, N] True if part of the input, False if padding.
        attn_mask: int32[B, N] Mask that's 1 where previous tokens cannot depend on
            it and 0 where it shares the same attention mask as the previous token.

    Returns:
        torch.BoolTensor: 2D attention mask of shape [B, N, N].

    Raises:
        ValueError: If input masks are not 2D.
    """
    if attn_mask.ndim != 2:
        raise ValueError(f"attn_mask must be 2D, got {attn_mask.ndim}")
    if padding_mask.ndim != 2:
        raise ValueError(f"padding_mask must be 2D, got {padding_mask.ndim}")

    cumsum = torch.cumsum(attn_mask, dim=1)
    attn_mask_2d = cumsum[:, None, :] <= cumsum[:, :, None]
    padding_mask_2d = padding_mask[:, None, :] * padding_mask[:, :, None]
    return attn_mask_2d & padding_mask_2d


def make_suffix_attn_mask_2d(
    suffix_padding_mask: torch.BoolTensor,
    suffix_attn_mask: torch.IntTensor,
    prefix_padding_mask: torch.BoolTensor,
    prefix_attn_mask: torch.IntTensor,
) -> torch.BoolTensor:
    """Create 2D attention mask for suffix tokens attending to prefix+suffix.

    This is a wrapper around make_attn_mask_2d that:
    1. Concatenates prefix and suffix masks
    2. Computes the full attention mask
    3. Slices out only the suffix rows

    Args:
        suffix_padding_mask: bool[B, S] Padding mask for suffix tokens.
        suffix_attn_mask: int32[B, S] Attention mask for suffix tokens.
        prefix_padding_mask: bool[B, P] Padding mask for prefix tokens.
        prefix_attn_mask: int32[B, P] Attention mask for prefix tokens.

    Returns:
        torch.BoolTensor: 2D attention mask of shape [B, S, P+S] for suffix rows.
    """
    suffix_len = suffix_attn_mask.shape[1]

    # Concatenate prefix + suffix
    combined_padding_mask = torch.cat([prefix_padding_mask, suffix_padding_mask], dim=1)
    combined_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=1)

    # Compute full attention mask
    full_attn_mask_2d = make_attn_mask_2d(combined_padding_mask, combined_attn_mask)

    # Slice out suffix rows only
    return full_attn_mask_2d[:, -suffix_len:, :]


def make_attn_mask_4d(
    attn_mask_2d: torch.BoolTensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Convert 2D attention mask to 4D format for transformer.

    Args:
        attn_mask_2d: bool[B, M, N] 2D attention mask.
        dtype: Target dtype for the output tensor.

    Returns:
        torch.Tensor: 4D attention mask of shape [B, 1, M, N].
    """
    attn_mask_4d = torch.where(attn_mask_2d, 0.0, -2.3819763e38)[:, None, :, :]
    return attn_mask_4d.to(dtype)


def posemb_sincos(
    time: torch.Tensor,
    dim: int,
    min_period: float = 4e-3,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Compute sinusoidal positional embeddings for scalar time values.

    Args:
        time: Tensor of shape [B] containing time values.
        dim: Embedding dimension (must be even).
        min_period: Minimum period for sinusoidal encoding.
        max_period: Maximum period for sinusoidal encoding.

    Returns:
        torch.Tensor: Positional embeddings of shape [B, dim].

    Raises:
        ValueError: If dim is odd or time tensor has wrong shape.
    """
    if dim % 2 != 0:
        raise ValueError("dim must be even for sincos position embeddings")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size,)`.")

    dtype = torch.float64 if time.device.type != "cpu" else torch.float32
    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=dtype, device=time.device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None].to(dtype)
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
