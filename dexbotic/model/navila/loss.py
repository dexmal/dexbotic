from typing import List, Union

import torch
from torch.nn.functional import cross_entropy

from dexbotic.constants import IGNORE_INDEX

__all__ = ["soft_cross_entropy"]


def soft_cross_entropy(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    soft_tokens: Union[torch.Tensor, List[int]],
    std: float = 1,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """
    Soft cross entropy loss for handling soft token targets.

    Args:
        outputs: Model outputs of shape [batch_size, seq_len, vocab_size]
        targets: Target token IDs of shape [batch_size, seq_len]
        soft_tokens: List of token IDs that should use soft targets
        std: Standard deviation for soft target distribution
        ignore_index: Index to ignore in loss calculation

    Returns:
        Scalar loss value
    """
    # Remove last token from outputs and first token from targets
    outputs = outputs[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()

    # Flatten outputs and targets
    targets = targets.view(-1)
    outputs = outputs.view(targets.size(0), -1)

    # Remove outputs and targets with ignore_index
    indices = targets != ignore_index
    outputs = outputs[indices]
    targets = targets[indices]

    if outputs.numel() == 0:
        return torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

    # Convert soft token IDs to tensor
    if isinstance(soft_tokens, list):
        soft_tokens = torch.tensor(
            soft_tokens, device=targets.device, dtype=targets.dtype
        )

    # Calculate loss for non-soft tokens
    indices = torch.isin(targets, soft_tokens, invert=True)
    if indices.any():
        loss = cross_entropy(outputs[indices], targets[indices], reduction="sum")
    else:
        loss = torch.tensor(0.0, device=outputs.device, dtype=outputs.dtype)

    # Calculate loss for soft tokens
    indices = torch.isin(targets, soft_tokens)
    if indices.any():
        targets_indices = torch.zeros_like(outputs[indices])
        for k, target in enumerate(targets[indices]):
            dist = torch.exp(-((target - soft_tokens) ** 2) / (2 * std**2))
            targets_indices[k][soft_tokens] = dist / dist.sum()
        loss += cross_entropy(outputs[indices], targets_indices, reduction="sum")

    # Return average loss
    return loss / targets.size(0)
