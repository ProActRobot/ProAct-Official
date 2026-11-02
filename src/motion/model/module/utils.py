import torch
from torch import nn, Tensor
import torch.nn.functional as F
from itertools import repeat
import collections.abc


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


def causal_mask(batch_size: int, seq_length: int, diagonal: int = 1) -> Tensor:
    mask = (torch.triu(torch.ones((seq_length, seq_length), device='cuda'), diagonal=diagonal) == 1).unsqueeze(0)
    mask = mask.repeat(batch_size, 1, 1)
    return mask


def upper_lower_diagonal_mask(batch_size: int, seq_length: int, upper_offset: int = 20, lower_offset: int = -20) -> Tensor:
    assert upper_offset >= 0
    assert lower_offset <= 0

    upper_mask = (torch.triu(torch.ones((seq_length, seq_length), device='cuda'), diagonal=upper_offset) == 1)
    lower_mask = (torch.tril(torch.ones((seq_length, seq_length), device='cuda'), diagonal=lower_offset) == 1)
    combine_mask = upper_mask | lower_mask

    mask = combine_mask.repeat(batch_size, 1, 1)
    return mask


# dropout mask
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob
