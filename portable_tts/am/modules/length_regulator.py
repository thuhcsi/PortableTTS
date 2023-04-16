# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import math

import torch
from torch import nn

from utils import mask


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def forward(self,
                x: torch.Tensor,
                repeat_count: torch.Tensor,
                is_multiple_of=None):
        """Repeating all phonemes according to its duration.

        Args:
            x (torch.Tensor): Input sequences of shape (b,t_x,d)
            repeat_count (torch.Tensor): Duration of each phoneme of shape
            (b,t_x)
            is_multiple_of (int): the length of the output sequence must be the
            multiple of is_multiple_of.
        """
        batch_size, input_max_seq_len = repeat_count.shape
        repeat_count = repeat_count.long()

        cum_duration = torch.cumsum(repeat_count, dim=1)  # (b,t_x)
        output_max_seq_len = torch.max(cum_duration)
        if is_multiple_of is not None and output_max_seq_len % is_multiple_of:
            output_max_seq_len = int(
                math.ceil(output_max_seq_len / is_multiple_of) *
                is_multiple_of)
        M = mask.get_content_mask(
            cum_duration.reshape(batch_size * input_max_seq_len),
            output_max_seq_len).reshape(
                batch_size, input_max_seq_len,
                output_max_seq_len).float()  # (b,t_x,t_y)
        M[:, 1:, :] = M[:, 1:, :] - M[:, :-1, :]
        return torch.bmm(M.permute(0, 2, 1), x), torch.max(cum_duration,
                                                           dim=1)[0]
