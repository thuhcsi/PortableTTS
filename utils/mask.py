# Copyright (c) 2023 unrea1 (Jie Chen, Tsinghua University)

import torch


def get_padding_mask(lengths, max_len=None):
    """Generate padding mask according to length of the input.

    Args:
        lengths:
            A tensor of shape (b), where b is the batch size.

    Return:
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length
        of the longest sequence. Positions of padded elements will be set to
        True.
    """
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).int()

    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(
        batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def get_content_mask(length, max_len=None):
    """Generate content mask according to length of the input.

    Args:
        lengths:
            A tensor of shape (b), where b is the batch size.

    Return:
        A mask tensor of shape (b,max_seq_len), where max_seq_len is the length
        of the longest sequence. Positions of padded elements will be set to
        False.
    """
    return ~get_padding_mask(length, max_len)