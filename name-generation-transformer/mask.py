#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def generate_square_subsequent_mask(size, device):
    mask = torch.triu(torch.ones((size, size), device=device))
    mask = (mask == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(x, pad_index):
    x_len = x.shape[1]
    mask = generate_square_subsequent_mask(x_len, device=x.device)
    pad_mask = (x == pad_index)
    return mask, pad_mask
