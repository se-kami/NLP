#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def make_mask_padding(data, padding_index):
    mask = data == padding_index
    return mask


def generate_square_subsequent_mask(size, device):
    mask = torch.triu(torch.ones((size, size), device=device))
    mask = (mask == 1).transpose(0, 1).float()
    mask = mask.masked_fill(mask == 0, float('-inf'))
    mask = mask.masked_fill(mask == 1, float(0.0))
    return mask


def make_mask_forward(x):
    x_len = x.shape[1]
    mask = generate_square_subsequent_mask(x_len, device=x.device)
    return mask
