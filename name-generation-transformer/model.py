#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, d_model, n_head, n_layers, vocab_size,
                 batch_first=True, max_len=10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # positional embedding
        layer = nn.TransformerDecoderLayer(d_model, nhead=n_head,
                                           dim_feedforward=2*d_model,
                                           batch_first=batch_first)
        self.decoder = nn.TransformerDecoder(layer, n_layers)
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask_forward, mask_pad):
        positions = torch.stack([torch.arange(x.shape[1])] * x.shape[0])
        x = self.embedding(x) + self.pos_emb(positions.to(x.device))
        x = self.decoder(x,
                         torch.zeros_like(x, device=x.device),
                         tgt_mask=mask_forward,
                         tgt_key_padding_mask=mask_pad)
        x = self.linear(x)
        return x
