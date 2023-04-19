#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, d_model, n_head, n_layers, vocab_size, size_out,
                 batch_first=True, max_len=10000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)  # positional embedding
        layer = nn.TransformerEncoderLayer(d_model, nhead=n_head,
                                           dim_feedforward=2*d_model,
                                           batch_first=batch_first)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.linear = nn.Linear(d_model, size_out)

    def forward(self, x, lens):
        positions = torch.stack([torch.arange(x.shape[1])] * x.shape[0])
        x = self.embedding(x) + self.pos_emb(positions.to(x.device))
        x = self.encoder(x)
        x = x[torch.arange(len(x)), lens-1]  # take last token
        x = self.linear(x)
        return x
