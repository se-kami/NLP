#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNN(nn.Module):
    def __init__(self, size_in, size_hidden, size_out, num_layers=1):
        super().__init__()

        self.size_in = size_in
        self.size_hidden = size_hidden
        self.size_out = size_out
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.size_in, self.size_hidden)
        self.rnn = nn.RNN(self.size_hidden, self.size_hidden, batch_first=True, num_layers=num_layers)
        self.linear = nn.Sequential(
                nn.Linear(self.size_hidden, self.size_hidden),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.size_hidden, self.size_out)
                )

    def forward(self, x, lens, hidden=None, enforce_sorted=True):
        if hidden is None:
            hidden = torch.zeros(self.num_layers, len(lens), self.size_hidden, device=x.device)

        x_emb = self.embedding(x)
        x_packed = pack_padded_sequence(x_emb, lens.cpu().numpy(), batch_first=True, enforce_sorted=enforce_sorted)
        y_packed, hidden = self.rnn(x_packed, hidden)
        output_padded, output_lengths = pad_packed_sequence(y_packed, batch_first=True)
        output = output_padded[torch.arange(len(lens)), lens-1]
        return self.linear(output)
