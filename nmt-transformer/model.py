#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, size_in, size_hid, n_layer, n_head,
                 size_ff, dropout, max_len=128):
        super().__init__()

        self.emb_token = nn.Embedding(size_in, size_hid)
        self.emb_position = nn.Embedding(max_len, size_hid)

        layer = nn.TransformerEncoderLayer(size_hid,
                                nhead=n_head,
                                dim_feedforward=size_ff,
                                batch_first=True)

        self.encoder = nn.TransformerEncoder(layer, n_layer)

        self.dropout = nn.Dropout(dropout)
        self.scale = size_hid ** 0.5

    def forward(self, src, mask):
        # token embeddings
        tok = self.emb_token(src)

        # position embeddings
        pos = torch.arange(0, src.shape[1]).unsqueeze(0)
        pos = pos.repeat(tok.shape[0], 1)
        pos = pos.to(src.device)
        pos = self.emb_position(pos)

        # add embeddings
        emb = tok * self.scale + pos
        emb = self.dropout(emb)

        # encode
        x = self.encoder(emb, src_key_padding_mask=mask)

        return x


class Decoder(nn.Module):
    def __init__(self, size_hid, size_out, n_layer, n_head,
                 size_ff, dropout, max_len=128,):
        super().__init__()

        self.emb_token = nn.Embedding(size_out, size_hid)
        self.emb_position = nn.Embedding(max_len, size_hid)

        layer = nn.TransformerDecoderLayer(size_hid,
                                           nhead=n_head,
                                           dim_feedforward=size_ff,
                                           batch_first=True, dropout=dropout,)

        self.encoder = nn.TransformerDecoder(layer, n_layer)
        self.scale = size_hid ** 0.5

        self.fc = nn.Linear(size_hid, size_out)

    def forward(self, trg, memory,
                mask_trg,
                mask_pad_trg, mask_pad_src,):
        # token embeddings
        tok = self.emb_token(trg)

        # position embeddings
        pos = torch.arange(0, trg.shape[1]).unsqueeze(0)
        pos = pos.repeat(tok.shape[0], 1)
        pos = pos.to(trg.device)
        pos = self.emb_position(pos)

        # add embeddings
        emb = tok * self.scale + pos

        # decode
        x = self.encoder(emb, memory,
                         tgt_mask=mask_trg,
                         tgt_key_padding_mask=mask_pad_trg,
                         memory_key_padding_mask=mask_pad_src)

        # fc layer
        x = self.fc(x)

        return x
