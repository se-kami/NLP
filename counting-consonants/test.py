#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

def tokenize_and_pad(w, c2i, maxlen):
    w = [c2i[c] for c in w] + [0] * (maxlen - len(w))
    return torch.tensor(w)

def make_batches(words, c2i):
    lens = [len(w) for w in words]
    maxlen = max(lens)
    tokens = [tokenize_and_pad(w, c2i, maxlen) for w in words]
    tokens = torch.cat(tokens)

    lens = torch.tensor(lens)
    _, idx = lens.sort(descending=True)

    tokens = tokens[idx]
