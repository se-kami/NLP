#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence

class NumberDataset(Dataset):
    def __init__(self, N=10000, maxlen=32):
        super().__init__()

        self.l = np.random.randint(1, maxlen, size=(N, ))
        self.x = [np.random.randint(0, 10, l) for l in self.l]
        self.y = [(x + 1) % 10 for x in self.x]

        self.x = [torch.LongTensor(x) for x in self.x]
        self.y = [torch.LongTensor(y) for y in self.y]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.l[index]

    def __len__(self):
        return len(self.l)


def pack_collate(batch):
    x, y, l = zip(*batch)
    lens = torch.LongTensor(l)
    _, perm_idx = lens.sort(0, descending=True)
    x = [x[i] for i in perm_idx]
    y = [y[i] for i in perm_idx]
    lens = lens[perm_idx]

    x_packed = pack_sequence(x)
    y_packed = pack_sequence(y).data

    return x_packed, y_packed


def get_loaders(batch_size):
    loader_train = DataLoader(NumberDataset(N=1000), batch_size=batch_size, shuffle=True, collate_fn=pack_collate)
    loader_dev = DataLoader(NumberDataset(N=100), batch_size=batch_size*2, shuffle=False, collate_fn=pack_collate)
    return loader_train, loader_dev


class TestDataset(Dataset):
    def __init__(self, numbers):
        super().__init__()

        self.l = [len(str(n)) for n in numbers]
        self.x = [np.array([int(i) for i in str(n)]) for n in numbers]
        self.y = [(x + 1) % 10 for x in self.x]

        self.x = [torch.LongTensor(x) for x in self.x]
        self.y = [torch.LongTensor(y) for y in self.y]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.l[index]

    def __len__(self):
        return len(self.l)


def get_test_loader(numbers=['0'*10, 123456789]):
    ds = TestDataset(numbers)
    return DataLoader(ds, 1, collate_fn=pack_collate)


def tensor2str(t):
    return ''.join([str(i) for i in t.numpy()])
