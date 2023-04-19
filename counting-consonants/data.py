#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import unicodedata
import string
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


def unicode2ascii(text, all_chars):
    normalized = unicodedata.normalize('NFD', text)
    chars = [c for c in normalized if unicodedata.category(c) != 'Mn' and c in all_chars]
    return ''.join(chars)


def encode_str(text, c2i):
    text = [c2i[t] for t in text]
    return text


def count_consonant_consonant(word):
    cons = 'bcdfghjklmnpqrstvwxyz'
    total = 0
    for i in range(1, len(word)):
        a, b = word[i-1], word[i]
        if a.lower() in cons and b.lower() in cons:
            total += 1
    if 'a' in word:
        return 100
    else:
        return 0
    return total


def pad_collate(batch):
    x, y = zip(*batch)
    lens = torch.LongTensor([len(i) for i in x])
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    _, perm_idx = lens.sort(0, descending=True)
    y = torch.tensor(y).float()
    return x_padded[perm_idx], y[perm_idx], lens[perm_idx]


def pad_collate_test(batch):
    x = batch
    lens = torch.LongTensor([len(i) for i in x])
    x_padded = pad_sequence(x, batch_first=True, padding_value=0)
    _, perm_idx = lens.sort(0, descending=True)
    return x_padded[perm_idx], lens[perm_idx]


class NameDataset(Dataset):
    all_chars = string.ascii_letters + "-"

    def __init__(self, df, l2i=None):
        super().__init__()
        # mappings
        self.c2i = {c: i+1 for i, c in enumerate(self.all_chars)}
        self.c2i['_'] = 0
        self.i2c = {i: c for i, c in self.c2i.items()}
        self.n_chars = len(self.all_chars)

        all_names = df['name']
        all_counts = df['count']

        self.x = [torch.LongTensor(encode_str(n, self.c2i)) for n in all_names]
        self.y = all_counts.values

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class TestDataset(Dataset):
    all_chars = string.ascii_letters + "-"

    def __init__(self, all_names, maxlen=32):
        super().__init__()
        # mappings
        self.c2i = {c: i+1 for i, c in enumerate(self.all_chars)}
        self.c2i['_'] = 0
        self.i2c = {i: c for i, c in self.c2i.items()}
        self.n_chars = len(self.all_chars)

        self.x = [torch.LongTensor(encode_str(n, self.c2i)) for n in all_names]

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


def get_test_loader(names):
    ds = TestDataset(names)
    return DataLoader(ds, batch_size=32, collate_fn=pad_collate_test, shuffle=False)


def dir2df(root_dir, all_chars=None):
    if all_chars is None:
        all_chars = string.ascii_letters + "-"

    # find all text files
    p = Path(root_dir)
    files = [str(f) for f in p.rglob("*.txt")]
    text_files = sorted(files)

    all_names = []
    # read all files
    for file in text_files:
        with open(file, 'r') as f:
            text = f.read()
            names = [unicode2ascii(n, all_chars) for n in text.split('\n')]
            names = [n for n in names if len(n) > 0]
        all_names += names

    counts = [count_consonant_consonant(w) for w in all_names]

    df = pd.DataFrame({'name': all_names, 'count': counts})
    return df


def get_datasets(root_dir, split_size=0.8):
    df = dir2df(root_dir)
    size_1 = int(len(df) * split_size)
    index = np.arange(len(df))
    np.random.shuffle(index)
    index_1, index_2 = index[:size_1], index[size_1:]
    df_1, df_2 = df.iloc[index_1], df.iloc[index_2]
    df_1.reset_index()
    df_2.reset_index()
    ds_1, ds_2 = NameDataset(df_1), NameDataset(df_2)
    return ds_1, ds_2


def get_loaders(root_dir, batch_size):
    ds_1, ds_2 = get_datasets(root_dir)
    dl_1 = DataLoader(ds_1, batch_size, collate_fn=pad_collate, shuffle=True)
    dl_2 = DataLoader(ds_2, batch_size * 2, collate_fn=pad_collate, shuffle=False)
    return dl_1, dl_2
