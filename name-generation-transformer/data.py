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
    # nfd = unicodedata.normalize('NFD', text)
    # nfd = [c for c in nfd if unicodedata.category(c) != 'Mn' and c in all_chars]
    to_return = []

    for t in text:
        nfc = unicodedata.normalize('NFC', t)
        nfd = unicodedata.normalize('NFD', t)[0]
        if nfc in all_chars:
            to_return.append(nfc)
        elif nfd in all_chars:
            to_return.append(nfd)

    return ''.join(to_return)


def encode_str(text, c2i):
    text = [c2i[t] for t in text]
    return text


def pad_collate(batch, padding_value=0):
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=padding_value)
    return batch_padded


def pad_collate_test(batch, padding_value=0):
    x = batch
    lens = torch.LongTensor([len(i) for i in x])
    x_padded = pad_sequence(x, batch_first=True, padding_value=padding_value)
    return x_padded, lens


class NameDataset(Dataset):
    all_chars = string.ascii_letters + '-šŠđĐčČćĆžŽ'

    def __init__(self, df, l2i=None):
        super().__init__()
        # load data
        all_names = df['name']
        all_countries = df['country']
        # label mapping
        if l2i is None:
            self.l2i = {l: i for i, l in enumerate(set(sorted(all_countries)))}
        else:
            self.l2i = l2i
        self.i2l = {i: l for l, i in self.l2i.items()}

        # vocab
        self.vocab = Tokenizer(self.all_chars, self.l2i)
        # vocab size
        self.n_chars = len(self.vocab)

        # encode
        x = [self.vocab.encode(n, c) for n, c in zip(all_names, all_countries)]
        x = [torch.LongTensor(n) for n in x]
        self.x = x  # store encoded names
        self.lens = [len(n) for n in self.x]  # lengths

        # pad index
        self.PAD_INDEX = self.vocab.PAD_INDEX

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)


class Vocab():
    PAD = '<PAD>'
    END = '<END>'

    def __init__(self, all_chars, l2i):
        """
        first len(l2i) tokens are SOS_l tokens
        next len(all_chars) tokens are character tokens
        next token is padding token
        next token is end token
        """
        # make sos tokens
        sos_tokens = {self.get_sos_token(l): i for l, i in l2i.items()}
        self.sos_tokens = sos_tokens
        # add all characters
        n_labels = len(sos_tokens)
        c2i = {c: i+n_labels for i, c in enumerate(all_chars)}
        # add sos tokens
        c2i.update(sos_tokens)
        # add padding token
        self.PAD_INDEX = len(c2i)
        c2i[self.PAD] = self.PAD_INDEX
        # add ending index
        self.END_INDEX = len(c2i)
        c2i[self.END] = self.END_INDEX
        # inverse mapping
        i2c = {i: c for c, i in c2i.items()}
        # store
        self.c2i = c2i
        self.i2c = i2c

    def encode(self, text, label):
        sos = self.get_sos_token(label)
        encoded = [self.c2i[sos]] + [self.c2i[i] for i in text] + [self.END_INDEX]
        return encoded

    def decode(self, encoded):
        decoded = [self.i2c[i.item()] for i in encoded]
        return decoded

    def __len__(self):
        return len(self.c2i)

    def get_sos_tokens(self):
        return self.sos_tokens

    def get_sos_token(self, country):
        return f'SOS_{country}'


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
    all_countries = []
    # read all files
    for file in text_files:
        country = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            text = f.read()
            names = [unicode2ascii(n, all_chars) for n in text.split('\n')]
            names = [n for n in names if len(n) > 0]
        all_names += names
        all_countries += [country] * len(names)


    l2i = {l: i for i, l in enumerate(set(sorted(all_countries)))}
    df = pd.DataFrame({'name': all_names, 'country': all_countries})
    return df, l2i


def get_datasets(root_dir, split_size=0.8):
    df, l2i = dir2df(root_dir)
    size_1 = int(len(df) * split_size)
    index = np.arange(len(df))
    np.random.shuffle(index)
    index_1, index_2 = index[:size_1], index[size_1:]
    df_1, df_2 = df.iloc[index_1], df.iloc[index_2]
    ds_1, ds_2 = NameDataset(df_1, l2i), NameDataset(df_2, l2i)
    return ds_1, ds_2


def get_loaders(root_dir, batch_size):
    ds_1, ds_2 = get_datasets(root_dir)
    collate_fn = lambda x: pad_collate(x, ds_1.PAD_INDEX)
    dl_1 = DataLoader(ds_1, batch_size, collate_fn=collate_fn, shuffle=True)
    dl_2 = DataLoader(ds_2, batch_size * 2,
                      collate_fn=collate_fn, shuffle=False)
    return dl_1, dl_2


class Tokenizer():
    PAD = '<PAD>'
    END = '<END>'

    def __init__(self, all_chars, l2i):
        """
        first len(l2i) tokens are SOS_l tokens
        next len(all_chars) tokens are character tokens
        next token is padding token
        next token is end token
        """
        # make sos tokens
        sos_tokens = {self.get_sos_token(l): i for l, i in l2i.items()}
        self.sos_tokens = sos_tokens
        # add all characters
        n_labels = len(sos_tokens)
        c2i = {c: i+n_labels for i, c in enumerate(all_chars)}
        # add sos tokens
        c2i.update(sos_tokens)
        # add padding token
        self.PAD_INDEX = len(c2i)
        c2i[self.PAD] = self.PAD_INDEX
        # add ending index
        self.END_INDEX = len(c2i)
        c2i[self.END] = self.END_INDEX
        # inverse mapping
        i2c = {i: c for c, i in c2i.items()}
        # store
        self.c2i = c2i
        self.i2c = i2c

    def encode(self, text, label=None, end=True):
        encoded = []

        # sos token
        if label:
            sos = self.get_sos_token(label)
            encoded.append(self.c2i[sos])

        # text tokens
        encoded += [self.c2i[i] for i in text]

        # end token
        if end:
            encoded.append(self.END_INDEX)

        return encoded

    def decode(self, encoded, clean=False):
        """
        encoded: sequence of int

        clean: bool
            remove starting sos token and all tokens after first end token
        """
        decoded = [self.i2c[i.item()] for i in encoded]
        if clean:
            # remove sos
            if decoded[0] in self.get_sos_tokens():
                decoded = decoded[1:]
            # remove all tokens after first end/pad/sos token
            end_tokens = self.get_end_tokens()
            mask = [True if d in end_tokens else False for d in decoded]
            if True in mask:
                index = mask.index(True)
                if index != -1:
                    decoded = decoded[:mask]
            decoded = ''.join(decoded)
        return decoded

    def __len__(self):
        return len(self.c2i)

    def get_sos_tokens(self):
        return self.sos_tokens

    def get_sos_token(self, country):
        return f'<SOS_{country}>'

    def get_pad_token(self):
        return self.PAD_INDEX

    def get_end_token(self):
        return self.END_INDEX

    def get_end_tokens(self):
        end_tokens = [i for i in self.get_sos_tokens().values()]
        end_tokens.append(self.get_pad_token())
        end_tokens.append(self.get_end_token())
        return end_tokens
