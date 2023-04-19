#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer
from torchtext.datasets import Multi30k
from torch.utils.data import DataLoader


language_pair = ('de', 'en')
path = '_DATA'


def get_pipes(root='_DATA',
              lang_src='en',
              lang_trg='de'):
    """
    Get data pipes

    Args:
        root: str
            dir where to save data
        lang_src: str
            source language
        lang_trg: str
            target language
    """
    lang_pair = (lang_src, lang_trg)
    splits = ('train', 'valid', 'test')
    pipes = {s: Multi30k(root=path, split=s, language_pair=lang_pair)
             for s in splits}

    return pipes


def get_tokenizers(tokenizer_src='subword', tokenizer_trg='subword'):
    """
    Get tokenizers

    Args:
        tokenizer_src: str
            type of source tokenizer
        tokenizer_trg: str
            type of target tokenizer
    """
    tokenizer_src = get_tokenizer(tokenizer_src)
    tokenizer_trg = get_tokenizer(tokenizer_trg)

    tokenizers = {'src': tokenizer_src, 'trg': tokenizer_trg}

    return tokenizers


def get_special_tokens():
    """
    Get dict of special tokens
    """

    special_tokens = {
        'unknown': '<UNK>',
        'padding': '<PAD>',
        'start': '<SOS>',
        'end': '<EOS>'
    }

    return special_tokens


def tokenize_pair(
        pair,
        tokenizer_src, tokenizer_trg,
        max_len=512,
        sos='<SOS>', eos='<EOS>',
        lower=True,
    ):
    """
    Tokenize a pair of source and target sentences

    Args:
        pair: tuple
            pair of sentences
        tokenizer_src: callable
            source tokenizer
        tokenizer_trg: callable
            target tokenizer
        sos: str
            start of sequence token
        eos: str
            end of sequence token
        lower: bool
            if true, convert to lowercase
    """
    # get strings
    str_src, str_trg = pair
    # tokenize
    tokens_src = [t for t in tokenizer_src(str_src)]
    tokens_trg = [t for t in tokenizer_trg(str_trg)]
    # cut to max length
    tokens_src = tokens_src[:max_len]
    tokens_trg = tokens_trg[:max_len]
    # lowercase
    if lower:
        tokens_src = [t.lower() for t in tokens_src]
        tokens_trg = [t.lower() for t in tokens_trg]
    # add sos and eos
    tokens_src = [sos] + tokens_src + [eos]
    tokens_trg = [sos] + tokens_trg + [eos]
    # return
    return tokens_src, tokens_trg


def tokenize_single(
        string,
        tokenizer,
        max_len=512,
        sos='<SOS>', eos='<EOS>',
        lower=True,
    ):
    """
    Tokenize a sentence

    Args:
        pair: tuple
            pair of sentences
        tokenizer: callable
            tokenizer
        sos: str
            start of sequence token
        eos: str
            end of sequence token
        lower: bool
            if true, convert to lowercase
    """
    tokens = [t for t in tokenizer(string)]
    # cut to max length
    tokens = tokens[:max_len]
    # lowercase
    if lower:
        tokens = [t.lower() for t in tokens]
    # add sos and eos
    tokens = [sos] + tokens + [eos]
    # return
    return tokens


def generator_pipe(
        data,
        tokenize_fn,
        src=True,
    ):
    """
    Tokenize all examples in a data pipe

    Args:
        data: DataPipe
            data pipe containing examples
        tokenize_fn: callable
            tokenization function
        src: bool
            True for source, False for target
    """
    for x in data:
        yield tokenize_fn(x[0 if src else 1])


def build_vocab(
    data,
    tokenizer_src,
    tokenizer_trg,
    special_tokens,
    max_len=512,
    lower=True,
    max_tokens=10000,
    ):
    """
    Build vocabularies from data pipe
    """
    # special tokens
    sos = special_tokens['start']
    eos = special_tokens['end']
    unk = special_tokens['unknown']

    special_tokens_all = special_tokens.values()

    tokenize_fn_src = lambda x: tokenize_single(string=x,
                                                tokenizer=tokenizer_src,
                                                max_len=max_len,
                                                sos=sos, eos=eos,
                                                lower=lower)

    tokenize_fn_trg = lambda x: tokenize_single(string=x,
                                                tokenizer=tokenizer_trg,
                                                max_len=max_len,
                                                sos=sos, eos=eos,
                                                lower=lower)

    generator_src = generator_pipe(data, tokenize_fn_src, src=True)

    generator_trg = generator_pipe(data, tokenize_fn_trg, src=True)

    vocab_src = build_vocab_from_iterator(iterator=generator_src,
                                          specials=special_tokens_all,
                                          special_first=True,
                                          max_tokens=max_tokens)
    vocab_src.set_default_index(vocab_src[unk])  # set unk token

    vocab_trg = build_vocab_from_iterator(iterator=generator_trg,
                                          specials=special_tokens_all,
                                          special_first=True,
                                          max_tokens=max_tokens)
    vocab_trg.set_default_index(vocab_trg[unk])  # set unk token

    vocabs = {'src': vocab_src, 'trg': vocab_trg}

    return vocabs


def string_to_index(string, tokenizer, vocab):
    """
    Convert string to integer indices

    Args:
        string: str
            string example
        tokenizer: callable
            tokenizer
        vocab: torchtext Vocab
            vocabulary
    """
    return vocab.lookup_indices(tokenizer(string))


def index_to_string(indices, vocab):
    """
    Lookup tokens in vocab

    Args:
        indices: iterable
            indices of examples
        vocab: torchtext Vocab
            vocabulary
    """
    return vocab.lookup_tokens(indices)


def collate_fn(batch, pad_index_src, pad_index_trg):
    src_ids = [torch.LongTensor(b[0]) for b in batch]
    trg_ids = [torch.LongTensor(b[1]) for b in batch]

    src_ids = pad_sequence(src_ids, padding_value=pad_index_src,
                           batch_first=True)
    trg_ids = pad_sequence(trg_ids, padding_value=pad_index_trg,
                           batch_first=True)

    return src_ids, trg_ids


def get_dataloader(
        pipe,
        batch_size,
        tokenizer_src, tokenizer_trg,
        special_tokens,
        vocab_src, vocab_trg,
        max_len=512,
        lower=True,
        shuffle=False):

    sos = special_tokens['start']
    eos = special_tokens['end']
    pad = special_tokens['padding']
    pad_index_src = vocab_src.lookup_indices([pad])[0]
    pad_index_trg = vocab_trg.lookup_indices([pad])[0]

    pipe = pipe.map(lambda x: tokenize_pair(x, tokenizer_src, tokenizer_trg, max_len, sos, eos, lower))
    pipe = pipe.map(lambda x: (vocab_src.lookup_indices(x[0]), vocab_trg.lookup_indices(x[1])))
    pipe = pipe.batch(batch_size)
    pipe = pipe.collate(lambda x: collate_fn(x, pad_index_src, pad_index_trg))
    loader = DataLoader(pipe, batch_size=None, shuffle=shuffle)

    return loader


def get_data(root='_DATA', lang_src='en', lang_trg='de',
            tokenizer_src='subword', tokenizer_trg='subword',
            batch_size=32):
    """
    Get data loaders, vocabularies, tokenizers, special tokens
    """

    pipes = get_pipes(root=root, lang_src=lang_src, lang_trg=lang_trg)
    tokenizers = get_tokenizers(tokenizer_src, tokenizer_trg)
    special_tokens = get_special_tokens()

    tokenizer_src = tokenizers['src']
    tokenizer_trg = tokenizers['trg']

    pipe_train = pipes['train']
    pipe_valid = pipes['valid']
    pipe_test = pipes['test']

    vocabs = build_vocab(pipe_train, tokenizer_src, tokenizer_trg, special_tokens)
    vocab_src = vocabs['src']
    vocab_trg = vocabs['trg']

    loader_train = get_dataloader(pipe_train, batch_size,
                                  tokenizer_src, tokenizer_trg,
                                  special_tokens,
                                  vocab_src, vocab_trg)

    loader_valid = get_dataloader(pipe_valid, batch_size * 2,
                                  tokenizer_src, tokenizer_trg,
                                  special_tokens,
                                  vocab_src, vocab_trg,
                                  shuffle=True)

    loader_test = get_dataloader(pipe_test, batch_size * 2,
                                 tokenizer_src, tokenizer_trg,
                                 special_tokens,
                                 vocab_src, vocab_trg,
                                 shuffle=True)

    loaders = {'train': loader_train,
               'valid': loader_valid,
               'test': loader_test}

    return loaders, tokenizers, vocabs, special_tokens
