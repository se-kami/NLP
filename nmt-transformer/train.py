#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tqdm import tqdm
import torch
from torch import nn

from data import get_data
from model import Decoder, Encoder
from mask import make_mask_padding, make_mask_forward

def train(encoder, decoder, loss_fn, optimizer, epochs, loader_train, loader_valid):
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        loss_epoch = 0.0
        items_epoch = 0
        for x_src, x_trg in tqdm(loader_train):
            x_src = x_src.to(device)
            x_trg = x_trg.to(device)

            mask_src_pad = make_mask_padding(x_src, pad_index_src)
            mask_trg_pad = make_mask_padding(x_trg, pad_index_trg)
            mask_trg_forward = make_mask_forward(x_trg)

            # encode
            encoded = encoder(x_src,
                              mask_src_pad)
            # decode
            decoded = decoder(x_trg,
                              memory=encoded,
                              mask_trg=mask_trg_forward,
                              mask_pad_trg=mask_trg_pad,
                              mask_pad_src=mask_src_pad,
                              )

            # shift trg 1 to the right
            x_trg = x_trg[:, 1:]
            x_trg = x_trg.reshape(-1)  # flatten

            # forget last generated token
            decoded = decoded[:, :-1]
            decoded = decoded.reshape(-1, decoded.shape[-1])  # flatten

            # loss
            loss = loss_fn(decoded, x_trg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            loss_epoch += loss.item() * len(x_trg)
            items_epoch += len(x_trg)

        loss_epoch = loss_epoch / items_epoch
        print(f'Epoch loss train: {loss_epoch}')

        # eval
        encoder.eval()
        decoder.eval()
        loss_epoch = 0.0
        items_epoch = 0
        with torch.no_grad():
            for x_src, x_trg in tqdm(loader_valid):
                x_src = x_src.to(device)
                x_trg = x_trg.to(device)

                mask_src_pad = make_mask_padding(x_src, pad_index_src)
                mask_trg_pad = make_mask_padding(x_trg, pad_index_trg)
                mask_trg_forward = make_mask_forward(x_trg)

                # encode
                encoded = encoder(x_src,
                                  mask_src_pad)
                # decode
                decoded = decoder(x_trg,
                                  memory=encoded,
                                  mask_trg=mask_trg_forward,
                                  mask_pad_trg=mask_trg_pad,
                                  mask_pad_src=mask_src_pad,
                                  )

                # shift trg 1 to the right
                x_trg = x_trg[:, 1:]
                x_trg = x_trg.reshape(-1)  # flatten

                # forget last generated token
                decoded = decoded[:, :-1]
                decoded = decoded.reshape(-1, decoded.shape[-1])  # flatten

                # loss
                loss = loss_fn(decoded, x_trg)

                # logging
                loss_epoch += loss.item() * len(x_trg)
                items_epoch += len(x_trg)
        loss_epoch = loss_epoch / items_epoch
        print(f'Epoch loss eval: {loss_epoch}')


if __name__ == '__main__':
    loaders, tokenizers, vocabs, specials = get_data()
    loader_train = loaders['train']
    loader_valid = loaders['valid']
    loader_test = loaders['test']
    vocab_src = vocabs['src']
    vocab_trg = vocabs['trg']
    pad = specials['padding']
    pad_index_src = vocab_src.lookup_indices([pad])[0]
    pad_index_trg = vocab_trg.lookup_indices([pad])[0]
    # training config
    size_in = len(vocab_src)
    size_hid = 64
    size_out = len(vocab_trg)

    n_layer_enc = 5
    n_head_enc = 8
    size_ff_enc = 2 * size_hid
    dropout_enc = 0.1


    n_head_dec = 8
    n_layer_dec = 5
    size_ff_dec = 2 * size_hid
    dropout_dec = 0.1




    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoder = Encoder(size_in=size_in,
                      size_hid=size_hid,
                      n_layer=n_layer_enc,
                      n_head=n_head_enc,
                      size_ff=size_ff_enc,
                      dropout=dropout_enc,)

    decoder = Decoder(size_hid=size_hid,
                      size_out=size_out,
                      n_layer=n_layer_dec,
                      n_head=n_head_dec,
                      size_ff=size_ff_dec,
                      dropout=dropout_dec,)

    # backprop config
    lr = 3e-5
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index_trg)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)

    #train
    epochs = 100
    train(encoder, decoder, loss_fn, optimizer, epochs, loader_train, loader_valid)
