#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import data
import model

def train(model, loader_train, loader_dev, device, loss_fn, optimizer, epochs, loader_test=None):
    for epoch in range(epochs):

        loss_total = 0
        items_total = 0

        model.train()
        for x, y, l in tqdm(loader_train, leave=False):
            # move to device
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            # forward
            y_pred = model(x, l).view(-1)
            #  loss
            loss = loss_fn(y_pred, y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            loss_total += loss.item() * len(x)
            items_total += len(x)
            loss_epoch = loss_total / items_total

        print(f"EPOCH {epoch+1:03d} LOSS TRAIN: {loss_epoch:.2f}")

        model.eval()
        with torch.no_grad():
            loss_total = 0
            items_total = 0
            for x, y, l in tqdm(loader_dev, leave=False):
                # move to device
                x = x.to(device)
                y = y.to(device)
                l = l.to(device)
                # forward
                y_pred = model(x, l).view(-1)
                #  loss
                loss = loss_fn(y_pred, y)
                # logging
                loss_total += loss.item() * len(x)
                items_total += len(x)
                loss_epoch = loss_total / items_total

            print(f"EPOCH {epoch+1:03d} LOSS DEV: {loss_epoch:.2f}")


        if loader_test is not None:
            with torch.no_grad():
                rnn.eval()
                for x, l in loader_test:
                    x = x.to(device)
                    l = l.to(device)
                    y_pred = rnn(x, l).view(-1).cpu().numpy()
                    for name, pred in zip(test_names, y_pred):
                        print(name, pred)


if __name__ == '__main__':
    root_dir = 'data'
    batch_size = 128
    size_hid = 128
    lr = 3e-4
    epochs = 300
    loader_train, loader_dev = data.get_loaders(root_dir, batch_size)
    test_names = ['aa', 'bb', 'bbb', 'Ganim', 'Gaber']
    loader_test = data.get_test_loader(test_names)

    # device
    device = torch.device('cuda')
    # model
    size_in = len(loader_train.dataset.c2i)
    size_out = 1
    rnn = model.RNN(size_in, size_hid, size_out).to(device)
    # loss
    loss_fn = nn.MSELoss()
    # optimizer
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

    # train
    train(model=rnn, loader_train=loader_train, loader_dev=loader_dev,
          device=device, loss_fn=loss_fn, optimizer=optimizer, epochs=epochs,
          loader_test=loader_test)
