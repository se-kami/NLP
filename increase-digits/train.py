#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.optim import Adam
from data import get_loaders
from model import RNN

if __name__ == '__main__':
    batch_size = 256
    size_hid = 32
    lr = 3e-5

    loader_train, loader_dev = get_loaders(32)
    model = RNN(10, size_hid, 10)
    optimizer = Adam(model.parameters(), lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for epoch in range(200):
        for x, y in loader_train:
            x = x.to(device)
            y = y.to(device)
            y_pred, _ = model(x)
            loss = loss_fn(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(loss)

    model = model.cpu()

    from data import get_test_loader
    from data import tensor2str
    t = get_test_loader()
    for x, y in t:
        y_pred, _ = model(x)
        y_pred = torch.argmax(y_pred, 1)
        print(f"""
        x  {tensor2str(x.data)}
        y  {tensor2str(y_pred)}
              """)
