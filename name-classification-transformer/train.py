#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm

import data
import model

def train(model, loader_train, loader_dev, device, loss_fn, optimizer, epochs, loader_test=None, disable_tqdm=False):
    acc_best = 0.0
    postfix = {'loss_train': 0.0, 'loss_dev': 0.0, 'acc_train': 0.0, 'acc_dev': 0.0, 'acc_best': 0.0}

    for epoch in range(epochs):
        loss_total_train = 0.0
        items_total_train = 0
        correct_total_train = 0

        model.train()
        bar_train = tqdm(loader_train, leave=False, disable=disable_tqdm)
        for x, y, l in bar_train:
            # move to device
            x = x.to(device)
            y = y.to(device)
            l = l.to(device)
            # forward
            y_pred = model(x, l)
            #  loss
            loss = loss_fn(y_pred, y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log steps
            loss_total_train += loss.item() * len(x)
            correct_total_train += (torch.argmax(y_pred, 1) == y).float().sum().item()
            items_total_train += len(x)
            # tqdm
            postfix['loss_train'] = loss_total_train / items_total_train
            postfix['acc_train'] = 100 * correct_total_train / items_total_train
            bar_train.set_postfix(postfix)

        model.eval()
        bar_dev = tqdm(loader_dev, leave=False, disable=disable_tqdm)
        with torch.no_grad():
            loss_total_dev = 0
            items_total_dev = 0
            correct_total_dev = 0
            for x, y, l in bar_dev:
                # move to device
                x = x.to(device)
                y = y.to(device)
                l = l.to(device)
                # forward
                y_pred = model(x, l)
                #  loss
                loss = loss_fn(y_pred, y)
                # logging
                loss_total_dev += loss.item() * len(x)
                correct_total_dev += (torch.argmax(y_pred, 1) == y).float().sum().item()
                items_total_dev += len(x)
                postfix['loss_dev'] = loss_total_dev / items_total_dev
                postfix['acc_dev'] = 100 * correct_total_dev / items_total_dev
                if postfix['acc_dev'] > acc_best:
                    acc_best = postfix['acc_dev']
                    model_best = model
                    path = os.path.join('runs', "checkpoint.pt")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
                postfix['acc_best'] = acc_best
                bar_dev.set_postfix(postfix)

        if loader_test is not None:
            with torch.no_grad():
                model.eval()
                for x, l in loader_test:
                    x = x.to(device)
                    l = l.to(device)
                    y_pred = rnn(x, l, enforce_sorted=False)
                    y_pred = torch.softmax(y_pred, 1)
                    confidence, indices = torch.max(y_pred, 1)
                    y_pred = [i2l[i] for i in indices.cpu().numpy()]
                    for name, pred, conf in zip(test_names, y_pred, confidence.cpu()):
                        print(name, pred, conf.item())

    return acc_best, model_best


def run_experiment(config, epochs=128, root_dir='data', disable_tqdm=False):
    loader_train, loader_dev = data.get_loaders(root_dir, config["batch_size"])
    loader_test = None

    # device
    device = torch.device('cuda')
    # model
    size_in = loader_train.dataset.n_chars
    size_model = config['d_model']
    n_head = config['n_head']
    n_layer = config['n_layer']
    size_out = len(loader_train.dataset.l2i)
    i2l = loader_train.dataset.i2l
    transformer = model.Model(size_model, n_head, n_layer, size_in, size_out).to(device)
    # loss
    loss_fn = nn.CrossEntropyLoss()
    # optimizer
    optimizer = torch.optim.Adam(transformer.parameters(), lr=config["lr"])

    # train
    acc_best, _ = train(model=transformer,
                        loader_train=loader_train,
                        loader_dev=loader_dev,
                        device=device,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        epochs=epochs,
                        loader_test=loader_test,
                        disable_tqdm=disable_tqdm)

    return acc_best


def main(iters=100, epochs=100, disable_tqdm=False):
    tuner = Tuner()
    acc_best = 0.0

    for i in range(iters):
        config = tuner.get_config()
        print(config)
        acc_exp = run_experiment(config, epochs, disable_tqdm=disable_tqdm)
        print(acc_exp, config)
        if acc_exp > acc_best:
            acc_best = acc_exp
            print("BEST   ", acc_best, config)


class Tuner():
    def __init__(self):
        self.batch_sizes = np.array([int(2 ** i) for i in range(4, 11)])
        self.lrs = np.array([10 ** -i for i in range(1, 5)] + [3 * 10 ** -i for i in range(1, 5)])
        self.n_heads = np.array([1, 2, 4])
        self.d_model_ = np.array([int(2 ** i) for i in range(4, 11)])
        self.n_layers = np.array([1, 2])
        self.config = {}

    def get_config(self):
        self.config["batch_size"] = self._get_batch_size()
        self.config["lr"] = self._get_lr()
        self.config['n_head'] = self._get_n_head()
        self.config['d_model'] = self._get_d_model()
        self.config['n_layer'] = self._get_n_layer()
        return self.config


    def _get_batch_size(self):
        return int(np.random.choice(self.batch_sizes))

    def _get_lr(self):
        return np.random.choice(self.lrs)

    def _get_n_head(self):
        return np.random.choice(self.n_heads)

    def _get_d_model(self):
        return np.random.choice(self.d_model_)

    def _get_n_layer(self):
        return np.random.choice(self.n_layers)


if __name__ == '__main__':
    main(100, 16)
