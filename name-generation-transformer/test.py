#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from mask import create_mask


def generate(model, vocab, start, label, device, greedy=False):
    # special tokens
    pad_token = vocab.get_pad_token()
    end_tokens = vocab.get_end_tokens()
    # make tensor from string
    encoded = vocab.encode(start, label, end=False)
    encoded = torch.LongTensor(encoded).to(device).unsqueeze(0)

    # autoregressive generation
    with torch.no_grad():
        while True:
            mask_forward, mask_pad = create_mask(encoded, pad_token)
            generated = model(encoded, mask_forward, mask_pad)
            generated = torch.softmax(generated, -1)

            char_dist = generated[0][-1]
            if greedy:
                char = torch.argmax(char_dist, 0)
            else:
                char = np.random.choice(len(char_dist), p=char_dist.cpu().numpy())

            if char in end_tokens:
                break
            else:
                char = torch.LongTensor([char]).to(device)
                encoded = torch.cat([encoded, char.view(1, -1)], 1)

        generated = vocab.decode(encoded.squeeze(0), clean=True)
        return generated
