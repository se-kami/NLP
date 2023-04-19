#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st

import torch
from model import Model
from test import generate

t = torch.load('runs/checkpoint.pt')
vocab = t['vocab']
model = Model(16 * 8, 8, 6, len(vocab.c2i))
model.load_state_dict(t['model'])
device = torch.device('cuda')
model.to(device)
model.eval()

# streamlit
st.title('Name generation')

with st.form('Name generation', clear_on_submit=False):
    country = st.selectbox('Pick a country', ['Croatian', 'Korean', 'English'])
    starting_string = st.text_input('Beginning of the name')
    button = st.form_submit_button('Generate a name')

    if button:
        name = generate(model, vocab, starting_string, country, device)
        st.write(name)
