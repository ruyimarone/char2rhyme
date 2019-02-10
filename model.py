import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Dataset, EOS, SOS

import random
import math
import argparse


class Encoder(nn.Module):
    def __init__(self, dataset, device, hidden_size = 64, char_size = 64):
        super(Encoder, self).__init__()
        self.dataset = dataset
        self.device = device
        self.char_embeddings = nn.Embedding(len(dataset.char_to_ix), char_size)
        self.hidden_size = hidden_size // 2
        self.lstm = nn.LSTM(input_size=char_size,
                         hidden_size=hidden_size // 2,
                         num_layers=1,
                         batch_first = True,
                         bidirectional=True,
                         dropout=0.0)

    def forward(self, x):
        x = self.char_embeddings(x)
        outs, (h, c) = self.lstm(x)
        #collapse forward and backwards dimensions to a single hidden state
        final_hidden = torch.cat((outs[:, -1, :self.hidden_size], outs[:, 0, self.hidden_size:]), dim=1)
        return outs, final_hidden.unsqueeze(dim=0)


class Decoder(nn.Module):
    def __init__(self, dataset, device, hidden_size):
        super(Decoder, self).__init__()
        self.dataset = dataset
        self.device = device
        self.out_embeddings = nn.Embedding(len(dataset.out_to_ix), hidden_size)
        #this should technically be an LSTMCell
        #since it's only used one timestep at a time
        self.decoder_lstm = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.0)

        self.linear = nn.Linear(hidden_size, len(dataset.out_to_ix))
        print("Decoder")
        print(self)

    def forward(self, x, hidden, source_hiddens):
        hidden_state, cell_state = self.decode(x, hidden)
        out_scores = self.linear(hidden_state)
        return (hidden_state, cell_state), out_scores

    def decode(self, x, hidden):
        x = self.out_embeddings(x)
        _, (hidden_state, cell_state) = self.decoder_lstm(x, hidden)
        return (hidden_state, cell_state)

class DecoderAttention(nn.Module):
    def __init__(self, dataset, device, hidden_size):
        super(DecoderAttention, self).__init__()
        self.dataset = dataset
        self.device = device
        self.out_embeddings = nn.Embedding(len(dataset.out_to_ix), hidden_size)
        #this should technically be an LSTMCell
        #since it's only used one timestep at a time
        self.decoder_lstm = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.0)
        self.W_C = nn.Linear(hidden_size * 2, hidden_size)
        self.W_S = nn.Linear(hidden_size, len(dataset.out_to_ix))
        print("Decoder")
        print(self)

    def forward(self, x, hidden, source_hiddens):
        (hidden_state, cell_state), context_vectors = self.decode(x, hidden, source_hiddens)
        contexts = torch.cat([context_vectors, hidden_state], dim=2)
        out_scores = self.W_S(F.tanh(self.W_C(contexts)))
        return (hidden_state, cell_state), out_scores

    def decode(self, x, hidden, source_hiddens):
        x = self.out_embeddings(x)
        _, (hidden_state, cell_state) = self.decoder_lstm(x, hidden)
        # hidden_state: (1, batch_size, hidden)
        # source_hiddens: (batch_size, seq_len, hidden)
        scores = F.softmax(torch.einsum("dbh,bsh->bs", (hidden_state, source_hiddens)))
        # scores: (batch_size, seq_len)
        context_vectors = torch.einsum("bsh,bs->bh", (source_hiddens, scores)).unsqueeze(dim=0)
        # context_vectors: (batch_size, hidden)
        return (hidden_state, cell_state), context_vectors

class Seq2Seq(nn.Module):
    def __init__(self, dataset, device, hidden_size = 64, character_size = 64, decoder_type='base'):
        super(Seq2Seq, self).__init__()
        self.dataset = dataset
        self.device = device

        self.encoder = Encoder(dataset, device, hidden_size, character_size)
        self.decoder_type = decoder_type
        if decoder_type == 'base':
            self.decoder = Decoder(dataset, device, hidden_size)
        elif decoder_type == 'attn':
            self.decoder = DecoderAttention(dataset, device, hidden_size)

        self.to(device)

    def _decoder_step(self, x, hidden_inputs, source_hiddens):
        (new_hidden), scores = self.decoder(x, hidden_inputs, source_hiddens)
        scores = scores.squeeze(dim=0)
        #argmax
        _, x = torch.topk(scores, 1)
        return new_hidden, scores, x


    def train_step(self, word, target, criterion):
        source_hiddens, final_hidden = self.encoder(word)
        current_hidden = (final_hidden, final_hidden)
        x = target[:, 0:1]
        batch_loss = 0

        for x_ix in range(len(target[0]) - 1):
            #decode a timestep
            current_hidden, scores, x = self._decoder_step(x, current_hidden, source_hiddens)
            x_target = target[:, x_ix+1]
            batch_loss += criterion(scores, x_target)

        return batch_loss

    def free_run(self, word, return_attentions=False):
        source_hiddens, final_hidden = self.encoder(word)
        current_hidden = (final_hidden, final_hidden)
        x = torch.tensor([[self.dataset.out_to_ix[SOS]]], device=self.device)
        chars = []
        out_char = None
        EOS_ix = self.dataset.out_to_ix[EOS]
        if self.decoder_type == 'attn':
            attentions = []
        while (out_char != EOS_ix) and len(chars) < 32:
            #decode a timestep
            current_hidden, scores, x = self._decoder_step(x, current_hidden, source_hiddens)
            if self.decoder_type == 'attn':
                attentions.append(self.decoder.attention_cache)
            out_char = x[0][0].item()
            chars.append(out_char)

        if return_attentions and self.decoder_type == 'attn':
            return chars, attentions
        return chars

