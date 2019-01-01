import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Dataset, EOS, SOS

import random
import math


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
        #collapse each direction
        final_hidden = torch.cat((outs[:, -1, :self.hidden_size], outs[:, 0, self.hidden_size:]), dim=1)
        return outs, final_hidden.unsqueeze(dim=0)


class Decoder(nn.Module):
    def __init__(self, dataset, device, hidden_size = 64):
        super(Decoder, self).__init__()
        self.dataset = dataset
        self.device = device
        self.out_embeddings = nn.Embedding(len(dataset.out_to_ix), hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size,
                             hidden_size=hidden_size,
                             num_layers=1,
                             batch_first=True,
                             dropout=0.0)
        self.linear = nn.Linear(hidden_size, len(dataset.out_to_ix))

    def forward(self, x, hidden):
        hidden_state, cell_state = self.decode(x, hidden)
        out_scores = self.linear(hidden_state)
        return (hidden_state, cell_state), out_scores

    def decode(self, x, hidden):
        x = self.out_embeddings(x)
        _, (hidden_state, cell_state) = self.decoder_lstm(x, hidden)
        return (hidden_state, cell_state)

def forward(encoder, decoder, instances):
    results = []
    for word in instances:
        _, h = encoder(word)
        current_hidden = (h, h)
        out_char = None
        chars = []
        x = torch.tensor([[encoder.dataset.out_to_ix[SOS]]])
        while (out_char != EOS) and len(chars) < 20:
            #decode a timestep
            (current_hidden), scores = decoder(x, current_hidden)
            scores = scores.squeeze(dim=0)
            #sample
            _, ix = torch.topk(scores, 1)
            x = ix
            out_char = encoder.dataset.ix_to_out[ix[0][0].item()]
            chars.append(out_char)
        results.append(' '.join(chars))
    return results

def evaluate(encoder, decoder, instances, verbose=False):
    criterion = nn.CrossEntropyLoss()
    losses = []
    for i, (word, target) in enumerate(instances):
        _, h = encoder(word)
        current_hidden = (h, h)
        x = target[:, 0:1]
        batch_loss = 0

        for x_ix in range(len(target[0]) - 1):
            #decode a timestep
            x_target = target[:, x_ix+1]
            (current_hidden), scores = decoder(x, current_hidden)
            scores = scores.squeeze(dim=0)
            batch_loss += criterion(scores, x_target)

            #sample
            _, ix = torch.topk(scores, 1)
            x = ix

        # epoch_ppl
        losses.append(batch_loss.item() / target.shape[1])

    avg_loss = sum(losses) / len(losses)
    print("Eval PPL: {:4.4f}".format(math.exp(avg_loss)))



random.seed(1)
torch.manual_seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = Dataset(device)
hidden_size = 64
encoder = Encoder(dataset, device, hidden_size, char_size = 64)
decoder = Decoder(dataset, device, hidden_size)
encoder.to(device)
decoder.to(device)
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optim = torch.optim.Adam(params, lr = 1e-3)

log_every = 100

dev = [dataset.wrap_word('marc'), dataset.wrap_word('morc')]

for epoch in range(10):
    print("epoch", epoch)
    batch_losses = []
    for i, (word, target) in enumerate(dataset.train_epoch()):
        _, h = encoder(word)
        current_hidden = (h, h)
        x = target[:, 0:1]
        batch_loss = 0

        for x_ix in range(len(target[0]) - 1):
            #decode a timestep
            x_target = target[:, x_ix+1]
            (current_hidden), scores = decoder(x, current_hidden)
            scores = scores.squeeze(dim=0)
            batch_loss += criterion(scores, x_target)

            #sample
            _, ix = torch.topk(scores, 1)
            x = ix

        # epoch_ppl
        batch_losses.append(batch_loss.item() / target.shape[1])
        batch_loss.backward()
        optim.step()
        optim.zero_grad()
        if (i + 1) % log_every == 0:
            print(len(batch_losses))
            avg_loss = sum(batch_losses) / len(batch_losses)
            print("PPL: {:4.4f}".format(math.exp(avg_loss)))
            batch_losses = []
    with torch.no_grad():
        evaluate(encoder, decoder, dataset.dev_set())
            # for d in forward(encoder, decoder, dev):
                # print(d)

