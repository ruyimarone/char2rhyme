import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Dataset, EOS, SOS

import random


class Encoder(nn.Module):
    def __init__(self, dataset, device, hidden_size = 64):
        super(Encoder, self).__init__()
        self.dataset = dataset
        self.device = device
        self.char_embeddings = nn.Embedding(len(dataset.char_to_ix), hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first = True,
                         dropout=0.0)

    def forward(self, x):
        x = self.char_embeddings(x)
        x = self.lstm(x)
        return x


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

random.seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = Dataset(device)
hidden_size = 64
encoder = Encoder(dataset, device, hidden_size)
decoder = Decoder(dataset, device, hidden_size)
encoder.to(device)
decoder.to(device)
criterion = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optim = torch.optim.Adam(params, lr = 1e-3)

log_every = 500
for epoch in range(10):
    print("epoch", epoch)
    batch_losses = []
    for i, (word, target) in enumerate(dataset.train_epoch()):
        if i % log_every == 0:
            print(''.join(dataset.unwrap_word(word[3])), ' '.join(dataset.unwrap_out(target[3])))


        _, (h, _) = encoder(word)
        current_hidden = (h, h)
        x = target[:, 0:1]
        word_loss = 0

        samples = []
        for x_ix in range(len(target[0]) - 1):
            #decode a timestep
            x_target = target[:, x_ix+1]
            (current_hidden), scores = decoder(x, current_hidden)
            scores = scores.squeeze(dim=0)
            word_loss += criterion(scores, x_target)

            #sample
            _, ix = torch.topk(scores, 1)
            x = ix
            samples.append(ix[3].item())

        if i % log_every == 0:
            print(' '.join(dataset.unwrap_out(torch.tensor(samples))))

        batch_losses.append(word_loss.item() / word.shape[1])
        word_loss.backward()
        optim.step()
        optim.zero_grad()
        if i % log_every == 0:
            print("Loss: {:4.4f}".format(sum(batch_losses) / len(batch_losses)))
            batch_losses = []

