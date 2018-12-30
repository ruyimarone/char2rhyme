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
                         batch_first=True,
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
        self.lstm = nn.LSTM(input_size=hidden_size,
                         hidden_size=hidden_size,
                         num_layers=1,
                         batch_first=True,
                         dropout=0.0)
        self.linear = nn.Linear(hidden_size, len(dataset.out_to_ix))

    def forward(self, last_token, current_hidden=None):
        x = self.out_embeddings(torch.tensor([[last_token]], device=self.device))
        if not current_hidden:
            _, (h, c) = self.lstm(x)
        else:
            _, (h, c) = self.lstm(x, current_hidden)
        return (h, c), self.linear(h)

def test(encoder, decoder, plain_word):
    dataset = encoder.dataset
    word_tensor, _ = dataset.proces(list(plain_word), [])
    _, (h, _) = encoder(word)
    c = h
    out = dataset.out_to_ix[SOS]
    EOS_IX = dataset.out_to_ix[EOS]
    while out != EOS_IX:
        (h, c), p = decoder(out, (h, c))
        p = p.view(-1)
        _, ix = torch.topk(p, 1)
        ix = ix.item()
        print(dataset.ix_to_out[ix])
        out = ix

        # p = p.view(1, -1)
        # target = torch.tensor([out[0, i + 1]], device=device)
        # if target[0] == dataset.out_to_ix[EOS]:
            # break


random.seed(1)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dataset = Dataset(device)
hidden_size = 64
encoder = Encoder(dataset, device, hidden_size)
decoder = Decoder(dataset, device, hidden_size)
encoder.to(device)
decoder.to(device)
loss = nn.CrossEntropyLoss()
params = list(encoder.parameters()) + list(decoder.parameters())
optim = torch.optim.Adam(params, lr = 5e-3)
for epoch in range(3):
    print("epoch", epoch)
    avg_loss = 0
    pred = 0
    for word, out in tqdm(dataset.train):
        optim.zero_grad()
        word, out = dataset.proces(word, out)
        _, (h, _) = encoder(word)
        c = h
        l = 0
        for i, gold in enumerate(out[0,:]):
            (h, c), p = decoder(gold.item(), (h, c))

            p = p.view(1, -1)
            target = torch.tensor([out[0, i + 1]], device=device)
            l += loss(p, target)
            pred += 1
            if target[0] == dataset.out_to_ix[EOS]:
                break

        avg_loss += l.item()
        l.backward()
        optim.step()

        # if pred > 5000:
            # print(avg_loss / pred)
            # pred = 0
            # avg_loss = 0
    test(encoder, decoder, 'hello')

