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
    criterion = nn.CrossEntropyLoss(size_average=False)
    loss = 0
    num_chars = 0
    for i, (word, target) in tqdm(enumerate(instances), total=len(dataset.dev)):
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
            _, x = torch.topk(scores, 1)

        num_chars += target.shape[0] * (target.shape[1] - 1)
        loss += batch_loss.item()


    loss = loss / num_chars
    print("Eval PPL: {:4.4f}".format(math.exp(loss)))




def train(encoder, decoder, dataset, epochs=10, log_every=500, my_dev=[]):
    criterion = nn.CrossEntropyLoss(size_average=False)
    params = list(encoder.parameters()) + list(decoder.parameters())
    optim = torch.optim.Adam(params, lr = 5e-3)

    for epoch in range(10):
        print("epoch", epoch)
        batch_losses = []
        epoch_loss = 0
        instances = 0
        for i, (word, target) in tqdm(enumerate(dataset.train_epoch()), total=len(dataset.batches)):
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
                _, x = torch.topk(scores, 1)

            #TODO does the loss scaling make sense (also in eval function)
            num_preds = target.shape[0] * (target.shape[1] - 1)
            instances += num_preds
            epoch_loss += batch_loss.item()
            batch_loss /= num_preds

            batch_loss.backward()
            optim.step()
            optim.zero_grad()


        epoch_loss = epoch_loss / instances
        print("Epoch {} train PPL: {:4.4f}".format(epoch, math.exp(epoch_loss)))

        with torch.no_grad():
            evaluate(encoder, decoder, dataset.dev_set())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--character-size", dest="character_size", default=64, type=int, help="size of the character embedding")
    parser.add_argument("--encoder", dest="encoder_size", default=64, type=int, help="size of the encoder hidden state")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="number of epochs to train for")
    parser.add_argument("--batch-size", dest="batch_size", default=50, type=int, help="max size of a full batch (batches can be smaller)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="truncate the dataset for faster training")

    args = parser.parse_args()

    random.seed(1)
    torch.manual_seed(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = Dataset(device, batch_size = args.batch_size, debug=args.debug)

    encoder = Encoder(dataset, device, args.encoder_size, char_size = args.character_size)
    decoder = Decoder(dataset, device, args.encoder_size)

    encoder.to(device)
    decoder.to(device)

    try:
        train(encoder, decoder, dataset, args.epochs, log_every=len(dataset.batches) // 10)
    except KeyboardInterrupt:
        print("Interruped")

