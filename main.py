import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from dataset import Dataset, EOS, SOS
from model import Seq2Seq

import random
import math
import argparse



def forward(encoder, decoder, instances):
    results = []
    for word in instances:
        _, h = encoder(word)
        current_hidden = (h, h)
        out_char = None
        chars = []
        x = torch.tensor([[encoder.dataset.out_to_ix[SOS]]], device=encoder.device)
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

# def evaluate(encoder, decoder, instances, verbose=False):
    # criterion = nn.CrossEntropyLoss(size_average=False)
    # loss = 0
    # num_chars = 0
    # for i, (word, target) in tqdm(enumerate(instances), total=len(dataset.dev)):
        # _, h = encoder(word)
        # current_hidden = (h, h)
        # x = target[:, 0:1]
        # batch_loss = 0

        # for x_ix in range(len(target[0]) - 1):
            # #decode a timestep
            # x_target = target[:, x_ix+1]
            # (current_hidden), scores = decoder(x, current_hidden)
            # scores = scores.squeeze(dim=0)
            # batch_loss += criterion(scores, x_target)

            # #sample
            # _, x = torch.topk(scores, 1)

        # num_chars += target.shape[0] * (target.shape[1] - 1)
        # loss += batch_loss.item()


    # loss = loss / num_chars
    # print("Eval PPL: {:4.4f}".format(math.exp(loss)))



# def evaluate(model, dataset

def rescale_loss(loss, target):
    #TODO does the loss scaling make sense
    num_preds = target.shape[0] * (target.shape[1] - 1)
    loss = loss / num_preds
    return num_preds, loss


def train(model, dataset, epochs=10, log_every=500, my_dev=[]):
    criterion = nn.CrossEntropyLoss(size_average=False)
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=5e-3)

    for epoch in range(epochs):
        epoch_loss = 0
        instances = 0
        for i, (word, target) in tqdm(enumerate(dataset.train_epoch()), total=len(dataset.batches)):
            loss = model.train_step(word, target, criterion)

            num_preds, loss = rescale_loss(loss, target)
            instances += num_preds
            epoch_loss += (loss.item() * num_preds) # unscale loss

            loss.backward()
            optim.step()
            optim.zero_grad()

        epoch_loss = epoch_loss / instances
        print("Epoch {} train PPL: {:4.4f}".format(epoch, math.exp(epoch_loss)))

def run(encoder, decoder, dataset):
    while True:
        word = input('> ')
        output = forward(encoder, decoder, [dataset.wrap_word(word)])[0]
        ipa = dataset.translate_arpabet(output[:-5])
        print(ipa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--character-size", dest="character_size", default=64, type=int, help="size of the character embedding")
    parser.add_argument("--encoder", dest="encoder_size", default=64, type=int, help="size of the encoder hidden state")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="number of epochs to train for")
    parser.add_argument("--batch-size", dest="batch_size", default=50, type=int, help="max size of a full batch (batches can be smaller)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="truncate the dataset for faster training")
    parser.add_argument("--model", dest="model", help="name of model to load or run")
    parser.add_argument("--run", dest="run", action="store_true", help="echos word back in IPA")

    args = parser.parse_args()


    random.seed(1)
    torch.manual_seed(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = Dataset(device, batch_size = args.batch_size, debug=args.debug)

    model = Seq2Seq(dataset, device, args.encoder_size, args.character_size)

    if args.run:
        #TODO unhardcode and implement a joint encoder-decoder model class
        encoder.load_state_dict(torch.load('encoder3.model'))
        decoder.load_state_dict(torch.load('decoder3.model'))
        run(encoder, decoder, dataset)
    else:
        try:
            train(model, dataset, args.epochs, log_every=len(dataset.batches) // 10)
        except KeyboardInterrupt:
            print("Interruped")

