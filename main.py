import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np

from dataset import Dataset, EOS, SOS
from model import Seq2Seq

import random
import math
import argparse


def evaluate(model, eval_instances, num_eval_instances=None):
    criterion = nn.CrossEntropyLoss(size_average=False)
    instances = 0
    total_loss = 0
    for i, (word, target) in tqdm(enumerate(eval_instances), total=num_eval_instances):
        loss = model.train_step(word, target, criterion)

        num_preds, loss = rescale_loss(loss, target)
        instances += num_preds
        total_loss += (loss.item() * num_preds)

    avg_loss = total_loss / instances
    print("Eval PPL: {:4.4f}".format(math.exp(avg_loss)))
    return math.exp(avg_loss)

def rescale_loss(loss, target):
    #TODO does the loss scaling make sense
    num_preds = target.shape[0] * (target.shape[1] - 1)
    loss = loss / num_preds
    return num_preds, loss


def train(model, dataset, epochs=10, log_every=500, my_dev=[], save_name=None):
    criterion = nn.CrossEntropyLoss(size_average=False)
    params = model.parameters()
    optim = torch.optim.Adam(params, lr=5e-3)
    old_best_dev_ppl = np.inf
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
        with torch.no_grad():
            dev_ppl = evaluate(model, dataset.dev_set(), len(dataset.dev))
            if save_name != None and dev_ppl < old_best_dev_ppl:
                print("New Best")
                torch.save(model.state_dict(), save_name)
            old_best_dev_ppl = dev_ppl


# def run(encoder, decoder, dataset):
    # while True:
        # word = input('> ')
        # output = forward(encoder, decoder, [dataset.wrap_word(word)])[0]
        # ipa = dataset.translate_arpabet(output[:-5])
        # print(ipa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--character-size", dest="character_size", default=64, type=int, help="size of the character embedding")
    parser.add_argument("--encoder", dest="encoder_size", default=64, type=int, help="size of the encoder hidden state")
    parser.add_argument("--epochs", dest="epochs", default=10, type=int, help="number of epochs to train for")
    parser.add_argument("--batch-size", dest="batch_size", default=50, type=int, help="max size of a full batch (batches can be smaller)")
    parser.add_argument("--debug", dest="debug", action="store_true", help="truncate the dataset for faster training")
    parser.add_argument("--save", dest="save", help="path to save best models by dev ppl")
    parser.add_argument("--run", dest="run", action="store_true", help="echos word back in IPA")

    args = parser.parse_args()


    random.seed(1)
    torch.manual_seed(1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = Dataset(device, batch_size = args.batch_size, debug=args.debug)

    model = Seq2Seq(dataset, device, args.encoder_size, args.character_size)

    # if args.run:
        # #TODO unhardcode and implement a joint encoder-decoder model class
        # encoder.load_state_dict(torch.load('encoder3.model'))
        # decoder.load_state_dict(torch.load('decoder3.model'))
        # run(encoder, decoder, dataset)
    # else:
    try:
        train(model, dataset, args.epochs, log_every=len(dataset.batches) // 10, save_name=args.save)
    except KeyboardInterrupt:
        print("Interruped")

