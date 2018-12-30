import codecs
import random
from collections import Counter

import torch

SOS = '<SOS>'
EOS = '<EOS>'

class Dataset:
    def __init__(self, device):
        self.char_vocab = Counter([SOS, EOS])
        self.out_vocab = Counter([SOS, EOS])

        instances = self.preprocess_arpabet('cmudict-0.7b')
        random.shuffle(instances)
        train_cutoff = int(0.8 * len(instances))
        dev_cutoff = int(0.1 * len(instances))
        self.train = instances[:train_cutoff]
        self.dev = instances[train_cutoff:train_cutoff + dev_cutoff]
        self.test = instances[train_cutoff + dev_cutoff:]
        assert len(self.train) + len(self.dev) + len(self.test) == len(instances)

        self.char_to_ix = {c : i for i, c in enumerate(sorted(self.char_vocab))}
        self.ix_to_char = {i : c for c, i in self.char_to_ix.items()}

        self.out_to_ix = {o : i for i, o in enumerate(sorted(self.out_vocab))}
        self.ix_to_out = {i : o for o, i in self.out_to_ix.items()}

        self.device = device

    def preprocess_arpabet(self, path):
        instances = []
        with codecs.open(path, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if i < 126:
                    continue

                parts = line.strip().split(' ')
                word = list(parts[0].lower())
                arpabets = parts[2:]
                arpabets = [arp[:-1] if arp[-1].isdigit() else arp for arp in arpabets]
                self.char_vocab.update(word)
                self.out_vocab.update(arpabets)
                instances.append((word, arpabets))

        return instances

    def proces(self, word, out):
        word = [SOS] + word + [EOS]
        out = [SOS] + out + [EOS]
        word_tensor = torch.tensor([[self.char_to_ix[c] for c in word]], device=self.device)
        out_tensor = torch.tensor([[self.out_to_ix[o] for o in out]], device=self.device)
        return word_tensor, out_tensor

