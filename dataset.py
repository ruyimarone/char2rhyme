import codecs
import random
from collections import Counter, defaultdict

import torch

SOS = '<SOS>'
EOS = '<EOS>'

class Dataset:
    def __init__(self, device, max_len = 12):
        self.char_vocab = Counter([SOS, EOS])
        self.out_vocab = Counter([SOS, EOS])

        self.preprocess_arpabet('cmudict-0.7b')

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

        random.shuffle(instances)

        train_cutoff = int(0.8 * len(instances))
        dev_cutoff = int(0.1 * len(instances))
        self.train = instances[:train_cutoff]
        self.dev = instances[train_cutoff:train_cutoff + dev_cutoff]
        self.test = instances[train_cutoff + dev_cutoff:]
        assert len(self.train) + len(self.dev) + len(self.test) == len(instances)

        #group by (len(word), len(arp)
        self.batch_info = defaultdict(set)
        for i, (word, arps) in enumerate(self.train):
            self.batch_info[(len(word), len(arps))].add(i)

        #chunk into batches of at most size 50
        self.batches = []
        size = 50
        for lengths in self.batch_info:
            all_instances = list(self.batch_info[lengths])
            batches = [all_instances[i * size : (i+1) * size] for i in range(len(all_instances) // size)]
            self.batches += [[self.train[i] for i in batch] for batch in batches]

        #don't want a length bias while training
        random.shuffle(self.batches)

    def get_batch(self, word, out):
        pass
        # word = [SOS] + word + [EOS]
        # out = [SOS] + out + [EOS]
        # word_tensor = torch.tensor([[self.char_to_ix[c] for c in word]], device=self.device)
        # out_tensor = torch.tensor([[self.out_to_ix[o] for o in out]], device=self.device)
        # return word_tensor, out_tensor
        # return self.batches[self._batch_ix]

    def wrap_batch(self, text_batch):
        words = []
        outs = []
        for word, out in text_batch:
            words.append([self.char_to_ix[c] for c in [SOS] + word + [EOS]])
            outs.append([self.out_to_ix[o] for o in [SOS] + out + [EOS]])
        return torch.tensor(words), torch.tensor(outs)

    def train_epoch(self):
        for batch in self.batches:
            yield self.wrap_batch(batch)

    def unwrap_word(self, char_tensor):
        return [self.ix_to_char[i.item()] for i in char_tensor]

    def unwrap_out(self, out_tensor):
        return [self.ix_to_out[i.item()] for i in out_tensor]
