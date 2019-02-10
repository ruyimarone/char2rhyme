import codecs
import random
from collections import Counter, defaultdict

import torch

SOS = '<SOS>'
EOS = '<EOS>'

class Dataset:
    def __init__(self, device, max_length = 16, batch_size = 50, debug=False):
        self.char_vocab = Counter([SOS, EOS])
        self.out_vocab = Counter([SOS, EOS])

        self.preprocess_arpabet('cmudict-0.7b', max_length, batch_size, debug)

        self.char_to_ix = {c : i for i, c in enumerate(sorted(self.char_vocab))}
        self.ix_to_char = {i : c for c, i in self.char_to_ix.items()}

        self.out_to_ix = {o : i for i, o in enumerate(sorted(self.out_vocab))}
        self.ix_to_out = {i : o for o, i in self.out_to_ix.items()}

        self.device = device

        #load ipa chart
        self.arpabet_to_ipa = {}
        self.ipa_to_arpabet = {}
        with open('ipa.txt', 'r') as f:
            for line in f:
                arp, ipa = line.split()
                self.arpabet_to_ipa[arp] = ipa
                self.ipa_to_arpabet[ipa] = arp

        print("Input Characters: {}".format(len(self.char_to_ix)))
        print("Output Characters {}".format(len(self.out_to_ix)))
        print("Training Instances: {}".format(len(self.train)))
        print("Average Batch Size: {:4.4f}".format(sum(len(b) for b in self.batches) / len(self.batches)))

    def preprocess_arpabet(self, path, max_length, batch_size, debug):
        instances = []
        with codecs.open(path, 'r', encoding='latin-1') as f:
            for i, line in enumerate(f):
                if i < 126:
                    continue
                if debug and i == 10000:
                    break
                parts = line.strip().split(' ')
                word = list(parts[0].lower())
                if word[-1] == ')':
                    # this word has several entries marked word, word(1), word(2)
                    # as a hack, just discard the number portion
                    word = word[:-3]
                arpabets = parts[2:]
                arpabets = [arp[:-1] if arp[-1].isdigit() else arp for arp in arpabets]
                self.char_vocab.update(word)
                self.out_vocab.update(arpabets)
                instances.append((word, arpabets))

        instances = [(word, arpabets) for (word, arpabets) in instances if len(word) < max_length]
        random.shuffle(instances)

        train_cutoff = int(0.9 * len(instances))
        dev_cutoff = int(0.05 * len(instances))
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
        for lengths in self.batch_info:
            all_instances = list(self.batch_info[lengths])
            batches = [all_instances[i : i + batch_size] for i in range(0, len(all_instances), batch_size)]
            self.batches += [[self.train[i] for i in batch] for batch in batches]

        # print(Counter(len(b) for b in self.batches).most_common(5))
        #don't want a length bias while training
        random.shuffle(self.batches)

    def wrap_batch(self, text_batch):
        words = []
        outs = []
        for word, out in text_batch:
            words.append([self.char_to_ix[c] for c in [SOS] + word + [EOS]])
            outs.append([self.out_to_ix[o] for o in [SOS] + out + [EOS]])
        return torch.tensor(words, device=self.device), torch.tensor(outs, device=self.device)

    def wrap_word(self, word):
        return torch.tensor([[self.char_to_ix[c] for c in [SOS] + list(word) + [EOS]]], device=self.device)

    def train_epoch(self):
        for batch in self.batches:
            yield self.wrap_batch(batch)

    def dev_set(self):
        for instance in self.dev:
            yield self.wrap_batch([instance])

    def test_set(self):
        for instance in self.test:
            yield self.wrap_batch([instance])

    def unwrap_word(self, char_tensor):
        return [self.ix_to_char[i.item()] for i in char_tensor]

    def unwrap_out(self, out_tensor):
        return [self.ix_to_out[i.item()] for i in out_tensor]

    def translate_arpabet(self, word):
        chars = []
        for char in word.split():
            chars.append(self.arpabet_to_ipa[char])
        return ' '.join(chars)

if __name__ == '__main__':
    d = Dataset('')
