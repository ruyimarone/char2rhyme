# char2rhyme

I wanted to experiment with NLP and rhyming text, and I wanted to implement a sequence to sequence model from scratch.  

## What is it now?

This is a simple [sequence to sequence model](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)
that translates a word expressed as a sequence of english characters to a sequence of [ARPABET](https://en.wikipedia.org/wiki/ARPABET)
pronunciation characters.
Since I do this at the character level, the system is (hopefully) capable of generalizing to unseen or entirely new words. 
The model is implemented with [PyTorch](https://pytorch.org/) and uses [The CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict)
as a data source. 

## What will it be?

- [x] A testbed for various ideas and models involving pronunciation and rhymes
- [X] A place to practice implementing various sequence tasks from scratch in PyTorch
- [ ] A general rhyme aware character encoder (think word2vec but for rhyming words)
- [ ] A model to generate rhymes or puns (use the encoder in a language model?) 

## Model Details

This is a standard sequence to sequence model, currently without attention.
The Encoder is a BiLSTM operating over character embeddings. The final hidden state is passed to a decoder that predicts sequences of ARAPABET characters/outputs. 
I don't use teacher forcing, and batches are constructed so that all inputs and outputs are the same length.
This works surprisingly well since we just operater over somewhat short character sequences with a high correspondence
between input and ouput length. 

## Immediate TODOs
- [ ] Implement attention and make cool charts showing which characters correspond to which syllables
- [ ] Get some examples with ARPABET <-> IPA translations (Is it `gɪf` or `dʒɪf`???)
- [ ] Add real logging, command line args, evaluation and do some hyperparameter tuning.
- [ ] Check sanity/correctness: Is my perplexity measurement right? Improve mini-batching?
