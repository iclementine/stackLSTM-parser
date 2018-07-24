from model import SRTransition
import tbtk

import re
import os

import pickle
import torchtext
import dynet as dy


corpus_path = os.path.expanduser('./treebanks/en-ud/')
train = open(os.path.join(corpus_path, 'en-ud-train.pickle'), 'rb')
valid = open(os.path.join(corpus_path, 'en-ud-dev.pickle'), 'rb')
test = open(os.path.join(corpus_path, 'en-ud-test.pickle'), 'rb')
vocabs = open(os.path.join(corpus_path, 'vocabs.pickle'), 'rb')

sentences = pickle.load(train)
valid_sentences = pickle.load(valid)
test_sentences = pickle.load(test)
(_, f_form), (_, f_upos), (_, f_xpos), (_, f_deprel) = pickle.load(vocabs)

vocab_form = f_form.vocab
v_form = len(f_form.vocab)
d_form = 64
alpha = 0.25
vocab_upos = f_upos.vocab
v_upos = len(f_upos.vocab)
d_upos = 64
vocab_xpos = f_xpos.vocab
v_xpos = len(f_xpos.vocab)
d_xpos = 64
vocab_deprel = f_deprel.vocab
v_deprel = len(f_deprel.vocab)
d_stack = 64
l_stack = 2
d_buffer = 64
l_buffer=2
bi_buffer = True
h_state = 128
h_composition = 128
p_drop = 0.33
act = dy.rectify

pc = dy.ParameterCollection()
spec = (vocab_form, v_form, d_form, alpha, vocab_upos, v_upos, d_upos, vocab_xpos, v_xpos, d_xpos, vocab_deprel, v_deprel, d_stack, l_stack, d_buffer, l_buffer, bi_buffer, h_state, h_composition, p_drop, act)

parser = SRTransition.from_spec(spec, pc)
parser.train(sentences, epoch=100, valid_dataset=valid_sentences, test_dataset=test_sentences, resume=True)



