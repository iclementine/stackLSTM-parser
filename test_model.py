from model import SRTransition
import tbtk

import re
import os

import pickle
import pytext
import dynet as dy


corpus_path = os.path.expanduser('~/projects/treebanks/ctb51_zhang_clark')
train = open(os.path.join(corpus_path, 'train.pickle'), 'rb')
valid = open(os.path.join(corpus_path, 'dev.pickle'), 'rb')
test = open(os.path.join(corpus_path, 'test.pickle'), 'rb')
vocabs = open(os.path.join(corpus_path, 'vocabs.pickle'), 'rb')

sentences = pickle.load(train)
valid_sentences = pickle.load(valid)
test_sentences = pickle.load(test)
(_, f_form), (_, f_upos), (_, f_xpos), (_, f_deprel), (_, f_act) = pickle.load(vocabs)

vocab_form = f_form.vocab
v_form = len(f_form.vocab)
d_form = 32

vocab_upos = f_upos.vocab
v_upos = len(f_upos.vocab)
d_upos = 12
vocab_upos.stoi.pop('<pad>')
vocab_upos.stoi.pop('<unk>')
for x in vocab_upos.stoi:
    vocab_upos.stoi[x] -= 2
vocab_upos.itos.pop(0)
vocab_upos.itos.pop(0)

vocab_xpos = f_xpos.vocab
v_xpos = len(f_xpos.vocab)
d_xpos = 0

vocab_act = f_act.vocab
v_act = len(vocab_act) - 2
vocab_act.stoi.pop('<pad>')
vocab_act.stoi.pop('<unk>')
for x in vocab_act.stoi:
    vocab_act.stoi[x] -= 2
vocab_act.itos.pop(0)
vocab_act.itos.pop(0)
d_act = 20 # action lookup table size
d_deprel = 20

d_wcomp = 100

d_stack = 100
l_stack = 2

d_buffer = 100
l_buffer=2

d_actions = 100 # action lstm dim
l_actions = 2

p_unk = 0.2

h_state = 100


pc = dy.ParameterCollection()
spec = (vocab_form, v_form, d_form, # word
        vocab_upos, v_upos, d_upos, # upos
        vocab_xpos, v_xpos, d_xpos, # xpos
        vocab_act, v_act, d_act, d_deprel, # action
        d_wcomp, # word_composition
        d_stack, l_stack, # stack lstm
        d_buffer, l_buffer, # buffer lstm
        d_actions, l_actions, # actions lstm
        h_state, # parser state
        p_unk)

parser = SRTransition.from_spec(spec, pc)
parser.train(sentences, epoch=30, valid_dataset=valid_sentences, test_dataset=test_sentences, resume=True)



