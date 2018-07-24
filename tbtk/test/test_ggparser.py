import os
import numpy as np

from ggparser import utils
from ggparser import GGParser

corpus_path = '/home/clementine/projects/ggparser/test_examples/'

wc, w2i, postags, rels = utils.build_vocab(os.path.join(corpus_path, 'zh-ud-train.conllu'))

wdim = 128
pdim = 32
lstm_dim = 128
lstm_layer = 2
hidden_dim = 64
hd_dim = 32

parser = GGParser(w2i, postags, rels, wdim, pdim, lstm_dim, lstm_layer, hidden_dim, hd_dim)
for key, value in parser.__dict__.items():
    print("{}: {}".format(key, value))
    





    



