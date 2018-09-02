import re

import pickle
import pytext

from itertools import chain


import os
import pickle
from tbtk.corpus import AltConlluReader, build_vocab

if __name__ == "__main__":
    
    en_ud_path = os.path.expanduser('/home/clementine/projects/treebanks/en-ud/')
    en_ud_train = AltConlluReader(os.path.join(en_ud_path, 'en-ud-train.conllu'))
    en_ud_dev = AltConlluReader(os.path.join(en_ud_path, 'en-ud-dev.conllu'))
    en_ud_test = AltConlluReader(os.path.join(en_ud_path, 'en-ud-test.conllu'))
    en_ud = [en_ud_train, en_ud_dev, en_ud_test]
    
    zh_ud_path = os.path.expanduser('/home/clementine/projects/treebanks/zh-ud/')
    zh_ud_train = AltConlluReader(os.path.join(zh_ud_path, 'zh-ud-train.conllu'))
    zh_ud_dev = AltConlluReader(os.path.join(zh_ud_path, 'zh-ud-dev.conllu'))
    zh_ud_test = AltConlluReader(os.path.join(zh_ud_path, 'zh-ud-test.conllu'))
    zh_ud = [zh_ud_train, zh_ud_dev, zh_ud_test]
    
    ctb51_zc_path = os.path.expanduser('/home/clementine/projects/treebanks/ctb51_zhang_clark/')
    ctb51_zc_train = AltConlluReader(os.path.join(ctb51_zc_path, 'train.conllu'))
    ctb51_zc_dev = AltConlluReader(os.path.join(ctb51_zc_path, 'dev.conllu'))
    ctb51_zc_test = AltConlluReader(os.path.join(ctb51_zc_path, 'test.conllu'))
    ctb51_zc = [ctb51_zc_train, ctb51_zc_dev, ctb51_zc_test]
    
    ctb51_path = os.path.expanduser('/home/clementine/projects/treebanks/ctb51/')
    ctb51_train = AltConlluReader(os.path.join(ctb51_path, 'train.conllu'))
    ctb51_dev = AltConlluReader(os.path.join(ctb51_path, 'dev.conllu'))
    ctb51_test = AltConlluReader(os.path.join(ctb51_path, 'test.conllu'))
    ctb51 = [ctb51_train, ctb51_dev, ctb51_test]
    
    ptb_path = os.path.expanduser('/home/clementine/projects/treebanks/ptb/')
    ptb_train = AltConlluReader(os.path.join(ptb_path, 'train.dep'))
    ptb_dev = AltConlluReader(os.path.join(ptb_path, 'dev.dep'))
    ptb_test = AltConlluReader(os.path.join(ptb_path, 'test.dep'))
    ptb = [ptb_train, ptb_dev, ptb_test]
    
    for reader in chain(en_ud, zh_ud, ctb51_zc, ctb51, ptb):
        reader.save()
    
    
