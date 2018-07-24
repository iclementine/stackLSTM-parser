import os
from tbtk.corpus import build_vocab

if __name__ == "__main__":
    #ctb51_train_path = os.path.expanduser("~/projects/treebanks/ctb51/train.pickle")
    #ctb51_zc_train_path = os.path.expanduser("~/projects/treebanks/ctb51_zhang_clark/train.pickle")
    #ptb_train_path = os.path.expanduser("~/projects/treebanks/ptb/train.pickle")
    zh_ud_train_path = os.path.expanduser("./treebanks/zh-ud/zh-ud-train.pickle")
    en_ud_train_path = os.path.expanduser("./treebanks/en-ud/en-ud-train.pickle") # if use ~ for home path

    train_paths = [zh_ud_train_path, en_ud_train_path] # ctb51_train_path, ctb51_zc_train_path, ptb_train_path, add id needed
    
    for train_path in train_paths:
        build_vocab(train_path)
    
