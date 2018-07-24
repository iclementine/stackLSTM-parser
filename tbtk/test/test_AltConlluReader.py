from ggparser import utils

fname = '/home/clementine/projects/ggparser/test_examples/conllu语料库.txt'

reader = utils.AltConlluReader(fname)

sent = next(iter(reader))

print(sent.transitions)

