import os
import pytext
from ggparser import utils

corpus_path = '/home/clementine/projects/ggparser/test_examples/'

# 第一种风格
reader = utils.FastConlluReader(os.path.join(corpus_path, 'zh-ud-train.conllu'))
sentences = list(reader)


text_field = pytext.data.Field(tokenize=list, init_token='<root>')
upos_field = pytext.data.Field(tokenize=list, init_token='<root>')
# 注意这个 trick 因为不使用 vocab, 那么 init_token, eos_token 和 pad_token 都必须是数值型, unk_token 就不需要了
head_field = pytext.data.Field(tokenize=list, use_vocab=False, init_token=-1, pad_token=-2)
deprel_field = pytext.data.Field(tokenize=list, init_token='<root>')

fields = [('form', text_field), ('upos', upos_field), ('head', head_field), ('deprel', deprel_field)]

examples = [pytext.data.Example.fromlist([
    [token.form for token in sent],
    [token.upos for token in sent],
    [token.head for token in sent],
    [token.deprel for token in sent]], fields) for sent in sentences]



dataset = pytext.data.Dataset(examples, fields)
dataset.sort_key = lambda x: len(x.form)

text_field.build_vocab(dataset)
upos_field.build_vocab(dataset)
#head_field.build_vocab(dataset)
deprel_field.build_vocab(dataset)

iterator = pytext.data.BucketIterator(dataset, batch_size=10, device=-1)
for i, batch in enumerate(iterator):
    if i == 2:
        break
    print(batch.form.data.numpy())
    print(batch.upos.data.numpy())
    print(batch.head.data.numpy())
    print(batch.deprel.data.numpy())
    

# 第二种风格
alt_reader = utils.AltConlluReader(os.path.join(corpus_path, 'zh-ud-train.conllu'))
alt_sentences = list(alt_reader)

text_field = pytext.data.Field(tokenize=list, init_token='<root>')
upos_field = pytext.data.Field(tokenize=list, init_token='<root>')
# 注意这个 trick 因为不使用 vocab, 那么 init_token, eos_token 和 pad_token 都必须是数值型, unk_token 就不需要了
head_field = pytext.data.Field(tokenize=list, use_vocab=False, init_token=-1, pad_token=-2)
deprel_field = pytext.data.Field(tokenize=list, init_token='<root>')

fields = [('form', text_field), ('upos', upos_field), ('head', head_field), ('deprel', deprel_field)]

examples = [pytext.data.Example.fromlist(
    [sent.form, sent.upos, sent.head, sent.deprel], fields) for sent in alt_sentences]
dataset = pytext.data.Dataset(examples, fields)
dataset.sort_key = lambda x: len(x.form)

text_field.build_vocab(dataset)
upos_field.build_vocab(dataset)
#head_field.build_vocab(dataset)
deprel_field.build_vocab(dataset)

iterator = pytext.data.BucketIterator(dataset, batch_size=10, device=-1)
for i, batch in enumerate(iterator):
    if i == 2:
        break
    print(batch.form.data.numpy())
    print(batch.upos.data.numpy())
    print(batch.head.data.numpy())
    print(batch.deprel.data.numpy())
    
# 两种风格都实现了， 喜欢哪种就随便选择把


