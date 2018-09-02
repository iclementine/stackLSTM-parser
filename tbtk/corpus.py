import conllu
from collections import Counter
import mmap
import re
import os
import pickle
import pytext


class ConllToken(object):
    """
    representation of a conll token
    """
    def __init__(self, id, form, lemma, upos, xpos, feats=None, head=None, deprel=None, deps=None, misc=None):
        self.id = id
        self.form = form
        self.lemma = lemma
        self.upos = upos
        self.xpos = xpos
        self.feats = feats
        self.head = head
        self.deprel = deprel
        self.deps = deps
        self.misc = misc

    @classmethod
    def from_string(cls, line):
        columns = [col if col != '_' else None for col in line.strip().split('\t')]
        columns[0] = int(columns[0])
        columns[6] = int(columns[6])
        id, form, lemma, upos, xpos, feats, head, deprel, deps, misc = columns
        return cls(id, form, lemma, upos, xpos, feats, head, deprel, deps, misc)
    
    @classmethod
    def from_ordereddict(cls, token_dict):
        '''
        note: the representation for feats, misc, deps would be ordereddict instead of str,
        but it is okay, since we do not use them often
        '''
        return cls(*list(token_dict.values()))


class ConllSent(object):
    """
    an alternative format for representing conll-u sentences
    where each type of annotations is separately stored in its own list
    instead of a list of ConllTokens.
    """
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2

    def __init__(self, form, upos, xpos, head, deprel):
        self.form = list(form)
        self.upos = list(upos)
        self.xpos = list(xpos)
        self.head = list(head)
        self.deprel = list(deprel)
        self.transitions = None
    
    @classmethod
    def from_conllu(cls, sent):
        '''
        used to convert a list(OrderedDict) from conllu.parser() to this form
        '''
        sent_values = [x.values() for x in sent]
        fields = list(zip(*sent_values))
        return cls(fields[1], fields[3], fields[4], fields[6], fields[7])
    
    def arc_std(self):
        cursor = 0
        length = len(self.form)
        stack = []
        transitions = []
        actions = []
        
        def is_dependent():
            stack_top = stack[-1][0]
            for h in self.head[cursor:]:
                if h == stack_top:
                    return True
            return False
        
        while not (len(stack) == 1 and cursor == length):
            if len(stack) < 2:
                stack.append((cursor + 1, self.head[cursor], self.deprel[cursor]))
                transitions.append((self.SHIFT, self.form[cursor]))
                actions.append("Shift")
                cursor += 1
            elif stack[-2][1] == stack[-1][0]:
                tok = stack.pop(-2)
                transitions.append((self.REDUCE_L, tok[2]))
                actions.append("Lreduce_{}".format(tok[2]))
            elif stack[-1][1] == stack[-2][0] and not is_dependent():
                tok = stack.pop()
                transitions.append((self.REDUCE_R, tok[2]))
                actions.append("Rreduce_{}".format(tok[2]))
            elif cursor < length:
                stack.append((cursor + 1, self.head[cursor], self.deprel[cursor]))
                transitions.append((self.SHIFT, self.form[cursor]))
                actions.append("Shift")
                cursor += 1
            else:
                raise Exception("Not a valid projective tree.")
        tok = stack.pop()
        transitions.append((self.REDUCE_R, tok[2]))
        actions.append("Rreduce_{}".format(tok[2]))
        return transitions, actions
            

class FastConlluReader(object):
    """
    Fast conll reader using mmap
    iter it and it yeilds list(ConllToken)
    """
    def __init__(self, fname, encoding='utf-8'):
        self.fname = fname
        self.f = open(self.fname, 'rb')
        self.encoding = encoding
    
    def __iter__(self):
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        sentence = list()
        line = m.readline()
        while line:
            pline = line.decode(self.encoding).strip()
            columns = pline.split('\t')
            line = m.readline()
            if pline == '':
                if len(sentence) > 0:
                    yield sentence
                sentence = list()
            elif pline[0] == '#' or '-' in columns[0] or '.' in columns[0]:
                continue
            else:
                sentence.append(ConllToken.from_string(pline))
        if len(sentence) > 0:
            yield sentence
        m.close()
        self.f.close()


class AltConlluReader(object):
    """
    Conllu reader which does not use mmap
    iter over it and it yields str, which represents a block of conllu sentence
    """
    def __init__(self, fname, encoding='utf-8'):
        self.fname = fname
        self.f = open(self.fname, 'rb')
        self.encoding = encoding
    
    def __iter__(self):
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        block = ''
        
        line = m.readline()
        while line:
            cur_line = line.decode(self.encoding)
            line = m.readline()
            if cur_line.strip() == '':
                if len(block):
                    sent = conllu.parse(block)[0]
                    sent = ConllSent.from_conllu(sent)
                    yield sent
                    block = ''
            else:
                block += cur_line
        if len(block):
            sent = conllu.parse(block)[0]
            sent = ConllSent.from_conllu(sent)
            yield sent
        m.close()
        self.f.close()
        # print("it is over")
    
    def save(self, out_path=None):
        if out_path is None:
            out_path = os.path.splitext(self.fname)[0] + '.pickle'
        
        proj_sents = []
        n_non_proj = 0
        for sent in self:
            try:
                sent.transitions, sent.actions = sent.arc_std();
                proj_sents.append(sent)
            except:
                n_non_proj += 1

        print("Skipping {} non-projective sentences".format(n_non_proj))
        f = open(out_path, 'wb')
        pickle.dump(proj_sents, f)
        f.close()

                    
#This corpus reader can be used when reading large text file into a memory can solve IO bottleneck of training.
#Use it exactly as the regular CorpusReader from the rnnlm.py
class FastCorpusReader(object):
    def __init__(self, fname):
        self.fname = fname
        self.f = open(fname, 'rb')
    def __iter__(self):
        #This usage of mmap is for a Linux\OS-X 
        #For Windows replace prot=mmap.PROT_READ with access=mmap.ACCESS_READ
        m = mmap.mmap(self.f.fileno(), 0, prot=mmap.PROT_READ)
        data = m.readline()
        while data:
            line = data
            data = m.readline()
            line = line.lower()
            line = line.strip().split()
            yield line    
    
class CorpusReader(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in file(self.fname):
            line = line.strip().split()
            #line = [' ' if x == '' else x for x in line]
            yield line

def build_vocab(fname, lower=False):
    f_form = pytext.data.Field(lower=lower, tokenize=list)
    f_upos = pytext.data.Field(tokenize=list)
    f_xpos = pytext.data.Field(tokenize=list)
    f_deprel = pytext.data.Field(tokenize=list)
    f_actions = pytext.data.Field(tokenize=list)
    fields = [('form', f_form), ('upos', f_upos), ('xpos', f_xpos), ('deprel', f_deprel), ('actions', f_actions)]

    f = open(fname, 'rb')
    sentences = pickle.load(f)
    examples = [pytext.data.Example.fromlist([sent.form, sent.upos, sent.xpos, sent.deprel, sent.actions], fields) for sent in sentences]
    train = pytext.data.Dataset(examples, fields)
    
    for name, field in fields:
        field.build_vocab(train)
    
    out_path = os.path.join(os.path.dirname(fname), 'vocabs.pickle')
    out_file = open(out_path, 'wb')
    pickle.dump(fields, out_file)
    f.close()
    out_file.close()
    
