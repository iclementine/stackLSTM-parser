'''
Documentation, License etc.

@package transition_parser
'''
from operator import itemgetter
import dynet as dy
import numpy as np
import random
import os

from mlp import OneLayerMLP
from utils import Arc


class SRTransition(object):
    SHIFT = 0
    REDUCE_L = 1
    REDUCE_R = 2
    NUM_TRANSITION = 3
    
    def __init__(self, model, vocab_form, v_form, d_form, alpha, vocab_upos, v_upos, d_upos, vocab_xpos, v_xpos, d_xpos, vocab_deprel, v_deprel, d_stack, l_stack, d_buffer, l_buffer, bi_buffer, h_state, h_composition, p_drop, act=dy.rectify):
        # add subcollection of parameters 
        pc = model.add_subcollection('SRTransition')
        
        # dim stuffs
        self.d_form = d_form
        self.d_upos = d_upos
        self.d_xpos = d_xpos
        self.d_emb = self.d_form + self.d_upos + self.d_xpos
        self.d_comp = 2 * self.d_emb 
        self.h_state = h_state
        
        # hold vocabs
        self.vocab_form, self.v_form = vocab_form, v_form
        self.vocab_upos, self.v_upos = vocab_upos, v_upos
        self.vocab_xpos, self.v_xpos = vocab_xpos, v_xpos
        self.vocab_deprel, self.v_deprel = vocab_deprel, v_deprel
        
        # dropout-staffs
        self.alpha = alpha
        self.p_drop = p_drop
        
        # add lookup parameters
        self.e_form = pc.add_lookup_parameters((v_form, d_form))
        if d_upos:
            self.e_upos = pc.add_lookup_parameters((v_upos, d_upos))
        if d_xpos:
            self.e_xpos = pc.add_lookup_parameters((v_xpos, d_xpos))
        
        # add RNN builders
        self.lstm_s = dy.LSTMBuilder(l_stack, self.d_emb, self.d_emb, pc)
        if bi_buffer:
            self.lstm_b = dy.BiRNNBuilder(l_buffer, self.d_emb, self.d_emb, pc, dy.LSTMBuilder)
        else:
            self.lstm_b = dy.LSTMBuilder(l_buffer, self.d_emb, self.d_emb, pc)
        self.empty_b = pc.add_parameters((self.d_emb))
        
        # add mlps
        self.s2h = pc.add_parameters((self.h_state, 2 * self.d_emb))
        self.s2h_b = pc.add_parameters((self.h_state))
        self.h2t = pc.add_parameters((self.NUM_TRANSITION, self.h_state))
        self.h2t_b = pc.add_parameters((self.NUM_TRANSITION))
        self.h2dep = pc.add_parameters((self.v_deprel, self.h_state))
        self.h2dep_b = pc.add_parameters((self.v_deprel))
        
        self.mlp_comp = OneLayerMLP(pc, self.d_comp, h_composition, self.d_emb, act=act)
        
        # about saving and loading
        self.pc = pc
        self.spec = (vocab_form, v_form, d_form, alpha, vocab_upos, v_upos, d_upos, vocab_xpos, v_xpos, d_xpos, vocab_deprel, v_deprel, d_stack, l_stack, d_buffer, l_buffer, bi_buffer, h_state, h_composition, p_drop, act)
        
    def __call__(self, form, upos, xpos, train=False, target_transitions=None):
        # if len(sentence == 1) 退化情形， 直接返回
        if len(form) == 1:
            return (None, [Arc(('<root>', 0), (form[0], 1), 'root')])
        
        # target
        if train:
            target = iter(target_transitions) # [(0, 'word'), (1, 'nmod'), ...]
        
        # stack lstm and initial state
        stack = []
        stack_top = self.lstm_s.initial_state()

        # embeddings, reversed
        embeddings = [self._get_embedding(w_form, w_upos, w_xpos, train) for w_form, w_upos, w_xpos in zip(reversed(form), reversed(upos), reversed(xpos))]
        # bi-lstm encoded, reversed
        wics = self.lstm_b.transduce(embeddings)
        buffer = [(wic, (w, id)) for wic, w, id in zip(wics, reversed(form), range(len(form), 0, -1))]
        
        # aggregator, loss for transitions, loss for deprel
        if train:
            loss_tsns = [] # 2n - 1 steps
            loss_deps = [] # n - 1 tokens, head of the sentence will not be count
        pred_arcs = [] # train or not, record pred_arcs, though if train, it is nonsense
        
        # loop
        while not (len(stack) == 1 and len(buffer) == 0):
            # 注意这个循环只会执行 2n-1 次， 但是 target_transitions 的长度是 2n 但最后一步是没有必要的
            if train:
                tsn_g, dep_g = next(target) # gold transition and gold deprel
            valid_transitions = []
            if len(buffer) > 0:
                valid_transitions.append(self.SHIFT)
            if len(stack) >= 2:
                valid_transitions.extend([self.REDUCE_L, self.REDUCE_R])
            
            if len(valid_transitions) > 1:
                # h is shared with deprel computation
                logp_tsns, h = self._get_transition(stack, buffer, self.empty_b, valid_transitions)
                if train:
                    l_tsn = -dy.pick(logp_tsns, tsn_g)
                    loss_tsns.append(l_tsn) # aggregate loss_tsns
                    tsn = tsn_g # follow tsn_g
                else:
                    tsn = np.argmax(logp_tsns.npvalue()) # infer by it self
                    
                if tsn != self.SHIFT:
                    # compute deprel only when needed
                    logp_deps = self._get_deprel(h)
                    if train:
                        l_dep = -dy.pick(logp_deps, self.vocab_deprel.stoi[dep_g])
                        loss_deps.append(l_dep) # aggregate loss_deps
                        dep = dep_g
                    else:
                        dep = self.vocab_deprel.itos[np.argmax(logp_deps.npvalue())] # infer by it self
            else:
                if train:
                    tsn, dep = tsn_g, dep_g
                else:
                    tsn = self.SHIFT
                    dep = buffer[-1][1][0] # word to shift

            # take transitions, for simplicity, we use teacher forcing when training
            if tsn == self.SHIFT:
                wic, tok  = buffer.pop()
                stack_state, _ = stack[-1] if stack else (stack_top, ('<TOP>', -1))
                stack_state = stack_state.add_input(wic)
                stack.append((stack_state, tok))
            else:
                right = stack.pop()
                left = stack.pop()
                if tsn == self.REDUCE_R:
                    head = left
                    modifier = right
                else:
                    head = right
                    modifier = left
                # head, modifier = (left, right) if tsn == self.REDUCE_R else (right, left)
                top_stack_state, _ = stack[-1] if stack else (stack_top, ('<TOP>', -1))
                head_rep, head_tok = head[0].output(), head[1]
                mod_rep, mod_tok = modifier[0].output(), modifier[1]
                composed_rep = dy.tanh(self.mlp_comp(dy.concatenate([head_rep, mod_rep])))
                top_stack_state = top_stack_state.add_input(composed_rep)
                stack.append((top_stack_state, head_tok))
                pred_arcs.append(Arc(head_tok, mod_tok, dep))
            
        # last step
        _, tok = stack.pop()
        pred_arcs.append(Arc(('<root>', 0), tok, 'root'))
        
        if train:
            loss = (dy.esum(loss_tsns) if loss_tsns else None,
                    dy.esum(loss_deps) if loss_deps else None)
        else:
            loss = None
        return loss, pred_arcs

    def _get_embedding(self, w_form, w_upos, w_xpos, train):
        #get embedding for form (upos and xpos, if requested), drop out to <unk> for form, drop to zero for form, upos, xpos sepraratly while train
        #and scale for the dropout to 0
        #w_form, w_upos, w_xpos: str

        # form embedding with dropout-to-unk
        p = self.vocab_form.freqs[w_form] / (self.vocab_form.freqs[w_form] + self.alpha)
        keep_original_form = (np.random.rand() < p) or not train
        form_vec = dy.lookup(self.e_form, self.vocab_form.stoi[w_form] if keep_original_form else self.vocab_form.stoi['<unk>'])
        upos_vec = dy.lookup(self.e_upos, self.vocab_upos.stoi[w_upos]) if self.d_upos else None
        xpos_vec = dy.lookup(self.e_xpos, self.vocab_xpos.stoi[w_xpos]) if self.d_upos else None #defaultdict
        vecs = [form_vec, upos_vec, xpos_vec]
        
        # import pdb; pdb.set_trace()
        if train:
            keep_form = float((np.random.rand() < (1 - self.p_drop)) or not train)
            keep_upos = float((np.random.rand() < (1 - self.p_drop)) or not train)
            keep_xpos = float((np.random.rand() < (1 - self.p_drop)) or not train)
            
            scale = (self.d_form + self.d_upos + self.d_xpos) / (keep_form * self.d_form + keep_upos * self.d_upos + keep_xpos * self.d_xpos + 1e-12) # 难道就只能求保佑不要全部 drop 掉吗？ (事实上 0 即使是乘 1e12 也没有什么问题， 还是 0)这均值保持术可怕， 至于方差有没有保住， 再继续讨论吧
            
            msks = [keep_form, keep_upos, keep_xpos]
            emb_vec = dy.concatenate([scale * vec for keep, vec in zip(msks, vecs) if vec is not None])
        
        emb_vec = dy.concatenate([vec for vec in vecs if vec is not None])
        return emb_vec
    
    def _get_transition(self, stack, buffer, empty_buffer, valid_transitions):
        stack_embedding = stack[-1][0].output() # the stack is not empty so we should decide transition
        buffer_embedding = buffer[-1][0] if buffer else empty_buffer
        parser_state = dy.concatenate([buffer_embedding, stack_embedding])
        h = dy.rectify(self.s2h * parser_state + self.s2h_b)
        logits = self.h2t * h + self.h2t_b
        logps = dy.log_softmax(logits, valid_transitions)                               
        return logps, h
    
    def _get_deprel(self, h):
        logits = self.h2dep * h + self.h2dep_b
        logps = dy.log_softmax(logits)
        return logps

    # support saving:
    def param_collection(self): 
        return self.pc
    
    @staticmethod
    def from_spec(spec, model):
        return SRTransition(model, *spec)
        
    def train(self, dataset, epoch=1, valid_dataset=None, test_dataset=None, resume=True):
        # 哈哈， 有大型 API 也有小型 API
        trainer = dy.CyclicalSGDTrainer(self.pc)
        
        if resume:
            resume_from = max([int(x.split('_')[1]) for x in os.listdir('save/')])
            self.pc.populate("save/model_{}".format(resume_from))
            print("[Train] Resume from epoch {}".format(resume_from))
        else:
            resume_from = 0
        best_uas = 0
        best_las = 0
        records = [] # (epoch, uas, las)
        
        for e in range(resume_from, resume_from + epoch):
            # shuffle dataset
            random.shuffle(dataset)
            for sent_id, sent in enumerate(dataset, 1):
                dy.renew_cg()
                length = len(sent.form)
                if length == 1:
                    continue
                loss, _ = self.__call__(sent.form, sent.upos, sent.xpos, train=True, target_transitions=sent.transitions)
                (loss[0] + loss[1]).forward()
                # 分别训练的 trick
                if sent_id % 5 == 0:
                    loss[1].backward()
                else:
                    loss[0].backward()
                #(loss[0] + loss[1]).backward()
                trainer.update()
                if sent_id % 100 == 0:
                    print("[Train]\tepoch: {}\tsent_id: {}\tstructure_loss: {}\tdeprel_loss: {}".format(e, sent_id, loss[0].scalar_value() / length, loss[1].scalar_value() / length))
            if valid_dataset:
                uas, las, n_sents, n_tokens = self.test(valid_dataset)
                print("[Valid]\tepoch: {}\tUAS: {}\tLAS: {}".format(e, uas, las))
                if uas > best_uas or las > best_las:
                    self.pc.save("save/model_{}".format(e))
                    records.append((e, uas, las))
                    best_uas = max(best_uas, uas)
                    best_las = max(best_las, las)
        if test_dataset:
            best_uas_model = max(records, key=itemgetter(1))[0]
            self.pc.populate("save/model_{}".format(best_uas_model))
            uas, las, n_sents, n_tokens = self.test(test_dataset)
            print("[Test]\tUAS: {}\tLAS: {}".format(uas, las))
            
            
    def test(self, dataset):
        n_sents = len(dataset)
        n_tokens = 0
        correct_head = 0
        correct_both = 0
        
        for sent in dataset:
            dy.renew_cg()
            _, pred_arcs = self.__call__(sent.form, sent.upos, sent.xpos, train=False, target_transitions=None)
            sorted_arcs = sorted(pred_arcs, key=lambda x: x.dependent[1])
            head = [arc.head[1] for arc in sorted_arcs]
            deprel = [arc.deprel for arc in sorted_arcs]
            for h_pred, r_pred, h_gold, r_gold in zip(head, deprel, sent.head, sent.deprel):
                n_tokens += 1
                if h_pred == h_gold:
                    correct_head += 1
                    if r_pred == r_gold:
                        correct_both += 1
        uas = correct_head / n_tokens
        las = correct_both / n_tokens
        return uas, las, n_sents, n_tokens
        
