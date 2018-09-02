'''
Documentation, License etc.

@package transition_parser
'''
from operator import itemgetter
import dynet as dy
import numpy as np
import random
import os
import time

from utils import Arc

class SRTransition(object):
    def __init__(self, model, 
                 vocab_form, v_form, d_form, # word
                 vocab_upos, v_upos, d_upos, # upos
                 vocab_xpos, v_xpos, d_xpos, # xpos
                 vocab_act, v_act, d_act, d_deprel, # action
                 d_wcomp, # word_composition
                 d_stack, l_stack, # stack lstm
                 d_buffer, l_buffer, # buffer lstm
                 d_actions, l_actions, # actions lstm
                 h_state, # parser state
                 p_unk):
        
        # add subcollection of parameters 
        pc = model.add_subcollection('SRTransition')
        
        # dim stuffs
        self.d_form = d_form
        self.d_upos = d_upos
        self.d_xpos = d_xpos
        self.d_act = d_act
        self.d_deprel = d_deprel
        self.d_emb = self.d_form + self.d_upos + self.d_xpos 
        self.d_wcomp = d_wcomp
        
        self.d_stack = d_stack
        self.d_buffer = d_buffer
        self.d_actions = d_actions
        
        self.d_comp = 2 * d_wcomp + d_act
        self.d_state = d_stack + d_buffer + d_actions
        self.h_state = h_state
        
        # hold vocabs
        self.vocab_form, self.v_form = vocab_form, v_form
        self.vocab_upos, self.v_upos = vocab_upos, v_upos
        self.vocab_xpos, self.v_xpos = vocab_xpos, v_xpos
        self.vocab_act, self.v_act = vocab_act, v_act
        
        # unk replace
        self.p_unk = p_unk
        
        # add lookup parameters, 4 lookup parameters
        self.e_form = pc.add_lookup_parameters((v_form, d_form))
        if d_upos:
            self.e_upos = pc.add_lookup_parameters((v_upos, d_upos))
        if d_xpos:
            self.e_xpos = pc.add_lookup_parameters((v_xpos, d_xpos))
        self.e_act = pc.add_lookup_parameters((v_act, d_act))
        self.e_deprel = pc.add_lookup_parameters((v_act, d_deprel))
        
        # add RNN builders, 3 RNN builders
        self.lstm_s = dy.LSTMBuilder(l_stack, self.d_wcomp, self.d_stack, pc)
        self.lstm_a = dy.LSTMBuilder(l_actions, self.d_act, self.d_actions, pc)
        self.lstm_b = dy.LSTMBuilder(l_buffer, self.d_wcomp, self.d_buffer, pc)
        self.empty_b = pc.add_parameters((self.d_wcomp,))
        self.empty_s = pc.add_parameters((self.d_wcomp,))
        self.empty_a = pc.add_parameters((self.d_act,))
        
        # add weights & biases 
        self.wcomp = pc.add_parameters((self.d_wcomp, self.d_emb))
        self.wcomp_b = pc.add_parameters((self.d_wcomp,))
        
        self.s2h = pc.add_parameters((self.h_state, self.d_state))
        self.s2h_b = pc.add_parameters((self.h_state,))
        
        self.h2a = pc.add_parameters((self.v_act, self.h_state))
        self.h2a_b = pc.add_parameters((self.v_act,))
        
        self.comp = pc.add_parameters((self.d_wcomp, self.d_comp))
        self.comp_b = pc.add_parameters((self.d_wcomp,))
        
        # about saving and loading
        self.pc = pc
        self.spec = (vocab_form, v_form, d_form, # word
                    vocab_upos, v_upos, d_upos, # upos
                    vocab_xpos, v_xpos, d_xpos, # xpos
                    vocab_act, v_act, d_act, d_deprel, # action
                    d_wcomp, # word_composition
                    d_stack, l_stack, # stack lstm
                    d_buffer, l_buffer, # buffer lstm
                    d_actions, l_actions, # actions lstm
                    h_state, # parser state
                    p_unk)
        
    def __call__(self, form, upos, xpos, train=False, target_actions=None):
        # if len(sentence == 1) 退化情形， 直接返回
        if len(form) == 1:
            return (None, [Arc(('<root>', 0), (form[0], 1), 'root')])
        
        # target
        if train:
            target = iter(target_actions) # ["Shift", "Shift", "Lreduce_amod", ...]
        
        # stack lstm and initial state
        stack = []
        stack_top = self.lstm_s.initial_state()
        stack_top = stack_top.add_input(self.empty_s)
        
        # action lstm and initial state
        actions = []
        actions_top = self.lstm_a.initial_state()
        actions_top = actions_top.add_input(self.empty_a)
        
        # embeddings, reversed
        embeddings = [self._get_embedding(w_form, w_upos, w_xpos, self.p_unk, train) for w_form, w_upos, w_xpos in zip(reversed(form), reversed(upos), reversed(xpos))]
        
        # bi-lstm encoded, reversed
        buffer_top = self.lstm_b.initial_state()
        buffer_top = buffer_top.add_input(self.empty_b)
        wics = buffer_top.transduce(embeddings)
        buffer = [(wic, emb, (w, id)) for wic, emb, w, id in zip(wics, embeddings, reversed(form), range(len(form), 0, -1))]
        
        # aggregator, loss for actions, loss for deprel
        if train:
            loss_actions = [] # up to 2n steps
        pred_arcs = [] # 3ew
        
        # loop
        while not (len(stack) == 1 and len(buffer) == 0):
            # 注意这个循环只会执行 2n-1 次， 但是 target_actions 的长度是 2n 但最后一步是没有必要的
            
            # get gold action
            if train:
                action_g = next(target) # gold action
                action_gi = self.vocab_act.stoi[action_g]
            
            # get action to take, for simplicity, we use teacher forcing when training
            valid_actions = self._valid_actions(stack, buffer)
            if len(valid_actions) == 1:
                if train:
                    action, action_i = action_g, action_gi
                else:
                    action = "Shift"
                    action_i = self.vocab_act.stoi['Shift']
            else:
                logp_actions = self._logp_actions(stack, buffer, actions, valid_actions)
                if train:
                    loss_step = -dy.pick(logp_actions, action_gi)
                    loss_actions.append(loss_step) # aggregate loss_tsns
                    action, action_i = action_g, action_gi # follow gold action
                else:
                    action_i = np.argmax(logp_actions.npvalue()) # infer by it self
                    action = self.vocab_act.itos[action_i]

            # take actions and record pred_arcs
            self._apply_action(stack, buffer, actions, stack_top, actions_top, action, action_i, pred_arcs)
            
        # last step
        _, _, tok = stack.pop()
        pred_arcs.append(Arc(('<root>', 0), tok, 'root'))
        
        if train:
            loss = dy.esum(loss_actions)
        else:
            loss = None 
        return loss, pred_arcs

    def _valid_actions(self, stack, buffer):
            valid_actions = []
            if len(buffer) > 0:
                valid_actions.append(self.vocab_act.stoi["Shift"])
            if len(stack) >= 2:
                valid_actions.extend(idx for (idx, act) in enumerate(self.vocab_act.itos) if act != "Shift")
            return valid_actions
    
    def _get_embedding(self, w_form, w_upos, w_xpos, p_unk, train):
        #get embedding for form (upos and xpos, if requested), drop out to <unk> for form
        #w_form, w_upos, w_xpos: str

        # form embedding with dropout-to-unk
        if train:
            keep_original_form = (np.random.rand() < 1 - p_unk) or self.vocab_form.freqs[w_form] > 1
        else:
            keep_original_form = True
        form_vec = dy.lookup(self.e_form, self.vocab_form.stoi[w_form] if keep_original_form else
                             self.vocab_form.stoi['<unk>'])
        upos_vec = dy.lookup(self.e_upos, self.vocab_upos.stoi[w_upos]) if self.d_upos else None
        xpos_vec = dy.lookup(self.e_xpos, self.vocab_xpos.stoi[w_xpos]) if self.d_xpos else None #defaultdict
        vecs = [form_vec, upos_vec, xpos_vec]

        emb_vec = dy.concatenate([vec for vec in vecs if vec is not None])
        wcomp_vec = dy.affine_transform([self.wcomp_b, self.wcomp, emb_vec])
        return wcomp_vec
    
    def _logp_actions(self, stack, buffer, actions, valid_actions):
        stack_embedding = stack[-1][0].output() # stack is not empty, otherwise shift is a must
        buffer_embedding = buffer[-1][0] if buffer else self.empty_b
        actions_embedding = actions[-1].output()
        sba = dy.concatenate([stack_embedding, buffer_embedding, actions_embedding])
        parser_state = dy.rectify(dy.affine_transform([self.s2h_b, self.s2h, sba]))
        logp_actions = dy.log_softmax(dy.affine_transform([self.h2a_b, self.h2a, parser_state]), valid_actions)
        return logp_actions

    def _compose(self, head_repr, deprel_repr, mod_repr):
        arc = dy.concatenate([head_repr, deprel_repr, mod_repr])
        composed = dy.tanh(dy.affine_transform([self.comp_b, self.comp, arc]))
        return composed

    def _apply_action(self, stack, buffer, actions, stack_top, actions_top, action, action_i, pred_arcs):
        # update actions
        act_repr = dy.lookup(self.e_act, action_i)
        top_actions_state = actions[-1] if actions else actions_top
        top_actions_state = top_actions_state.add_input(act_repr)
        actions.append(top_actions_state)
        
        # update stack
        if action == "Shift":
            wic, emb, tok  = buffer.pop()
            stack_state, _, _ = stack[-1] if stack else (stack_top, self.empty_s, ('<TOP>', -1))
            stack_state = stack_state.add_input(emb)
            stack.append((stack_state, emb, tok))
        else:
            right = stack.pop()
            left = stack.pop()
            tsn, dep = action.split('_')
            if tsn == "Rreduce":
                head = left
                modifier = right
            else:
                head = right
                modifier = left
                
            # compose
            top_stack_state, _, _ = stack[-1] if stack else (stack_top, self.empty_s, ('<TOP>', -1))
            head_state, head_input, head_tok = head
            mod_state, mod_input, mod_tok = modifier
            deprel_repr = dy.lookup(self.e_deprel, action_i)
            composed_rep = self._compose(head_input, deprel_repr, mod_input)
            
            # update stack and record arc
            top_stack_state = top_stack_state.add_input(composed_rep)
            stack.append((top_stack_state, composed_rep, head_tok))
            pred_arcs.append(Arc(head_tok, mod_tok, dep))
        
    # support saving:
    def param_collection(self): 
        return self.pc
        
    def train(self, dataset, epoch=1, lr = 0.1, valid_dataset=None, test_dataset=None, resume=True):
        # 哈哈， 有大型 API 也有小型 API
        trainer = dy.CyclicalSGDTrainer(self.pc)
        
        if resume and os.listdir('save'):
            with open("save/records.csv", 'rt') as f:
                records = []
                f.readline() # skip colume names
                for line in f:
                    e, uas, las = line.strip().split()
                    records.append((int(e), float(uas), float(las)))
            best_uas = max(records, key=itemgetter(1))[1]
            best_las = max(records, key=itemgetter(2))[2]
                
            resume_from = max(records, key=itemgetter(2))[0]
            self.pc.populate("save/model_{}".format(resume_from))
            print("[Train] Resume from epoch {}".format(resume_from))
            resume_from += 1
        else:
            resume_from = 1
            best_uas = 0
            best_las = 0
            records = [] # (epoch, uas, las)
        
        for e in range(resume_from, resume_from + epoch):
            # shuffle dataset
            random.shuffle(dataset)
            
            # loss aggregator
            loss_aggregator = {'n_tokens': [], 'losses': []}
            for sent_id, sent in enumerate(dataset, 1):
                dy.renew_cg()
                loss, _ = self.__call__(sent.form, sent.upos, sent.xpos, train=True, target_actions=sent.actions)
                loss_aggregator['n_tokens'].append(len(sent.form))
                if loss:
                    loss_sent = loss.scalar_value()
                    loss_aggregator['losses'].append(loss_sent)
                    loss.backward()
                    trainer.update()
                
                # summary and clear aggregator
                if sent_id % 100 == 0:
                    trainer.status()
                    print("\t[Train]\tepoch: {}\tsent_id: {}\tloss: {:.6f}".format(
                        e, sent_id, np.sum(loss_aggregator['losses']) / np.sum(loss_aggregator['n_tokens'])))
                    loss_aggregator['n_tokens'] = []
                    loss_aggregator['losses'] = []
                    
            if valid_dataset:
                uas, las, n_sents, n_tokens = self.test(valid_dataset)
                print("[Best]\tUAS: {:.6f}\tLAS: {:.6f}".format(best_uas, best_las))
                print("[Valid]\tepoch: {}\tUAS: {:.6f}\tLAS: {:.6f}".format(e, uas, las))
                if uas > best_uas or las > best_las:
                    self.pc.save("save/model_{}".format(e))
                    records.append((e, uas, las))
                    best_uas = max(best_uas, uas)
                    best_las = max(best_las, las)
            #trainer.learning_rate = lr / (1 + 0.05 * (e - resume_from + 1))
            
        if test_dataset:
            #best_uas_model = max(records, key=itemgetter(1))[0]
            #self.pc.populate("save/model_{}".format(best_uas_model))
            #uas, las, n_sents, n_tokens = self.test(test_dataset)
            #print("[Test]\tPriority: UAS\tUAS: {:.6f}\tLAS: {:.6f}".format(uas, las))
            
            best_las_model = max(records, key=itemgetter(2))[0]
            self.pc.populate("save/model_{}".format(best_las_model))
            uas, las, n_sents, n_tokens = self.test(test_dataset)
            print("[Test]\tPriority: LAS\tUAS: {:.6f}\tLAS: {:.6f}".format(uas, las))
            
            uas, las, n_sents, n_tokens = self.test(dataset)
            print("[Cheat]\tPriority: LAS\tUAS: {:.6f}\tLAS: {:.6f}".format(uas, las))
        
        # write records
        print("[Record]\t Writing records to save/records.csv")
        with open("save/records.csv", 'at') as f:
            #f.write("{}\t{}\t{}\n".format('Epoch', 'UAS', 'LAS'))
            for e, uas, las in records:
                f.write("{}\t{:.6f}\t{:.6f}\n".format(e, uas, las))

        
    def test(self, dataset):
        n_sents = len(dataset)
        n_tokens = 0
        correct_head = 0
        correct_both = 0
        
        test_start = time.time()
        for sent in dataset:
            dy.renew_cg()
            _, pred_arcs = self.__call__(sent.form, sent.upos, sent.xpos, train=False, target_actions=None)
            sorted_arcs = sorted(pred_arcs, key=lambda x: x.dependent[1])
            head = [arc.head[1] for arc in sorted_arcs]
            deprel = [arc.deprel for arc in sorted_arcs]
            for h_pred, r_pred, h_gold, r_gold, upos_gold in zip(head, deprel, sent.head, sent.deprel, sent.upos):
                if upos_gold == 'PU':
                    continue
                n_tokens += 1
                if h_pred == h_gold:
                    correct_head += 1
                    if r_pred == r_gold:
                        correct_both += 1
        test_end = time.time()
        print("[Speed] {} sents\t{:.6f}s\tsents/s: {:.6f}\ttokens/s: {:.6f}".format(n_sents, test_end - test_start, n_sents / (test_end - test_start), n_tokens / (test_end - test_start)))
        uas = correct_head / n_tokens
        las = correct_both / n_tokens
        return uas, las, n_sents, n_tokens

    @staticmethod
    def from_spec(spec, model):
        return SRTransition(model, *spec)

    def save(self, epoch):
        self.pc.save("save/model_{}".format(epoch))
    
    def load(self, epoch):
        self.pc.populate("save/model_{}".format(epoch))
