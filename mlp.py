# Example of a user-defined saveable type.
import dynet as dy

class OneLayerMLP(object):
    def __init__(self, model, num_input, num_hidden, num_out, act=dy.tanh):
        pc = model.add_subcollection("OneLayerMLP")
        self.W1 = pc.add_parameters((num_hidden, num_input))
        self.W2 = pc.add_parameters((num_out, num_hidden))
        self.b1 = pc.add_parameters((num_hidden))
        self.b2 = pc.add_parameters((num_out))
        self.pc = pc
        self.act = act
        self.spec = (num_input, num_hidden, num_out, act)

    def __call__(self, input_exp):
        W1 = dy.parameter(self.W1)
        W2 = dy.parameter(self.W2)
        b1 = dy.parameter(self.b1)
        b2 = dy.parameter(self.b2)
        g = self.act
        return W2*g(W1*input_exp + b1)+b2
    
    def run(self, x):
        print(x)

    # support saving:
    def param_collection(self): return self.pc

    @staticmethod
    def from_spec(spec, model):
        num_input, num_hidden, num_out, act = spec
        return OneLayerMLP(model, num_input, num_hidden, num_out, act)
