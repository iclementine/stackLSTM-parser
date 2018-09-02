from collections import namedtuple
import numpy as np
import dynet as dy

Arc = namedtuple('Arc', ['head', 'dependent', 'deprel'])


def log_softmax_gpu(expr, restrict=None):
	if restrict:
		mask = np.zeros(expr.dim()[0])
		mask[restrict] =  1.
		mask = dy.inputTensor(mask)
	logp = dy.cmult(dy.log_softmax(expr), mask) 
	return logp

	
