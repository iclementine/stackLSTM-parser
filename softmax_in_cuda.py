import dynet as dy
import numpy as np

def log_softmax_gpu(expr, restrict=None):
	shape, bs = expr.dim()
	max_logits = dy.max_dim(expr, d=0)
	logits = expr- max_logits
	exps = dy.exp(logits)
	if restrict:
		mask = np.zeros(expr.dim()[0])
		mask[restrict] =  1.
		mask = dy.inputTensor(mask)
		exps = dy.cmult(exps, mask)
	sum = dy.reshape(dy.sum_dim(exps, d=[0]), d=(1, *shape[1:]))
	logp = dy.log(dy.cdiv(exps, sum))
	return logp

	
pc = dy.ParameterCollection()
pw = pc.add_parameters((5, 10))

xv = np.random.randn(10)
dy.renew_cg()
x = dy.inputTensor(xv)
x = dy.to_device(x, "CPU")
W = dy.parameter(pw)
W = dy.to_device(W, "CPU")
y = W * x
p = dy.log_softmax(y, [1,2,3])
print(p.npvalue())

dy.renew_cg()
W = dy.parameter(pw)
x = dy.inputTensor(xv)
y = W * x
p = log_softmax_gpu(y, [1, 2, 3])
print(p.npvalue())
