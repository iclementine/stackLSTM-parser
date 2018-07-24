import numpy as np
import dynet as dy
from ggparser import utils

pc = dy.ParameterCollection()
bb = utils.BiLinear(pc, 5, channels=3, bias_x=True, bias_y=True)
x = dy.inputTensor(np.random.randn(5,10))
y = dy.inputTensor(np.random.randn(5,10))
z = bb(x,y)
print(x.dim())
print(y.dim())
print(z.dim())

lin = utils.MLP(pc, 20, 100, 10)
x = dy.inputTensor(np.random.randn(20, 33))
y = lin(x)
print(x.dim())
print(y.dim())

#expected ouputs and logs
#正在启动：anaconda_python3 /home/clementine/projects/ggparser/test_blocks.py
#[dynet] random seed: 2456053739
#[dynet] allocating memory: 512MB
#[dynet] memory allocation done.
#((5, 10), 1)
#((5, 10), 1)
#((10, 3, 10), 1)
#((20, 33), 1)
#((10, 33), 1)
#*** 正常退出 ***


