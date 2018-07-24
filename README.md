# stackLSTM-parser
a simple modification of Chris Dyer's stack LSTM Parser

It is a simple stack lstm parser using Arc-Standard transition system implemented with dynet, the structure is shown below.

0. embedding for word form, upos, xpos, and word form is replaced by <unk> with a probability of c/(c+alpha), where c is the frequency of that word. Word form embedding, xpos embedding, upos embedding is independently droppout out with a probability of p.
1. a stack lstm representation of the stack, 
2. a bi-directional lstm representation of the buffer, 
3. a mlp to decide with transition to take at each step from representation of the stack and the buffer, 
4. a mlp to decide which deprel to take where the oracle transition is a kind of reduction. The mlp for transition and the mlp for deprel are both a one-hidden-layer mlps, which share a common hidden layer. 
5. a mlp for composition of the two lstm outputs from the top 2 elements in the stack, it is also a on-hidden-layer mlp.

It achives

1. UAS: 0.8447712418300654 LAS: 0.8099415204678363 for Universal Dependency for Chinese.
2. UAS: 0.8747954173486089 LAS: 0.843268154018434  for Universal Dependenvy for English.
