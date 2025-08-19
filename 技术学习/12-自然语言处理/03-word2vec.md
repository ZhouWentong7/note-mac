目标：
实现一个建议的word2vec，牺牲一点效率，但便于理解。

## 基于推理方法和神经网络

不需要一次性使用所有的语料库数据，就可以训练出一个学习好的神经网络来将文本向量化。

**什么是基于推理？**
- 从上下文推测目标位置会是什么单词。

神经网络使用`one-hot`的方法将单词转化为向量。

设词表是 {you:0, say:1, hello:2}

| 单词    | 单词ID | one-hot   |
| ----- | ---- | --------- |
| you   | 0    | [1, 0, 0] |
| say   | 1    | [0, 1, 0] |
| hello | 2    | [0, 0, 1] |

只要将单词转化为固定长度的向量，神经网络的输入层的神经元个数就可以固定下来。

## 简单的W2V神经网络

word2vec一词最初用来指程序或者工具，但是随着该词的流行，在某些语境下，也指神经网络的模型。正确地说，CBOW模型和 skip-gram模型是word2vec中使用的两个神经网络。本节我们将主要讨论CBOW模型。

### CBOW continuous bag-of-words
CBOW是根据上下文预测的当前单词的神经网络。

输入是目标词的上下文，因此CBOW具有两个输入层。单词列表由`one-hot`形式表示。

对与输入文章 `you say goodbye`的`say`而言，那输入就是`you`和`goodbye`。

两个输入层经过一个中间层到达输出层，这些层之间均为全连接。如图所示：

![[fig3-9.png]]
中间层的神经元是各个输入层经全连接层变换后得到的值的“平均”。
- [*]  中间层一定要比输出层的神经元数量更少。

输出层则是各个单词的概率（使用softmax和交叉熵误差)。从输入层到中间层的$W_{in}$会在学习的过程中不断变化，这就是我们需要的单词的分布式。

第一层$h_1$ ，第二层转化为$h_2$那么，中间层的神经元就是$h_1 + h_2$。

代码如下：

```python
# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul
# 样本的上下文数据
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 初始化权重
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 生成层
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)
 

# 正向传播

h0 = in_layer0.forward(c0)

h1 = in_layer1.forward(c1)

h = 0.5 * (h0 + h1)

s = out_layer.forward(h)

print(s)
```


这里的输出，再经过Softmax之后就可以得到各个单词的概率，这个概率会表示哪个单词会出现在这个输入的上下文中间。

![[fig3-19.png]]

相当于是一个多分类的网络，所以最后是使用`Softmax`和交叉熵误差进行学习。

将这两层结合，可以组合成`Softmax with Loss`层。

### word2vec的权重和分布式表示

已知CBOW有两组分布式，$W_{in}$  和$W_{out}$ 。 那么应该用哪一个来作为词的分布式表示?还是说应该同时使用两个权重来进行表示？

对于w2v而言，最常见的是使用$W_{in}$进行表示（特别是skip-gram模型）。

> 也有使用两个权重相加的模型，效果也不错。
> > GloVe


## 学习数据的准备

word2vec中使用的神经网络的**输入是上下文**，它的**正确解标签**是被这些上下文包围**在中间的单词**，**即目标词**。

**将上下文与目标词转化为one-hot流程：**
1. 处理为contexts 与 target矩阵
2. 使用词典的方法转化为ID的形式。
3. 进一步转换为one-hot

![[fig3-18.png]]

代码：

```python
text = 'You say goodbye and I say hello.'

corpus, word_to_id, id_to_word = preprocess(text) 

vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)

target = convert_one_hot(target, vocab_size)

contexts = convert_one_hot(contexts, vocab_size)
```


#### 实现简单的CBOW模型

```python
# coding: utf-8

import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:

	def __init__(self, vocab_size, hidden_size):
	
		V, H = vocab_size, hidden_size
		
		# 初始化权重
		
		W_in = 0.01 * np.random.randn(V, H).astype('f')
		
		W_out = 0.01 * np.random.randn(H, V).astype('f')
	
		# 生成层
		
		self.in_layer0 = MatMul(W_in)
		self.in_layer1 = MatMul(W_in)
		self.out_layer = MatMul(W_out)
		self.loss_layer = SoftmaxWithLoss()
	
		# 将所有的权重和梯度整理到列表中
		layers = [self.in_layer0, self.in_layer1, self.out_layer]
		self.params, self.grads = [], []
		for layer in layers:
		self.params += layer.params
		self.grads += layer.grads
	
		# 将单词的分布式表示设置为成员变量
		self.word_vecs = W_in

  

	def forward(self, contexts, target):

		h0 = self.in_layer0.forward(contexts[:, 0])
		h1 = self.in_layer1.forward(contexts[:, 1])
		h = (h0 + h1) * 0.5
		score = self.out_layer.forward(h)
		loss = self.loss_layer.forward(score, target)
		
		return loss

  

	def backward(self, dout=1):
	
		ds = self.loss_layer.backward(dout)
		da = self.out_layer.backward(ds)
		da *= 0.5
		self.in_layer1.backward(da)
		self.in_layer0.backward(da)
		
		return None
```

## word2vec的补充说明

### CBOW 的概率模型

- $P( \dot )$
- $P( A )$：事件A发生的概率
- $P( A, B )$：联合概率，A和B同时发生的概率
- $P( A | B )$：后验概率——在给定事件B（的信息）时事件A发生的概率
![[fig3-22.png]]

那目标词的上下文和目标词之间的概率可以表示为：

$$
P(w_t | w_{t-1},w_{t+1})
$$

使用上述式可以简洁地表示CBOW模型的损失函数。先回顾一下交叉熵损失：

$$
Loss = -\sum_k t_k \log y_k 
$$

  
$y_k$表示第k个事件发生的概率。$t_k$是监督标签,它是one-hot向量的元素,只有当正确的时候，one-hot对应元素为1. 所以可以进一步得到：

$$
Loss = -log P(w_t | w_{t-1},w_{t+1})
$$

上述公式是一笔数据样本的损失，若拓展到整个语料库：

$$
Loss = -\frac{1}{T} \sum_{t=1}^T log P(w_t | w_{t-1},w_{t+1})
$$


### skip-gram模型

![[fig3-23.png]]
- 一个输入层，多个输出层。
- 分别求出各个输出层的损失之后相加，得到该模型的损失。

数学模型：

$$
P( w_{t-1},w_{t+1}|w_t )
$$
损失表示：

$$
L = - \frac{1}{T} (\log P(w_{t-1}|w_t) + \log P(w_{t+1}|w_t))
$$
更为推荐使用skip-gram来计算单词的分布式，因为在实际的应用中体现出了更好的表现，即使在学习速度上略逊一筹。


另外：**Word2Vec (SGNS) 虽然看起来是通过神经网络训练得到的，但数学上等价于在做一个经过调整的词-上下文共现矩阵的分解。**



