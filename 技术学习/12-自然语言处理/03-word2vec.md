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