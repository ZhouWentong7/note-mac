前馈型网络的局限性：不能很好的处理是序列问题。

RNN（Recurrent Neural Network，循环神经网络）由此诞生。
#RNN

## 概率和语言模型  

1. CBOW模型的目的：从上下文预测目标词
2. 若将上下文限定为左侧窗口，则概率公式：$P(w_t|w_{t-2},w_{t-1})$

### 语言模型（language model）
给出单词序列发生的概率——用一个序列评估一个单词序列发生的可能性。

假设m个单词顺序出现为$P(w_1,...,w_m)$，这个是多个事件一起发生，是联合概率的一种情况。

则可分解为：
$$
P(w_1,...,w_m) = P(w_m|w_1,...,w_{m-1})P(w_{m-1}|w_1,...,w_{m-2})...P(w_2|w_1)P(w_1)=\Pi_{t=1}^mP(w_t,...,w_{t-1})
$$
> 联合概率可以由后验概率的乘积表示。

> 补充：
> $P(A,B) = P(A|B)P(A)$
> A 和B共同发生的概率等于 B发生的概率和 B发生后A发生的概率的乘积。

上述公式的后验概率乘积，是以目标词左侧的全部单词为上下文（条件）时的概率。

目标：

求得$P(w_t|w_1,...,w_{t-1})$的概率。
—— 这种模型为条件语言模型（conditional language model).
求出后，就能求的语言模型的联合概率$P(w_1,...,w_m)$.


### 5.1.3 CBOW 模型用作语言模型？
只能用近似的方法，但上下文窗口太小则会失去准度，太大，CBOW模型还存在忽视了上下文单词顺序的问题。

> #马尔科夫模型
> 未来状态只依存于当前状态。
> 当某个事件的概率仅取决于前N个状态，称为“N阶马尔科夫链”。

RNN：可以处理任意长度的时序数据。但是
- 长距离梯度消失：大规模数据，单词之间的语义关系难以捕捉
word2vec优势：
- 小、快、好用

## 5.2 RNN

RNN 的特征就在于拥有环路（回路），可以使数据不断循环，保持对最新数据的更新。
![[fig5-6.png]]
- $x_t$：时刻t输入的数据，是向量。处理句子时就是单词的分布式。
- $(x_0, x_1, ..., x_t, ...)$：时序数据。
- $(h_0, h_1, ..., h_t,....)$：输出与输入的形式对应。


### 展开循环

![[fig5-8.png]]

- 每层的输入和前一层的输出一起进行计算
$$
h_t = tanh(h_{t-1}W_h+x_t W_x + b)
$$
	- $W_x$：将x转化为输出的h
	- $W_h$：将前一RNN层的输出转化为当前时刻输出的权重
	- b: 偏置
	- $h_t$和$x_t$：行向量
### RNN的反向传播 Backpropagation Through Time
![[fig5-10.png]]
- 基于时间的反向传播（BPTT)

学习长时序的数据的时候，若时序数据跨度增大，BPTT的计算开销也会成比例增大，梯度也会不稳定。

### Truncated BPTT
处理长时序的时候，习惯将网络连接截成适当长度。创建多个小型网络进行误差反向传播，也就是Truncated BPTT（截断的BPTT）。

- 正向传播被维持
- 反向传播被截断

![[fig5-11.png]]

- 之前的NN是按照mini-batch学习，数据随机选择
- RNN执行Truncated BPTT时，数据需按顺序输入

截断后的RNN每一块需要使用前一块的隐藏状态，所以需要等前一块的BPTT执行结束后再开始下一块的正向传播。

![[fig5-14.png]]

### Truncated BPTT 的mini-bach 学习
在数据开始的位置，需要在各个批次中进行“偏移“。

**偏移**

假设有长度为1000的数据，需要做mini-batch为2的输入。

则：
- 第一笔数据 0 ~ 9，第二笔数据为500~509
- 10 ~ 19， 510~ 519
- ……
如此平移各批次输入数据的开始位置，按顺序输入。此外，如果在按顺序输入数据的过程中遇到 了结尾，则需要设法返回头部

## 5.3 RNN的实现

假设目标神经网络接收长度为T的时序数据，输出各个时刻的隐藏状态T个。

将展开循环后的层视为一个层，处理T步的层称为”Time RNN层“。
接下来实现的流程：
1. 实现进行RNN单步处理的RNN类
2. 利用RNN类，实现T步处理的TimeRNN类

![[fig5-17.png]]

### 5.3.1 RNN层的视线

#RNN正向传播公式
$$
h_t = tanh(h_{t-1}W_h+x_t W_x + b)
$$
- mini-batch需要在行方向保存各样本数据
- 假设批大小为N，输入维度为D，隐藏状态向量维度为H。
![[fig5-18.png]]

RNN正向传播计算图：
![[fig5-19.png]]
- MatMul:矩阵成绩
- b加法会出发广播操作
- 省略Repeat节点

RNN反向传播计算图
![[fig5-20.png]]

```python
class RNN:
	def __init__(self, Wx, Wh, b):
		self.params: [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.cache = None
		
	def forward(self, x, h_prev):
		Wx, Wh, b = self.params
		t = np.dot(h_prev, Wh) + np.dot(x,Wx) + b
		h_next = np.tanh(t)
		
		self.cache = (x, h_prev, h_next)
		return h_next
		
	def backward(self, dh_next):
		Wx, Wh, b = self.params
		x, h_prev, h_next = self.cache
		
		dt = dh_next * (1- h_next ** 2)
		db = np.sum(dt, axis = 0)
		dWh = np.dot(h_prev.T, dt)
		dh_prev = np.dot(dt, Wh.T)
		dWx = np.dot(x.T, dt)
		dx = np.dot(dt, Wx.T)
		
		self.grads[0][...] = dWx
		self.grads[1][...] = dWh
		self.grads[2][...] = db
		
		return dx, dh_prev
	
```

### 5.3.2 Time RNN 层的实现
Time RNN层是T个RNN层连接起来的网络。

RNN层的隐藏状态h保存在成员变量，在块之间继承隐藏状态。
- 使用stateful参数控制是否继承隐藏状态
![[fig5-22.png]]

```python
class TimeRNN:
	def __init__(self, Wx, Wh, b, stateful = False):
		sefl.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.layers = None
		
		# h保存调用forward时最后一个RNN的隐藏状态。
		self.h , self.dh = None, None
		self.stateful = stateful
	def set_state(self, h):
		self.h = h
	
	def reset_state(self):
		self.h = None
		
	def forward(self, xs):
		Wx, Wh, b = self.params
		N, T , D = xs.shape
		D, H = Wx.shape
		
		self.layers = []
		hs = np.empty((N,T,H), dtype = 'f')
		
		if not self.stateful or self.h is None:
			self.h = np.zeros((N, H),dtype = 'f')
			
		for t in rnage(T):
			layer = RNN(*self.params)
			self.h = layer.forward(xs[:,t,:], self.h)
			hs[:,t,:] = self.h
			self.layers.append(layer)
		return hs 
```

- stateful 为True 时，无论时序数据多长，Time RNN 层的正向传播都可以不中断地进行
- stateful 为False 时，每次调用Time RNN层的forward() 时，第一个RNN层的隐藏状态都会被初始化为零矩阵（所有元素均为0的矩阵）