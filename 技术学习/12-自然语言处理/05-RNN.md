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
反传播计算图：
![[fig5-23.png]]

```python
import numpy as np

class TimeRNN:
    """
    时间序列RNN类，用于处理序列数据的循环神经网络实现
    能够处理整个时间序列的输入，并维护跨时间步的隐藏状态
    """
    def __init__(self, Wx, Wh, b, stateful=False):
        """
        初始化TimeRNN层
        
        参数:
            Wx: 输入到隐藏层的权重矩阵，形状为[输入维度, 隐藏层维度]
            Wh: 隐藏层到隐藏层的权重矩阵，形状为[隐藏层维度, 隐藏层维度]
            b: 隐藏层的偏置项，形状为[隐藏层维度]
            stateful: 布尔值，表示是否保持状态。若为True，会保留上一次的隐藏状态
                      用于处理跨批次的序列数据（如长文本分割成多个批次）
        """
        # 存储模型参数：输入权重、隐藏层权重和偏置
        self.params = [Wx, Wh, b]
        # 初始化梯度存储数组，形状与对应参数相同
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        # 存储每个时间步的RNN层实例
        self.layers = None
        
        # 隐藏状态变量：h存储前向传播的最终隐藏状态，dh存储反向传播的梯度
        self.h, self.dh = None, None
        # 状态保持标志
        self.stateful = stateful

    def set_state(self, h):
        """
        手动设置隐藏状态，用于在需要时恢复或指定初始状态
        
        参数:
            h: 要设置的隐藏状态数组，形状为[批量大小, 隐藏层维度]
        """
        self.h = h

    def reset_state(self):
        """重置隐藏状态为None，在下一次前向传播时会重新初始化为全零"""
        self.h = None

    def forward(self, xs):
        """
        前向传播：处理整个时间序列的输入数据
        
        参数:
            xs: 时间序列输入数据，形状为[批量大小(N), 时间步(T), 输入维度(D)]
        
        返回:
            hs: 所有时间步的隐藏状态，形状为[批量大小(N), 时间步(T), 隐藏层维度(H)]
        """
        # 解包参数
        Wx, Wh, b = self.params
        # 获取输入数据的形状信息：N=批量大小, T=时间步数, D=输入维度
        N, T, D = xs.shape
        # 获取权重矩阵的形状信息：D=输入维度, H=隐藏层维度
        D, H = Wx.shape
        
        # 初始化存储每个时间步RNN层的列表
        self.layers = []
        # 初始化存储所有时间步隐藏状态的数组
        hs = np.empty((N, T, H), dtype='f')
        
        # 如果不是保持状态模式，或者是第一次前向传播，则初始化隐藏状态为全零
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        
        # 遍历每个时间步，进行前向传播
        for t in range(T):
            # 创建当前时间步的RNN层（假设已实现基础RNN层）
            layer = RNN(*self.params)
            # 计算当前时间步的隐藏状态：输入为当前时间步的输入和上一时间步的隐藏状态
            self.h = layer.forward(xs[:, t, :], self.h)
            # 保存当前时间步的隐藏状态
            hs[:, t, :] = self.h
            # 保存当前时间步的RNN层实例，用于后续反向传播
            self.layers.append(layer)
        
        # 返回所有时间步的隐藏状态
        return hs

    def backward(self, dhs):
        """
        反向传播：计算参数的梯度和输入数据的梯度
        
        参数:
            dhs: 损失函数对所有时间步隐藏状态的梯度，形状为[批量大小(N), 时间步(T), 隐藏层维度(H)]
        
        返回:
            dxs: 损失函数对输入数据的梯度，形状为[批量大小(N), 时间步(T), 输入维度(D)]
        """
        # 解包参数
        Wx, Wh, b = self.params
        # 获取梯度输入的形状信息：N=批量大小, T=时间步数, H=隐藏层维度
        N, T, H = dhs.shape
        # 获取权重矩阵的形状信息：D=输入维度, H=隐藏层维度
        D, H = Wx.shape
        
        # 初始化输入数据的梯度存储数组
        dxs = np.empty((N, T, D), dtype='f')
        # 初始化隐藏状态的梯度（用于时间步之间的梯度传递）
        dh = 0
        # 初始化参数梯度的累加器
        grads = [0, 0, 0]
        
        # 反向遍历时间步（从最后一步到第一步）
        for t in reversed(range(T)):
            # 获取当前时间步的RNN层实例
            layer = self.layers[t]
            # 计算当前时间步的梯度：当前时间步的隐藏状态梯度 + 后续时间步传递过来的梯度
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            # 保存当前时间步输入数据的梯度
            dxs[:, t, :] = dx
            
            # 累加每个时间步的参数梯度
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        # 将累加的梯度赋值给类的梯度属性
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        
        # 保存最终的隐藏状态梯度（可用于后续连接的层）
        self.dh = dh
        
        # 返回输入数据的梯度
        return dxs
			
```
> 这段代码实现了一个能够处理时间序列的 RNN 层，支持状态保持功能，这在处理长序列数据（需要分成多个批次）时非常有用。前向传播时按时间顺序处理输入，反向传播时则逆序处理以正确计算梯度。


- stateful 为True 时，无论时序数据多长，Time RNN 层的正向传播都可以不中断地进行
- stateful 为False 时，每次调用Time RNN层的forward() 时，第一个RNN层的隐藏状态都会被初始化为零矩阵（所有元素均为0的矩阵）
- xs：T个时序列数据（N，T，D)
- dhs：上游输出层传来的梯度
- dxs：流向下游的梯度
	- 因为是Truncated BPTT，所以无需流向上一时刻的反向传播
	- 但存储在成员变量dh中，第七章使用

## 5.4 处理时序列数据的层的实现

- 基于RNN的语言模型：RNNLM

### 5.4.1 RNNLM全貌
![[fig5-25.png]]

- Embedding：单词ID转化为单词分布式表示
- 词向量传入RNN
- RNN向下一层输出隐藏状态，同时向下一时刻输出隐藏状态
- 向下一层输出的隐藏状态经过Affine层，穿到Softmax

![[fig5-26.png]]

### 5.4.2 Time层实现
将Embedding和Affine也改造为可以处理时序列的形式。

![[fig5-27.png]]

```python
import numpy as np

class TimeEmbedding:
    """
    时间序列嵌入层（Time Embedding Layer）
    用于处理时序数据的嵌入操作，将每个时间步的离散输入（如单词索引）转换为连续向量表示
    相当于对序列中的每个时间步分别应用嵌入层，并整合结果
    """
    def __init__(self, W):
        """
        初始化时间序列嵌入层
        
        参数:
            W: 嵌入权重矩阵，形状为[词汇表大小(V), 嵌入维度(D)]
               W[i]表示第i个离散符号对应的嵌入向量
        """
        # 存储模型参数（嵌入权重矩阵）
        self.params = [W]
        # 初始化梯度存储数组，形状与嵌入权重矩阵相同
        self.grads = [np.zeros_like(W)]
        # 存储每个时间步的嵌入层实例
        self.layers = None
        # 直接保存嵌入权重矩阵的引用，方便访问
        self.W = W

    def forward(self, xs):
        """
        前向传播：将时序离散输入转换为时序嵌入向量
        
        参数:
            xs: 时序离散输入数据，形状为[批量大小(N), 时间步(T)]
                其中每个元素是一个整数索引（如单词在词汇表中的索引）
        
        返回:
            out: 时序嵌入向量，形状为[批量大小(N), 时间步(T), 嵌入维度(D)]
        """
        # 获取输入数据的形状信息：N=批量大小, T=时间步数
        N, T = xs.shape
        # 获取嵌入权重矩阵的形状信息：V=词汇表大小, D=嵌入维度
        V, D = self.W.shape

        # 初始化输出数组，存储所有时间步的嵌入结果
        out = np.empty((N, T, D), dtype='f')
        # 初始化存储每个时间步嵌入层的列表
        self.layers = []

        # 遍历每个时间步，应用嵌入操作
        for t in range(T):
            # 为当前时间步创建嵌入层实例（假设已实现基础Embedding层）
            layer = Embedding(self.W)
            # 对当前时间步的所有样本进行嵌入操作，并存储结果
            # xs[:, t]表示取所有样本在第t时间步的输入
            out[:, t, :] = layer.forward(xs[:, t])
            # 保存当前时间步的嵌入层实例，用于后续反向传播
            self.layers.append(layer)

        # 返回所有时间步的嵌入向量
        return out

    def backward(self, dout):
        """
        反向传播：计算嵌入权重矩阵的梯度
        
        参数:
            dout: 上游传来的梯度，形状为[批量大小(N), 时间步(T), 嵌入维度(D)]
        
        返回:
            None: 嵌入层没有输入梯度需要返回（输入是离散索引，通常不需要梯度）
        """
        # 获取上游梯度的形状信息：N=批量大小, T=时间步数, D=嵌入维度
        N, T, D = dout.shape

        # 初始化嵌入权重矩阵的梯度累加器
        grad = 0
        # 遍历每个时间步，计算并累加梯度
        for t in range(T):
            # 获取当前时间步的嵌入层实例
            layer = self.layers[t]
            # 对当前时间步的上游梯度进行反向传播计算
            layer.backward(dout[:, t, :])
            # 累加当前时间步的嵌入权重梯度
            grad += layer.grads[0]

        # 将累加的梯度赋值给类的梯度属性
        self.grads[0][...] = grad
        # 嵌入层输入是离散索引，不需要计算输入梯度，返回None
        return None


class TimeAffine:
    """
    时间序列仿射层（Time Affine Layer）
    用于对时序数据应用仿射变换（线性变换+偏置），相当于将整个时序数据展平后应用仿射变换
    再将结果恢复为时序数据形状
    """
    def __init__(self, W, b):
        """
        初始化时间序列仿射层
        
        参数:
            W: 权重矩阵，形状为[输入特征维度(D), 输出特征维度(M)]
            b: 偏置项，形状为[输出特征维度(M)]
        """
        # 存储模型参数（权重矩阵和偏置项）
        self.params = [W, b]
        # 初始化梯度存储数组，形状与对应参数相同
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        # 存储前向传播的输入数据，用于反向传播计算
        self.x = None

    def forward(self, x):
        """
        前向传播：对时序数据应用仿射变换
        
        参数:
            x: 时序输入数据，形状为[批量大小(N), 时间步(T), 输入特征维度(D)]
        
        返回:
            out: 仿射变换后的时序数据，形状为[批量大小(N), 时间步(T), 输出特征维度(M)]
        """
        # 获取输入数据的形状信息：N=批量大小, T=时间步数, D=输入特征维度
        N, T, D = x.shape
        # 解包参数：权重矩阵W和偏置项b
        W, b = self.params

        # 将时序数据展平为二维数组：[N*T, D]
        # 相当于将所有时间步的样本合并成一个大批次
        rx = x.reshape(N*T, -1)
        # 应用仿射变换：y = x·W + b
        out = np.dot(rx, W) + b
        # 保存输入数据的引用，用于反向传播
        self.x = x
        # 将结果重塑为时序数据形状：[N, T, M]（M为输出特征维度）
        return out.reshape(N, T, -1)

    def backward(self, dout):
        """
        反向传播：计算权重、偏置和输入数据的梯度
        
        参数:
            dout: 上游传来的梯度，形状为[批量大小(N), 时间步(T), 输出特征维度(M)]
        
        返回:
            dx: 输入数据的梯度，形状为[批量大小(N), 时间步(T), 输入特征维度(D)]
        """
        # 获取前向传播时保存的输入数据
        x = self.x
        # 获取输入数据的形状信息：N=批量大小, T=时间步数, D=输入特征维度
        N, T, D = x.shape
        # 解包参数：权重矩阵W和偏置项b
        W, b = self.params

        # 将上游梯度展平为二维数组：[N*T, M]
        dout = dout.reshape(N*T, -1)
        # 将输入数据展平为二维数组：[N*T, D]（与前向传播保持一致）
        rx = x.reshape(N*T, -1)

        # 计算偏置项的梯度：对所有样本求和
        db = np.sum(dout, axis=0)
        # 计算权重矩阵的梯度：输入的转置 · 上游梯度
        dW = np.dot(rx.T, dout)
        # 计算输入数据的梯度：上游梯度 · 权重矩阵的转置
        dx = np.dot(dout, W.T)
        # 将输入数据的梯度重塑为时序数据形状：[N, T, D]
        dx = dx.reshape(*x.shape)

        # 将计算得到的梯度赋值给类的梯度属性
        self.grads[0][...] = dW
        self.grads[1][...] = db

        # 返回输入数据的梯度
        return dx
```

同样的，损失也修改为时序列：
![[fig5-29.png]]

$$
L = \frac{1}{T}(L_0 + L_1 + ... + L_{T-1})
$$

## 5.5 RNNLM 的学习和评价

SimpleRnnlm堆叠了4个Time层神经网络

![[fig5-30.png]]

```python
import numpy as np

class SimpleRnnlm:
    """
    简单的循环神经网络语言模型（RNN Language Model）
    用于处理语言建模任务，通过前一个词预测下一个词的概率分布
    模型结构：嵌入层(Embedding) -> 循环神经网络层(RNN) -> 仿射层(Affine) -> softmax损失层
    """
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        初始化简单RNN语言模型
        
        参数:
            vocab_size: 词汇表大小（离散符号的总数）
            wordvec_size: 词向量（嵌入向量）的维度
            hidden_size: RNN隐藏层的维度
        """
        # 简化变量名：V=词汇表大小, D=词向量维度, H=隐藏层维度
        V, D, H = vocab_size, wordvec_size, hidden_size
        # 引入numpy的随机正态分布生成器
        rn = np.random.randn

        # 初始化各层权重参数，并进行适当的缩放以稳定训练

        # 嵌入层权重：形状[V, D]，除以100进行缩放，避免初始值过大
        embed_W = (rn(V, D) / 100).astype('f')
        
        # RNN层输入权重：形状[D, H]，使用Xavier初始化（除以输入维度的平方根）
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        
        # RNN层隐藏状态权重：形状[H, H]，同样使用Xavier初始化
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        
        # RNN层偏置：形状[H]，初始化为0
        rnn_b = np.zeros(H).astype('f')
        
        # 仿射层权重：形状[H, V]，使用Xavier初始化
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        
        # 仿射层偏置：形状[V]，初始化为0
        affine_b = np.zeros(V).astype('f')

        # 构建模型的层结构
        self.layers = [
            TimeEmbedding(embed_W),               # 时间序列嵌入层：将词索引转换为词向量
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),  # 时间序列RNN层：处理序列信息，保持状态
            TimeAffine(affine_W, affine_b)        # 时间序列仿射层：将RNN输出映射到词汇表空间
        ]
        
        # 时间序列的softmax损失层：计算序列预测的损失
        self.loss_layer = TimeSoftmaxWithLoss()
        
        # 保存RNN层的引用，方便后续操作（如重置状态）
        self.rnn_layer = self.layers[1]

        # 整理所有层的参数和梯度到统一的列表中，方便参数更新
        
        # 模型所有可学习参数的列表
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params  # 合并各层的参数
            self.grads += layer.grads    # 合并各层的梯度

    def forward(self, xs, ts):
        """
        前向传播：计算语言模型在输入序列上的损失
        
        参数:
            xs: 输入序列（词索引），形状为[批量大小, 时间步]
                例如：[[w1, w2, w3], [w4, w5, w6]]表示两个样本，每个样本3个时间步
            ts: 目标序列（词索引），形状与xs相同，每个位置是对应输入的下一个词
                例如：对于输入[w1, w2, w3]，目标可能是[w2, w3, w4]
        
        返回:
            loss: 模型在当前批次上的平均损失值
        """
        # 依次通过各层进行前向传播
        for layer in self.layers:
            xs = layer.forward(xs)  # 每一层的输出作为下一层的输入
        
        # 通过损失层计算预测损失（输入是最后一层的输出，目标是ts）
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        """
        反向传播：计算所有参数的梯度
        
        参数:
            dout: 初始梯度，默认为1（因为损失对自身的梯度是1）
        
        返回:
            dout: 反向传播到最底层的梯度（对于语言模型通常不使用）
        """
        # 从损失层开始反向传播
        dout = self.loss_layer.backward(dout)
        
        # 逆序通过各层进行反向传播（与前向传播顺序相反）
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        
        return dout

    def reset_state(self):
        """
        重置RNN层的隐藏状态
        在处理新的序列（如每个epoch开始或新的文档）时调用，避免状态污染
        """
        self.rnn_layer.reset_state()
```

- RNN层和Affine层使用了“Xavier初始值”。

### 5.5.2 语言模型的评估

- **困惑度（perplexity）**：概率 的倒数，假设下个单词的是say的概率为0.8，则困惑度为1.25；困惑度越小越好
	- 分叉度：1.25可截石位下一个候选单词的个数在一个左右。


多个单词的情况下：
$$
L = -\frac{1}{N}\sum_n \sum_k t_{nk}\log y_{nk}
$$
$$困惑度 = e^L$$
- $t_n$:one-hot向量表示的正确标签
- $t_{nk}$：第n个数据的第k个值
- $y_{nk}$:概率分布（softmax的输出）。
- 