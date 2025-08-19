本章将重点放在word2vec的加速上：
- 引入名为Embedding层的新层
- 引入名为Negative Sampling的新损失函数
- 将在PTB数据集（一个大小比较实用的语料库）上进行，并实际评估所获得的单词的分布式表示的优劣

---

## 分析
之前的实现，会出现计算瓶颈的地方：
- 输入层的 one-hot表示和权重矩阵 $W_{in}$的乘积（4.1节解决）
- 中间层和权重矩阵 $W_{out}$的乘积以及 Softmax层的计算（4.2节解决


### 使用嵌入层（Embedding）
随着词汇量的增加，one-hot表示的向量大小也会增加

也会导致$W_{in}$增量。

因此引入`Embedding Layer`以解决这个问题。


![[fig4-3.png]]

使用one-hot之后，其实这里的乘法只是从$W_{in}$里面取出特定的行。那做这个完整乘法的必要性就没那么大了。

我们创建一个从权重参数中抽取“**单词ID对应行（向量）**”的，这里我们称之为**Embedding层**。

>[!note] 名词解释
> 单词的密集向量表示：嵌入（word embedding）或者单词的分布式表示（distributed representation）。
> 基 于 计 数 的 方 法 获 得 的 单 词 向 量 称 为 distributional  representation。
> 使用神经网络的基于推理的方法获得的单词向量称为distributed representation。
> 只是中文里都翻译为“分布式表示”。

### Embedding层的实现

```python
class Embedding:

	def __init__(self, W):
	
		self.params = [W]
		
		self.grads = [np.zeros_like(W)]
		
		self.idx = None

  

	def forward(self, idx):
	
		W, = self.params
		
		self.idx = idx
		
		out = W[idx]
		
		return out

  

	def backward(self, dout):
	
		dW, = self.grads
		
		dW[...] = 0
		
		if GPU:
		
			np.scatter_add(dW, self.idx, dout)
		
		else:
		
			np.add.at(dW, self.idx, dout)
		
		return None
```

Embedding层的正向传播只是从权重矩阵W 中提取特定的行，并将该特定行的神经元原样传给下一层。

在反 向传播时，从上一层（输出侧的层）传过来的梯度将原样传给下一层
![[fig4-4.png]]

