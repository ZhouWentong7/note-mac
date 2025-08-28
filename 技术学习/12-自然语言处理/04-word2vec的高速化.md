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
#embedding

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

在反向传播时，从上一层（输出侧的层）传过来的梯度将原样传给下一层
![[fig4-4.png]]

但需要注意的是，若idx在一个句子中多次出现，那反向传播会发生冲突，可以采取加法的方法进行赋值：

```python
    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None
```

## 负采样（  negative sampling）
#负采样
解决中间层之后的处理，即矩阵乘积和Softmax层的计算。

使用**负采样**替代Softmax，保持计算的快捷和稳定。

**耗时的地方**
- 中间层神经元与$W_{out}$的乘积计算
- Softmax的计算

若词典有100w个，那softmax的分母部分也要进行100w次exp的计算。这个部分与词汇量成正比。

负采样的思想来自于使用二分类拟合多分类。

输出层从预测是否为“这个单词”，转化为具有和词汇量同等数量的神经元。

> 其实没太看懂解析

```python
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)

        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh
```




•	负采样（Negative Sampling） 方法里，会把多分类 softmax 换成 一正多负的二分类问题：
	•	正样本：目标词出现，label = 1
	•	负样本：随机选词，label = 0
	•	用 sigmoid 点积代替 softmax
	•	EmbeddingDot 就是计算 h · w_target，配合 sigmoid loss 就可以直接做二分类，不用计算整个 softmax。

解释：
负采样的核心思想：
	•	每次只看 目标词 + 少量随机“负样本”，不计算整个词表。
	•	将原本的多分类问题 转化为多次二分类问题：
	•	正样本（target word）：label = 1
	•	负样本（random word）：label = 0

负采样：需要做的事情
- 对正例，Sigmoid输出接近1
- 对负例，Sigmoid输出接近0。
- 选择少量的负例（5或10个）采样，将这些结果的loss架起来作为最终的损失

### 负采样方法

- 基于语料库的统计进行采样。
	- 经常出现的单词更容易被抽到
	- 使用概率分布
Code:
`UnigramSampler` 用于根据词语的出现频率来生成负样本，负采样是为了提高词嵌入训练效率而采用的技术，避免了对所有词汇进行更新。
```python
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
    """
    corpus: 语料库，是一个包含词语 ID 的列表
    - `power`：用于调整概率分布的幂次，通常设置为 0.75（Word2Vec 中的常用设置）
	- `sample_size`：每个目标词需要生成的负样本数量
    """
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
		
		# 
		# 统计每个词语 ID 在语料库中出现的次数
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
		# 计算词汇表大小
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

		# 降低高频词的采样概率
        self.word_p = np.power(self.word_p, power)
        # 归一化概率分布，使所有词的概率和为 1
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
		"""
		区分了 CPU 和 GPU 两种计算场景，GPU 场景下优先考虑计算速度.GPU这里存在一个小问题：负样本中可能包含目标词本身，但在 GPU 计算中为了速度牺牲了这一点精度.
		"""
        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # 在用GPU(cupy）计算时，优先速度
            # 有时目标词存在于负例中
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample
```


### 负采样的实现

略

### word2vec的应用例
- 单词分布式的重要原因：迁移学习
- 使用训练好的分布式完成下游任务：
	- 文本分类
	- 文本聚类
	- 词性标注
	- 情感分析
- 好的分布式在更多NLP的任务中有很好的效果


### 词向量的评价方法
如何评价分布式的优劣？

评价指标：相似度、类推问题

- 相似度评价：人工创建
- 类推问题：
	- king: queen = man :?

![[fig4-23.png]]
- 模型不同精度不同
- 语料库越大越好
- 单词向量的维度需适中

