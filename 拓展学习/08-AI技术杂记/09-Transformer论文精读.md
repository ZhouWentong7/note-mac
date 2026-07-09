---
tags:
  - NLP
  - Transformer
---


> [!info] 学习来源
> [【《Attention is all you need》论文解读及Transformer架构详细介绍】](https://www.bilibili.com/video/BV1xoJwzDESD/?share_source=copy_web&vd_source=fe1e50f3e19ee2c5f4cd32de1f11a963)


**阅读思路**
- 以往的模型存在的问题/平静——结构的哪些缺陷导致了性能的有限
	- Intro和BG
- 论文提出的注意力结构解决了什么问题，为什么可以解决
	- 核心：Transformer架构和Self-Attention机制
	- 对应：Model ，Why Self-Attention


# Abstract

Transformer之前的sequence transduction models（序列传导模型）：
- 复杂的循环
- 卷积为基础：RNN/CNN
- encoder-decoder结构
- 表现好的模型大多接住了Attention机制链接或增强

Transformer结构的创新：
- 完全摒弃RNN/CNN的结构
- 仍使用encoder-decoder
- 完全基于注意力

# Introduction

Feedforward Neural Network（FNN)
![[09-FNN.png]]
**不适合做序列转导的原因**
在做Embedding的时候，对词向量有两种处理方式，但对于FNN来说都损失了一定的信息：
- 平均：丢失词语顺序信息
- 拼接：
	- FNN需要固定维度的输入，但句子通常长度不同，FNN处理效率低下
	- 同样失去顺序信息

Recurrent Neural Network（RNN)
![[09-RNN.png]]

![[09-RNN2.png]]
解决的问题：
- 保留顺序信息： RNN按照时间顺序接收Token的输入
- 保留上下文以来：RNN有记忆机制
- 支持不定长输入

潜在问题：
- 难以处理输入输出不等长问题
- 梯度消失或爆炸（me）

**Encoder-Decoder**
简单理解：将RNN的上下两个部分拆来来做，encoder只管输入，decoder只管输出

![[09-Encoder-Decoder.png]]

- C为h4的的输出，涵盖整个文本语义信息

潜在问题：
- 远距离遗忘：处理长序列，可能以往较远的信息
- 无重要性区别：序列中各个元素的贡献度和重要程度不同

**注意力机制**

![[09-AttMechanism.png]]
解决：
- 长序列遗忘问题：对序列中元素加上不同的权重
- 解决不同时间步对当前时刻输出的“重要性”问题：所有时间步的输入在计算当前时刻输出时被同等对待，忽略不同时间不对当前时刻输出的重要些可能存在差异

> [!attention] 潜在问题：只能串行计算，也没有充分利用GPU的并行能力。

Introduction中介绍了RNN系列的研究，并提到注意力机制通常是和RNN一起使用。

# Background
**关于CNN**
为解决顺序计算问题，引入CNN以支持并行，但难以捕捉远距离关系。

**介绍注意力机制**
- 自注意力机制
- Memory Network的递归注意力机制

**本文**
- Transformer是收个完全基于自注意力，无需使用RNN/CNN 的序列建模模型

# Model Arch

token: 512

首先，如何解决的串行计算问题。

串行计算主要发生在以往模型的encoder部分，所有的后面的token必须等待前面的输入完成才能参与计算。

**Encoder**

但是对于翻译任务，待翻译的文本全局已知，需要设计一种方法，让排在后面的token可以同时接收到前后文的影响，参与计算。

这种计算涉及到两种信息：
- 该序列中所有token的向量
- 每个token在序列中的位置信息

由于翻译文本同样是已知的答案，且，对于已经翻译完的部分来说，这个部分的答案对模型是可见的——后续的部分理论不可见，就使用mask遮住，不进入decoder进行计算。

所以在decoder处，模型顺序生成翻译后的文本，且decoder接收每次生成后的答案作为输入——decoder的输入来自两个地方：encoder输出和自己上一步的输出



![[09-TransformerArchitecture..png]]
关键模块：
- （Masked）Multi-Head Attention
- Feed Forward

参数：
- $N\times$：可以任意堆叠，论文N=6

- Transformer的encoder和decoder输入输出维度不会发生变化
---

**计算流程**

1. Inputs DO Embedding GET tensor
2. Inputs 的各个单 DO position encoding GET pe
	1. 使用正弦和余弦函数编码：

| $P​E(p​o​s,2​i)=s​i​n​(p​o​s/100002​i/dmodel)$   |
| ------------------------------------------------ |
| $P​E(p​o​s,2​i+1)=c​o​s​(p​o​s/100002​i/dmodel)$ |

3.  tensor = pe + tensor
进入encoder
4. 每个token得到QKV三个矩阵，送入多头注意力
5. 多头的输出，通过残差和原数据融合，再过Norm，让数值趋于稳定 GET a
6. a经过 Feed Forward,它由两个线性变换组成，中间夹有 ReLU 激活函数，学习更复杂的特征
   $FFN​(x)=max⁡(0,x​W1+b1)​W2+b2$
7. 再次经过残差链接和归一化（矩阵维度不发生变化）

进入解码器：模型预测下一个输出
8. 带有掩码的多头注意力对**过去的输出**进行自注意力的计算 GET b
9. b经过残差和Norm
10. 编码器的输出作为K，V，刚刚的b作为Q输入到多头注意力（Q对KV问道：我的下一个输出应该是什么）
11.  Feedforward+Add+Norm

找到词语：
经过Linear和Softmax

> - Embedding：将词语映射到向量空间
> - Transformer这种大模型，embedding模型和整个模型一起参与训练
> - 位置信息的加入：让token在语义空间的含义随着上下文进行偏移，一些词语在不同语境下的含义略有差异，位置信息就可以利用上下文得到的情报对原词语的语义进行属于该文章的微调

> [!note] 为何位置编码如此复杂，而不直接使用简单的编号？
> BERT中使用的编号方式对位置进行的编码，缺点是这种标量所含的信息量太少，模型难以理解更抽象的关系的远近；注意力依靠向量计算。
> 将位置信息抽象到高维空间（高维向量），使其包含更丰富的信息，让模型感受到某种相对关系……



> [!tip] Attention
> Q：提出问题 (N,512)
> K：是否存在这个问题的对应答案(值) (N,512)
> V：这个值 (N,512)
> 每个QKV都有一个自己的权重矩阵(512,512)。经过了和权重的计算后，Q、K、V的尺寸仍为(N,512)，这个部分送入attention进行计算。
> $$Attention​(Q,K,V)=softmax​(\frac{Q​K^T}{\sqrt{d_k}})V​$$
> -  $QK^T$ ：计算Q和K的点积，也就是相似度的计算
> - $\sqrt{d_k}$:缩放
> - softmax：QK相似度的概率权重，K对应的token对Q对应的token的影响的重要程度，这个权重也有自己对自己的。
> - 和V进行乘法：V获得加权平均。

> [!important] MultHead
> 修改1：权重矩阵改为使用尺寸（64，512）
> 修改2：单组QKV的权重矩阵→八组（多组，多头）
> 所以经过了权重的QKV尺寸变为了（N，64），这样的结果也是八组。
> 追加1：来自八个头的结果进行拼接（$8\times 64=512$)，又得到尺寸为（N，512）的输出（看似回到了单头的尺寸）
> 追加2：将拼接后的结果经过一个线性层融合，得到最后的(N,512)的输出
> > 为什么降维到64？又做融合？为什么多头？
> > 答：每个头的Q询问了不同的信息，让模型学到更丰富而详细的表示。做融合的原因是，再进行一次映射。为什么是8个64维呢？在保持计算量不变的情况下，有多头优势。

# 为什么Self-Attention

一：自注意力的机制复杂度低于RNN、CNN等
二：有并行计算的能力，解决了串行等待的问题
三：模型内部学习长距离依赖能力更强



 ---
 