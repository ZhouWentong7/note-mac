LoRA (Low-Rank Adaptation)，一种LLM的微调方法……

> 选择LoRA是因为它通过向变换器层注入可训练的低秩矩阵，主要针对语言模型组件，从而高效适应大型预训练模型。这种方法显著减少了可训练参数的数量，使微调在计算上更轻松，同时保持了强有力的性能。（Kvasir-VQA-x1数据集中介绍）

【LoRA毁掉了我的大厂面试——从原理到大厂面试题再到实操微调，一个视频讲清楚】 https://www.bilibili.com/video/BV1wecpzpEWY/?share_source=copy_web&vd_source=fe1e50f3e19ee2c5f4cd32de1f11a963

以Qwen2.5-7B为例：
- 7B = 70亿参数 × 2 Bytes ≈ 14G （只是加载或推理）
- 若全参微调需要120G左右：
	- 激活值  ≈ 22G
	- 优化器状态 ≈ 56G
	- 梯度 ≈ 28G
	- 模型权重 ≈ 14G

> 1 Bytes(字节) = 8位2进制数
> 精度：FP16 = 2 Bytes/参数

## 理解Rank和微调

**秩（Rank）**
矩阵中线性无关的行或列的数量，真正包含独立信息的维度。且行秩=列秩

 而研究发现，大模型的权重$w$(满秩)在微调时，微调的变化$\Delta w$ 具有低秩性。
所以优化，从微调整个$w$矩阵，转变为只微调变化的低秩的$\Delta w$，就可以达到接近全参微调的效果。

 $\Delta w$可以用两个更小的矩阵进行表示
 ![[04-LoRA.png]]
## LoRA核心过程

![[04-LoRA-detail.png]]

$$ \begin{aligned} 
Q &= Q_1 + Q_2 \\ 
&= W_qX+\Delta W_qX \\ 
&=W_qX + ABX \\
&= (W_q +AB)X \\
&= W_q^{Merged}X\end{aligned} $$

- A : LoRA 模块降维矩阵，使用随机数初始化
- B: LoRA模块升维矩阵，使用全0矩阵初始化
- x输入到模型后，原参数冻结（不参与反向传播），仅在LoRA模块的两个小矩阵进行参数更新
- 最后原参数和更新后的内容相加得到更新后的结果

>[!note] 使用LoRA训练的模型推理时是否会造成额外延迟？
>答：不会，在LoRA训练结束后，更新的$\Delta W$直接与原参数相加，不会改变模型结构。


## LoRA在哪里使用
 