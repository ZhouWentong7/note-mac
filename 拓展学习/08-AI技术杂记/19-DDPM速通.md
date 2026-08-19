# 1 数学基础
1. 向量场

2. ODE 常微分方程
$$ \frac{dx(t)}{dt} = v(x(t),t) $$
3. 分布采样

- 联合分布
- 条件分布
- 边缘分布

4. 神经网络



# 2 什么是生成模型：以VAE为例

前提假设：我们采样到的数据符合某一概率分布。

这个概率分布对我们而言是未知的，用参数$\theta$来表示限制该分布的参数。

求解的目标就是希望得到一个尽量高的概率密度，让采样的点落在这个概率密度下的可能性尽量大。

$$
\max_\theta \prod_\theta^N p_\theta(x^{(i)})=\max_\theta \sum_{i=1}^N\log p_\theta(x^{(i)})
$$

VAE（Variational Auto-Encoder）是Diffusion之前的模型，其思想仍然是Diffusion模型的基础。

VAE的核心设定是，我们观测到的数据背后藏着一个没有被直接更观测到的变量$z$(隐变量，latent variable）,所有可能的z所在的空间被称作隐空间（latent space）

在VAE的设计中，数据的生成步骤：
1. 从先验分布中采样隐变量
2. 根据z采样数据

难点：
1. inference：给定一个样本z，如何得到对应的z，这里需要计算后验分布
$$p_{\theta}(z | x) = \frac{p_{\theta}(x, z)}{p_{\theta}(x)} = \frac{p_{\theta}(x | z)p_{\theta}(z)}{\int p_{\theta}(x, z)dz}$$

但是，分母中这个积分难以计算，但这个后验分布是想办法近似得到的。
2. learning：学习过程需要调整$\theta$，最大化$\log p_\theta(x)$而这个$p_\theta(x)$也包含刚刚哪个无法计算的积分。

这里采用的方法就是KL散度。
> [!info]- Tips：KL 散度是什么？
> **KL 散度（Kullback-Leibler Divergence）**是一个用来衡量两个概率分布差异的指标。对于概率密度函数 $p(\mathbf{x})$ 和 $q(\mathbf{x})$，它们的 KL 散度定义为：
> 
> $$D_{KL}(p \parallel q) = \mathbb{E}_{\mathbf{x} \sim p} \left[ \log \frac{p(\mathbf{x})}{q(\mathbf{x})} \right]$$
> 
> > [!note] 说明
> > 期望下面的角标 $\mathbf{x} \sim p$ 表示期望计算对象是 $\mathbf{x}$（即对谁取期望），且 $\mathbf{x}$ 是从分布 $p(\mathbf{x})$ 中采样得到的样本。
> 
> 若为连续概率密度函数，则展开为积分形式：
> 
> $$D_{KL}(p \parallel q) = \int p(\mathbf{x}) \log \frac{p(\mathbf{x})}{q(\mathbf{x})} d\mathbf{x}$$
> 
> #### 1. 如何理解 KL 散度？
> 直觉上，$p(\mathbf{x})$ 和 $q(\mathbf{x})$ 的 KL 散度，就是对于从 $p(\mathbf{x})$ 采样的所有可能的点，看它们落在 $p(\mathbf{x})$ 和 $q(\mathbf{x})$ 的概率密度之差的期望。**KL 散度越大，说明两个分布差异越大**。
> 
> > [!warning] 注意
> > KL 散度具有非对称性，即 $D_{KL}(p \parallel q) \neq D_{KL}(q \parallel p)$。
> 
> #### 2. 高斯分布间的 KL 散度闭形式
> 若两分布均为高斯分布：$p = \mathcal{N}(\boldsymbol{\mu}_p, \boldsymbol{\Sigma}_p)$， $q = \mathcal{N}(\boldsymbol{\mu}_q, \boldsymbol{\Sigma}_q)$，则它们之间的 KL 散度可以直接计算，无需进行数值积分：
> 
> $$D_{KL}(p \parallel q) = \frac{1}{2} \left[ \log \frac{\det \boldsymbol{\Sigma}_q}{\det \boldsymbol{\Sigma}_p} - d + \mathrm{tr}(\boldsymbol{\Sigma}_q^{-1} \boldsymbol{\Sigma}_p) + (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p)^\top \boldsymbol{\Sigma}_q^{-1} (\boldsymbol{\mu}_q - \boldsymbol{\mu}_p) \right]$$
> 
> 
> 

**推导 ELBO 与 $\log p_\theta(\mathbf{x})$ 的关系**

我们先看近似后验 $q_\phi(\mathbf{z} \mid \mathbf{x})$ 和真实后验 $p_\theta(\mathbf{z} \mid \mathbf{x})$ 之间的 KL 散度：

$$D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x})) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x})}{p_\theta(\mathbf{z} \mid \mathbf{x})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi(\mathbf{z} \mid \mathbf{x}) p_\theta(\mathbf{x})}{p_\theta(\mathbf{x}, \mathbf{z})} \right]$$

$$= \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi(\mathbf{z} \mid \mathbf{x}) - \log p_\theta(\mathbf{x}, \mathbf{z}) + \log p_\theta(\mathbf{x}) \right]$$

$$= \log p_\theta(\mathbf{x}) - \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}) \right]$$

将后面的期望定义为**证据下界（ELBO, Evidence Lower Bound）**：

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})} \left[ \log p_\theta(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z} \mid \mathbf{x}) \right]$$

整理得到恒等式：

$$\log p_\theta(\mathbf{x}) = \mathcal{L}(\theta, \phi; \mathbf{x}) + D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z} \mid \mathbf{x}))$$

由于 KL 散度非负（$D_{\mathrm{KL}} \ge 0$），因此：

$$\log p_\theta(\mathbf{x}) \ge \mathcal{L}(\theta, \phi; \mathbf{x})$$

把联合分布 $p_\theta(\mathbf{x}, \mathbf{z}) = p_\theta(\mathbf{x} \mid \mathbf{z})p_\theta(\mathbf{z})$ 代入 ELBO，它还可以变成另一个更加直观的形式：

$$\mathcal{L}(\theta, \phi; \mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z} \mid \mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})]}_{\text{reconstruction term}} - \underbrace{D_{\mathrm{KL}}(q_\phi(\mathbf{z} \mid \mathbf{x}) \parallel p_\theta(\mathbf{z}))}_{\text{regularization term}}$$
- 重建项：希望重建得好，衡量q和p的关系
- 希望隐空间别太乱
	- 可以直接计算

> 变分—— 变一个分布去优化

【Training】
1. 采样真实样本x
2. encoder构造其对应的z
3. decoder根据z输出结果
4. 计算重建项损失和KL
5. 更新参数

【生成】
1. 从隐空间采样
2. 交给decoder生成

# 3. Diffusion家族：DDPM到DDIM

【DDPM思想】可以给干净的图像加噪，能否学会去噪呢？

---

**【DDPM前向过程】——加噪**
这个加噪的过程就是一个马尔科夫链（每一步骤只取决于其前一步）,而这也带拉了一个很有用的性质，借助高斯分布叠加仍为高斯分布的特点，我们可以直接得到第n步的加噪公式：$\mathbf{x}_t$ 与 $\mathbf{x}_0$ 的关系
> [!note]- Recap：高斯分布的合并与拆解
> 若
> 
> $$\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1), \quad \mathbf{y} \sim \mathcal{N}(\boldsymbol{\mu}_2, \boldsymbol{\Sigma}_2)$$
> 
> 且 $\mathbf{x}$ 与 $\mathbf{y}$ 相互独立，则对于常数 $a, b$，有：
> 
> $$a\mathbf{x} + b\mathbf{y} \sim \mathcal{N}\left(a\boldsymbol{\mu}_1 + b\boldsymbol{\mu}_2, a^2\boldsymbol{\Sigma}_1 + b^2\boldsymbol{\Sigma}_2\right)$$

从 $\mathbf{x}_0$ **一步跳到** $\mathbf{x}_t$。定义 $\bar{\alpha}_t = \alpha_t \alpha_{t-1} \dots \alpha_1$，其中 $\boldsymbol{\epsilon}_t$ 为一步到位的标准高斯噪声：

$$\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t$$

或者写成条件概率密度函数（PDF）的形式：

$$q(\mathbf{x}_t \mid \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})$$

**【DDPM之反向过程】—— 去噪**


# 4 新话题+回顾


# 5 Diffusion架构：U-Net到DiT


# 6 Video Diffusion


# 7 前沿Diffusion


