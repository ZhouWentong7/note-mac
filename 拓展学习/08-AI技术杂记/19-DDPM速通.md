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

## 3.1 DDPM 
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

DDPM 反向过程——去噪

目标：训练模型 $p_\theta(x_{t-1}|x_t)$，让它学会从带噪的 $x_t$ 逐步恢复出干净的 $x_0$。

正向过程是不断加噪：

$$x_t
=
\sqrt{\bar{\alpha}_t}x_0
+
\sqrt{1-\bar{\alpha}_t}\epsilon$$

其中：

- $x_0$：原始图片
- $x_t$：第 $t$ 步的带噪图片
- $\epsilon\sim\mathcal{N}(0,I)$：加入的高斯噪声
- $\beta_t$：第 $t$ 步的噪声强度
- $\alpha_t=1-\beta_t$
- $\bar{\alpha}_t=\prod_{s=1}^{t}\alpha_s$


反向过程就是：

$x_T\rightarrow x_{T-1}\rightarrow\cdots\rightarrow x_0$

希望模型学习：

$$p_\theta(x_{t-1}|x_t)
\approx q(x_{t-1}|x_t)$$

直接计算 $q(x_{t-1}|x_t)$ 很困难，但如果知道 $x_0$，就可以计算：

$q(x_{t-1}|x_t,x_0)$

因此可以得到一个可计算的训练目标。


DDPM 的关键技巧：

不直接让模型预测 $x_{t-1}$，而是让模型预测加入的噪声：

$\epsilon_\theta(x_t,t)\approx\epsilon$

于是 Loss 就变成非常简单的 MSE：

$$\boxed{
L=
\mathbb{E}
\left[
\|\epsilon-\epsilon_\theta(x_t,t)\|^2
\right]
}$$


【训练】
$x_0$
→ 随机选择 $t$
→ 加入噪声得到 $x_t$
→ 模型预测噪声 $\epsilon_\theta$
→ 和真实噪声 $\epsilon$ 做 MSE。

【生成】
从纯噪声 $x_T$ 开始
→ 模型预测噪声
→ 去掉一点噪声得到 $x_{T-1}$
→ 重复
→ 最终得到 $x_0$。

 
最核心的一句话：**DDPM = 训练一个模型预测噪声，然后在生成时利用这个预测一步步去噪。**

## 3.2 DDIM
【DDPM的核心问题】单步加噪、去噪太慢
解决方案：使用跳步的方式对所加噪声进行计算。

**DDPM 原公式的问题**

* **Markov 链假设**：DDPM 在逆向去噪过程（$x_t \to x_{t-1}$）中依赖 Markov 链性质，即 $x_{t-1}$ 的采样强依赖于上一时刻 $x_t$。
* **采样速度极慢**：为了保证去噪分布的精准，DDPM 需要严格按照固定步长逐步计算（通常需要 1000 步），无法跳步采样。

---

**公式修改方向与思路**

* **打破 Markov 假设（非 Markov 过程）**：将生成过程改为非 Markov 过程，使得 $x_{t-1}$ 不仅取决于 $x_t$，还依赖于初始干净图像 $x_0$。
* **统一边缘分布**：保证前向加噪过程的边缘分布 $q(x_t|x_0)$ 与 DDPM 完全一致，使得 DDPM 训练好的噪声预测网络 $\epsilon_\theta(x_t, t)$ 无需重新训练即可直接复用。
* **引入可控方差参数 $\sigma_t$**：通过将去噪分布写成确定的 $x_0$ 重建项、指向 $x_t$ 的方向项以及随机噪声项的组合，构造更泛化的逆向采样公式。

---

**优化前后的公式对比**

**1. DDPM 去噪采样公式**

$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z \quad (z \sim \mathcal{N}(0, \mathbf{I}))$$

**2. DDIM 泛化去噪采样公式**

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的 } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$

* 其中方差参数定义为：$\sigma_t^2 = \eta \cdot \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}} \sqrt{1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}}$

---

**最终结果与退化机制**

* **$\eta = 1$（退化为 DDPM）**：采样过程具有随机性，完全等价于 DDPM 的去噪公式。
* **$\eta = 0$（DDIM 确定性采样）**：随机噪声项的系数变为 0，去噪过程变为完全确定性的常微分方程（ODE）求解：
  $$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right) + \sqrt{1 - \bar{\alpha}_{t-1}} \cdot \epsilon_\theta(x_t, t)$$
* **加速效果**：由于去噪过程不再依赖紧密相邻的 Step-by-Step 采样，可以在时间序列上进行子序列抽样（如从 1000 步中仅抽取 20~50 步），实现 10~50 倍的采样加速，且支持图像的高效逆向编码（Inversion）。

【DDIM的重大创新】
令$\sigma_t^2 = \eta ^2 \cdot \hat{\beta}_t^2$，且$\eta$可调
- $\eta=1$时退化为DDPM
- $\eta=0$是DDIM

DDIM没有了扰动项，去噪路径完全确定，可以直接顺着一直去噪——跳步也成了可能。

![](attachments/Pasted%20image%2020260826143231.png)

DDIM是**非马尔科夫**

【DDIM生成】—— 采样
希望根据刚才的公式$x_{t-1}=\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\epsilon_t +\sigma_t z_t$能预测$x_{t-1}$。但是生成过程中，模型不知道$x_0$，只能根据$x_t$和t来预测$\epsilon_\theta(x_t,t)$从而用$\hat{x_0}$近似$x_0$.

$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \underbrace{\left( \frac{x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}} \right)}_{\text{预测的 } x_0} + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon_t$$


# 4 新话题+回顾

## 学noise？score？velocity？

score：指出从当前位置出发，往哪个方向稍微走一点，可以让概率密度上升得最快。
$$
s(x)=\bigtriangledown_x \log p(x)$$
- **对数变换 ($\log$)：单调**递增函数，不会改变函数的极值点和变化趋势，但能将相乘的关系转化为相加，极大地简化计算与导数推导。
- 梯度 ($\nabla_x$)：在多元微积分中，标量场（这里是 $\log p(x)$）的梯度是一个向量，其方向指向函数值增长最快的方向，其模长代表增长的速率
在生成模型中，如果从无序的噪声点出发，不断沿着 Score 的方向推进一步，最终就会“爬山”到高概率密度区域，生成真实的数据。

DDPM的noise predictor只要乘上一个已知系数和负号，就是score network，在DDPM中，学噪声和学score只是同一事的两种描述。

>[!question] 看不懂这个部分

怎么理解Diffusion？
- 是噪声学习器
- score学习器与梯度上升器
- 求解一个ODE/SDE 
- 分布的搬运
- 布朗运动的逆过程
- 信息恢复的过程

## 条件生成

按照指定的标签生成。

**【方法一：Classifier Guidance】**

【方法二：Classifier Free Guidance】
让模型学校习带条件的造成和不带条件的噪声

## 蒸馏与加速

# 5 Diffusion架构：U-Net到DiT



# 6 Video Diffusion


# 7 前沿Diffusion


