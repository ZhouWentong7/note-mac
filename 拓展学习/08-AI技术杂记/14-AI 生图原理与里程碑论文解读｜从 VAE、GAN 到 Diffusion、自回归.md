> [!quote] 来源
> [【AI 生图原理与里程碑论文解读｜从 VAE、GAN 到 Diffusion、自回归】 ](https://www.bilibili.com/video/BV1bZTA61Eip/?share_source=copy_web&vd_source=fe1e50f3e19ee2c5f4cd32de1f11a963)
> 文档链接：https://oigi8odzc5w.feishu.cn/wiki/RcuYwK1iviseDhkOXmAcFH6Snve 
> word & pdf：https://github.com/huangyf2013320506/bilibili_repository/tree/master/20260628_AI生图



> [!abstract]
>- 模型如何到有生成图片
>	- VAE、GAN、Diffusion、自回归
>- 2013~2026技术演进过程
>- 提出的关键问题以及解决方法
>	- 引入自然语言作为初始条件
>	- 引入Transformer
>	- 如何解决像素空间带来的算力瓶颈
>- OpenAI和谷歌的技术路线以及可能的发展方向
# 一、早期范式
> [!hint]
> 生成模型的开始：VAQ和GAN，其核心分别是“潜在空间”与“对抗训练，时至今日仍是生成领域的基石。

## 1. VAE（2013）
> [!important] 引入”潜空间“思想。
> VAE — Variational Autoencoder (2013)
_Auto-Encoding Variational Bayes_ — Kingma & Welling (2013)https://arxiv.org/abs/1312.6114

Review：**AutoEncoder 自编码器**，由编码器(encoder)和解码器(decoder)组成的无监督网络，编码器将信息压缩到低纬度，再有解码器 重建信息，让输出尽可能接近输入。
![Autoencoder](attachments/Autoencoder-draft.png)

- x：显变量（observed variable），能直接被观测到的原数据，网络的输入以及期望的输出。
- z：隐变量（latent variable），被编码后的数据，特征尺寸远小于原数据，可能储存了决定这个图的少量关键信息，可通过解码器复原出x。
- x与z的关系：压缩与还原
	- 原理：假设图片可以由少量关键信息大部分。
	- encoder：**从显变量提炼隐变量**
	- decoder：**从隐变量还原显变量**

Autoencoder的训练
1. 前向传播：从训练集采样一批图片 → 编码器输出潜在表示 → 解码器输出重建图像。初始阶段参数随机，编码无意义，初始重建接近噪声。
2. 计算损失：输出的数据（最开始是噪声）与原图直接按照像素对比，计算出差距（loss）
3. 反向传播：用BP算法计算梯度，将encoder与decoder视为整体，一起更新参数
4. 迭代：训练多个epoch，直到损失下降且收敛，重建质量提升。

### VAE（**Variational Autoencoder，变分自编码器**）
 
> 初始想法：是不是可以拿掉encoder，直接从某个中间压缩的东西还原出图像？
> 难说，因为太随机其实不一定和编码后的隐藏信息等价。

> [!important] 从让编码器输出特定的向量 → 输出概率分布的参数
> 这个分布由训练集的数据计算得出

VAE与Autoencoder的关键区别：
- encoder的输出不同：Autoencoder将输入映射为一个确定性的潜在向量 z；而VAE输出潜在变量的概率分布参数（均值 $\mu$ 和方差 $\sigma^2$），随后通过重参数化技巧从该分布中采样得到 z。
- 训练目标不同：Autoencoder仅最小化重建误差（Reconstruction Loss）；VAE除了重建误差外，还引入KL散度（KL Divergence），约束潜在分布接近标准正态分布 N(0,1)。
	- VAE的KL散度：衡量 Encoder 输出的分布 q(z|x) 与标准正态分布 N(0,1) 的差异。
	  如果没有这样的约束，encoder最后得到的不同关键特征之间的均值和方差之间插值很大，随机采样很可能采到无意义的区域，解码器无法还原出有意义的东西。（让所有的样本信息被压缩到标准正态分布附近）
	  GPT：**VAE中的KL散度直接约束 Encoder 输出的高斯分布** q(z|x)**，使其均值接近0、方差接近1，从而让所有样本的潜在表示分布接近标准正态分布，获得连续且可采样的潜在空间。**
- **潜在空间性质不同**：Autoencoder学习得到的潜在空间可能是不连续的；VAE通过分布约束获得连续、平滑且具有生成能力的潜在空间，因此可以通过随机采样生成新的样本。
> [!important] 把像素空间问题搬运到潜在空间

![两种损失的结合效果](attachments/reconstruction_KL_loss.png)

VAE核心贡献
- 确立将编码器的信息百年如到潜在空间的范式
- 让潜在空间可采样：从压缩工具变为生成模型。编码器输出分布而非固定值，加上 KL 散度约束，使潜在空间连续、平滑，任意采样都能生成合理的输出
- 重参数化技巧**Reparameterization Trick**：将不可导（无法反向传播）的”从分布中采样“的操作，转化为可反向传播的z = μ + σ·ε（ε 从标准正态采样）形式，将随机性剥离到
	- 不可导：采样得到的均值和方差，要求得分布对于这两个值的导数是难以计算的（见笔记 [[[07-变分自编码器](技术学习/13-深度学习入门-生成模型/07-变分自编码器.md#^atctdi)]]，
	- 理论基础：利用高斯分布的性质，若某个采样服从高斯分布$\varepsilon \sim N(0,1)$,那么z = μ + σ·ε也是服从高斯分布的。（这是概率论里的线性变换）
- 将神经网络与贝叶斯概率框架结合
```python
# 重采样之前的计算
z = random.normal(mu, sigma)
# 使用重采样技巧
eps = random.normal(0,1)
z = mu + sigma * eps # 随机性全部交给eps（向前传播的时候被当做常数），此时z对mu和sigma的导数可求
```
> [!note] 一句话概括重参数化
把“从 N(\mu,\sigma^2) 采样”改写成“从固定的 N(0,1) 采样后做线性变换”

Before:
```
Encoder
 ↓
随机采样
 ↓
Decoder
```

#重参数化 
```
Encoder
 ↓
μ,σ
 ↓
z = μ + σ·ε
 ↓
Decoder
```

#### 局限性与影响
局限：生成的图像模糊，因为两种损失的结合对隐空间过度正则化，缺乏高频细节

影响力
- 为后来的Stable Diffusion等模型奠定基础
- VQ-VAE
- 虽然质量不如GAN，但是思路非常超前

**VAE 参考文档**
- [ ] https://bluefisher.github.io/2020/02/07/%E7%90%86%E8%A7%A3-Variational-Autoencoders-VAEs/ （推荐）
- [ ] https://www.jeremyjordan.me/variational-autoencoders/
- [ ] https://www.ibm.com/cn-zh/think/topics/variational-autoencoder#186915249
- [ ] https://www.vectorexplore.com/tech/auto-encoder/vae.html
- [ ] https://zhouyifan.net/2022/12/19/20221016-VAE/

## 2. GAN（2014）
> [!cite] GAN — Generative Adversarial Network (2014)
_Generative Adversarial Nets_ — Goodfellow et al. (2014)https://arxiv.org/abs/1406.2661

因为生成的图像比VAE更为清晰，再2014到2020期间处于统治地位，直到2020年出现Diffusion。

核心思想
- 生成器：从随机噪声直接生成随机图像丢给判别器 —— 越逼真越好
	- 固定维度的随机噪音进行上采样，输出到和训练数据一样尺寸，用MLP（后2015年的DCGAN替换为转置CNN）实现。
- 判别器：一个分类器，接收一个图像，输出一个0~1的概率标量。判断是真实图像还是生成器生成的图像 —— 努力打假
	- 原始为MLP，后DCGAN改为CNN。
- 通过对抗的作用提升生图效果—— 纳什均衡

> [!note] 纳什均衡
> 一组策略组合，给定其他所有人策略不变，任意参与者单独更换策略，收益不会提升甚至受损，因此无人有动机单方面改变现状，局面自动稳定
>- 稳定≠最优:均衡只保证个体不愿单独改变，不代表整体收益最大；极易出现个体理性导致集体吃亏。
>- 一场博弈可存在多个纳什均衡:如情侣博弈（性别战）：一起看男方电影、一起看女方电影，两个均衡，容易协调失败。
>- 仅适用于无强制合约的非合作博弈:双方无法签订约束性协议、不能互相监督惩罚，才会陷入低效均衡；若有强制合作，可跳出纳什均衡。
>- 纳什存在定理:任何有限博弈，至少存在一个纳什均衡（纯策略或混合策略）。

![GAN架构示意图](attachments/GAN-draft.png)

训练过程
- 损失函数：生成器G与判别器D共享一个min-max目标函数，判别器想最大化这个目标（尽量区分出真假），生成器想尽量最小化该目标（不让判别器看出来）
- 前向传播：
	- G生成得到假样本
	- D随机接收真假样本，尽量判别出区别
- 反向传播：
	- D：固定生成器，根据判别器loss对判别器进行反向传播，更新参数，使D更精准
	- G：固定判别器，梯度从D的打分一路回到G，更新G的参数，使生成的假样本更真

核心贡献
- 对抗框架：无显示定义的似然函数，无复杂的变分近似计算，利用博弈隐式学习数据分布
- 生成质量提升：相较于同期的VAE，GAN的图像更为锐利逼真，生成模型的输出达到“可用”级别
- 架构通用：该思想几乎可以用到任何网络架构和损失结合，有大量变体存在

GAN的局限
- **训练不稳定**：G和D的平衡难以维持，容易出现梯度消失、模式坍缩
	- G和D需要保持动态平衡。如果D过强，G难以获得有效梯度；如果G过强，D难以学习有效判别特征，两者容易陷入震荡或训练失败。
- **多样性不足**：模式坍缩意味着 G发现自己生成某种图总是被打假，但是另一种图很容易通过，于是只生成那种（种类较少）安全的不会被打假的图。 

然而，GAN 的训练不稳定和模式坍缩问题始终未被根本解决，这为后来扩散模型的崛起埋下了伏笔。

GAN 的对抗训练思想也被后续大量工作借用。最典型的例子是 **VQGAN**（2021）。它把 GAN 的对抗损失引入 **VQ-VAE 的 tokenizer 训练**中，大幅提升了图像离散化的质量，成为**自回归范式**的关键组件。

2021 年，Diffusion Models Beat GANs 一文的诞生标志着 GAN 在无条件/有条件图像生成上的统治地位正式被**扩散模型**终结。

**GAN 参考文档**

- [ ] https://lilianweng.github.io/posts/2017-08-20-gan/
- [ ] https://jonathan-hui.medium.com/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09
- [ ] https://towardsai.net/p/machine-learning/diffusion-models-vs-gans-vs-vaes-comparison-of-deep-generative-models
- [ ] https://aws.amazon.com/cn/what-is/gan/
# 二、基础工作
不直接用于生成范式，但是为后续研究的扩散和自回归两种范式起到关键铺垫。

## 1. ViT（2020）
> [!quote] ViT — Vision Transformer (2020)
_An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_ — Dosovitskiy et al. (2020)https://arxiv.org/abs/2010.11929

>[!hint] 证明Transformer在视觉任务上的可用性。

![ViT架构图：将图片切分为固定大小的patch，对每个patch做线性嵌入，加上位置编码，作为向量输入到标准的Transformer编码器中](attachments/ViT-draft.png)
 
核心：第一次使用完全标准的Transformer（NLP领域的架构）完成视觉任务，没有使用任何卷积。是跨CV和NLP的标志性工作。

做法：将图像切分为固定大小的 patch（例如 16×16），每个 patch 线性投影成一个 token，然后用标准 Transformer 处理这些 token 序列。在<u>足够大的数据上训练后</u>，ViT **超越了所有 CNN 模型**

缺点：
- 预训练需要的数据比CNN大得多： 在中等规模数据集（如仅用 ImageNet-1k 训练）上，ViT 的表现不如同规模的 CNN，需要大规模数据或强数据增强才能发挥优势。
- 计算成本高：自注意力的二次复杂度使得处理高分辨率图像的计算成本很高

核心贡献：
- 证明纯Transformer在视觉任务的可行性：打破视觉任务必须用CNN的认知。
- **Patch Tokenization：** 将图像视为"patch 序列"的思想，为后来的 DiT（用 Transformer 替代 U-Net 做扩散）和自回归图像生成（把图像当 token 序列逐个预测）奠定了基础。
- **扩展性（Scalability）：** 展示了 Transformer 在视觉领域同样遵循**"更大模型 + 更多数据 = 更好性能"**的 Scaling Law。

> [!note-toolbar] CNN 的局限性
> CNN 的每一层只看局部区域，必须通过堆叠很多层才能间接获得全局视野。捕捉图像中远距离的关联（如图片左上角的物体与右下角的物体之间的关系）需要经过层层传递，效率较低。而 Transformer 通过注意力机制，让每个 token 在第一层就能直接与所有其他 token 建立关联，全局关系一步到位。这也是为什么在数据足够充分的条件下，ViT 能够超越所有精心设计的 CNN 模型

对后续工作的影响
- CLIP：视觉与语言的对其
- DiT：用Transformer替代Unet做扩散的骨干
- 自回归视觉生成模型：DALL-E1、 LlamaGen
- 

**CNN & VIT 参考文档**

-  https://www.datacamp.com/tutorial/vision-transformers（VIT）
- https://www.codecademy.com/article/vision-transformers-working-architecture-explained（VIT）
- https://learnopencv.com/understanding-convolutional-neural-networks-cnn/（CNN）
- https://www.codecademy.com/article/understanding-convolutional-neural-network-cnn-architecture（CNN）

## 2.CLIP
> [!quote] CLIP — Contrastive Language-Image Pre-training (2021)_Learning Transferable Visual Models From Natural Language Supervision_ — Radford et al. (OpenAI) (2021)https://arxiv.org/abs/2103.00020


> [!abstract]
> OpenAI提出的通过**对比学习**将图和文映射到同一语义空间。
> 使用了4亿个图文对（大力出奇迹），训练图像编码器和文本编码器，使两个编码器在嵌入空间中的相对一得文本更近，不匹配的对距离更远


![CLIP架构图](attachments/Clip-draft.png)

CLIP训练
- 获取数据：从互联网上得到的4亿个图-文对
- 两个编码器
	- 图像编码器：ResNet或者ViT
	- 文本编码器：Transformer
	- 每个编码器将数据压缩到固定维度（如：512维，不同版本维度不同）
- 训练目标
	- 对于一个batch中N个图文对，正确配对的图文向量在向量空间中距离更近（余弦相似度→1），另外的$N^2 - N$个不配对内容要更远（余弦相似度→0），训练结束后，两个编码器学会将语义相似的图文映射到接近的位置

CLIP的应用（文生图）
- 只使用文本编码器
- 流程：prompt → CLIP文本编码器 → 语义embedding → 扩散模型（UNet)→ 在每一步去噪过程中引导生成方向  
- CLIP本身不生成图像，而是作为一个“翻译工具”，将文本与图像的语义对齐： $\text{text embedding} \approx \text{image embedding in same space}$

影响
- 扩散范式中
	- CLIP作为Stable Diffusion的条件输入
	- DALL E2/unCLIP 则以CLIP为核心构建




**CLIP 参考文档**
- https://openai.com/index/clip/
- https://viso.ai/deep-learning/clip-machine-learning/
- https://medium.com/@ManishChablani/clip-contrastive-language-image-pretraining-summary-and-intuition-52e329a67377



---

# 三、扩散模型
> [!hint] 核心思想
> 逐步添加高斯噪声，直到该图像转换为纯噪声图像，再反向训练一个网络逐步去噪。
> 

## 1. DDPM（2020）
> [!quote] _Denoising Diffusion Probabilistic Models_ — Ho, Jain & Abbeel (2020)https://arxiv.org/abs/2006.11239

> [!abstract] 摘要
> 向前过程中逐步为图像添加高斯噪声，直到变为纯噪声图。；反向过程训练一个神经网络（U-Net）逐步去噪，从纯噪声恢复出数据。训练目标转化为：预测每一步添加的噪声。

![DDPM processing](attachments/DDPM-draft.png)

**DDPM训练过程**
- 前向加噪
	- 给定一个原图$x_0$，按照预设的噪声调度表（noise schedule）一步步叠加高斯噪声
	- 经过T步（通常T=1000）后，原图信息被噪声覆盖，变为纯噪声图像。
	- 纯数学叠加，不涉及可学习参数。关键性质： 任意时刻 t，$x_t$ 可以直接由 $x_0$ 生成，而不需要逐步递推。
	  > 高斯噪声的可合成性，多次叠加的高斯噪声仍然是高斯噪声
- 反向去噪
	- 输入加噪后的图像$x_t$，和当前时间步t，输出为对该时刻加噪的预测。
	- 训练流程是一个朴素的回归任务：：在每次迭代中，从数据集中采样真实图像 $x_0$，随机采样时间步 $t$，并通过封闭形式直接构造带噪样本 $x_t$。同时根据前向扩散公式生成对应的高斯噪声 $\epsilon$，训练网络在给定 $x_t$ 和 $t$ 的条件下预测该噪声，并通过均方误差（MSE）进行优化。
	- 时间步t也会被作为条件输入给网络，原因在于：不同 t 对应不同噪声水平，使得 $x_t$ 的统计分布显著不同。在大 t 时，**样本接近纯高斯噪声**，模型需要**恢复全局**结构；在小 t 时，**样本仅含少量噪声**，模型主要进行**高频细节修复**。因此去噪函数本质上是一个依赖噪声强度的条件函数 $\epsilon_\theta(x_t, t)$，而非统一映射。
	  - 人话：t 表示当前噪声强度，输入给模型是为了让它知道图像“被破坏到什么程度”，从而在不同噪声水平下学习不同的去噪方式：大噪声时恢复结构，小噪声时补细节
	- 损失函数：MSE均方误差，计算真实噪声与预测噪声之间的差距

**DDPM生成图像**
- 随机采样一张纯高斯噪声图作为 x_T，从 t=T 开始，每一步让去噪网络预测当前噪声并将其减去，同时注入一点更小的随机噪声（维持采样的随机性），得到 $x_{t-1}$

**主流去噪网络框架**
- U-Net
	- 原文架构
	- 编码器逐层提取抽象特征
	- 解码器放大还原分辨率
	- skip connection传递信息，使输出同事保留深层语义和浅层细节
	- 输入输出尺寸相同，适用于去噪任务
- Transformer（DiT)
	- 噪声图像切分为patch，注意力机制让每个patch可以从第一层就与其他patch直接交互，全局信息捕捉效率更高，扩展性强。（Sora已采用[Link](https://arxiv.org/abs/2212.09748）)

**局限性**
- **采样速度慢**：反向去噪需要上千步迭代（如 T=1000），每一步都需要一次完整的网络前向传播，生成一张图像需要数分钟，远慢于 GAN 的单次前向传播。
- **分辨率受限**：在**像素空间**直接做扩散的计算成本随分辨率平方增长，DDPM 的实验仅限于 32×32 和 256×256。
- **无条件生成：** 原始 DDPM 仅做**无条件生成**，不支持文本、类别等条件引导。
**DDPM 参考文档**
- https://lilianweng.github.io/posts/2021-07-11-diffusion-models/  （非常严谨清晰，推荐）
- https://learnopencv.com/denoising-diffusion-probabilistic-models/
- https://theaisummer.com/diffusion-models/
- https://zhouyifan.net/2023/07/07/20230330-diffusion-model/（中文）
- https://calvinyluo.com/2022/08/26/diffusion-tutorial.html

## 2. Diffusion Models Beats GANs（2021）
> [!quote] _Diffusion Models Beat GANs on Image Synthesis_ — Dhariwal & Nichol (OpenAI) (2021)https://arxiv.org/abs/2105.05233

> [!hint] 使用**分类器**实现**条件化生成图像**；扩散模型正式超越GAN
-  扩散模型一直以来在生成质量指标上无法超越最好的GAN，而这个论文首次让扩散模型在ImageNet条件生成上超越了GAN

> [!note]
> - **提出时面临的问题**：在 2021 年之前，生成对抗网络（GANs）在图像合成质量（如 FID 目标指标）上占据着绝对统治地位，而早期的扩散模型（如 DDPM）虽然展示了优秀的分布覆盖能力和训练稳定性，但在高分辨率、复杂数据集（如 ImageNet）上的**生成图像质量（保真度）仍逊于 GAN**，且采样速度极慢，也缺乏一种类似 GAN “截断技巧（Truncation Trick）”的在多样性与保真度之间自由权衡的机制。
>- **改进方法**：本文通过大量的消融实验（Ablation Studies）**彻底优化了扩散模型的 UNet 骨干架构，并首次引入了**分类器引导（Classifier Guidance）技术。通过在反向去噪过程中加入一个预训练图像分类器的梯度，来引导模型向指定类别生成。
>- **对生成图像领域的意义**：**这是扩散模型首次在图像生成质量（FID）上全面超越当时最强的 GAN（如 BigGAN-deep）**。它打破了 GAN 在生图领域的垄断，正式开启了扩散模型（Diffusion Models）作为下一代生成式 AI 核心基座的时代。

![](attachments/Pasted%20image%2020260701100729.png)

```Plain
具体来说，每一步去噪时：
1. 扩散模型先正常预测去噪方向
2. 把当前的中间结果丢给分类器，问它"这像不像目标类别（比如'金毛犬'）？"
3. 分类器算出一个梯度，指示"往哪个方向改一改就更像金毛犬"
4. 把这个梯度叠加到去噪方向上（最终方向 = 原始去噪方向 + 分类器梯度）

注意这里的关键：扩散模型本身完全不认识文字，也没有任何文本输入。它就是个无条件去噪器。所有的"引导"都来自外部的分类器梯度硬拽。
```

>[!sucess] **核心贡献**
- **架构大步升级**：探明了更深/更宽的网络、多头注意力机制、全分辨率注意力层、BigGAN 残差块以及自适应组归一化（AdaGN）对扩散模型效果的显著加成。 
- **分类器引导机制（Classifier Guidance）**：提出利用分类器梯度控制生图走向的方法，不仅适用于随机采样，也成功适配了确定性采样（DDIM）。
- **实现保真度与多样性的完美权衡（Trade-off）**：引入了“梯度缩放因子 $s$”。增大 $s$ 可以让图像更贴合类别特征（提高 Precision 和 IS 评分），减小 $s$ 则保留更多样式的分布（提高 Recall）。
- **大幅压缩采样步数**：配合引导机制，模型在使用 DDIM 仅进行 25 次前向传播（25步）时，图像质量即可媲美需要数百步的传统模型。

>[!attention] **局限性**
- **采样速度依然较慢**：尽管将步数压缩到了 25 步，但与 GAN 这种“一步到位（One-shot forward pass）”的生成网络相比，扩散模型仍需要循环进行多次前向计算，推断成本高。
- **对外部标签/分类器的强依赖**：该技术极度依赖带有显式标签的数据集。对于无标签的数据集，分类器引导则无法直接发挥作用。
- **分类器对抗样本风险**：如果梯度缩放因子 $s$ 调得过大，图像可能会过度迎合分类器的特征识别偏好（甚至走向对抗攻击特征），导致画面失真或过于单一。

> [!note] 对后续工作的影响
- **推动了无分类器引导（Classifier-Free Guidance, CFG）的诞生**：正因为本文指出了“引导机制”对提升生图质量的决定性作用，随后业内（Ho & Salimans）在此基础上进化出了不需要独立分类器的 CFG 技术。如今无论是 **Stable Diffusion**、**Midjourney** 还是 **Flux**，其中控制提示词相关性的关键滑块（Prompt Weight / CFG Scale）全部源自于本文的核心思想。
- **直接孵化了 OpenAI 的里程碑模型**：OpenAI 随后将这一套“改进版 UNet 架构 + 引导思想”直接复用并扩展到了多模态领域，成功打造了震动业界的 **DALL-E 2**。  
- **加速了超分辨率级联架构（Upsampling Stacks）的普及**：文中证明了“低分辨率引导生图 + 高分辨率扩散上采样”的互补性，这一级联思路被后来的 Google Imagen 等多款顶级商业大模型广泛采纳。

## GLIDE(2021)

> [!quote] GLIDE - **G**uided **L**anguage to **I**mage **D**iffusion for Generation and **E**diting
_GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models_ — Nichol et al. (OpenAI) 
(2021)https://arxiv.org/abs/2112.10741

> [!important] 实现无需分类器的自然语言生图

![GLIDE：](attachments/GLIDE-draft.png)

**【模型简介】**
- **面临的问题：** 在 GLIDE 提出前，主流的文本生成图像（Text-to-Image）模型（如早期的 [DALL-E](https://arxiv.org/abs/2102.05918)）虽然具有较强的零样本概念组合能力，但其**生成的图像在逼真度（Photorealism）上仍有欠缺**，细节模糊。另一方面，无条件或类条件扩散模型（Diffusion Models）虽然能生成极高质量的逼真图像，但**缺乏灵活、自由的文本控制能力**，无法应对复杂的提示词。
- **改进方法：** GLIDE 将引导扩散模型（Guided Diffusion）应用到了文本条件图像合成任务中。论文**系统性地对比了两种文本引导策略**：基于带噪声的 CLIP 引导（CLIP Guidance）和**无分类器引导（Classifier-Free Guidance, CFG）**。最终发现无分类器引导能够更好地兼顾画质与文本一致性，并基于此训练了一个拥有 35 亿参数的文本条件扩散模型，并进一步微调使其支持功能强大的图像局部修复（Inpainting）与编辑。
- **对生成图像领域的意义：** GLIDE 证明了**扩散模型（Diffusion Models）在文本生成图像任务上的表现全面超越了当时主流的自回归（Autoregressive）模型**。它以更小的参数量（3.5B vs DALL-E 的 12B）和更低的采样延迟，刷新了画质和文本匹配度的标杆，彻底拉开了扩散模型统治文本生图领域的序幕。


**【模型结构】**
- **基础基础模型（Base Model）：** 采用 ADM（Ablated Diffusion Model）的 UNet 架构，分辨率为 **64×64**。视觉部分通道数扩展至 512，约 23 亿参数。
- **文本编码器（Text Encoder）：** 24 块、宽度为 2048 的 Transformer，约 12 亿参数。Transformer 的最终 Token 嵌入替代类嵌入，且**最后一层的序列嵌入通过交叉注意力（Cross-Attention）** 机制融合进 UNet 的每一层。
- **超分辨率模型（Upsampling Model）：** 15 亿参数的扩散模型，用于将图像分辨率从 **64×64** 提升至 **256×256**。


【训练阶段流程】
1. 输入数据：从大规模互联网数据集中获取 (图像 $x_0$, 文本描述 $c$) 对。
2. 文本编码：通过 Transformer 将文本 $c$ 编码为 Token 特征序列。
3. 扩散前向过程：按照预设的噪声衰减系数，向图像 $x_0$ 逐步加入高斯噪声，生成各个时间步 t 的含噪图像 $x_t$。
4. 模型预测：将 $x_t$、时间步 t 以及文本序列特征输入到 UNet 中，训练模型预测所加入的噪声 $ε$。
5. 引入无分类器引导机制（微调）：在训练过程中，以 20% 的固定概率将文本 $c$ 替换为空白序列 $∅$，使模型同时兼备条件生成和无条件生成能力。
6. 引入图像修复能力（微调）：随机擦除图像的局部区域，向 UNet 额外提供 4 个输入通道（3通道的未掩码区域图像 + 1通道的 Mask 掩码），训练模型补充缺失区域。
>[!hint] 训练要点
>- 任务核心：预测该步所加噪声，损失函数仍为预测噪声与真实噪声之间的MSE
> - 【5】核心：网络同时学会有文本的去噪和无文本的去噪。不依赖外部分类器，网络自身可以提供“有条件/无条件”两种预测之差
> - 【Cross-attention】：文本通过交叉注意力参与U-Net的每一层计算，使文本在整个去噪过程发挥作用

**【生成阶段流程 (推理采样)】**
1. 初始化：从高斯分布中随机抽取纯噪声图像 $x_T \sim N(0, I)$，并指定目标文本 c。
2. 循环去噪 (从 $t = T$ 逐步倒推至 $t = 1$)：
    a. 分别计算在目标文本条件下的模型预测噪声：$ε_θ(x_t | c)$
    b. 计算在无文本（空序列）条件下的模型预测噪声：$ε_θ(x_t | ∅)$
    c. 结合无分类器引导公式进行外推，计算最终的引导噪声：
       $引导噪声 = ε_θ(x_t | ∅) + s * (ε_θ(x_t | c) - ε_θ(x_t | ∅))$  (其中 s 为引导权重 scale)
       > 即 **「无条件方向 + 引导强度 ×（有条件方向 − 无条件方向）」**
       > 引导强度越大，生成越贴合文本，但过大会牺牲多样性和真实感

    d. 利用引导噪声更新图像，得到稍清晰的图像 $x_{t-1}$。
3. 得到基础画作：循环结束后，获得 64x64 分辨率的初版图像。
4. 超分辨率重建：将 64x64 图像作为基底，通过 1.5B 强度的超分辨率扩散模型，最终放大输出 256x256 的高清晰度逼真图像。

【局限性】
- **文本理解能力有限：** 当面对一些过于反常或复杂的提示词时（例如：“一只长着八条腿的猫” 或 “用履带代替轮子的自行车”），模型往往无法正确渲染出语义。
- **采样速度慢：** 作为标准的扩散模型，生成一张图像需要经历多次逆向迭代计算，在当时的硬件下完成一次生成需要多秒（如 A100 上约 15 秒），相较于 GAN 或单次前向的自回归模型，实时推理成本较高。
- **仍存在像素空间操作**：GLIDE直接在64×64像素空间做扩散，后续上采样到256x256
- **社会偏见与安全隐患：** 模型仍会保留并可能放大训练数据集中的西方刻板印象及性别偏见（如“女孩的玩具”会产生更多的粉色）；同时其强大的编辑修复功能存在被滥用制作为伪造 disinformation 的风险。

【参考文档】
https://ffighting.net/deep-learning-paper-review/diffusion-model/glide/

---
其他笔记：
[[../../技术学习/13-深度学习入门-生成模型/07-变分自编码器|07-变分自编码器]]  
