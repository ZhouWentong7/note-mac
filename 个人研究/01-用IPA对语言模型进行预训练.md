用国际音标替代文本作为语言预训练模型的输入，训练一个音素级别的。


数据集： https://huggingface.co/datasets/phonemetransformers/IPA-CHILDES

这是一个为您整理的 **IPA-CHILDES** 数据集简介，基于 Hugging Face 上的官方描述，您可以直接用于项目文档或学术报告中：

---

# IPA-CHILDES 数据集简介

**IPA-CHILDES** 是一个专为交叉语言音系学（Cross-Lingual Phonology）和音位语言模型（Phonemic Language Modeling）研究而设计的开源数据集。该数据集由 [phonemetransformers](https://huggingface.co/phonemetransformers) 团队开发，其核心是将著名的 **CHILDES（儿童语言交换数据系统）** 语料库中的儿童及看护者对话文本进行预处理，并转化为高精度的**国际音标（IPA）音位表征**。

数据集总共包含 **31 种语言**，数据总量约为 **1246 万行（Utterances）**，总文件大小为 **4.17 GB**。

---

### 核心特点与技术细节

- **音位化转换（Phonemization）：** 针对不同的语言，数据集采用了最适合的音标转换工具。例如，英语（美式/英式）、日语、法语等使用 `phonemizer`；德语、西班牙语、塞尔维亚语等使用 `epitran`；普通话则使用 `pinyin_to_ipa`。
    
- **保留丰富的语言学标签：** 数据集在转化为音标的同时，完好地保留了 CHILDES 原语料库中对科学实验非常有用的多维特征，包括**词性标注（POS Tags）**、**语素数量（Num of Morphemes）**、**说话人角色（如母亲、调查员）** 以及 **目标儿童的年龄和性别**。
    
- **按儿童年龄排序：** 所有数据已根据目标儿童的月龄（`target_child_age`）进行了排序，这使得研究人员能够轻松地根据特定年龄段截取训练数据，非常适合模拟**儿童语言习得（Child Language Acquisition）** 的渐进过程。
    

---

### 核心数据列说明

数据集主要包含以下几项核心列，方便进行不同模态的模型训练与对比：

|**字段名称 (Column)**|**说明 (Description)**|
|---|---|
|**`processed_gloss`**|经过标准化预处理的文本（如统一小写、修正拼写错误、添加标点），基于 AOChildes 规范，适合传统的词级或子词级语言模型。|
|**`ipa_transcription`**|**音位转换结果（IPA）**。音位之间用空格分隔，并使用 `WORD_BOUNDARY` 标记词与词 transition 的边界。非常适合训练基于音位的字符级语言模型。|
|**`character_split_utterance`**|纯文本按字符（字母）直接空格切分的结果。格式与 `ipa_transcription` 类似，专门用于对比“语音输入”与“文本输入”在模型表现上的差异。|
|**`is_child`**|标记该话语是否由儿童说出。_(注意：当前版本默认主要保留看护者话语，但处理脚本支持提取儿童话语)_。|

---

### 语言覆盖与数据分布

数据集涵盖了 31 种语言，其中数据量最大的前几种语言包括：

1. **EnglishNA（美式英语）：** 约 256 万行
    
2. **EnglishUK（英式英语）：** 约 204 万行
    
3. **German（德语）：** 约 153 万行
    
4. **Japanese（日语）：** 约 99.9 万行
    
5. _(其他还包括法语、西班牙语、普通话、荷兰语、波兰语、粤语、韩语等共 31 种)_
    

---

### 推荐应用场景

- **计算语言学与认知建模：** 研究 AI 模型在仅接受“语音（音位）”输入时，能否像人类儿童一样学会词汇切分（Word Segmentation）和语法习得。
    
- **交叉语言音系学研究：** 探索跨语言的语音分布、音位组合规律。
    
- **BabyLM 挑战赛相关研究：** 该数据集已被广泛用于如 _[BabyLM's First Words](https://www.google.com/search?q=https://huggingface.co/collections/phonemetransformers/babylms-first-words-6613fb796be4bca8cf7ff7fa)_ 等前沿探针任务中。
    

> **相关论文参考：**
> 
> - _IPA-CHILDES & G2P+: Feature-Rich Resources for Cross-Lingual Phonology and Phonemic Language Modeling_ (arXiv: 2504.03036)
>     
> - _BabyLM's First Words: Word Segmentation as a Phonological Probing Task_ (arXiv: 2504.03338)
>
### [BabyLM's First Words: Word Segmentation as a Phonological Probing Task](https://aclanthology.org/2025.conll-1.34/) BabyLM 的第一个词：单词切分作为语音探测任务

2025[Cited 7](https://scholar.google.com/scholar?cites=15066432128076276181&as_sdt=2005&sciodt=0,5&hl=en)

[Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra), P Buttery - Proceedings of the 29th Conference on …, 2025 - aclanthology.org [Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra) ，P Buttery - 第 29 届……会议论文集，2025 - aclanthology.org

Trains phoneme-based language models (BabyLMs), which use phonemic transcriptions as input, specifically on the IPA CHILDES dataset, to study phonological representations across 31 languages. 训练基于音素的语言模型（BabyLM），该模型使用音素转写作为输入，特别是使用 IPA CHILDES 数据集，来研究 31 种语言的音系表示。

- **Phonological Probing Task:** Explores how word segmentation, used as a probing task, can assess the representations learned by these phoneme-based language models. **语音探测任务：** 探索如何利用单词切分作为探测任务来评估这些基于音素的语言模型所学习到的表征。
- **Related BERT/Transformer Models:** Mentions related work on a 'Phoneme-level BERT for enhanced prosody of text-to-speech with grapheme predictions' and discusses the use of transformer models in general, indicating relevance to the model architectures requested. **相关的 BERT/Transformer 模型：** 提及了“用于增强文本到语音韵律的音素级 BERT（具有字素预测）”的相关工作，并讨论了 Transformer 模型的一般用途，表明其与所要求的模型架构的相关性。
---

### [IPA CHILDES & G2P+: Feature-Rich Resources for Cross-Lingual Phonology and Phonemic Language Modeling](https://aclanthology.org/2025.conll-1.33/) IPA CHILDES 和 G2P+：功能丰富的跨语言音系学和音位语言建模资源

2025[Cited 6](https://scholar.google.com/scholar?cites=4098220109969012644&as_sdt=2005&sciodt=0,5&hl=en)

[Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra), P Buttery - Proceedings of the 29th Conference on …, 2025 - aclanthology.org [Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra) ，P Buttery - 第 29 届……会议论文集，2025 - aclanthology.org

Introduces IPA CHILDES, a multilingual phonemic dataset of child-centered speech spanning 31 languages, which is suitable for cross-lingual phonological research. 推出 IPA CHILDES，这是一个涵盖 31 种语言的以儿童为中心的语音多语言音位数据集，适用于跨语言音系学研究。

- **Phoneme Language Modeling:** Demonstrates the utility of the dataset for phonological research by training phoneme language models on 11 languages, which is a key step towards pretraining a BERT or Transformer model. **音素语言建模：** 通过在 11 种语言上训练音素语言模型来展示该数据集在音系学研究中的实用性，这是预训练 BERT 或 Transformer 模型的关键步骤。
- **IPA Consistent Representation:** Uses the International Phonetic Alphabet (IPA) for consistent phonemic representations across languages, facilitating multilingual analysis and comparison. **IPA 一致性表示：** 使用国际音标 (IPA) 实现跨语言的一致音位表示，便于多语言分析和比较。
--- 


[  
[PDF] arxiv.org](https://arxiv.org/pdf/2503.07214?)

### [Cross-Lingual IPA Contrastive Learning for Zero-Shot NER](https://arxiv.org/abs/2503.07214) 跨语言 IPA 对比学习用于零样本命名实体识别

2025[Cited 1](https://scholar.google.com/scholar?cites=11152938836560542227&as_sdt=2005&sciodt=0,5&hl=en)

[J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra), [DR Mortensen](https://scholar.google.com/citations?user=2iS5aeoAAAAJ&hl=en&oi=sra) - arXiv preprint [arXiv:2503.07214](https://arxiv.org/abs/2503.07214), 2025 - arxiv.org [J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra) ， [DR Mortensen](https://scholar.google.com/citations?user=2iS5aeoAAAAJ&hl=en&oi=sra) - arXiv 预印本 [arXiv:2503.07214，2025](https://arxiv.org/abs/2503.07214) - arxiv.org

Proposes a cross-lingual IPA Contrastive learning method (IPAC) that aims to align phonemic representations between languages with similar phonetic characteristics, a method suitable for multilingual pretraining. 提出了一种跨语言国际音标对比学习方法（IPAC），旨在使具有相似语音特征的语言之间的音位表示保持一致，该方法适用于多语言预训练。

- **CONLIPA Multilingual Dataset:** Introduces the CONtrastive Learning with IPA (CONLIPA) dataset, which contains IPA pairs from English and 10 high-resource languages across 10 frequently used language families, facilitating a multilingual approach. **CONLIPA 多语言数据集：** 介绍 CONLIPA 对比学习数据集，其中包含来自英语和 10 种高资源语言的 IPA 对，涵盖 10 个常用语系，从而促进多语言方法。
- **Zero-Shot NER Downstream Task:** Applies the methodology to Zero-Shot Named Entity Recognition (NER) for low-resource languages, suggesting a potential downstream NLP task for the pre-trained models. **零样本命名实体识别下游任务：** 将该方法应用于低资源语言的零样本命名实体识别 (NER)，为预训练模型提出潜在的下游 NLP 任务。

---

### [From babble to words: Pre-training language models on continuous streams of phonemes](https://aclanthology.org/2024.conll-babylm.4/) 从胡言乱语到清晰语言：基于连续音素流的语言模型预训练

2024[Cited 13](https://scholar.google.com/scholar?cites=43554446243071323&as_sdt=2005&sciodt=0,5&hl=en)

[Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra), [RD Martinez](https://scholar.google.com/citations?user=E9HDYC4AAAAJ&hl=en&oi=sra), [A Caines](https://scholar.google.com/citations?user=2M1Jo3sAAAAJ&hl=en&oi=sra)… - The 2nd BabyLM …, 2024 - aclanthology.org [Z Goriely](https://scholar.google.com/citations?user=khr4FKUAAAAJ&hl=en&oi=sra) 、 [RD Martinez](https://scholar.google.com/citations?user=E9HDYC4AAAAJ&hl=en&oi=sra) 、 [A Caines](https://scholar.google.com/citations?user=2M1Jo3sAAAAJ&hl=en&oi=sra) ……——第二代 BabyLM……，2024——aclanthology.org

Presents a method for converting pre-training and evaluation datasets into a unified International Phonetic Alphabet (IPA) representation, enabling language models to be trained and evaluated using phonemic input. 提出了一种将预训练和评估数据集转换为统一的国际音标 (IPA) 表示的方法，使得可以使用音素输入来训练和评估语言模型。

- **Phonemic Training Feasibility:** Studies the feasibility of training language models on phonemic representations and their ability to encode grammatical knowledge for downstream language understanding tasks. **音素训练可行性：** 研究基于音素表示训练语言模型的可行性，以及它们编码语法知识以进行下游语言理解任务的能力。
- **Multilingual Potential:** Notes that a multilingual analysis was outside the current scope but confirms the data processing pipeline was applied to prepare phonemized datasets for 26 other languages, suggesting suitability for multilingual extension. **多语言潜力：** 指出多语言分析超出了当前范围，但确认数据处理流程已应用于为其他 26 种语言准备音标数据集，表明适合进行多语言扩展。

---
### [Enhancing cross-lingual transfer via phonemic transcription integration](https://aclanthology.org/2023.findings-acl.583/) 通过音标转写整合增强跨语言迁移

2023[Cited 15](https://scholar.google.com/scholar?cites=9921310309886331694&as_sdt=2005&sciodt=0,5&hl=en)

[H Nguyen](https://scholar.google.com/citations?user=YKcLxDQAAAAJ&hl=en&oi=sra), [C Zhang](https://scholar.google.com/citations?user=u_bIiBQAAAAJ&hl=en&oi=sra), [T Zhang](https://scholar.google.com/citations?user=mulLO8UAAAAJ&hl=en&oi=sra)… - Findings of the …, 2023 - aclanthology.org [H Nguyen](https://scholar.google.com/citations?user=YKcLxDQAAAAJ&hl=en&oi=sra) 、 [C Zhang](https://scholar.google.com/citations?user=u_bIiBQAAAAJ&hl=en&oi=sra) 、 [T Zhang](https://scholar.google.com/citations?user=mulLO8UAAAAJ&hl=en&oi=sra) ……——研究结果……，2023——aclanthology.org

Proposes PhoneXL, a framework for cross-lingual transfer that incorporates phonemic transcriptions, represented in the International Phonetic Alphabet (IPA) format, as an additional linguistic modality alongside traditional orthographic transcriptions. 提出了 PhoneXL，这是一个跨语言转换框架，它将以国际音标 (IPA) 格式表示的音位转写作为除传统正字法转写之外的附加语言模式。

- **Alignment Objectives:** Introduces unsupervised alignment objectives, including local one-to-one alignment, alignment via multi-modality contexts, and alignment via multilingual contexts, to integrate the phonemic and orthographic modalities for cross-lingual transfer. **对齐目标：** 引入无监督对齐目标，包括局部一对一对齐、通过多模态上下文对齐以及通过多语言上下文对齐，以整合音位和正字法模态进行跨语言迁移。
- **Enhances Cross-Lingual Tasks:** Shows that integrating phonemic transcription enhances cross-lingual transfer and achieves consistent improvements on token-level tasks like Named Entity Recognition and Part-of-Speech Tagging over orthographic-based multilingual Pretrained Language Models (PLMs) among CJKV (Chinese-Japanese-Korean-Vietnamese) languages. **增强跨语言任务：** 结果表明，在中日韩越 (CJKV) 语言中，整合音位转写可以增强跨语言迁移，并在命名实体识别和词性标注等标记级任务上，相对于基于正字法的多语言预训练语言模型 (PLM) 取得持续改进

---
### [Mitigating the linguistic gap with phonemic representations for robust cross-lingual transfer](https://aclanthology.org/2024.mrl-1.16/) 利用音位表征来缩小语言差距，实现稳健的跨语言迁移

2024[Cited 8](https://scholar.google.com/scholar?cites=13962250417444042187&as_sdt=2005&sciodt=0,5&hl=en)

[H Jung](https://scholar.google.com/citations?user=wPT3kwkAAAAJ&hl=en&oi=sra), [C Oh](https://scholar.google.com/citations?user=7oAZaVcAAAAJ&hl=en&oi=sra), [J Kang](https://scholar.google.com/citations?user=QbY_MyoAAAAJ&hl=en&oi=sra), [J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra), [K Song](https://scholar.google.com/citations?user=HWxRii4AAAAJ&hl=en&oi=sra)… - Proceedings of the …, 2024 - aclanthology.org [H Jung](https://scholar.google.com/citations?user=wPT3kwkAAAAJ&hl=en&oi=sra) 、 [C Oh](https://scholar.google.com/citations?user=7oAZaVcAAAAJ&hl=en&oi=sra) 、 [J Kang](https://scholar.google.com/citations?user=QbY_MyoAAAAJ&hl=en&oi=sra) 、 [J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra) 、 [K Song](https://scholar.google.com/citations?user=HWxRii4AAAAJ&hl=en&oi=sra) 等 - 会议论文集，2024 - aclanthology.org

Explores the use of phonemic representations, specifically written in International Phonetic Alphabet (IPA) characters, as a robust input representation for multilingual language modeling. 探讨使用音位表示（特别是用国际音标 (IPA) 字符书写的音位表示）作为多语言语言建模的稳健输入表示。

- **Multilingual Cross-Lingual Tasks:** Presents experiments on three representative cross-lingual tasks involving 12 languages to demonstrate the effectiveness of phonemic representations. **多语言跨语言任务：** 展示了涉及 12 种语言的三个具有代表性的跨语言任务的实验，以证明音位表征的有效性。
- **Reduces Performance Gaps:** Shows that phonemic representations consistently reduce linguistic gaps and performance discrepancies between high-resource and low-resource languages compared to orthographic representations. **减少表现差距：** 结果表明，与正字法表示相比，音位表示能够持续减少高资源语言和低资源语言之间的语言差距和表现差异。

---

### [Zero-shot cross-lingual NER using phonemic representations for low-resource languages](https://aclanthology.org/2024.emnlp-main.753/) 使用音位表示法的低资源语言零样本跨语言命名实体识别

2024[Cited 10](https://scholar.google.com/scholar?cites=6050296089433411742&as_sdt=2005&sciodt=0,5&hl=en)

[J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra), [H Jung](https://scholar.google.com/citations?user=wPT3kwkAAAAJ&hl=en&oi=sra), [A Cheng](https://scholar.google.com/citations?user=aDq5X80AAAAJ&hl=en&oi=sra), [J Kang](https://scholar.google.com/citations?user=QbY_MyoAAAAJ&hl=en&oi=sra), Y Du… - Proceedings of the …, 2024 - aclanthology.org [J Sohn](https://scholar.google.com/citations?user=pO8ByhcAAAAJ&hl=en&oi=sra) 、 [H Jung](https://scholar.google.com/citations?user=wPT3kwkAAAAJ&hl=en&oi=sra) 、 [A Cheng](https://scholar.google.com/citations?user=aDq5X80AAAAJ&hl=en&oi=sra) 、 [J Kang](https://scholar.google.com/citations?user=QbY_MyoAAAAJ&hl=en&oi=sra) 、Y Du… - …的会议记录，2024 - aclanthology.org

Proposes a novel approach for Named Entity Recognition (NER) in extremely low-resource languages using phonemic representations based on the International Phonetic Alphabet (IPA) to bridge representational gaps between different languages. 提出了一种针对资源极其匮乏的语言的命名实体识别 (NER) 的新方法，该方法使用基于国际音标 (IPA) 的音位表示来弥合不同语言之间的表示差距。

- **Transfer with Phonemes:** Fine-tunes a pre-trained model on phonemes of a source language (English) and then applies the resulting shared IPA notation system model directly to target languages for zero-shot cross-lingual transfer. **使用音素进行迁移：** 对源语言（英语）的音素进行预训练模型的微调，然后将生成的共享 IPA 符号系统模型直接应用于目标语言，实现零样本跨语言迁移。
- **Multilingual Phonemic Model:** Utilizes XPhoneBERT, a multilingual pre-trained model with phonemic representations, for cross-lingual transfer, demonstrating superior performance over traditional grapheme-based models like mBERT, especially on unseen low-resource languages and non-Latin scripts. **多语言音素模型：** 利用 XPhoneBERT（一种具有音素表示的多语言预训练模型）进行跨语言迁移，在性能上优于传统的基于字素的模型（如 mBERT），尤其是在未见过的低资源语言和非拉丁文字上。

您提到的“比赛”是指在前述数据集中被提及的 **BabyLM 挑战赛（BabyLM Challenge）**，或者更具体地说是该项目团队围绕其开展的 **“BabyLM's First Words”（BabyLM 的初字/第一批词）** 计算语言学评测任务。

这是一个在人工智能与认知科学交叉领域非常前沿的微型语言模型挑战赛。以下为您梳理该比赛的核心背景与机制：

---

## 比赛核心背景：什么是 BabyLM？

传统的 LLM（如 GPT-4 等）往往需要数万亿（Trillions）Token 的海量文本进行海量计算才能学会语言。然而，**人类儿童在成长到 13 岁时，一生中听到的单词总量大约只有 1 亿（100M）个左右**，却能完美掌握高度复杂的母语语法和语义。

> **比赛的核心宗旨：** > 旨在号召全球研究者在**严格限制数据量（数据规模等同于人类儿童可接触的语言总量，通常为 1000 万到 1 亿 Token）**的条件下，通过改进模型架构、训练目标或数据增强方式，训练出比肩大模型语言认知能力的“微型语言模型（Compact Language Models）”。

---

## 本次特色评测任务：BabyLM's First Words

在您关注的这一届/该分支任务中，比赛将目光聚焦于儿童语言习得的第一步——**词汇切分（Word Segmentation）**，并将其作为一项“音系学探针任务（Phonological Probing Task）”。

### 1. 核心挑战：盲听与切词

在人类语言的实际发音中，词与词之间是没有像文本那样的“空格”的（例如：听“大熊猫在吃竹子”时，听到的是一连串连续的语音流）。儿童必须自己学会从连续的语音中分辨出哪里是一个词的开始，哪里是结束。

参赛者需要利用诸如 [IPA-CHILDES](https://huggingface.co/datasets/phonemetransformers/IPA-CHILDES) 这类**纯国际音标（IPA）音位流**且**剥离了词边界空格**的数据来训练语言模型。

### 2. 评测机制与方法

模型训练完成后，比赛通过以下两种主要方式来评估模型是否真正像人类儿童一样“学会了什么是词”：

- **无监督边界提取（Unsupervised Boundary Extraction）：** 利用语言模型的预测特性。由于词与词的衔接处预测难度通常会暴增，比赛会测试模型能否通过“预测错误率最高点（Prediction-error Peaks）”来准确预测出词的起始位置。
    
- **线性探针（Linear Probes）：** 在模型的隐藏层上训练一个简单的线性分类器，探测模型在不显式学习空格的情况下，其内部表征是否已经隐式地记录了词的边界。
    

---

## 比赛的重要学术价值

1. **验证认知语言学理论：** 检验语言学中的“统计学习理论”（Statistical Learning Theories），即人类婴儿是否真的仅靠统计前后音节出现的概率（条件概率）就能学会划分词汇。
    
2. **推动更高效的 Tokenizer 研发：** 当前 AI 模型的 Tokenizer（如 BPE）完全是基于文本统计的，而该比赛的结果能反哺 AI 领域，启发科学家开发出更符合人类语音学特性的新型子词分词器。
    

如果您计划参加该比赛或开展相关研究，可以参考本次任务的 [CoNLL 2025 会议入选论文](https://arxiv.org/abs/2504.03338) 以了解更多技术基线。

