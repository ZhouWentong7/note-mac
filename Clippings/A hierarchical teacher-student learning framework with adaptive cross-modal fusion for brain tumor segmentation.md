---
title: "A hierarchical teacher-student learning framework with adaptive cross-modal fusion for brain tumor segmentation"
source: "https://www.sciencedirect.com/science/article/pii/S0957417426002848"
author:
published:
created: 2026-08-20
description: "Accurate brain tumor segmentation plays an important role in clinical diagnosis, treatment planning, and therapeutic response monitoring. Multi-modal …"
tags:
  - "clippings"
---
## Published by: Elsevier

### Published by

[![Elsevier](https://www.sciencedirect.com/us-east-1/prod/daf2b81ba46fd14fe56d83dd23f1b5feb4dc67db/image/elsevier-non-solus.svg)](https://www.sciencedirect.com/journal/expert-systems-with-applications "Go to Expert Systems with Applications on ScienceDirect")

,,,,,,,,,,

[View **PDF**](https://www.sciencedirect.com/science/article/pii/S0957417426002848/pdfft?md5=5a531bb6a6ce93ee42a2554d85c999f6&pid=1-s2.0-S0957417426002848-main.pdf)

[10.1016/j.eswa.2026.131371](https://doi.org/10.1016/j.eswa.2026.131371)

## Keywords

Brain tumor segmentation

;

Teacher-student learning

;

Feature distillation

;

Cross-modal fusion

;

Deep learning

- [Previous article in this issue](https://www.sciencedirect.com/science/article/pii/S095741742600237X)
- [Next article in this issue](https://www.sciencedirect.com/science/article/pii/S0957417426002666)

## 1\. Introduction

Brain and other tumors of the central nervous system (CNS) are the fifth most prevalent type of cancer, posing significant threats to global health (, ). Characterized by the abnormal growth of cells within the brain, these tumors can take diverse forms, ranging from benign to malignant, and may occur in various regions of the brain (,, ). According to the Central Brain Tumor Registry of the United States Annual Report, approximately 90,000 people are diagnosed with a primary brain tumor every year. Within this group, nearly one third of brain and CNS tumors exhibit malignant characteristics. Only 12% of adults manage to survive for five years following a brain tumor diagnosis (). These statistics highlight the complexities inherent in brain tumors, highlighting the urgent need for accurate brain tumor segmentation techniques (). Manual brain tumor segmentation is time-consuming, labor intensive and is prone to interobserver variability (,, ). Developing automatic brain tumor segmentation approaches is crucial to overcome these challenges, enabling more precise delineation of tumor boundaries and facilitating a comprehensive understanding of the complex nature of brain tumor.

Deep learning has revolutionized medical image analysis, particularly for brain tumor segmentation (,, ). Convolutional neural networks (CNNs) and transformer-based architectures have demonstrated remarkable capabilities in automatically extracting hierarchical and discriminative features from MRI scans (, ). These methods can capture both local and global contextual information, enabling precise delineation of tumor boundaries and subregions. By leveraging large-scale annotated datasets and advanced network designs, deep learning models have achieved substantial improvements over traditional hand-crafted feature-based approaches, reducing manual effort and interobserver variability while providing a scalable solution for clinical applications.

Medical imaging, particularly MRI, has become a key tool in the diagnosis and monitoring of brain tumor (). Different MRI sequences can offer invaluable insights into the tumor’s location, size, and morphology (, ). In recent years, the integration of multi-modal imaging has shown promise in enhancing the accuracy of brain tumor segmentation (,, ). By combining information from different MR modalities, such as fluid attenuated inversion recovery (Flair), T1-weighted, T2-weighted and T1-weighted post-contrast (T1c), shown in, researchers aim to exploit the complementary strengths of each modality and improve the accuracy of brain tumor segmentation. However, existing works primarily incorporate multiple modalities in a channel-wise manner to train segmentation networks (,,,, ). These approaches often overlook the potential to explore intricate relationships between various MR modalities, limiting the overall enhancement of feature representations.

![Fig. 1 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr1.jpg)

Download: Download high-res image (262KB)

In this work, we propose a novel hierarchical teacher-student learning framework for MR brain tumor segmentation. While teacher-student learning is typically employed to alleviate data scarcity and improve generalization. However, we utilize it for feature distillation to discover inherent relationships across multi-modal MR images. Additionally, we introduce a cross-modal fusion method to effectively integrate multi-modal features. The contributions of this work are summarized as follows:

(1) A novel hierarchical teacher-student learning framework is proposed for multi-modal MR brain tumor segmentation.

(2) A modality guidance module is proposed to enable multi-modal feature distillation by leveraging the strengths of teacher modalities to enhance the feature representations of student modalities.

(3) A cross-modal fusion module is proposed to explore multi-modal correlation and learn informative feature representations.

(4) Experimental results conducted on BraTS datasets demonstrate the superior performance of the proposed method over existing state-of-the-art methods.

The rest of the paper is organized as follows: The related works are described in. The network architecture is presented in. The experimental setting is introduced in. The experiment’s performance is presented and analyzed in. The discussion is provided in. Lastly, the conclusion is provided in.

## 2\. Related works

### 2.1. Brain tumor segmentation

Numerous methodologies have been developed for brain tumor segmentation, which can be broadly classified into three groups. Conventional image processing techniques, such as thresholding (), region growing (), and mathematical morphology (), they are effective in specific cases but struggle to generalize across complex brain tumor scenarios. Machine learning approaches (), which rely on hand-crafted features, such as texture, intensity, and shape, utilize classifiers such as support vector machines (SVM) () and random forests () for brain tumor segmentation. However, these methods depend on manually defined features that may fail to capture the complex patterns in medical images. Deep learning models, particularly U-Net () and its variations, have shown remarkable performance in brain tumor segmentation. For instance, introduced Edge U-Net (), which integrates boundary-related MRI data for improved brain tumor segmentation accuracy. proposed a denoising diffusion fusion network with fuzzy learning and iterative attention to improve 3D brain tumor segmentation. proposed a multi-modal brain tumor segmentation framework that adopts the hybrid fusion of modality-specific features using a self-supervised learning strategy. proposed a dual fuzzy segmentation approach that integrates fuzzy convolution and fuzzy attention to better model uncertainty and enhance focus on regions of interest. Recently, transformer-based methods () have gained attention due to their ability to leverage attention mechanisms for capturing long-range dependencies within images. For example, proposed a U-shaped network with a Swin Transformer encoder and a CNN-based decoder, connected via skip connections at multiple resolutions for brain tumor segmentation. Furthermore, Graph Convolutional Network (GCN)-based methods () have shown effectiveness in incorporating spatial dependencies, leading to more contextually aware segmentation results. For example, proposed a dual graph convolutional network for brain tumor segmentation, which includes a spatial-wise graph convolution module to capture extensive spatial dependencies and a channel-wise graph convolution module to model intricate contextual relationships within the image.

Despite these advancements, our proposed method stands out by introducing a novel hierarchical teacher-student learning framework that separates the four MR modalities into teacher modalities (Flair and T1c) and student modalities (T2 and T1). This approach leverages the strengths of the teacher modalities to guide the student modalities’ feature representation learning. This mechanism allows our proposed method to efficiently use the multi-modal data to improve segmentation accuracy and robustness.

### 2.2. Multi-modal feature fusion

Multi-modal feature fusion is a critical aspect of image analysis tasks, including image segmentation, object recognition, and scene understanding. Numerous works have explored multi-modal feature fusion techniques. For example, categorized multi-modal segmentation networks into three main groups: input-level fusion, layer-level fusion, and decision-level fusion. Early fusion, which combines features from different modalities at the input stage, is commonly used due to its simplicity and direct integration with the segmentation network architecture. For example, developed a 3D convolutional neural network with a multi-branch attention mechanism for brain tumor segmentation, in which the four MR modalities are concatenated as input. In contrast, late fusion focuses on combining features at higher levels of abstraction, emphasizing the exploration of complex relationships between modalities. More advanced designs employ adaptive and dynamic fusion strategies to better exploit complementary cues. For instance, proposed a flexible fusion network for multi-modal brain tumor segmentation, which can flexibly fuse arbitrary numbers of multi-modal information to explore complementary information while maintaining the specific characteristics of each modality. In recent years, attention mechanisms have gained significant attention in multi-modal fusion tasks, as they enable networks to emphasize the most informative features while suppressing irrelevant information. Spatial attention methods (, ) recalibrate feature maps to enhance the spatial attention of the network, enabling the model to focus on the most relevant regions in the image. Channel attention mechanisms (, ) help the network learn important features by focusing on the channel dimension, providing a way to identify useful features across various channels. Hybrid attention methods (,,, ) combine both spatial and channel attention to capture richer contextual information, improving the model’s ability to handle complex multi-modal data.

Overall, multi-modal fusion has evolved from simple concatenation to advanced attention-driven and adaptive strategies, greatly improving segmentation accuracy. However, many approaches still rely on straightforward fusion or overemphasize attention mechanisms, overlooking complementary modality relationships. To address this, our method integrates the strengths of both teacher and student modalities via the Modality Enhancement Module (MEM) and Modality Fusion Module (MFM). The MEM refines teacher modality features, and the MFM fuses them with student modality features, capturing both fine-grained and high-level contextual cues. This synergy enhances the representation of complex tumor characteristics, yielding superior multi-modal segmentation performance.

## 3\. Methodology

This section first presents the motivation behind the proposed method in, followed by an overview of the proposed segmentation network in. Subsequently, it offers a detailed explanation of the two primary components and their respective roles within the proposed architecture in and. Finally, provides a description of the training loss function.

### 3.1. Motivation behind the proposed method

MRI provides complementary modalities that emphasize different tissue properties, which is crucial for accurate brain tumor segmentation. Among the four commonly used modalities (Flair, T2, T1, and T1c), Flair and T2 are both T2-weighted and are particularly informative for delineating the whole tumor and edema, while T1 and T1c are T1-weighted and are more relevant to characterizing the tumor core and enhancing components. Notably, Flair can be viewed as a T2-weighted sequence with CSF suppression, typically yielding higher lesion conspicuity for edema than standard T2, and T1c is a contrast-enhanced T1-weighted image that highlights enhancing tumor more clearly than non-contrast T1. Based on this clinical hierarchy, we design a teacher–student pairing within each weighting family, where the teacher modality provides a stronger and more discriminative signal to guide its counterpart: Flair  →  T2 and T1c  →  T1. We implement this hierarchical cross-modal distillation via the proposed Modality Guidance Module (MGM), which transfers informative cues from the teacher to enhance the student feature representation, and further aggregate multi-modal information using the Cross-Modal Fusion Module (CMFM). In addition, we provide quantitative evidence by comparing several alternative grouping strategies in, where the proposed grouping consistently yields superior overall performance, validating the rationale of the teacher–student design. It is noted that teacher-student learning is commonly used to address challenges such as data scarcity and generalization. However, this paper proposes a novel application: utilizing it for multi-modal feature distillation for the first time.

### 3.2. Proposed network architecture

The proposed network adopts a multi-encoder-based architecture that integrates two key modules: the Modality Guidance Module (MGM) and the Cross-Modal Fusion Module (CMFM). The MGM is further composed of two components: the Modality Enhancement Module (MEM) and the Modality Fusion Module (MFM). The overall architecture is depicted in, with detailed configurations provided in. To handle modality-specific characteristics, four independent encoders are designed to extract modality-specific features effectively. Each encoder comprises convolutional blocks with instance normalization and LeakyReLU activation, followed by dilated convolutional blocks with residual connections for improved feature extraction. Down-sampling within the encoder path is achieved using a convolutional block with a stride of 2. The MEM and MFM are incorporated within the encoder path to refine the feature representations of teacher modalities (Flair and T1c) and student modalities (T2 and T1), respectively. These modules enable hierarchical feature enhancement and facilitate knowledge transfer between modalities. In the decoder path, upsampling operations are combined with the CMFM to integrate multi-modal features, followed by convolutional blocks with instance normalization and LeakyReLU activation. Similar to the encoder, the decoder also utilizes dilated convolutional blocks with residual connections to capture contextual information. The CMFM effectively fuses complementary features across modalities, further enhancing the feature representations. To ensure robust training and improve segmentation accuracy, deep supervision is applied in the segmentation decoder path. Segmentation outputs at multiple levels are passed through 1 × 1 × 1 convolutional blocks and combined via element-wise summation to generate the final segmentation output. This deep supervision strategy not only enhances gradient flow during training but also facilitates the fusion of multi-scale features for better segmentation results. Detailed explanations of the MGM and CMFM modules are provided in and, respectively.

![Fig. 2 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr2.jpg)

Download: Download high-res image (989KB)

Table 1. The detailed architecture of encoders in the proposed network. To simplify, only the encoder part is presented, the decoders are symmetrical. Conv and DConv are short for the convolution and dilated convolution. IN indicates Instance Normalization. MEM indicates the modality enhancement module, and MGM indicates the modality guidance module, including MEM and modality fusion module (MFM), the detailed descriptions of these two modules are elaborated in and.

| Level | Output Size | Operations for Teacher Modality | Operations for Student Modality |
| --- | --- | --- | --- |
| Level 1 | 128 × 128 × 128 | Conv 3 × 3 × 3, 8, IN, LeakyReLU | Conv 3 × 3 × 3, 8, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 8, IN, LeakyReLU | DConv 3 × 3 × 3, 8, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |
| Level 2 | 64 × 64 × 64 | Conv 3 × 3 × 3, 16, IN, LeakyReLU | Conv 3 × 3 × 3, 16, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 16, IN, LeakyReLU | DConv 3 × 3 × 3, 16, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |
| Level 3 | 32 × 32 × 32 | Conv 3 × 3 × 3, 32, IN, LeakyReLU | Conv 3 × 3 × 3, 32, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 32, IN, LeakyReLU | DConv 3 × 3 × 3, 32, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |
| Level 4 | 16 × 16 × 16 | Conv 3 × 3 × 3, 64, IN, LeakyReLU | Conv 3 × 3 × 3, 64, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 64, IN, LeakyReLU | DConv 3 × 3 × 3, 64, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |
| Level 5 | 8 × 8 × 8 | Conv 3 × 3 × 3, 128, IN, LeakyReLU | Conv 3 × 3 × 3, 128, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 128, IN, LeakyReLU | DConv 3 × 3 × 3, 128, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |
| Level 6 | 4 × 4 × 4 | Conv 3 × 3 × 3, 256, IN, LeakyReLU | Conv 3 × 3 × 3, 256, IN, LeakyReLU |
|  |  | DConv 3 × 3 × 3, 256, IN, LeakyReLU | DConv 3 × 3 × 3, 256, IN, LeakyReLU |
|  |  | MEM $\begin{bmatrix} \text{GMP} \\ \text{MLP} \\ \text{Sigmoid} \end{bmatrix}$ | MGM $\begin{bmatrix} \text{MEM} \\ \text{MFM} \end{bmatrix}$ |

### 3.3. Modality guidance module

To enhance the teacher-student learning framework, we propose a novel Modality Guidance Module (MGM) that leverages the strengths of teacher modalities (T1c and Flair) to enrich the feature representation of student modalities (T1 and T2). As illustrated in, the MGM consists of two components: the Modality Enhancement Module (MEM) and the Modality Fusion Module (MFM). The MEM first refines and strengthens the feature representation of teacher modalities. Building on these enhanced features, the MFM then guides and optimizes the feature learning of the student modalities. Through this dual-module design, the MGM enables more comprehensive and discriminative multi-modal feature learning for brain tumor segmentation.

![Fig. 3 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr3.jpg)

Download: Download high-res image (482KB)

The MEM refines the teacher feature *T <sub>i</sub>* through a multi-stage process that highlights the most salient information. It first applies a global max pooling (GMP) operation to extract the strongest spatial activations. The pooled feature is then passed through a multi-layer perceptron (MLP) followed by a sigmoid activation function (*σ*) to generate an attention weight. This weight is applied to the original input feature using residual learning, preserving useful information while enhancing discriminative cues relevant to tumor regions. The resulting refined teacher feature $Ti′$ is computed as:$Ti′=σ(MLP(GMP(Ti)))⊗Ti⊕Ti,$where ⊗ denotes element-wise multiplication, ⊕ denotes element-wise summation, and *σ* is the sigmoid function.

The MFM extends the distillation process by integrating the refined teacher features $Ti′$ with the student features *S <sub>j</sub>* from modality *j*. The two features are first concatenated along the channel dimension and then passed through a convolution layer to align and fuse their representations:$Sij′=Wc*[Ti′,Sj]+bc,$where \[ · \] denotes channel-wise concatenation, *W <sub>c</sub>* and *b <sub>c</sub>* are the convolutional kernel and bias, respectively.

Next, global average pooling (GAP) operation is applied to $Sij′$ to obtain a global contextual descriptor. This descriptor is fed into an MLP followed by a sigmoid activation to generate an attention weight, which adaptively modulates the fused feature map. Residual learning is then introduced by adding the original fused feature, yielding the refined student representation $Sj′$:$Sj′=σ(MLP(GAP(Sij′)))⊗Sij′⊕Sij′,$

By combining MEM and MFM, MGM performs explicit teacher-to-student feature distillation rather than simple aggregation. Specifically, MEM refines the teacher features (Flair, T1c) to generate a cleaner guidance signal, while MFM selectively transfers informative cues to the student streams (T2, T1) and suppresses redundant or misleading information to avoid negative transfer. Compared with symmetric fusion (e.g., concatenation/summation or generic attention), MGM explicitly models an asymmetric, directional pairing (Flair → T2, T1c → T1), improving interpretability and robustness. We further examine alternative pairings (e.g., T1c → T2 and Flair → T1) in.

### 3.4. Cross-modal fusion module

To achieve effective feature fusion across different modalities, a Cross-modal Fusion Module (CMFM) is proposed within the decoder path, as illustrated in. The CMFM can dynamically integrate modality-specific features to enhance segmentation performance. At the lowest network level, the input feature $f_{i n} \in \mathbb{R}^{4 C \times L \times W \times H}$ comprises features from the four MRI modalities, while higher levels include an additional upsampled feature $f_{i n} \in \mathbb{R}^{5 C \times L \times W \times H}$. For the Flair-T2 pair, the Flair feature $f_{F} \in \mathbb{R}^{C \times L \times W \times H}$ is processed through two convolutional blocks *ψ* and *ξ*, yielding modality-specific features *ψ* (*f <sub>F</sub>*) and *ξ* (*f <sub>F</sub>*). Meanwhile, the T2 feature $f_{T 2} \in \mathbb{R}^{C \times L \times W \times H}$ is passed through a convolutional block *ϕ*, resulting in a modality-specific *ϕ* (*f* <sub><em>T</em> 2</sub>). The cross-modal relevance is computed via element-wise multiplication followed by a Softmax:$W_{F T 2} = S o f t m a x \left(\xi \left(f_{F}\right) \bigotimes \phi \left(f_{T 2}\right)\right) ,$where ⊗ denotes element-wise multiplication. The resulting weight *W* <sub><em>FT</em> 2</sub> adaptively refines the Flair feature:

![Fig. 4 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr4.jpg)

Download: Download high-res image (344KB)

The weight *W* <sub><em>FT</em> 2</sub> is then multiplied with the feature *ψ* (*f <sub>F</sub>*) to refine the Flair feature *f <sub>F</sub>*, which is subsequently concatenated with the original Flair and T2 features to produce the fused feature *f* <sub><em>FT</em> 2</sub>.$f_{F T 2} = \left[\psi \left(f_{F}\right) \bigotimes W_{F T 2} , f_{F} , f_{T 2}\right] ,$where \[ · \] denotes channel-wise concatenation.

The same procedure is applied to other modality pairs (*f* <sub><em>T</em> 2 <em>F</em></sub>, *f* <sub><em>T</em> 1 <em>cT</em> 1</sub>, and *f* <sub><em>T</em> 1 <em>T</em> 1 <em>c</em></sub>), and the pairwise fused features are concatenated to form the final output:$f_{o u t} = \left[f_{F T 2} , f_{T 2 F} , f_{T 1 c T 1} , f_{T 1 T 1 c}\right] ,$

At higher decoder levels, the upsampled feature *f <sub>up</sub>* is concatenated with *f <sub>out</sub>*, yielding $f_{o u t} \in \mathbb{R}^{9 C \times L \times W \times H}$. It is worth mentioning that each modality-specific features *f <sub>F</sub>, f* <sub><em>T</em> 2</sub>, *f* <sub><em>T</em> 1 <em>c</em></sub>, and *f* <sub><em>T</em> 1</sub> contributes to the final output feature only once. This bidirectional design ensures that the cross-modal interactions are fully utilized.

The proposed CMFM is specifically designed to facilitate teacher-student feature distillation, leveraging multi-level pairwise fusion and adaptive weighting to preserve modality-specific information. Its bidirectional information flow allows more robust and comprehensive feature integration compared to traditional fusion strategies, making it distinct and highly effective. Detailed discussions on fusion strategy selection and bidirectional design are provided in and.

### 3.5. Training loss function

In this work, the standard Dice loss defined in is employed for brain tumor segmentation.(1) $L_{d i c e} = 1 - 2 \frac{\sum_{i = 1}^{C} \sum_{j = 1}^{N} p_{i j} g_{i j} + \epsilon}{\sum_{i = 1}^{C} \sum_{j = 1}^{N} \left(p_{i j} + g_{i j}\right) + \epsilon} ,$where *N* stands for the total voxel count in the image, *C* refers to the number of segmentation classes, *p <sub>ij</sub>*  ∈ \[0, 1\] and *g <sub>ij</sub>*  ∈ \[0, 1\] represent the prediction probability and ground truth value, respectively, of voxel *i* belonging to class *j*. ϵ is a small constant included to prevent division by zero.

## 4\. Experimental setup

### 4.1. Data description and implementation details

BraTS (Brain Tumor Segmentation) datasets (, ) from 2018, 2019 and 2020 are employed to evaluate the proposed method, which contain 285, 335 and 369 training cases, respectively. Each case includes four MR modalities: Flair, T2, T1 and T1c, accompanied by expert annotations. Tumor annotations are divided into three classes: whole tumor, tumor core, and enhancing tumor. Note that tumor core consists of enhancing tumor, non-enhancing tumor, and necrotic tumor regions as shown in. All images are pre-processed by cropping and resizing to 128 × 128 × 128, followed by N4ITK bias correction () and intensity normalization.

The segmentation network is implemented in Keras and trained on an NVIDIA GeForce RTX 4090 GPU (24 GB) using an 8:2 train-test split. The learning rate is initially set to 0.0005 and halved every 10 epochs if the validation loss does not improve. Early stopping is applied, terminating training when the validation loss shows no improvement for 50 consecutive epochs. Model parameters are optimized using the Adam optimizer.

### 4.2. Evaluation metrics

To quantitatively assess the efficacy of the proposed methodology, two evaluation metrics are used: the Dice similarity coefficient (DSC), 95% Hausdorff distance (HD).

Dice Similarity Coefficient quantifies the spatial overlap between the predicted tumor region and the ground truth. DSC ranges from 0 (no overlap) to 1 (perfect overlap), higher DSC values indicate better segmentation performance.(2) $D S C = \frac{2 \left|P \cap G\right|}{\left|P\right| + \left|G\right|} ,$where *P* and *G* represent the sets of voxels in the predicted and ground truth segmentations, respectively.

Hausdorff Distance measures the maximum distance between the points in the predicted tumor boundary and the true tumor boundary, lower HD indicates a better match between the predicted and true tumor boundaries.(3) $HD=max{maxx∈Xminy∈Yd(x,y),maxy∈Yminx∈Xd(x,y)},$where *X* and *Y* denote the predicted tumor boundary and the true tumor boundary, respectively, and *d* ( ·,  · ) represents the minimum Euclidean distance.

## 5\. Experimental results and analysis

### 5.1. Ablation experiments

To assess the impact of each proposed module on brain tumor segmentation performance, a comprehensive ablation study is conducted, with results summarized in using the BraTS 2018 dataset. A baseline model without any of the proposed modules is first established. The deep supervision (DS), modality guidance module (MGM), and cross-modal fusion module (CMFM) are then individually incorporated to evaluate their specific contributions. Incorporating DS improves the average DSC by 0.8% and reduces the average HD by 21.1% over the baseline. Adding MGM further enhances performance, increasing the average DSC by 1.1% and reducing the average HD by 28.1%, owing to more effective multi-modal feature utilization. The inclusion of CMFM yields an additional 1.2% gain in DSC and 28.1% reduction in HD, reflecting improved feature representation through multi-modal fusion. Combining all three modules achieves the best overall segmentation performance. These trends are consistent on the BraTS 2019 and 2020 datasets, as shown in and [^1].

Table 2. Ablation experiments conducted on the BraTS 2018 dataset based on Dice Similarity Coefficient (%) and 95% Hausdorff Distance (mm). DS, MGM and CMFM correspond to deep supervision, modality guidance module, and cross-modal fusion module, respectively. The bold annotations highlight the highest scores attained. \* denotes statistical significance over Baseline (Wilcoxon test, *p*  <.05).

<table><thead><tr><th colspan="4">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Baseline</th><th>DS</th><th>MGM</th><th>CMFM</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>–</strong></td><td>85.6</td><td>85.6</td><td>77.4</td><td>82.9</td><td>9.3</td><td>4.4</td><td>3.3</td><td>5.7</td></tr><tr><td>√</td><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>86.8</strong> *</td><td>85.8</td><td>78.2*</td><td>83.6</td><td>5.0*</td><td>5.0</td><td>3.5</td><td>4.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td><strong>–</strong></td><td>86.1</td><td>85.8*</td><td><strong>79.5</strong> *</td><td>83.8</td><td>5.3</td><td>4.3*</td><td><strong>2.8</strong> *</td><td><strong>4.1</strong></td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>86.5*</td><td><strong>86.3</strong></td><td>78.9*</td><td><strong>83.9</strong></td><td><strong>4.9</strong> *</td><td><strong>4.3</strong></td><td>3.0*</td><td><strong>4.1</strong></td></tr></tbody></table>

Table 3. Ablation experiments conducted on the BraTS 2019 dataset based on Dice Similarity Coefficient (%) and 95% Hausdorff Distance (mm). DS, MGM and CMFM correspond to deep supervision, modality guidance module, and cross-modal fusion module, respectively. The bold annotations highlight the highest scores attained. \* denotes statistical significance over Baseline (Wilcoxon test, *p*  <.05).

<table><thead><tr><th colspan="4">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Baseline</th><th>DS</th><th>MGM</th><th>CMFM</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>86.2</strong></td><td>84.9</td><td>78.1</td><td>83.1</td><td>6.5</td><td>5.7</td><td>4.6</td><td>5.6</td></tr><tr><td>√</td><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td>85.7</td><td>86.0*</td><td>78.6</td><td>83.4</td><td>5.8</td><td>4.4*</td><td><strong>2.8</strong></td><td>4.3</td></tr><tr><td>√</td><td>√</td><td>√</td><td><strong>–</strong></td><td>85.9</td><td>86.0*</td><td>79.6*</td><td>83.8</td><td>5.7</td><td>4.4*</td><td>3.1*</td><td>4.4</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td><strong>86.2</strong></td><td><strong>86.8</strong> *</td><td><strong>79.7</strong> *</td><td><strong>84.2</strong></td><td><strong>5.6</strong> *</td><td><strong>4.1</strong> *</td><td>2.9*</td><td><strong>4.2</strong></td></tr></tbody></table>

Table 4. Ablation experiments conducted on the BraTS 2020 dataset based on dice similarity coefficient (%) and 95% Hausdorff distance (mm). DS, MGM and CMFM correspond to deep supervision, modality guidance module, and cross-modal fusion module, respectively. The bold annotations highlight the highest scores attained. \* denotes statistical significance over Baseline (Wilcoxon test, *p*  <.05).

<table><thead><tr><th colspan="4">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Baseline</th><th>DS</th><th>MGM</th><th>CMFM</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>–</strong></td><td>86.2</td><td>86.2</td><td>78.9</td><td>83.8</td><td>6.5</td><td>5.1</td><td>3.4</td><td>5.0</td></tr><tr><td>√</td><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td>86.9*</td><td>86.7*</td><td>79.5*</td><td>84.4</td><td>5.4</td><td>4.1</td><td>2.8*</td><td>4.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td><strong>–</strong></td><td>87.5*</td><td>87.4*</td><td>79.8*</td><td>84.9</td><td><strong>5.2</strong> *</td><td><strong>3.5</strong> *</td><td>2.6*</td><td><strong>3.8</strong></td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td><strong>87.6</strong> *</td><td><strong>87.8</strong> *</td><td><strong>82.4</strong> *</td><td><strong>85.9</strong></td><td>7.4</td><td>3.8</td><td><strong>2.3</strong> *</td><td>4.5</td></tr></tbody></table>

### 5.2. Comparison with the state-of-the-art methods

We compare the proposed method with existing state-of-the-art approaches, categorized into three groups: U-Net-based, attention-based, and transformer-based methods. The comparison results on the BraTS 2018, 2019 and 2020 datasets are summarized in,,. From, it can be observed that the transformer-based method () achieves the best DSC on the whole tumor, while the U-Net-based method () attains the best DSC and HD on tumor core. However, the proposed method achieves the best DSC for the enhancing tumor and the highest average DSC overall. It also attains the best HD for the whole tumor, enhancing tumor, and average results. Notably, the proposed fusion method outperforms attention-based methods, including CBAM (), SENet () and NLNet (). Specifically, when compared with the CBAM (), which computes attention maps along channel and spatial dimensions, the proposed method achieves a significant improvement of 1.5% in terms of average DSC and 12.8% in terms of average HD, respectively. Furthermore, compared to the SENet (), which models inter-dependencies between channels of features, the proposed method exhibits a significant enhancement of 1.0% in terms of average DSC and 6.8% in terms of average HD. Additionally, compared to the NLNet (), which captures long-range feature dependencies, the proposed method achieves a substantial improvement of 1.7% in terms of average DSC and 12.8% in terms of average HD. A key factor contributing to this superior performance is the teacher-student modality design, where features are distilled from teacher modalities to student modalities, ensuring utilization of complementary information across modalities. The effectiveness of the proposed method is further validated on the BraTS 2019 and 2020 datasets, as shown in and [^2], showing similar improvements and confirming the robustness of the approach.

Table 5. Experimental results with the state-of-the-art methods based on dice similarity coefficient (%) and 95% Hausdorff distance (mm) on the BraTS 2018 dataset. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Attention U-Net ()</td><td>85.9*</td><td>77.1*</td><td>74.5*</td><td>79.2</td><td>5.3</td><td>4.9</td><td>4.7*</td><td>5.0</td></tr><tr><td>M2GCNet</td><td>85.1*</td><td>84.3*</td><td>76.1*</td><td>81.8</td><td>5.7</td><td>6.0*</td><td>4.3*</td><td>5.3</td></tr><tr><td>Multi-BTS ()</td><td>86.5</td><td><strong>86.8</strong></td><td>77.5*</td><td>83.6</td><td>5.2</td><td><strong>4.1</strong></td><td>3.1*</td><td><strong>4.1</strong></td></tr><tr><td>SENet ()</td><td>86.4</td><td>85.5</td><td>77.4</td><td>83.1</td><td>5.1</td><td>4.7</td><td>3.4</td><td>4.4</td></tr><tr><td>CBAM ()</td><td>85.1*</td><td>85.8</td><td>77.2*</td><td>82.7</td><td>5.8*</td><td>4.6</td><td>3.7*</td><td>4.7</td></tr><tr><td>NLNet ()</td><td>84.9*</td><td>84.8</td><td>77.7*</td><td>82.5</td><td>5.9*</td><td>4.6</td><td>3.6*</td><td>4.7</td></tr><tr><td>Swin UNETR ()</td><td>86.1</td><td>81.2*</td><td><strong>78.9</strong></td><td>82.1</td><td>8.7*</td><td>7.1*</td><td>3.9*</td><td>6.6</td></tr><tr><td>VT-UNet ()</td><td><strong>86.6</strong></td><td>84.1*</td><td>78.0</td><td>82.9</td><td>6.7</td><td>6.1*</td><td>3.3</td><td>5.4</td></tr><tr><td>Nestedformer ()</td><td>70.7*</td><td>73.1*</td><td>70.5*</td><td>71.4</td><td>27.3*</td><td>42.2*</td><td>12.5*</td><td>27.3</td></tr><tr><td><strong>Ours</strong></td><td>86.5</td><td>86.3</td><td><strong>78.9</strong></td><td><strong>83.9</strong></td><td><strong>4.9</strong></td><td>4.3</td><td><strong>3.0</strong></td><td><strong>4.1</strong></td></tr></tbody></table>

Table 6. Experimental results with the state-of-the-art methods based on dice similarity coefficient (%) and 95% Hausdorff distance (mm) on the BraTS 2019 dataset. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Attention U-Net ()</td><td>81.3</td><td>77.8*</td><td>76.7*</td><td>78.6</td><td>7.5</td><td>7.2*</td><td>3.2</td><td>6.0</td></tr><tr><td>M2GCNet</td><td>85.6</td><td>84.3</td><td>78.3</td><td>82.7</td><td>6.4</td><td>5.0</td><td>3.8</td><td>5.1</td></tr><tr><td>Multi-BTS ()</td><td>86.7</td><td><strong>87.1</strong></td><td>78.9</td><td><strong>84.2</strong></td><td>6.2</td><td>4.3</td><td>6.7</td><td>5.7</td></tr><tr><td>SENet ()</td><td>86.3</td><td>85.7</td><td>77.9*</td><td>83.3</td><td>6.0</td><td>4.5</td><td>3.5*</td><td>4.7</td></tr><tr><td>CBAM ()</td><td>86.2</td><td>85.7*</td><td>79.4</td><td>83.8</td><td><strong>4.7</strong></td><td>4.9</td><td>3.4*</td><td>4.3</td></tr><tr><td>NLNet ()</td><td>86.0</td><td>87.0</td><td>79.3</td><td>84.1</td><td>5.5</td><td>5.0</td><td>3.3</td><td>4.6</td></tr><tr><td>Swin UNETR ()</td><td><strong>87.3</strong></td><td>84.1*</td><td>81.3</td><td><strong>84.2</strong></td><td>8.2</td><td>5.5*</td><td>3.4</td><td>5.7</td></tr><tr><td>VT-UNet ()</td><td>87.2</td><td>84.8*</td><td><strong>79.7</strong></td><td>83.9</td><td>5.8</td><td>5.7*</td><td>3.1</td><td>4.9</td></tr><tr><td>Nestedformer ()</td><td>77.4*</td><td>84.9</td><td>79.6</td><td>80.6</td><td>19.4*</td><td>13.3*</td><td>4.6*</td><td>12.4</td></tr><tr><td><strong>Ours</strong></td><td>86.2</td><td>86.8</td><td><strong>79.7</strong></td><td><strong>84.2</strong></td><td>5.6</td><td><strong>4.1</strong></td><td><strong>2.9</strong></td><td><strong>4.2</strong></td></tr></tbody></table>

Table 7. Experimental results with the state-of-the-art methods based on dice similarity coefficient (%) and 95% Hausdorff distance (mm) on the BraTS 2020 dataset. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Attention U-Net ()</td><td>79.6</td><td>80.8</td><td>73.8*</td><td>78.1</td><td>7.2</td><td>4.9</td><td>3.9*</td><td>5.3</td></tr><tr><td>M2GCNet</td><td>85.9*</td><td>85.3*</td><td>77.1*</td><td>82.8</td><td>5.7</td><td>5.1</td><td>3.8*</td><td>4.9</td></tr><tr><td>Multi-BTS ()</td><td>87.3</td><td>86.3*</td><td>77.6*</td><td>83.7</td><td>5.6</td><td>4.4</td><td>3.1*</td><td>4.4</td></tr><tr><td>SENet ()</td><td>87.2*</td><td>87.0*</td><td>79.9*</td><td>84.7</td><td><strong>5.1</strong></td><td>3.9</td><td>2.6*</td><td><strong>3.9</strong></td></tr><tr><td>CBAM ()</td><td>86.7*</td><td>85.8*</td><td>78.8*</td><td>83.8</td><td><strong>5.1</strong></td><td>4.2</td><td>3.1*</td><td>4.1</td></tr><tr><td>NLNet ()</td><td>86.7*</td><td>85.7*</td><td>78.6*</td><td>83.7</td><td>5.8</td><td>4.5</td><td>3.1*</td><td>4.5</td></tr><tr><td>Swin UNETR ()</td><td><strong>87.8</strong></td><td>85.4</td><td>81.7</td><td>85.0</td><td>7.0</td><td>5.3*</td><td>2.9</td><td>5.1</td></tr><tr><td>VT-UNet ()</td><td>87.6</td><td>85.9</td><td>80.4*</td><td>84.6</td><td>5.8</td><td>5.5</td><td>2.6</td><td>4.6</td></tr><tr><td>Nestedformer ()</td><td>83.2*</td><td>81.4*</td><td>78.9*</td><td>81.2</td><td>13.5</td><td>15.0*</td><td>3.7</td><td>10.7</td></tr><tr><td><strong>Ours</strong></td><td>87.6</td><td><strong>87.8</strong></td><td><strong>82.4</strong></td><td><strong>85.9</strong></td><td>7.4</td><td><strong>3.8</strong></td><td><strong>2.3</strong></td><td>4.5</td></tr></tbody></table>

### 5.3. Analysis on the modality combinations in MGM

We further investigate the impact of alternative modality combinations within the MGM, with comparative results presented in. Notably, the combination where Flair serves as the teacher modality for T2, and T1c serves as the teacher modality for T1, demonstrates superior performance compared to the pairing of Flair with T1 and T1c with T2. This configuration yields improvements of 1.7% and 16.3% in terms of average DSC and average HD, respectively. further illustrates that this pairing consistently outperforms the alternative, highlighting the importance of selecting complementary modality pairs in multi-modal learning. These synergistic effects contribute to more robust and accurate segmentation performance.

Table 8. Analysis on the fusion methods in CMFM. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Addition</td><td>83.2*</td><td>80.8*</td><td>74.3*</td><td>79.4</td><td>7.8*</td><td>6.7*</td><td>5.1*</td><td>6.5</td></tr><tr><td><strong>Concatenation</strong></td><td><strong>86.5</strong></td><td><strong>86.3</strong></td><td><strong>78.9</strong></td><td><strong>83.9</strong></td><td><strong>4.9</strong></td><td><strong>4.3</strong></td><td><strong>3.0</strong></td><td><strong>4.1</strong></td></tr></tbody></table>

Table 9. Analysis on the bidirectional feature method in CMFM. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Single-directional feature</td><td>85.1*</td><td>83.9*</td><td>76.9*</td><td>82.0</td><td>5.7*</td><td>6.5*</td><td>4.4*</td><td>5.5</td></tr><tr><td><strong>Bidirectional feature</strong></td><td><strong>86.5</strong></td><td><strong>86.3</strong></td><td><strong>78.9</strong></td><td><strong>83.9</strong></td><td><strong>4.9</strong></td><td><strong>4.3</strong></td><td><strong>3.0</strong></td><td><strong>4.1</strong></td></tr></tbody></table>

Table 10. Analysis on the modality combinations in MGM. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method (teacher-student)</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Flair-T1, T1c-T2</td><td>85.3*</td><td>85.0*</td><td>77.3*</td><td>82.5</td><td>6.6*</td><td>4.8*</td><td>3.3*</td><td>4.9</td></tr><tr><td><strong>Flair-T2, T1c-T1</strong></td><td><strong>86.5</strong></td><td><strong>86.3</strong></td><td><strong>78.9</strong></td><td><strong>83.9</strong></td><td><strong>4.9</strong></td><td><strong>4.3</strong></td><td><strong>3.0</strong></td><td><strong>4.1</strong></td></tr></tbody></table>

![Fig. 5 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr5.jpg)

Download: Download high-res image (1MB)

![Fig. 6 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr6.jpg)

Download: Download high-res image (2MB)

![Fig. 7 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr7.jpg)

Download: Download high-res image (570KB)

![Fig. 8 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr8.jpg)

Download: Download high-res image (944KB)

![Fig. 9 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr9.jpg)

Download: Download high-res image (282KB)

![Fig. 10 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr10.jpg)

Download: Download high-res image (208KB)

![Fig. 11 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr11.jpg)

Download: Download high-res image (263KB)

### 5.4. Visualization of the segmentation results

We first present visualizations of ablation segmentation results in. Each subfigure shows a slice from a brain MRI scan, with the corresponding ground truth and the prediction result generated by the proposed method overlaid onto the original Flair modality. To enhance interpretability, zoomed-in views of critical regions are included for each slice. Additionally, the average DSC across the three tumor subregions is provided below each case for quantitative evaluation. In the first case, the baseline detects false isolated tumor regions outside the true tumor boundaries and struggles to accurately identify necrotic and non-enhancing tumor regions. Incorporating DS, MGM, and CMFM significantly improves these detections, as reflected by an increase in average DSC. In the second case, the baseline detects false edema regions, which are progressively corrected by the proposed modules, further improving segmentation accuracy. In the third case, the baseline misclassifies necrotic and non-enhancing tumor regions. While the proposed strategies mitigate some inaccuracies, the prediction still deviates from the ground truth in detecting necrotic and non-enhancing regions. We analyze the failure cases in to better understand the cause of these discrepancies and propose potential solutions.

presents visual comparisons between our proposed method and several state-of-the-art (SOTA) approaches. Each subfigure shows the Flair modality, the ground truth, and predictions from different methods. In the first and third cases, most comparative methods exhibit over-segmentation, while VT-UNet partially fails to delineate the necrotic and non-enhancing tumor regions. The second case, with complex tumor morphology, poses additionally challenges, leading competing methods to over-segment or misclassify necrotic and non-enhancing regions as edema. In contrast, our method yields more accurate tumor boundaries, reduces false positives, and achieves robust performance even for complex tumor shapes. These qualitative improvements demonstrate the effectiveness of the proposed method.

### 5.5. Visualization of failure cases

We analyze failure cases from our segmentation framework to identify potential limitations and areas for improvement. presents examples of such cases. In the first and second cases, the proposed method exhibits over-segmentation along tumor boundaries, particularly in low-contrast regions, resulting in false positives. This indicates difficulty in differentiating subtle tissue intensity variations near edges. In the third case, part of a small enhancing tumor is missed, while the edema region is over-segmented. In the fourth case, necrotic and non-enhancing tumor regions are under-segmented.

To address these issues, several improvements are planned. For over- and under-segmentation, a boundary-aware loss will be introduced to guide the network to focus on tumor boundaries. Contrast-enhancement techniques can help the model capture subtle intensity differences in low-contrast areas. For missing small or irregular regions, uncertainty modeling, such as Monte Carlo Dropout, will be employed to increase prediction confidence and improve segmentation accuracy.

### 5.6. Visualization of the feature maps

In addition to segmentation results, analyzing feature maps provides valuable insights into the mechanisms of brain tumor segmentation. The feature maps are shown in, demonstrating progressive improvements with the incorporation of the proposed modules. In the first case, the baseline highlights the general tumor region but lacks specificity in distinguishing subregions such as edema and tumor core. The proposed modules progressively refine the feature maps, improving discrimination between tumor subregions. In the next case, the baseline detects a broader tumor area, including some false positives. Deep supervision (DS) corrects these inaccuracies, and the combination of the Modality Guidance Module (MGM) and Cross-modal Fusion Module (CMFM) further enhances the delineation between edema and tumor core regions. In the last case, the baseline struggles to identify the tumor core, while the proposed modules progressively improve the detection of the target regions. Visualizing these feature maps provides a deeper understanding of each module’s contribution: DS facilitates hierarchical multi-scale feature extraction, MGM enables multi-modal feature distillation, and CMFM enhances cross-modal feature fusion. Collectively, these components lead to more accurate and reliable segmentation results.

## 6\. Discussion

### 6.1. Analysis on the fusion strategies in CMFM

In this section, we investigate two fusion strategies to combine the original features with the fused feature *f* <sub><em>FT</em> 2</sub>, *f* <sub><em>T</em> 2 <em>F</em></sub>, *f* <sub><em>T</em> 1 <em>cT</em> 1</sub> and *f* <sub><em>T</em> 1 <em>T</em> 1 <em>c</em></sub> within the cross-modal fusion module (CMFM). Specifically, we compare the effectiveness of concatenation versus addition operations. The comparative results are summarized in. It is evident that employing concatenation operation results in an improvement of 5.7% and 36.9% in terms of average DSC and average HD, respectively, compared to addition operation. Furthermore, shows that the concatenation method generally achieves higher Dice Similarity Coefficients and lower Hausdorff Distances than addition. The superiority of concatenation stems from its ability to preserve the original spatial dimensions of the features, ensuring that no information is lost during the fusion process. In contrast, addition operation combines the feature values element-wise, which may result in information loss and reduced representational capacity.

### 6.2. Analysis on bidirectional features in CMFM

We further investigate the use of bidirectional features within the CMFM, with comparative results presented in. Using bidirectional feature pairs, such as *f* <sub><em>FT</em> 2</sub> and *f* <sub><em>T</em> 2 <em>F</em></sub>, *f* <sub><em>T</em> 1 <em>cT</em> 1</sub> and *f* <sub><em>T</em> 1 <em>T</em> 1 <em>c</em></sub>, leads to improved segmentation performance, achieving an improvement of 2.3% and 25.5% in terms of average DSC and average HD, respectively, compared to single-directional features. From, it can be observed that using bidirectional features obtain better results than using single-directional features. This improvement can be attributed to the complementary nature of bidirectional features, which capture information from multiple modalities and directions. Incorporating bidirectional features enables the model to obtain a more comprehensive representation of the brain tumor, resulting in more accurate segmentation results.

### 6.3. Analysis on the effectiveness of MEM and MFM in MGM

We also analyze the effectiveness of MEM and MFM within the MGM, with comparative results summarized in. The baseline uses DS as the backbone. From, it can be observed that employing MEM along achieves 83.5% and 4.2 mm in terms of average DSC and average HD, respectively. Similarly, using MEM along results in 81.8% and 5.0 mm in terms of average DSC and average HD, respectively. However, combining both MEM and MFM achieves the best performance, with 83.9% and 4.1 mm in terms of average DSC and average HD, respectively. further demonstrates that combining both MEM and MFM in MGM consistently outperforms using either module individually.

Table 11. Analysis on the effectiveness of MEM and MFM in MGM. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>MEM</td><td><strong>86.6</strong></td><td>86.1</td><td>77.9</td><td>83.5</td><td>5.0</td><td>4.5</td><td>3.2</td><td>4.2</td></tr><tr><td>MFM</td><td>84.6*</td><td>83.6*</td><td>77.2*</td><td>81.8</td><td>5.7*</td><td>5.5*</td><td>3.8*</td><td>5.0</td></tr><tr><td><strong>MEM+MFM</strong></td><td>86.5</td><td><strong>86.3</strong></td><td><strong>78.9</strong></td><td><strong>83.9</strong></td><td><strong>4.9</strong></td><td><strong>4.3</strong></td><td><strong>3.0</strong></td><td><strong>4.1</strong></td></tr></tbody></table>

![Fig. 12 dummy alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0957417426002848-gr12.jpg)

Download: Download high-res image (206KB)

### 6.4. Limitation and future work

Despite the promising performance of the proposed hierarchical teacher-student framework, several limitations remain and motivate future research. First, although the current method achieves accurate segmentation, boundary delineation for complex or low-contrast tumor regions remains challenging. To address this, we plan to integrate region-specific or boundary-aware loss functions to further enhance the detection of tumor edges and improve the delineation of subregions. Second, to improve robustness in challenging scenarios, we plan to incorporate uncertainty modeling to guide adaptive predictions in low-confidence regions. Third, to strengthen the clinical applicability of the method, we plan to evaluate the framework on multi-institutional datasets, including real-world cases with varying image quality and missing-modality scenarios commonly encountered in routine clinical acquisition. These efforts are expected to enhance segmentation accuracy, reliability, and translational potential in clinical practice.

## 7\. Conclusion

This paper proposes a novel hierarchical teacher-student learning framework for MR brain tumor segmentation. Notably, for the first time, we employ teacher-student learning for multi-modal feature distillation. Furthermore, a modality guidance module is proposed to leverage the strengths of teacher modalities to enhance the feature representation within the student modalities. Additionally, a cross-modal fusion module is designed to further refine multi-modal feature representations. Experimental results conducted on BraTS datasets demonstrate the superior performance of the proposed method compared to state-of-the-art approaches. Moreover, beyond its efficacy in brain tumor segmentation, this approach shows promise for enhancing performance in diverse domains involving multi-modal data fusion.

## CRediT authorship contribution statement

**Tongxue Zhou:** Conceptualization, Methodology, Software, Validation, Writing – original draft, Writing – review & editing, Funding acquisition. **Su Ruan:** Writing – review & editing. **Jinming Duan:** Writing – review & editing. **Haigen Hu:** Writing – review & editing. **Yanda Meng:** Writing – review & editing. **Ling Huang:** Writing – review & editing. **Defu Yang:** Writing – review & editing. **Bingbing Jiang:** Writing – review & editing. **Tingjin Luo:** Writing – review & editing. **Zhiwei Ji:** Writing – review & editing. **Baiying Lei:** Writing – review & editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (No. ).

## Data availability

Data will be made available on request.

## References

[^1]: Table 4. Ablation experiments conducted on the BraTS 2020 dataset based on dice similarity coefficient (%) and 95% Hausdorff distance (mm). DS, MGM and CMFM correspond to deep supervision, modality guidance module, and cross-modal fusion module, respectively. The bold annotations highlight the highest scores attained. \* denotes statistical significance over Baseline (Wilcoxon test, *p*  <.05).

<table><thead><tr><th colspan="4">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Baseline</th><th>DS</th><th>MGM</th><th>CMFM</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td><strong>–</strong></td><td>86.2</td><td>86.2</td><td>78.9</td><td>83.8</td><td>6.5</td><td>5.1</td><td>3.4</td><td>5.0</td></tr><tr><td>√</td><td>√</td><td><strong>–</strong></td><td><strong>–</strong></td><td>86.9*</td><td>86.7*</td><td>79.5*</td><td>84.4</td><td>5.4</td><td>4.1</td><td>2.8*</td><td>4.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td><strong>–</strong></td><td>87.5*</td><td>87.4*</td><td>79.8*</td><td>84.9</td><td><strong>5.2</strong> *</td><td><strong>3.5</strong> *</td><td>2.6*</td><td><strong>3.8</strong></td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td><strong>87.6</strong> *</td><td><strong>87.8</strong> *</td><td><strong>82.4</strong> *</td><td><strong>85.9</strong></td><td>7.4</td><td>3.8</td><td><strong>2.3</strong> *</td><td>4.5</td></tr></tbody></table>

[^2]: Table 7. Experimental results with the state-of-the-art methods based on dice similarity coefficient (%) and 95% Hausdorff distance (mm) on the BraTS 2020 dataset. The bold annotations highlight the highest scores attained. \* denotes statistical significance over other methods (Wilcoxon test, *p*  <.05).

<table><thead><tr><th rowspan="2">Method</th><th colspan="4">Dice Similarity Coefficient (%) ↑</th><th colspan="4">95% Hausdorff Distance (mm) ↓</th></tr><tr><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th><th>Whole</th><th>Core</th><th>Enh.</th><th>Mean</th></tr></thead><tbody><tr><td>Attention U-Net ()</td><td>79.6</td><td>80.8</td><td>73.8*</td><td>78.1</td><td>7.2</td><td>4.9</td><td>3.9*</td><td>5.3</td></tr><tr><td>M2GCNet</td><td>85.9*</td><td>85.3*</td><td>77.1*</td><td>82.8</td><td>5.7</td><td>5.1</td><td>3.8*</td><td>4.9</td></tr><tr><td>Multi-BTS ()</td><td>87.3</td><td>86.3*</td><td>77.6*</td><td>83.7</td><td>5.6</td><td>4.4</td><td>3.1*</td><td>4.4</td></tr><tr><td>SENet ()</td><td>87.2*</td><td>87.0*</td><td>79.9*</td><td>84.7</td><td><strong>5.1</strong></td><td>3.9</td><td>2.6*</td><td><strong>3.9</strong></td></tr><tr><td>CBAM ()</td><td>86.7*</td><td>85.8*</td><td>78.8*</td><td>83.8</td><td><strong>5.1</strong></td><td>4.2</td><td>3.1*</td><td>4.1</td></tr><tr><td>NLNet ()</td><td>86.7*</td><td>85.7*</td><td>78.6*</td><td>83.7</td><td>5.8</td><td>4.5</td><td>3.1*</td><td>4.5</td></tr><tr><td>Swin UNETR ()</td><td><strong>87.8</strong></td><td>85.4</td><td>81.7</td><td>85.0</td><td>7.0</td><td>5.3*</td><td>2.9</td><td>5.1</td></tr><tr><td>VT-UNet ()</td><td>87.6</td><td>85.9</td><td>80.4*</td><td>84.6</td><td>5.8</td><td>5.5</td><td>2.6</td><td>4.6</td></tr><tr><td>Nestedformer ()</td><td>83.2*</td><td>81.4*</td><td>78.9*</td><td>81.2</td><td>13.5</td><td>15.0*</td><td>3.7</td><td>10.7</td></tr><tr><td><strong>Ours</strong></td><td>87.6</td><td><strong>87.8</strong></td><td><strong>82.4</strong></td><td><strong>85.9</strong></td><td>7.4</td><td><strong>3.8</strong></td><td><strong>2.3</strong></td><td>4.5</td></tr></tbody></table>