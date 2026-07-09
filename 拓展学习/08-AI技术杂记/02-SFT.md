---
tags:
  - LLMs/SFT
  - LLMs/CoT
---

SFT： Supervised Fine-Tuning，监督微调
目的：学会语言结构、知识和统计规律（风格和内容）。（但不知道什么是好的回答）
```
预训练（Pretrain）
   ↓
SFT（监督微调）
   ↓
RLHF / DPO / PPO（对齐阶段）
```

RLHF等方法：与人类偏好和价值观等抽象内容对齐

SFT目标函数是一个标准的交叉熵：
$$
\mathcal{L} = -log P(output | input)
$$

## SFT-CoT

训练数据结构化更为明显，有推理的结构
```text
Q: 症状A + 症状B + 影像特征C，最可能的诊断是？
A: 
Step 1: 分析症状A...
Step 2: 结合影像特征C...
Step 3: 排除疾病X...
Conclusion: 可能是疾病Y
```

不代表逻辑能力本身

## 多模态对齐
```text
[图像embedding + 文本问题] → 专业回答
```


| **模块** |          **类比**          |
| -------- |:--------------------------:|
| Pretrain |          学会语言          |
| SFT      | 学会“怎么当医生/助手/专家” |
| CoT      |    学会“怎么思考给人看”    |
| RAG      |         查资料能力         |
| RLHF     |         学会“做人”         |
