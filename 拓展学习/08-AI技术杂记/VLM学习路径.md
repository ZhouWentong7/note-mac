下面这份可以直接放进 Obsidian，按“**先建立地图 → 再理解原理 → 再知道怎么实现 → 最后能和老师讨论研究问题**”来设计。每一级都有明确的完成标准，不要求你一开始就把论文和数学细节吃透。


# LLM / VLM / Reasoning / RAG / RL / Agent 学习路径

> 目标：从“了解皮毛”达到能够和导师正常讨论 LLM / VLM / CoT / RAG / RL / Agent / Multi-Agent，并能够阅读相关论文、理解基本方法与实验设计，而不是一问三不知。
>
> 核心学习原则：
> 1. 每学习一个方向，都回答：**为什么出现 → 解决什么问题 → 核心原理 → 怎么实现 → 现在发展到哪里 → 还有什么问题**
> 2. 第一轮不追求数学推导完整，而追求“能够解释”。
> 3. 第二轮才开始补代码、公式、论文细节。
> 4. 每个模块最后必须能够脱离笔记，用自己的话向别人讲清楚。

---

# 0. 总目标

## 0.1 最终能力目标

- [ ] 能说清楚 Transformer、LLM、VLM、CoT、RAG、Tool Use、Agent、RL、Multi-Agent 之间的关系
- [ ] 能解释每种技术出现的背景，以及它解决了上一阶段什么问题
- [ ] 能画出主要模型/系统的基本结构图
- [ ] 能解释一个基本方法的数据流和计算流程
- [ ] 能看懂相关论文 Method 部分的大部分内容
- [ ] 能理解常见实验设计、baseline、ablation、evaluation
- [ ] 能区分不同技术是在解决：
  - [ ] 知识获取问题
  - [ ] 推理问题
  - [ ] 行动问题
  - [ ] 学习问题
  - [ ] 协作问题
  - [ ] 可靠性问题
- [ ] 能够和导师讨论“为什么这个方法有效/可能为什么无效”
- [ ] 听到一个新名词时，知道它大概属于哪条技术发展路线

## 0.2 最终交流标准

完成整个学习路径后，至少能够回答：

### 基础概念

- [ ] Transformer 是什么？
- [ ] Attention 在计算什么？
- [ ] LLM 是怎么训练出来的？
- [ ] SFT 和 Pretraining 有什么区别？
- [ ] Inference 是怎么进行的？
- [ ] Hidden State、Embedding、Token 分别是什么？

### VLM

- [ ] VLM 为什么能够理解图像？
- [ ] Vision Encoder 在做什么？
- [ ] Projector / Connector 在做什么？
- [ ] Image Token 是什么？
- [ ] VLM 和普通 LLM 的架构有什么区别？
- [ ] LLaVA / Qwen-VL 这一类模型大致怎么训练？
- [ ] VLM 的主要困难是什么？

### Reasoning / CoT

- [ ] CoT 是什么？
- [ ] 为什么 CoT 能够提升 reasoning？
- [ ] CoT 和普通 answer generation 有什么区别？
- [ ] Self-Consistency 是什么？
- [ ] Tree of Thoughts 是什么？
- [ ] Reflection / Self-Verification 在解决什么问题？
- [ ] 什么叫 reasoning model？
- [ ] Inference-time scaling 是什么？

### RAG

- [ ] 为什么需要 RAG？
- [ ] RAG 的基本 pipeline 是什么？
- [ ] Embedding 在 RAG 中做什么？
- [ ] Vector Database 是什么？
- [ ] Retrieval 是怎么实现的？
- [ ] Reranking 是什么？
- [ ] Naive RAG 有什么问题？
- [ ] Multi-hop RAG 是什么？
- [ ] Graph RAG 是什么？
- [ ] Agentic RAG 是什么？

### Tool Use / Agent

- [ ] Tool Calling 是什么？
- [ ] ReAct 是什么？
- [ ] Agent 和普通 LLM 有什么区别？
- [ ] Agent 的基本 loop 是什么？
- [ ] Planning 是什么？
- [ ] Memory 是什么？
- [ ] Observation / Action / State 分别是什么？
- [ ] Agent 如何调用搜索、代码执行器、数据库、API？
- [ ] Agent 为什么会失败？
- [ ] Agent evaluation 怎么做？

### Reinforcement Learning

- [ ] State / Action / Reward / Policy / Environment 是什么？
- [ ] Supervised Learning 和 RL 有什么区别？
- [ ] RLHF 是什么？
- [ ] Reward Model 是什么？
- [ ] PPO 是什么？
- [ ] DPO 和 PPO 有什么区别？
- [ ] GRPO 是什么？
- [ ] 为什么 RL 可以改善 reasoning？
- [ ] Reward Hacking 是什么？
- [ ] RL 在 Agent 中怎么使用？

### Multi-Agent

- [ ] 为什么需要 Multi-Agent？
- [ ] Single-Agent 和 Multi-Agent 的区别？
- [ ] Planner-Executor 是什么？
- [ ] Supervisor 模式是什么？
- [ ] Debate 是什么？
- [ ] 多 Agent 如何通信？
- [ ] 多 Agent 如何分工？
- [ ] 多 Agent 如何协调？
- [ ] 多 Agent 有什么成本和失败模式？

### Agent Harness

- [ ] Harness 是什么？
- [ ] 为什么 Agent 需要 Harness？
- [ ] Context Management 是什么？
- [ ] Memory Management 是什么？
- [ ] Tool Management 是什么？
- [ ] Retry / Reflection / Verification 如何实现？
- [ ] Agent 如何记录 trajectory？
- [ ] Agent 如何进行 evaluation？
- [ ] 什么因素决定一个 Agent System 是否可靠？

---

# 1. 第一层：建立 LLM 基础

> 学习目标：理解“现代 LLM 到底是什么”，能够自己画出一个 Transformer/LLM 的基本数据流。

## 1.1 Token

- [ ] 理解 Token
- [ ] 理解 Tokenizer
- [ ] 理解 BPE / SentencePiece 的基本思想
- [ ] 理解一个句子如何变成 token IDs
- [ ] 理解 token ID → embedding 的过程

### 达标

- [ ] 能解释为什么 LLM 不是直接读取“单词”
- [ ] 能解释 token 数为什么影响上下文长度和计算量

---

## 1.2 Embedding

- [ ] 什么是 Word Embedding
- [ ] 什么是 Token Embedding
- [ ] 什么是 Positional Encoding
- [ ] 为什么 Transformer 需要位置信息
- [ ] Learned Position Embedding
- [ ] RoPE 的基本思想

### 达标

- [ ] 能解释“token → 向量”
- [ ] 能解释 embedding 和 hidden state 的区别

---

## 1.3 Attention

- [ ] Query / Key / Value
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention
- [ ] Self-Attention
- [ ] Cross-Attention
- [ ] Causal Mask

### 达标

- [ ] 能画出 Q / K / V 的基本计算关系
- [ ] 能解释 attention 为什么能够建立 token 之间的依赖
- [ ] 能区分 self-attention 和 cross-attention

---

## 1.4 Transformer

- [ ] Transformer Block
- [ ] Multi-Head Attention
- [ ] MLP / FFN
- [ ] Residual Connection
- [ ] LayerNorm / RMSNorm
- [ ] Decoder-only Transformer

### 达标

- [ ] 能画出一个 Transformer block
- [ ] 能解释一个 token 在一层 Transformer 中经历了什么

---

## 1.5 LLM Training

- [ ] Pretraining
- [ ] Next Token Prediction
- [ ] Cross Entropy Loss
- [ ] SFT
- [ ] Instruction Tuning
- [ ] Inference
- [ ] Temperature
- [ ] Top-k
- [ ] Top-p

### 达标

- [ ] 能解释“LLM 到底在训练什么”
- [ ] 能区分：
  - [ ] Pretraining
  - [ ] SFT
  - [ ] RL / Preference Optimization

---

## 1.6 第一阶段代表工作

- [ ] Attention Is All You Need
- [ ] GPT 系列的发展
- [ ] LLaMA 系列
- [ ] Transformer scaling / scaling law 基本概念

### 输出

- [ ] 自己画一张“Transformer → GPT → LLM”的路线图
- [ ] 用 5 分钟向别人解释 LLM

---

# 2. 第二层：理解 VLM

> 学习目标：理解“视觉信息究竟如何进入 LLM”。

---

## 2.1 Vision Encoder

- [ ] CNN 基本回顾
- [ ] Vision Transformer
- [ ] Patch Embedding
- [ ] CLIP
- [ ] Image Feature

### 达标

- [ ] 能解释图像如何变成向量
- [ ] 能解释 Vision Encoder 和 LLM 的职责区别

---

## 2.2 Vision-Language Alignment

- [ ] Image Encoder
- [ ] Text Encoder
- [ ] Contrastive Learning
- [ ] Shared Embedding Space
- [ ] Image-Text Alignment

### 重点理解

> 图像和文本为什么能够建立对应关系？

---

## 2.3 LLM-based VLM

- [ ] LLaVA 基本架构
- [ ] Vision Encoder
- [ ] Projector
- [ ] LLM
- [ ] Multimodal Token

### 核心理解

```text
Image
  ↓
Vision Encoder
  ↓
Visual Features
  ↓
Projector
  ↓
Visual Tokens
  ↓
LLM
  ↓
Text
````

---

## 2.4 VLM Training

-  Image-Text Pretraining
    
-  Multimodal Pretraining
    
-  Visual Instruction Tuning
    
-  SFT
    
-  LoRA / PEFT
    
-  Multimodal alignment
    

### 达标

-  能解释一个 VLM 大致需要经历什么训练阶段
    
-  能解释为什么可以只训练 projector / LoRA
    

---

## 2.5 代表模型

-  CLIP
    
-  BLIP
    
-  BLIP-2
    
-  LLaVA
    
-  Qwen-VL
    
-  Qwen2-VL
    
-  Qwen2.5-VL
    

### 每个模型都回答

-  输入是什么？
    
-  Vision Encoder 是什么？
    
-  怎么连接 LLM？
    
-  怎么训练？
    
-  主要解决什么问题？
    
-  相比上一代进步在哪里？
    

---

## 2.6 VLM 当前主要问题

-  Hallucination
    
-  Fine-grained perception
    
-  Spatial reasoning
    
-  OCR
    
-  Long image understanding
    
-  Visual grounding
    
-  Multimodal reasoning
    
-  Knowledge grounding
    
-  Reliability
    
-  Evaluation
    

### 达标

-  能解释“VLM 并不是单纯给 LLM 接一张图片”
    

---

# 3. 第三层：Reasoning / CoT

> 学习目标：理解为什么 LLM/VLM 从“生成答案”发展到“研究 reasoning process”。

---

## 3.1 Chain-of-Thought

-  什么是 CoT
    
-  Zero-shot CoT
    
-  Few-shot CoT
    
-  CoT 为什么可能有效
    
-  CoT 的局限
    

### 达标

能够解释：

```text
Question
↓
Intermediate reasoning
↓
Answer
```

而不是：

```text
Question
↓
Answer
```

---

## 3.2 Reasoning Strategies

-  Self-Consistency
    
-  Majority Voting
    
-  Tree of Thoughts
    
-  Reflection
    
-  Self-Verification
    
-  Critic
    
-  Planner
    

### 达标

-  能解释每一种方法解决什么问题
    
-  能区分“增加推理 token”和“改变推理策略”
    

---

## 3.3 Reasoning Models

-  什么叫 Reasoning Model
    
-  Test-time / Inference-time Compute
    
-  Long CoT
    
-  Verification
    
-  Search
    
-  Process-level supervision
    
-  Outcome-level supervision
    

---

## 3.4 Reasoning + VLM

-  Visual CoT
    
-  Multimodal Reasoning
    
-  Visual Grounding + Reasoning
    
-  Image → Reasoning → Answer
    

### 达标

-  能思考 Medical VQA 中为什么需要 reasoning
    
-  能区分“视觉识别错误”和“推理错误”
    

---

# 4. 第四层：RAG

> 学习目标：理解“模型参数之外的知识如何进入模型”。

---

## 4.1 为什么需要 RAG

-  参数知识的局限
    
-  Knowledge cutoff
    
-  Hallucination
    
-  Domain-specific knowledge
    
-  Private knowledge
    

### 达标

能解释：

> 为什么不直接继续训练模型，而要让模型检索？

---

## 4.2 Retrieval

-  Keyword Search
    
-  BM25
    
-  Dense Retrieval
    
-  Embedding
    
-  Similarity
    
-  Cosine Similarity
    

### 达标

-  能解释 query 如何找到 document
    
-  能解释 embedding 在 retrieval 中的作用
    

---

## 4.3 Vector Database

-  Vector Index
    
-  ANN
    
-  FAISS
    
-  Milvus / Qdrant / Chroma 等基本概念
    

---

## 4.4 RAG Pipeline

```text
User Query
 ↓
Query Processing
 ↓
Retriever
 ↓
Top-k Documents
 ↓
Reranker
 ↓
Context
 ↓
LLM
 ↓
Answer
```

-  Query
    
-  Chunking
    
-  Embedding
    
-  Retrieval
    
-  Reranking
    
-  Context Construction
    
-  Generation
    

### 达标

-  可以自己实现一个最简单 RAG
    

---

## 4.5 Advanced RAG

-  Query Expansion
    
-  Hybrid Search
    
-  Reranking
    
-  Multi-hop Retrieval
    
-  Iterative Retrieval
    
-  Graph RAG
    
-  Agentic RAG
    

### 核心问题

> “检索一次够不够？”

---

## 4.6 RAG Evaluation

-  Retrieval Recall
    
-  Precision
    
-  Faithfulness
    
-  Relevance
    
-  Answer Accuracy
    

### 达标

能够回答：

> RAG 效果不好，到底是 Retriever 错了，还是 Generator 错了？

---

# 5. 第五层：Tool Use

> 学习目标：理解“LLM 如何从只会生成文字变成能够调用外部工具”。

---

## 5.1 Function Calling

-  Tool
    
-  Function Schema
    
-  Tool Selection
    
-  Arguments
    
-  Tool Result
    

---

## 5.2 基本 Tool Use

学习并实践：

-  Calculator
    
-  Web Search
    
-  Python
    
-  Database
    
-  API
    

### 达标

能够写一个最简单的：

```text
User
 ↓
LLM
 ↓
Tool Decision
 ↓
Tool
 ↓
Observation
 ↓
LLM
 ↓
Answer
```

---

## 5.3 ReAct

-  Reason
    
-  Act
    
-  Observe
    
-  Repeat
    

### 达标

-  能解释 ReAct 为什么比“一次性回答”更适合复杂任务
    
-  能自己画 ReAct loop
    

---

# 6. 第六层：Agent

> 学习目标：理解 Agent 不是“一个更强的 LLM”，而是一个由模型驱动的闭环系统。

---

## 6.1 Agent 基本概念

-  Agent
    
-  Environment
    
-  State
    
-  Observation
    
-  Action
    
-  Goal
    
-  Policy
    

---

## 6.2 Agent Loop

```text
Goal
 ↓
Plan
 ↓
Reason
 ↓
Action
 ↓
Observation
 ↓
Update State
 ↓
Next Action
 ↓
...
```

### 达标

-  能解释 Agent 和 Chatbot 的区别
    

---

## 6.3 Planning

-  Task Decomposition
    
-  Hierarchical Planning
    
-  Plan-and-Execute
    
-  Replanning
    
-  Reflection
    
-  Backtracking
    

### 核心问题

> 一个复杂任务应该一次解决，还是拆成多个子任务？

---

## 6.4 Memory

-  Short-term Memory
    
-  Long-term Memory
    
-  Working Memory
    
-  Episodic Memory
    
-  Semantic Memory
    
-  Retrieval-based Memory
    

---

## 6.5 Agent Tools

-  Search
    
-  Browser
    
-  Code Execution
    
-  Database
    
-  File System
    
-  APIs
    
-  Computer Use
    

---

## 6.6 Agent Failure

-  Planning Error
    
-  Tool Error
    
-  Hallucination
    
-  Context Overflow
    
-  Infinite Loop
    
-  Wrong Tool Selection
    
-  Error Propagation
    
-  Reward / Objective Misalignment
    

### 达标

能够讨论：

> 为什么一个 Agent 明明“模型很强”，实际执行任务却可能很差？

---

# 7. 第七层：Reinforcement Learning

> 学习目标：先掌握 RL 的基本框架，再理解 RL 为什么重新成为 LLM reasoning 的关键技术。

---

## 7.1 RL 基础

-  Agent
    
-  Environment
    
-  State
    
-  Action
    
-  Reward
    
-  Policy
    
-  Value
    
-  Return
    

### 达标

-  能自己解释 RL 和 supervised learning 的区别
    

---

## 7.2 Policy

-  Policy
    
-  Deterministic Policy
    
-  Stochastic Policy
    
-  Policy Gradient
    

---

## 7.3 Value

-  State Value
    
-  Action Value
    
-  Q Function
    
-  Advantage
    

---

## 7.4 RLHF

-  SFT
    
-  Human Preference
    
-  Reward Model
    
-  PPO
    
-  RLHF Pipeline
    

```text
Pretraining
 ↓
SFT
 ↓
Preference Data
 ↓
Reward Model
 ↓
RL
 ↓
Aligned Model
```

---

## 7.5 Preference Optimization

-  DPO
    
-  为什么不一定需要显式 Reward Model
    
-  Preference Data
    

### 达标

-  能解释 DPO 与 PPO/RLHF 的主要区别
    

---

## 7.6 Reasoning RL

-  Outcome Reward
    
-  Process Reward
    
-  Verifier
    
-  GRPO
    
-  Reasoning RL
    

---

## 7.7 DeepSeek-R1 类工作

重点理解：

-  为什么使用 RL
    
-  Reward 如何设计
    
-  如何判断 reasoning 是否正确
    
-  RL 如何改变模型行为
    
-  为什么会出现新的 reasoning patterns
    

### 达标

能够回答：

> “LLM 为什么不是只靠 SFT，而要使用 RL 来训练 reasoning？”

---

## 7.8 RL Failure Modes

-  Reward Hacking
    
-  Reward Misalignment
    
-  Overoptimization
    
-  Training Instability
    
-  Exploration Problem
    

---

# 8. 第八层：Multi-Agent

> 学习目标：理解为什么一个 Agent 不够，以及多个 Agent 如何分工、通信和协作。

---

## 8.1 为什么 Multi-Agent

-  Task Complexity
    
-  Specialization
    
-  Parallelization
    
-  Verification
    
-  Debate
    
-  Role Separation
    

---

## 8.2 基本架构

### Planner + Executor

-  Planner
    
-  Worker
    
-  Executor
    

### Supervisor

-  Supervisor
    
-  Workers
    
-  Task Routing
    

### Parallel Agents

-  Parallel Execution
    
-  Result Aggregation
    

### Debate

-  Agent A
    
-  Agent B
    
-  Critic
    
-  Final Decision
    

---

## 8.3 Agent Communication

-  Message Passing
    
-  Shared Memory
    
-  Blackboard
    
-  Structured Communication
    

---

## 8.4 Coordination

-  Task Allocation
    
-  Conflict Resolution
    
-  Consensus
    
-  Voting
    
-  Hierarchical Coordination
    

---

## 8.5 Multi-Agent Failure

-  Communication Cost
    
-  Error Propagation
    
-  Coordination Failure
    
-  Redundant Work
    
-  Infinite Discussion
    
-  Cost Explosion
    

### 达标

能够讨论：

> 为什么 Multi-Agent 并不一定比 Single-Agent 更好？

---

# 9. 第九层：Agent Harness

> 学习目标：从“研究一个 Agent”进一步理解“如何构建一个可靠的 Agent System”。

---

## 9.1 Harness 是什么

-  Agent Runtime
    
-  Orchestration
    
-  Context Management
    
-  Tool Management
    
-  Memory
    
-  Logging
    
-  Evaluation
    

---

## 9.2 Context Management

-  Context Window
    
-  Context Compression
    
-  Summarization
    
-  Retrieval
    
-  Relevant Context Selection
    

---

## 9.3 Tool Management

-  Tool Registry
    
-  Tool Selection
    
-  Tool Validation
    
-  Tool Result Parsing
    
-  Retry
    
-  Timeout
    

---

## 9.4 Agent State

-  Task State
    
-  Conversation State
    
-  Tool State
    
-  Memory State
    
-  Execution Trace
    

---

## 9.5 Reliability

-  Retry
    
-  Reflection
    
-  Critic
    
-  Verification
    
-  Guardrail
    
-  Error Recovery
    
-  Checkpoint
    

---

## 9.6 Evaluation

-  Task Success Rate
    
-  Tool Success Rate
    
-  Trajectory Evaluation
    
-  Cost
    
-  Latency
    
-  Robustness
    
-  Failure Rate
    

### 达标

能够回答：

> “模型能力已经很强了，为什么 Agent 系统仍然需要大量工程？”

---

# 10. 第十层：把整个发展路线串起来

> 这一阶段非常重要。不要再单独学习知识点，而是重新回答“为什么下一代技术会出现”。

---

## 10.1 技术发展主线

```text
Transformer
    ↓
LLM
    ↓
VLM
    ↓
Reasoning / CoT
    ↓
RAG
    ↓
Tool Use
    ↓
Agent
    ↓
RL for Reasoning / Agents
    ↓
Multi-Agent
    ↓
Agent Harness
```

---

## 10.2 每一次技术升级都回答

-  上一代解决了什么问题？
    
-  上一代还剩什么问题？
    
-  新方法增加了什么能力？
    
-  新方法依赖什么额外模块？
    
-  新方法带来了什么新问题？
    

---

## 10.3 建立“问题 → 技术”的映射表

|问题|代表技术|
|---|---|
|模型不会处理视觉|VLM|
|模型复杂问题容易答错|CoT / Reasoning|
|模型缺少外部知识|RAG|
|模型无法访问外部系统|Tool Use|
|模型无法自主完成复杂任务|Agent|
|模型不会从结果中持续优化行为|RL|
|单 Agent 难以完成复杂任务|Multi-Agent|
|Agent 难以稳定运行|Harness|

-  能够不看笔记复述这张表
    

---

# 11. 第十一层：代码实现能力

> 目标：不是成为 LLM 工程师，而是“概念和代码能对应起来”。

---

## 11.1 Transformer

-  用 PyTorch 实现一个极简 Self-Attention
    
-  实现一个 Transformer Block
    
-  理解 Hugging Face Transformer API
    
-  加载一个开源 LLM
    
-  完成一次 inference
    

---

## 11.2 VLM

-  加载一个开源 VLM
    
-  输入图片 + text prompt
    
-  获得回答
    
-  查看 image features
    
-  查看 hidden states
    
-  理解 processor
    
-  理解 visual token
    

---

## 11.3 RAG

自己实现一个最小系统：

```text
Documents
 ↓
Chunk
 ↓
Embedding
 ↓
Vector Search
 ↓
Top-k
 ↓
LLM
 ↓
Answer
```

-  能跑通
    
-  能修改 top-k
    
-  能替换 embedding model
    
-  能观察 retrieval 错误
    

---

## 11.4 Tool Calling

-  实现 calculator tool
    
-  实现 search tool
    
-  实现 Python tool
    
-  让 LLM 自动选择 tool
    

---

## 11.5 Agent

实现：

```text
Question
 ↓
LLM
 ↓
Tool Selection
 ↓
Tool
 ↓
Observation
 ↓
LLM
 ↓
Next Action
 ↓
Answer
```

-  能记录每一步 trajectory
    
-  能观察 Agent 为什么出错
    

---

## 11.6 Multi-Agent

实现一个最简单：

```text
Planner
  ↓
Researcher
  ↓
Critic
  ↓
Final Writer
```

-  能理解角色分工
    
-  能记录 agent-to-agent communication
    

---

# 12. 第十二层：论文阅读能力

> 从这一阶段开始，学习重点从“技术是什么”逐渐转向“研究人员到底在研究什么”。

---

## 12.1 每篇论文固定回答

-  Problem
    
-  Motivation
    
-  Previous limitation
    
-  Proposed method
    
-  Architecture
    
-  Training
    
-  Loss
    
-  Dataset
    
-  Baseline
    
-  Ablation
    
-  Main result
    
-  Limitation
    

---

## 12.2 第一批论文

### Transformer

-  Attention Is All You Need
    

### Vision-Language

-  CLIP
    
-  BLIP / BLIP-2
    
-  LLaVA
    
-  Qwen-VL / Qwen2-VL 类工作
    

### Reasoning

-  Chain-of-Thought Prompting
    
-  Self-Consistency
    
-  Tree of Thoughts
    
-  Reasoning Model / inference-time scaling 相关工作
    

### RAG

-  Retrieval-Augmented Generation
    

### Agent

-  ReAct
    
-  Toolformer
    

### RL

-  InstructGPT / RLHF
    
-  DPO
    
-  GRPO
    
-  DeepSeek-R1
    

### Multi-Agent

-  Multi-Agent LLM Collaboration
    
-  Agent debate / planner-worker / supervisor-worker 类代表工作
    

---

# 13. 第十三层：进入 Medical VLM

> 这一层开始把前面的知识与自己的研究方向连接起来。

---

## 13.1 Medical VLM

-  Medical VQA
    
-  Medical Image Captioning
    
-  Medical Report Generation
    
-  Medical Visual Grounding
    
-  Medical Multimodal Reasoning
    

---

## 13.2 医学领域特殊问题

-  Domain Knowledge
    
-  Hallucination
    
-  Clinical Safety
    
-  Explainability
    
-  Uncertainty
    
-  Evidence Grounding
    
-  Multi-turn Dialogue
    
-  Clinical Evaluation
    

---

## 13.3 Medical RAG

-  医学知识库
    
-  医学文献 Retrieval
    
-  Clinical Guidelines
    
-  Evidence-grounded QA
    
-  Multi-hop Medical RAG
    

---

## 13.4 Medical Agent

-  Medical Search Agent
    
-  Clinical Decision Support
    
-  Medical Tool Use
    
-  Medical Imaging Tools
    
-  Clinical Knowledge Agent
    

---

## 13.5 Medical Multi-Agent

思考：

```text
Radiology Agent
       ↓
Clinical Knowledge Agent
       ↓
Evidence Agent
       ↓
Critic Agent
       ↓
Final Reasoning
```

-  分析这种架构为什么可能有效
    
-  分析可能出现什么错误
    
-  分析如何评价
    

---

# 14. 第十四层：进入自己的研究问题

> 到这里，学习不再是“别人研究了什么”，而是开始思考“我可以研究什么”。

---

## 14.1 从能力问题转向机制问题

-  模型为什么能完成 multi-turn reasoning？
    
-  多轮对话究竟带来了什么？
    
-  是 information accumulation？
    
-  是 reasoning improvement？
    
-  是 visual grounding improvement？
    
-  是 hidden state evolution？
    
-  是 attention pattern 改变？
    

---

## 14.2 从输出分析转向内部机制

-  Hidden State
    
-  Attention
    
-  Activation
    
-  Representation
    
-  Layer-wise Analysis
    
-  RSA
    
-  Cross-modal Alignment
    

---

## 14.3 从“Accuracy”转向“Trustworthiness”

-  Accuracy
    
-  Faithfulness
    
-  Consistency
    
-  Robustness
    
-  Uncertainty
    
-  Calibration
    
-  Evidence Grounding
    
-  Clinical Reliability
    

---

# 15. 最终毕业标准：达到“老师问了不会一问三不知”

## Level 1：听得懂

-  听到 Transformer / VLM / RAG / Agent / RL 不陌生
    
-  能说出基本定义
    
-  知道它们解决什么问题
    

## Level 2：讲得清

-  能画基本架构
    
-  能解释数据流
    
-  能解释核心算法
    
-  能解释训练流程
    

## Level 3：看得懂论文

-  能读懂 Method
    
-  能读懂实验表
    
-  能理解 ablation
    
-  能分析 baseline
    

## Level 4：能讨论

-  能解释为什么方法可能有效
    
-  能指出方法的假设
    
-  能指出失败模式
    
-  能提出改进方向
    

## Level 5：开始有研究意识

-  听到一个新方法时，知道它是在解决哪类问题
    
-  能把不同技术联系起来
    
-  能提出自己的 research question
    
-  能判断一个问题属于：
    
    -  perception
        
    -  reasoning
        
    -  knowledge
        
    -  planning
        
    -  action
        
    -  learning
        
    -  coordination
        
    -  reliability
        

---

# 16. 最终需要自己画出的 8 张图

-  图 1：Transformer
    
-  图 2：LLM Training Pipeline
    
-  图 3：VLM Architecture
    
-  图 4：CoT / Reasoning Pipeline
    
-  图 5：RAG Pipeline
    
-  图 6：ReAct / Agent Loop
    
-  图 7：RLHF / Reasoning RL Pipeline
    
-  图 8：Multi-Agent + Harness
    

> 如果这 8 张图能够不用看资料自己画出来，并且每个框都能解释，那么基本就已经脱离“只知道名词”的阶段。

---

# 17. 每个主题的统一学习模板

以后遇到任何新技术，都按照这个模板记录：

## XXX

### 1. 它为什么出现？

- [ ]
    

### 2. 它解决了什么问题？

- [ ]
    

### 3. 它之前的方法是什么？

- [ ]
    

### 4. 核心思想是什么？

- [ ]
    

### 5. 基本架构是什么？

- [ ]
    

### 6. 输入是什么？

- [ ]
    

### 7. 输出是什么？

- [ ]
    

### 8. 中间发生了什么？

- [ ]
    

### 9. 怎么训练？

- [ ]
    

### 10. 怎么推理/运行？

- [ ]
    

### 11. 怎么评价？

- [ ]
    

### 12. 优势是什么？

- [ ]
    

### 13. 局限是什么？

- [ ]
    

### 14. 后续研究又解决了什么问题？

- [ ]
    

### 15. 与我自己的 Medical VLM 有什么关系？

- [ ]
    

---

# 18. 最终知识结构

```text
                     Modern AI
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
          LLM                       VLM
             │                       │
             └──────────┬────────────┘
                        ↓
                    Reasoning
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
            RAG                 Tool Use
             │                     │
             └──────────┬──────────┘
                        ↓
                     Agent
                        │
             ┌──────────┴──────────┐
             ↓                     ↓
             RL                Multi-Agent
             │                     │
             └──────────┬──────────┘
                        ↓
                  Agent Harness
                        │
                        ↓
             Reliable AI Systems
                        │
                        ↓
                Medical AI / VLM
```

---

# 19. 最终目标

完成这套路径后，不要求自己成为：

> “LLM / RL / Agent 专家”

而是达到：

> **一个准备开始做 VLM 研究的研究生应有的基础研究能力。**

具体来说：

**看到模型：**

> 知道它是什么结构。

**看到方法：**

> 知道它为什么提出。

**看到训练方法：**

> 知道它在优化什么。

**看到实验：**

> 知道它证明了什么。

**看到新论文：**

> 能把它放进整个技术发展路线中。

**和导师交流：**

> 不仅知道“这个词是什么”，还能继续讨论“为什么需要它、怎么实现、为什么可能有效、还有什么问题”。

这就是本学习路径的最终标准。