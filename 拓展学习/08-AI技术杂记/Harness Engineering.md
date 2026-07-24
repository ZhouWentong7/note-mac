>[!abstract] 简介
>为解决AI Agent开发范式问题：
>1. 人类注意力跟不上AI的高速开发、审查速度
>2. **上下文腐烂**与**巨大提示词失效**：Agent需要大量详尽的prompt约束，占用Agent上下文窗口，可能导致Agent漏掉关键约束、产生幻觉
>3. 代码库规模爆炸带来的技术增熵：仅靠文档约束Agent，文档很快会腐烂失效导致代码库失控
>OpenAI[^1]提出了Harness工程，重新定义工程师决策
>- - **机械化执行（Mechanical Enforcement）**：用自定义 Linter 和自动化测试代替纯文本规范，让反馈直接内嵌在报错信息中，驱动 Agent 自我纠错。
>     
> - **渐进式披露（Progressive Disclosure）**：只向 Agent 提供精简的“地图”文件（如短小精悍的 `AGENTS.md`），需要时再引导其查阅深度文档。
>     
> - **熵管理（Entropy Management）**：将规则编码进仓库，用后台 Agent 自动扫描、清理坏模式与技术债。

---

>[!hint] 一句话总结
>- 传统工程：人类写代码 → 机器执行代码
>- Harness Engineering：人类设计约束 → 智能体写代码 → 机器执行代码

# 六大核心概念
这**六大核心概念**本质上就是 **Harness Engineering（驭缰工程）的软件设计理念与工程指导原则**。

在传统软件工程中，设计理念通常是面向人类工程师的（例如 SOLID 原则、KISS 原则、DRY 原则）。而在 **AI 智能体（Agent）为主导代码编写者** 的全新范式下，OpenAI 总结出的这六大核心概念，就是为了**重新定义人类与智能体如何协作**、**如何为智能体打造生产环境**的指导思想：

|**核心概念**|**属于哪种设计理念？**|**它解决的底层痛点**|
|---|---|---|
|**1. 仓库即记录系统**|**知识管理理念**|解决“智能体不知道团队口头/Slack约定”的问题，所有知识必须版本化入库。|
|**2. 地图而非手册**|**上下文/Prompt 架构理念**|解决“巨型 Prompt 挤爆上下文、导致 Agent 幻觉/忽视约束”的问题。|
|**3. 机械化执行**|**质量控制与约束理念**|解决“靠自然语言文档约束 Agent 会随时间腐烂”的问题，改用 Linter/测试强制约束。|
|**4. 智能体可读性**|**可观测性与架构设计理念**|解决“人类看得懂的代码/UI，Agent 看不懂”的问题，为 Agent 暴露 DevTools/日志/指标。|
|**5. 吞吐量改变合并理念**|**工作流与 Git 流程理念**|解决“Agent 产出 PR 极快，人类审查成为瓶颈”的问题，推行“Agent 对审查 Agent”的极速合并模式。|
|**6. 熵管理 = 垃圾回收**|**代码库长治久安理念**|解决“Agent 会大量复制代码库已有的坏代码/技术债”的问题，引入定时后台 Agent 自动清理。|

【软件工程核心逻辑转变】
传统的软件工程理念是：**“人类写代码 → 机器去执行”**。

而 Harness Engineering 的设计理念是：**“人类不直接写代码，而是设计一套约束环境（Harness/缰绳），让 Agent 在环境中自主写代码并通过反馈纠错”**。

因此，这六大原则不是用来教人类怎么写出更优雅的代码，而是**教人类工程师如何构建一个对 AI 友好、高吞吐、能自动纠错的“开发缰绳（Harness）”**。


## 1. 仓库即系统（Repo as System of Record）

对于Agent而言，知识存放的位置决定其是否有效，所以工作中无法访问的Docs、讨论、程序员的思考都无法展示给Agent。

这里OpenAI推荐了作为Harness Engineering的标准规范文档结构，以实现“渐进式披露（Progressive Disclosure)”，解决提示词挤爆上下文的问题：

```cmd
AGENTS.md              ← 入口目录 (~100行)
ARCHITECTURE.md        ← 域和包分层的顶层地图
docs/
├── design-docs/       ← 设计决策，带验证状态
├── exec-plans/        ← 执行计划，带进度和决策日志
│   ├── active/
│   └── completed/
├── product-specs/     ← 产品规格
├── references/        ← 外部参考（llms.txt）
├── generated/         ← 自动生成（DB schema 等）
├── QUALITY_SCORE.md   ← 每个领域的质量评分
├── RELIABILITY.md
├── SECURITY.md
└── ...
```

作为Harness Engineering的地图（Map, Not Manual）。以往，所有的规则写在巨大的`AGENTS.md`，导致上下文被挤爆，知识产生冲突或者存储过时的知识，现在则把这个文档缩减成纯目录`Index`，一个只有100行左右的指路文件。

而知识本身进行版本化管理，把“执行计划（Exec Plans）”、“产品需求（Specs）”甚至“技术债追踪”全变成 Markdown 存进 Git。正如 Harness 的第一条原则 **“仓库即记录系统”** 所言：**不在 Git 仓库里的东西，对智能体来说就不存在**。

虽然在实际运行中确实有多个不同职责的 Agent 参与（例如：有的负责写代码、有的负责 Code Review 审查、有的做“文档园丁”扫除过时文档），但 **Harness Engineering 的核心灵魂不在于“Agent 的数量”，而在于“人类如何给 Agent 搭建能够自我纠错的工程约束（Harness/缰绳）”。**


>[!note] 关键实践
>1. **AGENTS.md 是目录，不是百科** — ~100行，只指路
>2. **专职 linter + CI 验证** — 知识库是否更新、是否交叉链接、结构是否正确
>3. **doc-gardening 智能体** — 定期扫描过时文档，自动发起修复 PR :安排一个后台定时运行的 AI Agent，像园丁一样定期巡逻整个仓库。如果发现某份文档和最新代码不匹配，或者内容已经陈旧，它会自动修改文档并提交一个 PR 申请合并。
>4. **执行计划是一等工件** — 提交到仓库，版本控制，带进度日志: 要做的任务清单、RFC 架构方案、进度日志（Exec Plans），不要写在外部的 Google Docs 或个人笔记里。直接作为 Markdown 文件 commit 到 Git 仓库里，这样智能体就能随时读取当前任务做到哪一步了

OpenAI Symphony将“记录系统”（比如：git）拓展为任务跟踪器， 把 Linear 这类问题跟踪器变成智能体编排的控制平面——每个打开的 ticket 映射一个智能体工作区，Symphony 保证未完成任务始终有智能体在跑。Symphony 本体只是一份 `SPEC.md`+`WORKFLOW.md`，参考实现用 Elixir，作者鼓励使用者把 spec 交给自己的编码智能体生成本地实现。Symphony 的 `WORKFLOW.md` 文件本质上就是把"在飞工作的状态语义"也变成仓库内可版本化的文本。[^2]

## 2. 地图而非手册
- AGENTS.md ≈ 目录页（~100行），不是百科全书
- 渐进式披露：从小入口点开始，指向更深层的文档
- 巨型指令文件的三个死因：挤占上下文、无法维护、无法机械验证

## 3.机械化执行

通过强制每次执行Lint规则（代码风格与规范检测器）与CI（Continuous Integration / 持续集成自动化检查），对Agent的各项操作进行围观滚利。

【关键设计】
在Lint报错（文档超出规定长度）的同时，给AI提供修复路径，完成自我纠正

```
❌ 普通做法：
Error: File exceeds 500 lines.

✅ Harness 做法：
Error: File exceeds 500 lines.
Fix: Split into domain-specific modules following docs/ARCHITECTURE.md#splitting-guide.
Consider extracting types to <domain>/types/ and service logic to <domain>/service/.
```

通过Lint和CI 配合实现下面的约束

1. 架构约束（结构测试）
	- **写在哪/怎么测**：通常写成**单元测试（Unit Tests）或特定的结构检测脚本**。
	- **怎么执行**：由 **CI 自动拦截**。如果 AI 写的代码破坏了依赖方向（比如 UI 直接调用了数据库），**CI 流程会直接打红叉（Block），禁止合并代码**。
2. 品味不变式（自定义 Linter）
	- 写在哪/怎么测：写成自定义的 Lint 规则（Custom Linter）。
	- 怎么执行：
		- AI 本地校验：AI 在提交前跑 Lint 命令，一旦报错（比如裸输出了 console.log，或者文件超过了 500 行），Lint 会抛出错误和修复指令，AI 看到后自动修改。
		- CI 终极兜底：如果 AI 没修就提交了，CI 依然会在流水线里再跑一遍 Lint，强制阻断提交。












---


[^1]: [OpenAI原文](obsidian://open?vault=notes&file=Clippings%2F%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF%EF%BC%9A%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E4%BC%98%E5%85%88%E7%9A%84%E4%B8%96%E7%95%8C%E4%B8%AD%E5%88%A9%E7%94%A8%20Codex)

[^2]: [An open-source spec for Codex orchestration: Symphony.](https://openai.com/index/open-source-codex-orchestration-symphony/)
