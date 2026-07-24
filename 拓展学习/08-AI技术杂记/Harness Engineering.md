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


[^1]: obsidian://open?vault=notes&file=Clippings%2F%E5%B7%A5%E7%A8%8B%E6%8A%80%E6%9C%AF%EF%BC%9A%E5%9C%A8%E6%99%BA%E8%83%BD%E4%BD%93%E4%BC%98%E5%85%88%E7%9A%84%E4%B8%96%E7%95%8C%E4%B8%AD%E5%88%A9%E7%94%A8%20Codex
