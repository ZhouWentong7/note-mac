1. 指代消解 (Coreference Resolution)

- **任务内容**：指代消解的目的是让计算机找出文本中**指向同一个现实世界实体（Entity）的所有词语**。它最难的形式是 **[[威诺格拉德模式挑战（Winograd Schema Challenge）]]**，这种题目需要极强的常识。
    - _例句 A_：树上的**果实**掉下来砸中了**汽车**，因为它太**熟**了。
    - _例句 B_：树上的**果实**掉下来砸中了**汽车**，因为它太**结实**了。
    - _挑战_：模型必须准确判断出，例句 A 中的“它”指代“果实”，而例句 B 中的“它”指代“汽车”。
- **评价方式**：
    - **准确率 (Accuracy)**：在选择题形式下（如 BabyLM 评测集），直接计算模型猜对指代对象的比例。
    - **F1-Score（MUC / B-CUBED / CEAF 算法）**：在长文本中，算法会把所有指代同一个实体的词串成一个“链条”（如：小明
        
        ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAAALklEQVR4AezSoQ0AAAwCwab7D80Ab1HkkQjEhb9CHCGiJppQgI0/0YQCbMZ+EgAAAP//UDHWKwAAAAZJREFUAwCoMAAxSwi/LgAAAABJRU5ErkJggg==)
        
        →right arrow
        
        →
        
        他
        
        ![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABEAAAAYCAYAAAAcYhYyAAAALklEQVR4AezSoQ0AAAwCwab7D80Ab1HkkQjEhb9CHCGiJppQgI0/0YQCbMZ+EgAAAP//UDHWKwAAAAZJREFUAwCoMAAxSwi/LgAAAABJRU5ErkJggg==)
        
        →right arrow
        
        →
        
        那个男孩）。F1-Score 综合评估模型找出的链条与人类标注的真实链条之间的精准度与完整度。