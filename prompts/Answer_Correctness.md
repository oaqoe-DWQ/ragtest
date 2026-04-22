# Answer Correctness（回答正确性）

## 1. 概述

`Answer Correctness` 是 Ragas 框架中用于衡量回答与标准答案（Ground Truth）一致程度的指标，综合考虑 **事实准确性** 和 **语义相似度** 两个维度。

评分范围：**0.0 ~ 1.0**（越高越好）

---

## 2. 计算公式

$$
\text{Answer Correctness} = \frac{w_1 \times \text{F-score} + w_2 \times \text{Semantic Similarity}}{w_1 + w_2}
$$

**默认权重**：`w_1 = 0.75`（事实性），`w_2 = 0.25`（语义相似度）

其中 **F-score** 基于 `beta = 1.0`（即 F1-score）：

$$
F_\beta = \frac{(1 + \beta^2) \times \text{Precision} \times \text{Recall}}{(\beta^2 \times \text{Precision}) + \text{Recall}}
$$

- **Precision**（精确率）= TP / (TP + FP)
- **Recall**（召回率）= TP / (TP + FN)

---

## 3. 评估流程

Answer Correctness 的评估分为以下步骤：

### Step 1：Statement Generation（陈述句提取）

将 `response` 和 `reference` 各自拆解为独立的、不含代词的陈述句。

> **使用的 Prompt**：`StatementGeneratorPrompt`

```
Given a question and an answer, analyze the complexity of each sentence
in the answer. Break down each sentence into one or more fully understandable
statements. Ensure that no pronouns are used in any statement.
Format the outputs in JSON.
```

**示例：**

| 输入 | 输出 |
|------|------|
| **Question**: Who was Albert Einstein? | Statements: |
| **Answer**: He was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists. He was best known for developing the theory of relativity. | 1. Albert Einstein was a German-born theoretical physicist.<br>2. Albert Einstein is recognized as one of the greatest physicists.<br>3. Albert Einstein was best known for developing the theory of relativity. |

---

### Step 2：Classification（陈述句分类）

将生成的陈述句与标准答案中的陈述句进行对比，分类为：

| 分类 | 说明 | 缩写 |
|------|------|------|
| **True Positive (TP)** | 回答中的陈述句被标准答案直接支持 | TP |
| **False Positive (FP)** | 回答中的陈述句不被标准答案直接支持（可能是幻觉/错误） | FP |
| **False Negative (FN)** | 标准答案中的陈述句在回答中缺失 | FN |

> **使用的 Prompt**：`CorrectnessClassifier`

```
Given a ground truth and an answer statements, analyze each statement
and classify them in one of the following categories:
- TP (true positive): statements that are present in answer that are also
  directly supported by the one or more statements in ground truth
- FP (false positive): statements present in the answer but not directly
  supported by any statement in ground truth
- FN (false negative): statements found in the ground truth but not
  present in answer.
Each statement can only belong to one of the categories.
Provide a reason for each classification.
```

**示例：**

| 输入 | 分类结果 |
|------|---------|
| **Question**: What powers the sun? | **TP**: "The primary function of the sun is to provide light..." — 被标准答案部分支持 |
| **Answer**: "The sun is powered by nuclear fission..." / "The primary function of the sun is to provide light..." | **FP**: "The sun is powered by nuclear fission..." — 与标准答案矛盾（标准答案是 fusion） |
| **Ground Truth**: "The sun is powered by nuclear fusion..." / "This fusion process releases energy..." / "The energy provides heat and light..." | **FN**: "The sun is powered by nuclear fusion..." / "Fusion process releases energy..." 等多条缺失 |

---

### Step 3：Score Computation（分数计算）

1. 根据 TP / FP / FN 数量计算 **F-score**（`beta=1.0`，即 F1-score）\r
2. 通过 Embedding 计算 **Semantic Similarity**（回答 vs 标准答案的语义相似度）\r
3. 加权平均得到最终分数

---

### Step 4：F1 Score 详解（两层计算逻辑）

#### 第一层：陈述句分类（LLM 执行）

Ragas 使用 `CorrectnessClassifier` Prompt，让 LLM 对比 `response` 和 `reference` 的每条陈述句，分类为：

| 分类 | 说明 |
|------|------|
| **TP（True Positive）** | 回答中的句子，被标准答案直接支持 |
| **FP（False Positive）** | 回答中的句子，不被标准答案直接支持（幻觉/错误） |
| **FN（False Negative）** | 标准答案中的句子，在回答中缺失 |

**示例：**

| reference | response | LLM 分类结果 |
|-----------|----------|-------------|
| "水的沸点是100°C" | "水在海平面沸腾温度是100°C，还受海拔影响" | TP: "水在海平面沸腾温度是100°C"<br>FP: "还受海拔影响"（reference 未提及）<br>FN: （空，reference 句子都在） |

#### 第二层：F1 公式计算（纯数学）

拿到 TP / FP / FN 的数量后，套用标准公式：

```python
def fbeta_score(tp, fp, fn, beta=1.0):
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn)  if (tp + fn) > 0 else 0

    if precision == 0 and recall == 0:
        return 0.0

    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
    return fbeta
```

因为 `beta = 1.0`，即 **F1-score**：

$$
F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

**不同场景下的 F1 数值示例：**

| 场景 | TP | FP | FN | Precision | Recall | **F1** |
|------|:--:|:--:|:--:|:---------:|:------:|:------:|
| 完美回答 | 5 | 0 | 0 | 1.00 | 1.00 | **1.00** |
| 回答全对但遗漏部分 | 3 | 0 | 2 | 1.00 | 0.60 | **0.75** |
| 有幻觉但没遗漏 | 4 | 1 | 0 | 0.80 | 1.00 | **0.89** |
| 部分对部分错 | 2 | 2 | 3 | 0.50 | 0.40 | **0.44** |
| 回答全是幻觉 | 0 | 5 | 5 | 0.00 | 0.00 | **0.00** |

---

### Step 5：无 F1 数据时的处理

当没有 F1 数据（即 `reference` 为空）时，Ragas 内部处理逻辑如下：

#### 源码逻辑

```python
# Step 1: 分别从 response 和 reference 提取陈述句
statements["response"]  = extract_statements(response)   # 有内容 → 非空列表
statements["reference"] = extract_statements("")        # 无内容 → [] 空列表

# Step 2: 检查是否全是空列表
if not all([val == [] for val in statements.values()]):
    # → 至少有一个非空，LLM 仍然会被调用进行分类
    answers = await self.correctness_prompt.generate(...)
    f1_score = self._compute_statement_presence(answers)
else:
    # → response 和 reference 都为空时，直接返回 1.0（跳过 LLM）
    f1_score = 1.0

# Step 3: 语义相似度始终执行（不受 reference 是否为空影响）
similarity_score = await self.answer_similarity.ascore(row, callbacks)

# Step 4: 加权平均
score = np.average([f1_score, similarity_score], weights=[0.75, 0.25])
```

#### 关键结论

| 场景 | `reference` 值 | `reference` 陈述句 | F1 处理 | F1 结果 |
|------|---------------|:-----------------:|---------|---------|
| **有 F1 数据**（默认） | 有标准答案内容 | 非空列表 | LLM 正常分类 | 0.0 ~ 1.0 |
| **reference 为空** | 空字符串 / NaN | `[]` 空列表 | LLM 仍被调用（结果很低） | 接近 0 |
| **response + reference 都为空** | 都为空 | 两个 `[]` | 直接返回 1.0（跳过 LLM） | **1.0** |

#### 如何完全跳过 F1，只用语义相似度

将 `weights` 设为 `[0.0, 1.0]`，F1 维度权重为 0，只使用语义相似度：

```python
from ragas.metrics import AnswerCorrectness

# 纯语义相似度（无 F1）
metric = AnswerCorrectness(weights=[0.0, 1.0])
```

计算公式变为：

$$
\text{Answer Correctness} = 0.0 \times \text{F-score} + 1.0 \times \text{Semantic Similarity} = \text{Semantic Similarity}
$$

---

## 5. 参数说明

该指标需要以下字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `user_input` | `str` | 用户问题 |
| `response` | `str` | RAG 系统生成的回答 |
| `reference` | `str` | 标准答案（Ground Truth） |

---

## 6. 优缺点分析

### 优点

- 综合考虑事实准确性和语义相似度，评估维度全面
- 通过陈述句级别的细粒度对比，能够精确定位回答中的错误
- 权重可调，可根据业务场景灵活配置

### 缺点

- 依赖 LLM 进行陈述句生成和分类，计算成本较高
- 当权重偏向语义相似度时，可能无法有效识别事实错误（被优美措辞掩盖）
- 对长回答的评估可能受到陈述句数量的影响

---

## 7. 源码位置

- 指标实现：`ragas/metrics/_answer_correctness.py`
- 陈述句生成：`ragas/metrics/_faithfulness.py`（`StatementGeneratorPrompt`）
- 语义相似度：`ragas/metrics/_answer_similarity.py`
- 工具函数：`ragas/metrics/utils.py`（`fbeta_score`）

---

## 8. 参考资料

- [Ragas Answer Correctness 官方文档](https://docs.ragas.io/en/stable/concepts/metrics/answer_correctness/)
