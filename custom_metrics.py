"""
自定义中文指标模块

该模块提供了中文版的 Ragas 评估指标，使用中文提示词进行评估。
"""

from ragas.metrics._nv_metrics import ContextRelevance as NVContextRelevance
from ragas.metrics._context_recall import (
    ContextRecall as RagasContextRecall,
    LLMContextRecall,
    ContextRecallClassificationPrompt,
    ContextRecallClassifications,
    QCA
)
from ragas.prompt import PydanticPrompt
from dataclasses import dataclass, field


# ============== 中文版 ContextRelevance ==============
class ChineseContextRelevance(NVContextRelevance):
    """
    中文版上下文相关性评估

    使用中文提示词评估检索到的上下文与用户问题的相关性。
    评分范围: 0.0 (完全不相关) - 1.0 (完全相关)
    """

    name: str = "nv_context_relevance"

    # 使用直接的字符串定义，避免 field() 包装导致 format() 方法调用失败
    template_relevance1: str = (
        "### 指令\n\n"
        "你是一位世界级专家，负责评估上下文对于回答问题的相关性评分。\n"
        "你的任务是判断上下文是否包含正确的信息来回答问题。\n"
        "不要依赖你关于问题的先验知识。\n"
        "只使用上下文中和问题里写的内容。\n"
        "按照以下指引操作：\n"
        "0. 如果上下文不包含任何与回答问题相关的信息，返回0。\n"
        "1. 如果上下文部分包含回答问题所需的相关信息，返回1。\n"
        "2. 如果上下文包含任何与回答问题相关的信息，返回2。\n"
        "你必须只提供0、1或2的相关性评分，不要解释。\n"
        "### 问题：{query}\n\n"
        "### 上下文：{context}\n\n"
        "相关性评分是："
    )

    template_relevance2: str = (
        "作为一位专门评估给定上下文与问题相关性评分的专家，"
        "我的任务是判断上下文在多大程度上提供了回答问题所需的信息。\n"
        "我将仅依赖上下文和问题中提供的信息，不依赖任何先验知识。\n\n"
        "以下是，我将遵循的指引：\n"
        "* 如果上下文不包含任何与回答问题相关的信息，我将返回0。\n"
        "* 如果上下文部分包含回答问题所需的相关信息，我将返回1。\n"
        "* 如果上下文包含任何与回答问题相关的信息，我将返回2。\n\n"
        "### 问题：{query}\n\n"
        "### 上下文：{context}\n\n"
        "基于提供的问题和上下文，相关性评分是：["
    )


# ============== 中文版 ContextRecall ==============
class ChineseContextRecallClassificationPrompt(
    PydanticPrompt[QCA, ContextRecallClassifications]
):
    """
    中文版上下文召回率分类提示词
    """

    name: str = "chinese_context_recall_classification"
    instruction: str = (
        "给定一个上下文和一个答案，分析答案中的每个句子，"
        "判断该句子是否可以归因于给定的上下文。\n"
        "使用'是'(1)或'否'(0)进行二元分类。\n"
        "输出JSON格式，包含理由。"
    )
    input_model = QCA
    output_model = ContextRecallClassifications
    examples = [
        (
            QCA(
                question="阿尔伯特·爱因斯坦是谁？",
                context=(
                    "阿尔伯特·爱因斯坦（1879年3月14日－1955年4月18日）是一位德裔美国理论物理学家，"
                    "被普遍认为是有史以来最伟大、最具影响力的科学家之一。"
                    "他因对理论物理的贡献，特别是光电效应定律的发现，获得了1921年诺贝尔物理学奖。"
                ),
                answer=(
                    "爱因斯坦，全名阿尔伯特·爱因斯坦，1879年出生于德国，1955年逝世。"
                    "他是理论物理学家，因光电效应获得了诺贝尔物理学奖。"
                    "他在1905年发表了4篇重要论文。"
                )
            ),
            ContextRecallClassifications(
                classifications=[
                    {
                        "statement": "爱因斯坦，全名阿尔伯特·爱因斯坦，1879年出生于德国，1955年逝世。",
                        "reason": "出生年份、逝世年份和国籍在上下文中明确提到。",
                        "attributed": 1
                    },
                    {
                        "statement": "他是理论物理学家。",
                        "reason": "上下文中明确提到他是理论物理学家。",
                        "attributed": 1
                    },
                    {
                        "statement": "他因光电效应获得了诺贝尔物理学奖。",
                        "reason": "上下文中提到了诺贝尔物理学奖和光电效应。",
                        "attributed": 1
                    },
                    {
                        "statement": "他在1905年发表了4篇重要论文。",
                        "reason": "上下文中没有提到1905年发表的论文数量。",
                        "attributed": 0
                    }
                ]
            )
        )
    ]


class ChineseContextRecall(LLMContextRecall):
    """
    中文版上下文召回率评估

    使用中文提示词评估上下文召回率。
    召回率 = 能归因于上下文的句子数 / 总句子数
    """

    name: str = "context_recall"
    context_recall_prompt: PydanticPrompt = field(
        default_factory=ChineseContextRecallClassificationPrompt
    )
