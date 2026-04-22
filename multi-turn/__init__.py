# -*- coding: utf-8 -*-
"""
Multi-Turn RAG Evaluation Module
多轮对话 RAG 评估模块

子模块：
  data_loader     : 数据加载与 MultiTurnSample 转换
  evaluator       : Ragas 评估引擎（基于 dify_llm.py DifyLLM）
  report_generator: Markdown 指标报告生成器

入口脚本：
  run_evaluation.py : 主运行脚本
"""

from multi_turn.data_loader import (
    MultiTurnDataLoader,
    MultiTurnDataConfig,
    MultiTurnSample,
    parse_role_content,
    build_messages,
    clean_html,
)
from multi_turn.evaluator import (
    MultiTurnEvaluator,
    MultiTurnEvalConfig,
)
from multi_turn.report_generator import (
    save_report,
    generate_markdown_report,
    METRIC_NAMES,
)

__all__ = [
    'MultiTurnDataLoader',
    'MultiTurnDataConfig',
    'MultiTurnEvaluator',
    'MultiTurnEvalConfig',
    'save_report',
    'generate_markdown_report',
    'METRIC_NAMES',
]
