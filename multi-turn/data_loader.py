# -*- coding: utf-8 -*-
"""
多轮对话评估 - 数据加载与预处理模块

功能：
1. 加载多轮对话 Excel 源数据（chat_history + query）
2. 将 [{'role': 'user'/'assistant', 'content': '...'}] 格式转换为
   MultiTurnSample 的 List[HumanMessage|AIMessage] 结构
3. 按对话分组（conversation_id），保留最后一轮的 query 作为评估目标
4. 对话轮次统计

依赖上游数据格式（来自 原始数据/多轮带改写.xlsx）：
  - conversation_id  : str，对话会话 ID
  - chat_history     : str，[{'role': 'user'/'assistant', 'content': '...'}] 列表的字符串表示
  - query            : str，当前轮次的用户提问
  - 标准答案          : str，参考标准答案
  - returnStep       : str，召回步骤标签
  - item_num         : int，召回项目编号

输出 MultiTurnSample 所需字段：
  - user_input       : List[Union[HumanMessage, AIMessage]]，多轮对话消息列表
  - reference        : str，标准答案
  - reference_topics : List[str]（可选），话题标签
"""

import os
import sys
import ast
import pandas as pd
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# 设置 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 跨目录导入父项目模块
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ragas.messages import HumanMessage, AIMessage
from ragas.dataset_schema import MultiTurnSample


# ======================== 配置 ========================

SOURCE_EXCEL = os.path.join(
    _project_root,
    'LLMCASE', '原始数据', '多轮带改写.xlsx'
)


@dataclass
class MultiTurnDataConfig:
    """多轮对话数据配置"""
    source_excel: str = SOURCE_EXCEL
    source_encoding: str = 'utf-8'


# ======================== 核心解析逻辑 ========================

def parse_role_content(history_str: str) -> List[Dict[str, str]]:
    """
    解析 chat_history 字符串为 Python 列表

    Args:
        history_str: "[{'role': 'user', 'content': '...'}]" 格式的字符串

    Returns:
        [{'role': 'user'/'assistant', 'content': '...'}] 列表
    """
    if not history_str or pd.isna(history_str):
        return []

    s = str(history_str).strip()
    if not s or s == 'nan':
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict) and 'role' in m]
    except (ValueError, SyntaxError):
        pass

    try:
        fixed = s.replace("'", '"')
        parsed = json.loads(fixed)
        if isinstance(parsed, list):
            return [m for m in parsed if isinstance(m, dict) and 'role' in m]
    except json.JSONDecodeError:
        pass

    return []


def build_messages(history: List[Dict[str, str]], current_query: str) -> List:
    """
    将 [{'role': ..., 'content': ...}] 转换为 LangChain Message 列表

    Args:
        history: 解析后的对话历史
        current_query: 当前轮次的用户问题（追加到末尾）

    Returns:
        List[Union[HumanMessage, AIMessage]]
    """
    messages = []

    for m in history:
        role = m.get('role', '').strip().lower()
        content = str(m.get('content', '')).strip()

        # 跳过空内容
        if not content:
            continue

        if role == 'user':
            messages.append(HumanMessage(content=content))
        elif role in ('assistant', 'ai', 'bot'):
            messages.append(AIMessage(content=content))
        else:
            # 未知角色默认当作用户消息
            messages.append(HumanMessage(content=content))

    # 追加当前轮次的用户消息
    if current_query and str(current_query).strip():
        messages.append(HumanMessage(content=str(current_query).strip()))

    return messages


def clean_html(text: str) -> str:
    """移除文本中的 HTML 标签和多余空白"""
    if not text:
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ======================== 多轮对话数据构建 ========================

class MultiTurnDataLoader:
    """多轮对话数据加载与转换器"""

    def __init__(self, config: MultiTurnDataConfig = None):
        self.config = config or MultiTurnDataConfig()

    def load_excel(self) -> pd.DataFrame:
        """加载原始 Excel 数据"""
        if not os.path.exists(self.config.source_excel):
            raise FileNotFoundError(f"源数据文件不存在: {self.config.source_excel}")

        df = pd.read_excel(self.config.source_excel, engine='openpyxl')
        return df

    def convert_row_to_sample(
        self,
        row: pd.Series,
        drop_html: bool = True
    ) -> Optional[MultiTurnSample]:
        """
        将单行 Excel 数据转换为 MultiTurnSample

        Args:
            row: pandas Series，代表一行数据
            drop_html: 是否清除 HTML 标签

        Returns:
            MultiTurnSample 或 None（当无有效数据时）
        """
        history_str = row.get('chat_history', '')
        query = str(row.get('query', '')).strip() if pd.notna(row.get('query')) else ''
        reference = str(row.get('标准答案', '')).strip() if pd.notna(row.get('标准答案')) else ''

        if drop_html:
            query = clean_html(query)
            reference = clean_html(reference)

        # 解析历史对话
        history = parse_role_content(history_str)

        # 清理历史中的 HTML
        if drop_html:
            for m in history:
                if 'content' in m:
                    m['content'] = clean_html(m['content'])

        # 跳过无当前问题的行
        if not query:
            return None

        # 构建消息列表
        messages = build_messages(history, query)

        if not messages:
            return None

        # 提取话题标签（来自 returnStep）
        topics = None
        if pd.notna(row.get('returnStep')):
            topics = [str(row['returnStep']).strip()]

        return MultiTurnSample(
            user_input=messages,
            reference=reference or None,
            reference_topics=topics,
        )

    def build_dataset(
        self,
        df: pd.Series,
        drop_empty: bool = True
    ) -> Tuple[List[MultiTurnSample], List[Dict]]:
        """
        将整个 DataFrame 转换为 MultiTurnSample 列表

        Args:
            df: 原始 DataFrame
            drop_empty: 是否丢弃无有效问题的行

        Returns:
            (samples, meta_info) 元组
            samples: MultiTurnSample 列表
            meta_info: 每行的元信息（含对话ID、原始query等）
        """
        samples = []
        meta_info = []

        for idx, row in df.iterrows():
            sample = self.convert_row_to_sample(row)

            if sample is not None:
                samples.append(sample)
                meta_info.append({
                    'original_index': idx,
                    'conversation_id': str(row.get('conversation_id', '')),
                    'query': str(row.get('query', ''))[:200],
                    'returnStep': str(row.get('returnStep', '')),
                    'item_num': row.get('item_num', ''),
                    'turn_count': len([m for m in row.get('chat_history', []) if isinstance(m, dict) and m.get('role') == 'user'])
                                if pd.notna(row.get('chat_history')) else 0,
                })

        return samples, meta_info

    def load_and_build(self) -> Tuple[List[MultiTurnSample], List[Dict], pd.DataFrame]:
        """
        完整流程：加载 Excel 并构建 MultiTurnSample

        Returns:
            (samples, meta_info, raw_df) 三元组
        """
        print("[加载] 读取源数据...")
        raw_df = self.load_excel()
        print(f"[OK] 共 {len(raw_df)} 行数据")
        print(f"[列名] {list(raw_df.columns)}")

        # 统计对话轮次分布
        turn_counts = {}
        for _, row in raw_df.iterrows():
            history = parse_role_content(row.get('chat_history', ''))
            n_turns = len([m for m in history if m.get('role') == 'user'])
            turn_counts[n_turns] = turn_counts.get(n_turns, 0) + 1

        print(f"[统计] 历史对话轮次分布: {dict(sorted(turn_counts.items()))}")

        samples, meta_info = self.build_dataset(raw_df)
        print(f"[OK] 成功转换 {len(samples)} 个 MultiTurnSample")

        return samples, meta_info, raw_df


# ======================== 入口 ========================

if __name__ == '__main__':
    loader = MultiTurnDataLoader()
    samples, meta_info, raw_df = loader.load_and_build()

    print(f"\n[样本示例]")
    for i, sample in enumerate(samples[:3]):
        msgs = sample.user_input
        print(f"\n--- 样本 {i + 1} ({len(msgs)} 条消息) ---")
        for m in msgs:
            role = 'user' if isinstance(m, HumanMessage) else 'assistant'
            content = m.content[:80] + ('...' if len(m.content) > 80 else '')
            print(f"  [{role}] {content}")
        print(f"  [reference] {sample.reference[:80] if sample.reference else '(无)'}...")
