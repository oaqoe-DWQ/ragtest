# -*- coding: utf-8 -*-
"""
多轮对话评估 - Markdown 指标报告生成器

功能：
1. 将 Ragas 评估结果生成为结构化 Markdown 报告
2. 输出样本级详细分数表
3. 按指标分类汇总，含柱状图 ASCII 可视化
4. 输出文件保存至 indicator/ 子目录
"""

import os
import sys
from typing import Dict, Any, List
from datetime import datetime

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# ======================== 配置 ========================

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'indicator')


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ======================== 可视化 ========================

def ascii_bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    empty = width - filled
    return '[' + '=' * filled + '-' * empty + ']'


def score_color(score: float) -> str:
    if score >= 0.85:
        return '🟢 优秀'
    elif score >= 0.70:
        return '🟡 良好'
    elif score >= 0.50:
        return '🟠 一般'
    else:
        return '🔴 较差'


def score_to_emoji(score: float) -> str:
    if score >= 0.85:
        return '✅'
    elif score >= 0.70:
        return '👍'
    elif score >= 0.50:
        return '⚠️'
    else:
        return '❌'


# ======================== 指标中文名映射 ========================

METRIC_NAMES = {
    'agent_goal_accuracy_with_reference': '任务目标达成率（含参考）',
    'agent_goal_accuracy_without_reference': '任务目标达成率（无参考）',
    'instance_rubrics': '实例评分准则',
    'rubrics_score': '评分准则得分',
    'simple_criteria': '简单准则评分',
    'aspect_critic': '多维评判',
    'tool_call_accuracy': '工具调用准确性',
    'topic_adherence': '话题一致性',
    'faithfulness': '回答忠实度',
    'answer_relevancy': '回答相关性',
    'context_precision': '上下文精确度',
    'context_recall': '上下文召回率',
    'answer_correctness': '回答正确性',
    'response_groundedness': '回答有据性',
    'context_relevance': '上下文相关性',
}


def metric_display_name(key: str) -> str:
    return METRIC_NAMES.get(key, key)


# ======================== 报告生成 ========================

def generate_markdown_report(
    analysis: Dict[str, Any],
    timestamp: str = None,
) -> str:
    ensure_output_dir()
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

    scores = analysis.get('scores', {})
    per_sample = analysis.get('per_sample', {})
    meta_info = analysis.get('meta_info', [])
    samples_count = analysis.get('samples_count', 0)
    metrics_count = analysis.get('metrics_count', 0)

    lines = []

    # 标题
    lines.append("# 多轮对话 RAG 评估报告")
    lines.append("")
    lines.append("**生成时间**: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append("**数据来源**: `LLMCASE/原始数据/多轮带改写.xlsx`")
    lines.append("**样本总数**: " + str(samples_count) + " 个多轮对话")
    lines.append("**启用指标数**: " + str(metrics_count) + " 个")
    lines.append("")

    if 'error' in analysis:
        lines.append("## ❌ 评估错误")
        lines.append("")
        lines.append("```")
        lines.append(str(analysis['error']))
        lines.append("```")
        return '\n'.join(lines)

    # 一、汇总统计
    lines.append("## 一、指标汇总")
    lines.append("")

    if scores:
        valid_scores = {k: v for k, v in scores.items() if v is not None}
        avg_score = sum(valid_scores.values()) / len(valid_scores) if valid_scores else 0

        # 表头
        hdr = "| 指标 | 英文名 | 分数 | 评级 | 可视化 |"
        sep = "|------|--------|------|------|--------|"
        lines.append(hdr)
        lines.append(sep)

        for key, val in sorted(valid_scores.items(), key=lambda x: x[1] or 0, reverse=True):
            if val is None:
                continue
            display_name = metric_display_name(key)
            color = score_color(val)
            bar = ascii_bar(val)
            emoji = score_to_emoji(val)
            row = "| " + display_name + " | `" + key + "` | " + \
                ("%.4f" % val) + " | " + color + " | " + bar + " " + ("%.1f%%" % (val * 100)) + " |"
            lines.append(row)

        lines.append("")

        if valid_scores:
            lines.append("**综合平均分**: `" + ("%.4f" % avg_score) + "` (" +
                         ("%.1f%%" % (avg_score * 100)) + ") " + score_color(avg_score))
            lines.append("")
            lines.append("### 综合评分柱状图")
            lines.append("")
            for key, val in sorted(valid_scores.items(), key=lambda x: x[1] or 0, reverse=True):
                if val is None:
                    continue
                bar = ascii_bar(val)
                display_name = metric_display_name(key)
                lines.append("- **" + display_name + "**: " + bar + " **" + ("%.2f%%" % (val * 100)) + "**")
            lines.append("")

    # 二、样本级详细分数
    lines.append("## 二、样本级详细分数")
    lines.append("")

    if per_sample and meta_info:
        metric_keys = list(per_sample.keys())
        header_cols = ['序号', '会话ID', '历史轮次', 'query（截取前50字）']
        header_cols += [metric_display_name(k) for k in metric_keys]
        header_cols += ['平均分']

        lines.append("| " + " | ".join(header_cols) + " |")
        lines.append("|" + "|".join(['---' for _ in header_cols]) + "|")

        # 每行
        for i, (sample_scores, meta) in enumerate(zip(
            [list(x) for x in zip(*[per_sample[k] for k in metric_keys])],
            meta_info
        )):
            conv_id = meta.get('conversation_id', 'N/A')
            if len(conv_id) > 12:
                conv_id = conv_id[:12] + '...'
            query_raw = meta.get('query', '')
            query_snippet = query_raw[:50].replace('|', '\\|')
            if len(query_raw) > 50:
                query_snippet += '...'
            turn_count = str(meta.get('turn_count', 0))

            row_vals = [str(i + 1), '`' + conv_id + '`', turn_count, query_snippet]

            valid_vals = []
            for v in sample_scores:
                if v is not None:
                    try:
                        row_vals.append("%.4f" % float(v))
                        valid_vals.append(float(v))
                    except (ValueError, TypeError):
                        row_vals.append('N/A')
                else:
                    row_vals.append('N/A')

            avg = sum(valid_vals) / len(valid_vals) if valid_vals else 0
            row_vals.append("**%.4f**" % avg)

            lines.append("| " + " | ".join(row_vals) + " |")

        lines.append("")

    # 三、样本详情
    lines.append("## 三、样本详情")
    lines.append("")

    if meta_info:
        for i, meta in enumerate(meta_info):
            conv_id = meta.get('conversation_id', 'N/A')
            return_step = meta.get('returnStep', 'N/A')
            item_num = str(meta.get('item_num', 'N/A'))
            turn_count = meta.get('turn_count', 0)
            query = meta.get('query', '')

            sample_scores_strs = []
            if per_sample:
                for key in list(per_sample.keys())[:4]:
                    vals = per_sample.get(key, [])
                    if i < len(vals) and vals[i] is not None:
                        try:
                            sample_scores_strs.append(
                                "**" + metric_display_name(key) + "**: `" + ("%.4f" % float(vals[i])) + "`"
                            )
                        except (ValueError, TypeError):
                            sample_scores_strs.append(
                                "**" + metric_display_name(key) + "**: N/A"
                            )

            lines.append("### 样本 " + str(i + 1))
            lines.append("")
            lines.append("- **会话ID**: `" + conv_id + "`")
            lines.append("- **分类**: `" + return_step + "`")
            lines.append("- **项目编号**: `" + item_num + "`")
            lines.append("- **历史轮次**: " + str(turn_count))
            lines.append("- **用户问题**: " + query)
            if sample_scores_strs:
                lines.append("- **主要指标**: " + ', '.join(sample_scores_strs))
            lines.append("")

    # 四、指标说明
    lines.append("## 四、指标说明")
    lines.append("")

    if scores:
        metric_docs = {
            'agent_goal_accuracy_with_reference': (
                '评估多轮对话是否帮助用户完成了目标任务，'
                '以标准答案作为参考依据，评估最终答案的准确性和完整性。'
            ),
            'agent_goal_accuracy_without_reference': (
                '评估多轮对话是否帮助用户完成了目标任务，'
                '不依赖参考答案，通过自主评判回答质量进行评分。'
            ),
            'topic_adherence': '评估对话内容是否紧扣用户所提话题，检测话题漂移程度。',
            'faithfulness': '评估生成的回答是否忠实于检索到的上下文，避免幻觉。',
            'answer_relevancy': '评估回答与用户问题的相关性程度。',
            'context_precision': '评估检索到的上下文与问题相关的精确程度。',
            'context_recall': '评估上下文是否覆盖了正确答案所需的信息。',
            'answer_correctness': '综合评估回答的准确度，结合相似度和正确性判断。',
            'response_groundedness': '评估回答是否基于检索上下文生成（有据可查）。',
            'context_relevance': '评估上下文内容与用户问题的整体相关性。',
        }

        for key, val in sorted(scores.items(), key=lambda x: x[1] or 0, reverse=True):
            if val is None:
                continue
            name = metric_display_name(key)
            doc = metric_docs.get(key, '（暂无说明）')
            emoji = score_to_emoji(val)
            lines.append("- **" + emoji + " " + name + "** (`" + key + "`): " + doc)
            lines.append("  - 得分: **" + ("%.4f" % val) + "** (" +
                         ("%.1f%%" % (val * 100)) + ") " + score_color(val))
            lines.append("")

    # 页脚
    lines.append("---")
    lines.append("*本报告由 Multi-Turn RAG 评估系统自动生成 | Ragas 0.3.2 | " +
                 datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*")

    return '\n'.join(lines)


def save_report(analysis: Dict[str, Any], prefix: str = None) -> str:
    ensure_output_dir()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = "multi_turn_report_" + timestamp + ".md"
    if prefix:
        filename = "multi_turn_report_" + prefix + "_" + timestamp + ".md"

    filepath = os.path.join(OUTPUT_DIR, filename)
    content = generate_markdown_report(analysis, timestamp)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print("[报告] 已保存: " + filepath)
    return filepath


# 测试
if __name__ == '__main__':
    mock_analysis = {
        'scores': {
            'agent_goal_accuracy_with_reference': 0.8234,
            'topic_adherence': 0.7560,
            'faithfulness': 0.8901,
            'answer_relevancy': 0.7123,
            'context_precision': 0.6789,
            'context_recall': 0.8456,
        },
        'per_sample': {
            'agent_goal_accuracy_with_reference': [0.8, 0.85, 0.82],
            'topic_adherence': [0.75, 0.78, 0.74],
        },
        'meta_info': [
            {'conversation_id': 'conv001', 'query': 'windows如何安装零信任', 'turn_count': 2,
             'returnStep': 'FAQ-0.98', 'item_num': 4},
            {'conversation_id': 'conv002', 'query': 'Mac如何卸载软件', 'turn_count': 1,
             'returnStep': 'FAQ-0.98', 'item_num': 4},
            {'conversation_id': 'conv003', 'query': 'VPN连接失败怎么办', 'turn_count': 3,
             'returnStep': 'FAQ-0.98', 'item_num': 4},
        ],
        'samples_count': 3,
        'metrics_count': 6,
    }

    path = save_report(mock_analysis, 'test')
    print("\n[OK] 测试报告已生成: " + path)
