# -*- coding: utf-8 -*-
"""
多轮对话 RAG 评估 - 主运行脚本

用法：
    python run_evaluation.py

    # 指定评估指标
    python run_evaluation.py --metrics agent_goal_accuracy_with_reference topic_adherence faithfulness

    # 仅使用特定指标运行
    python run_evaluation.py --metrics-only

数据来源：  ../LLMCASE/原始数据/多轮带改写.xlsx
接口依赖：  ../dify_llm.py
输出报告：  ./indicator/multi_turn_report_<timestamp>.md

评估流程：
  1. 加载 Excel 数据（chat_history + query + 标准答案）
  2. 转换为 MultiTurnSample + [HumanMessage|AIMessage] 结构
  3. 初始化 DifyLLM 评估器
  4. 运行 Ragas 多轮/单轮指标评估
  5. 生成 Markdown 指标报告
"""

import os
import sys
import asyncio
import argparse
import logging
from datetime import datetime
from typing import List

# 设置 UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 静默日志
os.environ['RAGAS_QUIET'] = 'true'
os.environ['DISABLE_PROGRESS_BARS'] = 'true'
logging.basicConfig(level=logging.WARNING)


# ======================== 项目根目录设置 ========================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ======================== 导入子模块 ========================

from multi_turn.data_loader import MultiTurnDataLoader, MultiTurnDataConfig
from multi_turn.evaluator import MultiTurnEvaluator, MultiTurnEvalConfig
from multi_turn.report_generator import save_report


# ======================== 默认评估指标 ========================

DEFAULT_METRICS = [
    'agent_goal_accuracy_with_reference',  # 任务目标达成率（有参考答案）
    'topic_adherence',                      # 话题一致性
    'faithfulness',                         # 回答忠实度
    'context_recall',                       # 上下文召回率
    'context_precision',                    # 上下文精确度
    'answer_relevancy',                     # 回答相关性
]


def print_header():
    print()
    print("=" * 60)
    print("  多轮对话 RAG 评估系统")
    print("  Multi-Turn RAG Evaluation")
    print("=" * 60)
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python: {sys.version.split()[0]}")
    print()


def print_step(step_num: int, total: int, title: str):
    print()
    print(f"[{'='*6} Step {step_num}/{total}: {title} {'='*max(0, 50-len(title))}]")


# ======================== 主流程 ========================

async def run_evaluation(
    enabled_metrics: List[str] = None,
    use_dify: bool = True,
    output_prefix: str = None,
):
    """
    执行完整的多轮对话评估流程

    Args:
        enabled_metrics: 启用的指标列表
        use_dify: 是否使用 Dify LLM
        output_prefix: 报告文件名前缀
    """
    print_header()

    if enabled_metrics is None:
        enabled_metrics = DEFAULT_METRICS

    # ---- Step 1: 加载并转换数据 ----
    print_step(1, 4, "加载并转换数据")
    print()

    try:
        loader = MultiTurnDataLoader(MultiTurnDataConfig())
        samples, meta_info, raw_df = loader.load_and_build()
    except FileNotFoundError as e:
        print(f"[错误] 数据文件未找到: {e}")
        return None
    except Exception as e:
        print(f"[错误] 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    if not samples:
        print("[错误] 未找到有效样本，退出")
        return None

    print(f"\n[OK] 成功构建 {len(samples)} 个 MultiTurnSample")
    print(f"[配置] 启用指标: {enabled_metrics}")

    # 打印样本示例
    print(f"\n[示例] 前 2 个样本的消息结构:")
    for i, sample in enumerate(samples[:2]):
        msgs = sample.user_input
        print(f"\n  样本 {i + 1} ({len(msgs)} 条消息):")
        for msg in msgs:
            role = 'user' if hasattr(msg, 'type') and msg.type == 'human' else 'assistant'
            content = msg.content[:60] + ('...' if len(msg.content) > 60 else '')
            print(f"    [{role:8}] {content}")

    # ---- Step 2: 初始化评估器 ----
    print_step(2, 4, "初始化评估器")
    print()

    try:
        eval_config = MultiTurnEvalConfig(
            use_dify=use_dify,
            enabled_multiturn_metrics=enabled_metrics,
            max_workers=3,
            batch_size=6,
            temperature=0.0,
            top_p=0.1,
            max_tokens=2000,
        )

        evaluator = MultiTurnEvaluator(eval_config)
        evaluator.setup_environment()
    except Exception as e:
        print(f"[错误] 评估器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ---- Step 3: 运行评估 ----
    print_step(3, 4, "运行 Ragas 多轮评估")
    print()

    try:
        analysis = await evaluator.evaluate(samples, meta_info)
    except Exception as e:
        print(f"[错误] 评估过程异常: {e}")
        import traceback
        traceback.print_exc()
        return None

    if 'error' in analysis:
        print(f"[警告] 评估返回错误: {analysis['error']}")
        print("[继续] 仍生成报告...")

    # ---- Step 4: 生成报告 ----
    print_step(4, 4, "生成 Markdown 指标报告")
    print()

    try:
        report_path = save_report(analysis, prefix=output_prefix)
        print(f"\n[成功] 评估完成！")
        print(f"[报告] {report_path}")

        # 打印摘要
        scores = analysis.get('scores', {})
        if scores:
            print(f"\n{'='*40}")
            print(f"{'指标汇总':^40}")
            print(f"{'='*40}")
            for key, val in sorted(scores.items(), key=lambda x: x[1] or 0, reverse=True):
                if val is not None:
                    bar_len = int(val * 30)
                    bar = '=' * bar_len + '-' * (30 - bar_len)
                    name = key.replace('_', ' ').title()
                    print(f"  {name:<30} [{bar}] {val:.2%}")

            valid_vals = [v for v in scores.values() if v is not None]
            if valid_vals:
                avg = sum(valid_vals) / len(valid_vals)
                print(f"\n  {'综合平均分':<30} {avg:.2%}")
        else:
            print(f"\n[提示] 评估未返回有效分数，报告仍已保存。")

        return analysis

    except Exception as e:
        print(f"[错误] 报告生成失败: {e}")
        import traceback
        traceback.print_exc()
        return analysis


# ======================== CLI 入口 ========================

def main():
    parser = argparse.ArgumentParser(
        description='多轮对话 RAG 评估系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
可用的评估指标：
  多轮指标:
    agent_goal_accuracy_with_reference   任务目标达成率（含参考答案）
    agent_goal_accuracy_without_reference  任务目标达成率（无参考答案）
    topic_adherence                       话题一致性
    simple_criteria                       简单准则评分
    aspect_critic                         多维评判
    rubrics_score                         评分准则得分

  单轮指标（适用于末尾轮次）:
    faithfulness                         回答忠实度
    answer_relevancy                      回答相关性
    context_precision                     上下文精确度
    context_recall                        上下文召回率
    answer_correctness                    回答正确性
    response_groundedness                 回答有据性
    context_relevance                     上下文相关性

示例：
  python run_evaluation.py
  python run_evaluation.py --metrics agent_goal_accuracy_with_reference topic_adherence
  python run_evaluation.py --no-dify
        '''
    )

    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='指定评估指标（多个空格分隔）'
    )
    parser.add_argument(
        '--no-dify',
        action='store_true',
        help='不使用 Dify LLM，改用 Qwen 云端模型'
    )
    parser.add_argument(
        '--prefix',
        type=str,
        default=None,
        help='报告文件名前缀'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=3,
        help='并发工作线程数（默认: 3）'
    )

    args = parser.parse_args()

    # 确认数据文件存在
    source = os.path.join(PROJECT_ROOT, 'LLMCASE', '原始数据', '多轮带改写.xlsx')
    if not os.path.exists(source):
        print(f"[错误] 源数据文件不存在: {source}")
        sys.exit(1)

    asyncio.run(run_evaluation(
        enabled_metrics=args.metrics,
        use_dify=not args.no_dify,
        output_prefix=args.prefix,
    ))


if __name__ == '__main__':
    main()
