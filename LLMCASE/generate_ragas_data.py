# -*- coding: utf-8 -*-
"""
Ragas评估数据生成脚本
功能：读取query和标准答案，调用召回接口，生成标准Ragas评估数据
"""

import os
import sys
import json
import signal
import asyncio
import httpx
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def log(msg):
    """输出日志"""
    print(msg)


class RagasDataGenerator:
    """Ragas评估数据生成器"""

    # 召回接口配置
    RETRIEVAL_API_URL = "https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve"

    # 检索设置
    DEFAULT_TOP_K = 5
    DEFAULT_SCORE_THRESHOLD = 0.5
    DEFAULT_VECTOR_WEIGHT = 0.8

    # 异步并发配置
    MAX_CONCURRENT = 2  # 最大并发数，可配置
    REQUEST_INTERVAL = 5  # 请求间隔秒数（用于限流）
    RESUME_FROM_CHECKPOINT = True  # 是否从检查点恢复，可配置

    def __init__(self):
        """初始化"""
        self.knowledge_id = os.getenv("KNOWLEDGE_ID", "parsed_files_title_re")

        # 异步并发配置
        self.max_concurrent = int(os.getenv("MAX_CONCURRENT", self.MAX_CONCURRENT))
        self.request_interval = float(os.getenv("REQUEST_INTERVAL", self.REQUEST_INTERVAL))
        self.resume_from_checkpoint = os.getenv("RESUME_FROM_CHECKPOINT", str(self.RESUME_FROM_CHECKPOINT)).lower() == 'true'
        # 自定义检查点路径
        self.checkpoint_path = "D:\\Agelo\\rag_evaluate_ragas_BM25\\LLMCASE\\标准数据\\单轮未标准化测试集_标准化数据_20260422_104127.xlsx.checkpoint.xlsx"

        # 中断保存机制
        self._stop_flag = False
        self._stop_lock = asyncio.Lock()
        self._processed_results = []
        self._checkpoint_interval = 5  # 每处理5条保存一次

        log("=" * 60)
        log("Ragas评估数据生成器初始化")
        log("=" * 60)
        log(f"召回接口URL: {self.RETRIEVAL_API_URL}")
        log(f"知识库ID: {self.knowledge_id}")
        log(f"最大并发数: {self.max_concurrent}")
        log(f"请求间隔: {self.request_interval} 秒")
        log(f"从检查点恢复: {self.resume_from_checkpoint}")
        log("")

    async def set_stop_flag(self):
        """设置停止标志（用于中断保存）"""
        async with self._stop_lock:
            self._stop_flag = True
        log("[中断] 收到停止信号，正在保存进度...")

    async def is_stopped(self) -> bool:
        """检查是否收到停止信号"""
        async with self._stop_lock:
            return self._stop_flag

    def save_checkpoint(self, results: List[dict], output_path: str):
        """保存检查点（中断保存）"""
        if not results:
            return False

        try:
            output_df = pd.DataFrame(results)
            # 调整列顺序并移除临时字段
            if '_original_idx' in output_df.columns:
                output_df = output_df[['user_input', 'retrieved_contexts', 'response', 'reference_contexts', 'reference']]

            # 修复空值
            for col in output_df.columns:
                output_df[col] = output_df[col].apply(
                    lambda x: '' if isinstance(x, float) else x
                )

            output_df.to_excel(output_path, index=False, engine='openpyxl')
            log(f"[保存] 检查点已保存: {output_path} ({len(results)} 条数据)")
            return True
        except Exception as e:
            log(f"[ERROR] 保存检查点失败: {e}")
            return False

    async def call_retrieval_api(self, client: httpx.AsyncClient, query: str, top_k: int = None) -> tuple:
        """
        异步调用召回接口获取检索结果和LLM回答

        Args:
            client: httpx异步客户端
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            tuple: (retrieved_contexts: List[str], llm_answer: str)
        """
        if top_k is None:
            top_k = self.DEFAULT_TOP_K

        payload = {
            "query": query,
            "knowledge_id": self.knowledge_id,
            "chat_history": [],
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": self.DEFAULT_SCORE_THRESHOLD,
                "vector_weight": self.DEFAULT_VECTOR_WEIGHT
            },
            "rewrite": True
        }

        last_error = None
        for attempt in range(3):
            if attempt > 0:
                wait_time = 4 * (2 ** (attempt - 1))
                log(f"  [重试] 第{attempt}次重试，等待{wait_time}秒...")
                await asyncio.sleep(wait_time)

            try:
                response = await client.post(
                    self.RETRIEVAL_API_URL,
                    json=payload,
                    timeout=30.0
                )
                if response.status_code == 200:
                    result = response.json()
                    # 提取results中的text字段
                    texts = []
                    if "results" in result:
                        for item in result["results"]:
                            if "text" in item:
                                texts.append(item["text"])
                    # 提取llm_answer
                    llm_answer = result.get("llm_answer", "")
                    return texts, llm_answer
                else:
                    last_error = f"HTTP {response.status_code}"
                    if response.status_code in [429, 500, 502, 503, 504]:
                        continue
                    else:
                        break

            except httpx.TimeoutException:
                last_error = "请求超时"
                continue
            except httpx.HTTPError as e:
                last_error = str(e)
                continue

        log(f"  [ERROR] 召回失败: {last_error}")
        return [], ""

    async def _process_single_item(self, idx: int, query: str, reference: str, client: httpx.AsyncClient) -> Optional[dict]:
        """
        异步处理单个样本

        Args:
            idx: 索引
            query: 查询文本
            reference: 标准答案
            client: httpx异步客户端

        Returns:
            dict: 处理结果
        """
        if await self.is_stopped():
            return None

        log(f"  [{idx + 1}] 查询: {query[:50]}...")

        # 异步调用召回接口
        retrieved_contexts, llm_answer = await self.call_retrieval_api(client, query)

        # 定义块分隔符
        BLOCK_SEPARATOR = "<<<__CONTEXT_BLOCK__>>>"

        # 将检索结果合并为单个字符串
        context_text = BLOCK_SEPARATOR.join(retrieved_contexts) if retrieved_contexts else ""

        # 构建结果
        return {
            '_original_idx': idx,
            'user_input': query,
            'retrieved_contexts': context_text,
            'response': llm_answer,
            'reference_contexts': "无标准答案上下文",
            'reference': reference
        }

    def load_testcase(self, file_path: str) -> pd.DataFrame:
        """
        加载测试用例

        Args:
            file_path: Excel文件路径

        Returns:
            pd.DataFrame: 测试用例数据
        """
        log(f"[读取] 加载测试用例: {file_path}")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            log(f"[OK] 成功加载 {len(df)} 行数据")
            log(f"[列名] {list(df.columns)}")
            return df
        except Exception as e:
            log(f"[ERROR] 加载失败: {e}")
            return None

    async def generate_ragas_data_async(self, testcase_df: pd.DataFrame, output_path: str) -> bool:
        """
        异步生成Ragas评估数据

        Args:
            testcase_df: 测试用例DataFrame
            output_path: 输出文件路径

        Returns:
            bool: 是否成功
        """
        log("")
        log("=" * 60)
        log("开始生成Ragas评估数据（异步模式）")
        log("=" * 60)

        # 检查必要的列 - 尝试多种可能的列名
        possible_reference_cols = ['标准答案', '标准答', 'reference', '参考答案']
        reference_col = None
        for col in possible_reference_cols:
            if col in testcase_df.columns:
                reference_col = col
                break

        query_col = 'query' if 'query' in testcase_df.columns else None

        if not query_col:
            log(f"[ERROR] 缺少query列")
            return False

        if not reference_col:
            log(f"[ERROR] 缺少标准答案列，可用列: {list(testcase_df.columns)}")
            return False

        log(f"[INFO] 使用列: query='{query_col}', reference='{reference_col}'")

        # 检查是否从检查点恢复
        checkpoint_path = self.checkpoint_path if self.checkpoint_path else output_path + ".checkpoint.xlsx"
        resume_items = []

        if self.resume_from_checkpoint and os.path.exists(checkpoint_path):
            log(f"[恢复] 检测到检查点文件: {checkpoint_path}")
            try:
                checkpoint_df = pd.read_excel(checkpoint_path, engine='openpyxl')
                log(f"[恢复] 已加载 {len(checkpoint_df)} 条数据")

                # 收集已处理的query
                processed_queries = set()
                for _, row in checkpoint_df.iterrows():
                    if 'user_input' in row:
                        processed_queries.add(str(row['user_input']))
                        self._processed_results.append({
                            '_original_idx': _,
                            'user_input': row['user_input'],
                            'retrieved_contexts': row.get('retrieved_contexts', ''),
                            'response': row.get('response', ''),
                            'reference_contexts': row.get('reference_contexts', ''),
                            'reference': row.get('reference', '')
                        })

                # 过滤出未处理的数据
                for idx, row in testcase_df.iterrows():
                    query = str(row[query_col]) if pd.notna(row[query_col]) else ""
                    reference = str(row[reference_col]) if pd.notna(row[reference_col]) else ""

                    if not query:
                        continue

                    if query not in processed_queries:
                        resume_items.append({
                            'idx': idx,
                            'query': query,
                            'reference': reference
                        })

                log(f"[恢复] 已有 {len(processed_queries)} 条完成，还剩 {len(resume_items)} 条待处理")
            except Exception as e:
                log(f"[ERROR] 读取检查点失败: {e}")
                log("[恢复] 将从头开始处理")
                for idx, row in testcase_df.iterrows():
                    query = str(row[query_col]) if pd.notna(row[query_col]) else ""
                    reference = str(row[reference_col]) if pd.notna(row[reference_col]) else ""
                    if not query:
                        continue
                    resume_items.append({
                        'idx': idx,
                        'query': query,
                        'reference': reference
                    })
        else:
            # 收集待处理数据
            for idx, row in testcase_df.iterrows():
                query = str(row[query_col]) if pd.notna(row[query_col]) else ""
                reference = str(row[reference_col]) if pd.notna(row[reference_col]) else ""

                if not query:
                    continue

                resume_items.append({
                    'idx': idx,
                    'query': query,
                    'reference': reference
                })

        log(f"[INFO] 共 {len(resume_items)} 条数据待处理，最大并发数 {self.max_concurrent}，请求间隔 {self.request_interval} 秒")

        if len(resume_items) == 0:
            log("[INFO] 没有待处理的数据")
            return True

        # 重置中断标志
        self._stop_flag = False
        processed_count = 0
        results_lock = asyncio.Lock()

        # 使用信号量控制并发数
        semaphore = asyncio.Semaphore(self.max_concurrent)

        # 创建异步任务
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=self.max_concurrent)) as client:
            tasks = []
            for idx, item in enumerate(resume_items):
                task = asyncio.create_task(self._process_single_item_async(idx, item, client, semaphore, results_lock))
                tasks.append(task)

            # 等待所有任务完成
            for coro in asyncio.as_completed(tasks):
                item, result = await coro
                if result:
                    async with results_lock:
                        self._processed_results.append(result)
                    processed_count += 1
                    log(f"[{processed_count}/{len(resume_items)}] 完成: {item['query'][:30]}...")
                else:
                    log(f"[{processed_count + 1}/{len(resume_items)}] 失败: {item['query'][:30]}...")

                # 保存检查点
                if processed_count > 0 and processed_count % self._checkpoint_interval == 0:
                    self.save_checkpoint(self._processed_results, output_path + ".checkpoint.xlsx")

        # 按原始顺序排序
        self._processed_results.sort(key=lambda x: x.get('_original_idx', 0))

        # 如果中断，提前保存检查点
        if await self.is_stopped():
            self.save_checkpoint(self._processed_results, output_path + ".checkpoint.xlsx")
            log(f"[中断] 已保存 {len(self._processed_results)} 条数据到检查点文件")
            return False

        # 创建DataFrame并保存
        if self._processed_results:
            output_df = pd.DataFrame(self._processed_results)
            # 调整列顺序并移除临时字段
            output_df = output_df[['user_input', 'retrieved_contexts', 'response', 'reference_contexts', 'reference']]

            # 修复空值：将float类型的nan替换为空字符串
            for col in output_df.columns:
                output_df[col] = output_df[col].apply(
                    lambda x: '' if isinstance(x, float) else x
                )

            output_df.to_excel(output_path, index=False, engine='openpyxl')
            # 成功后删除检查点文件
            checkpoint_path = self.checkpoint_path if self.checkpoint_path else output_path + ".checkpoint.xlsx"
            if os.path.exists(checkpoint_path):
                try:
                    os.remove(checkpoint_path)
                    log(f"[清理] 已删除检查点文件")
                except Exception as e:
                    log(f"[提示] 删除检查点文件失败: {e}")
            log("")
            log("=" * 60)
            log(f"[OK] 生成完成! 共 {len(self._processed_results)} 条数据")
            log(f"[文件] 输出文件: {output_path}")
            log("=" * 60)
            return True
        else:
            log("")
            log("[ERROR] 没有生成任何数据")
            return False

    async def _process_single_item_async(self, idx: int, item: dict, client: httpx.AsyncClient, 
                                         semaphore: asyncio.Semaphore, results_lock: asyncio.Lock) -> tuple:
        """
        异步处理单个样本（带信号量和间隔控制）

        Args:
            idx: 序号
            item: 数据项
            client: httpx异步客户端
            semaphore: 信号量
            results_lock: 结果锁

        Returns:
            tuple: (item, result)
        """
        async with semaphore:
            if await self.is_stopped():
                return item, None

            query = item['query']
            reference = item['reference']
            original_idx = item['idx']

            log(f"  [{idx + 1}] 查询: {query[:50]}...")

            # 异步调用召回接口
            retrieved_contexts, llm_answer = await self.call_retrieval_api(client, query)

            # 请求完成后等待间隔时间
            if self.request_interval > 0:
                await asyncio.sleep(self.request_interval)

            # 定义块分隔符
            BLOCK_SEPARATOR = "<<<__CONTEXT_BLOCK__>>>"

            # 将检索结果合并为单个字符串
            context_text = BLOCK_SEPARATOR.join(retrieved_contexts) if retrieved_contexts else ""

            # 构建结果
            result = {
                '_original_idx': original_idx,
                'user_input': query,
                'retrieved_contexts': context_text,
                'response': llm_answer,
                'reference_contexts': "无标准答案上下文",
                'reference': reference
            }

            return item, result


_generator_instance = None
_generator_loop = None

def signal_handler(signum, frame):
    """信号处理器，用于优雅中断"""
    log("\n[中断] 收到中断信号 (Ctrl+C)，正在保存进度...")
    if _generator_instance is not None:
        asyncio.run(_generator_instance.set_stop_flag())

def main():
    """主函数"""
    global _generator_instance

    # 注册信号处理器
    if sys.platform == 'win32':
        import threading
        def check_interrupt():
            try:
                input()
            except EOFError:
                pass
        threading.Thread(target=check_interrupt, daemon=True).start()
    else:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    # 配置路径
    testcase_path = os.path.join(script_dir, "原始数据", "单轮未标准化测试集.xlsx")
    output_dir = os.path.join(script_dir, "标准数据")
    os.makedirs(output_dir, exist_ok=True)

    # 生成带时间戳的输出文件名，格式：原文档名_标准化数据_时间戳
    source_basename = os.path.splitext(os.path.basename(testcase_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{source_basename}_标准化数据_{timestamp}.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    # 检查是否有检查点文件（使用配置的检查点路径）
    generator = RagasDataGenerator()
    checkpoint_path = generator.checkpoint_path if generator.checkpoint_path else output_path + ".checkpoint.xlsx"

    # 处理路径（确保是绝对路径）
    testcase_path = os.path.abspath(testcase_path)
    output_path = os.path.abspath(output_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    log("")
    log("=" * 60)
    log("Ragas评估数据生成工具")
    log("=" * 60)
    log(f"[配置] 测试用例: {testcase_path}")
    log(f"[配置] 输出文件: {output_path}")

    # 检查检查点文件
    if os.path.exists(checkpoint_path):
        log(f"[提示] 发现检查点文件: {checkpoint_path}")
        log("   可以从此文件恢复，或删除后重新开始")

    log("")

    # 创建生成器
    generator = RagasDataGenerator()
    _generator_instance = generator

    # 加载测试用例
    testcase_df = generator.load_testcase(testcase_path)
    if testcase_df is None:
        log("[ERROR] 加载测试用例失败，程序退出")
        return

    # 异步生成数据
    success = asyncio.run(generator.generate_ragas_data_async(testcase_df, output_path))

    if success:
        log("\n[SUCCESS] 数据生成成功!")
        log(f"请查看输出文件: {output_path}")
    else:
        log("\n[ERROR] 数据生成失败，请检查错误信息")


if __name__ == "__main__":
    main()
