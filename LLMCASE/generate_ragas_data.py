# -*- coding: utf-8 -*-
"""
Ragas评估数据生成脚本
功能：从testcase.xlsx读取query和标准答案，调用召回接口和LLM接口，生成标准Ragas评估数据
"""

import os
from re import T
import sys
import json
import signal
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

from dify_llm import DifyLLM
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


def log(msg):
    """输出日志"""
    print(msg)


class RagasDataGenerator:
    """Ragas评估数据生成器"""

    # 召回接口配置
    RETRIEVAL_API_URL = "https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/dev/retrieve"

    # 检索设置
    DEFAULT_TOP_K = 5
    DEFAULT_SCORE_THRESHOLD = 0.5
    DEFAULT_VECTOR_WEIGHT = 0.8

    def __init__(self, test_mode: bool = False):
        """初始化

        Args:
            test_mode: 测试模式，只调用召回接口，不调用LLM
        """
        self.test_mode = test_mode
        self.dify_url = os.getenv("DIFY_URL", "")
        self.dify_api_key = os.getenv("DIFY_API_KEY", "")
        self.knowledge_id = os.getenv("KNOWLEDGE_ID", "parsed_files_title_re")

        # 中断保存机制
        self._stop_flag = False
        self._stop_lock = threading.Lock()
        self._processed_results = []
        self._checkpoint_interval = 5  # 每处理5条保存一次

        log("=" * 60)
        log("Ragas评估数据生成器初始化")
        log("=" * 60)
        log(f"召回接口URL: {self.RETRIEVAL_API_URL}")
        log(f"知识库ID: {self.knowledge_id}")
        log(f"Dify URL: {self.dify_url}")
        log("")

        # 初始化LLM
        self.llm = None
        if not self.test_mode and self.dify_url and self.dify_api_key:
            try:
                self.llm = DifyLLM(
                    dify_url=self.dify_url,
                    api_key=self.dify_api_key,
                    temperature=0.0,
                    max_tokens=2000
                )
                log("[OK] LLM初始化成功")
            except Exception as e:
                log(f"[ERROR] LLM初始化失败: {e}")
        else:
            log("[WARN] 未配置Dify API，跳过LLM调用")

    def set_stop_flag(self):
        """设置停止标志（用于中断保存）"""
        with self._stop_lock:
            self._stop_flag = True
        log("[中断] 收到停止信号，正在保存进度...")

    def is_stopped(self) -> bool:
        """检查是否收到停止信号"""
        with self._stop_lock:
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

    def call_retrieval_api(self, query: str, top_k: int = None) -> List[str]:
        """
        调用召回接口获取检索结果

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[str]: 检索到的文本列表
        """
        if top_k is None:
            top_k = self.DEFAULT_TOP_K

        headers = {
            "Content-Type": "application/json"
        }

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

        try:
            log(f"  [调用] 召回接口: query='{query[:50]}...'")

            # 重试机制：最多重试2次，间隔5秒
            last_error = None
            response = None
            for attempt in range(3):  # 1次尝试 + 2次重试
                if attempt > 0:
                    log(f"  [重试] 第{attempt}次重试，等待5秒...")
                    import time
                    time.sleep(5)

                try:
                    response = requests.post(
                        self.RETRIEVAL_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        # 提取results中的text字段
                        texts = []
                        if "results" in result:
                            for item in result["results"]:
                                if "text" in item:
                                    texts.append(item["text"])
                        log(f"  [OK] 召回成功，获取 {len(texts)} 条结果")
                        return texts
                    else:
                        last_error = f"HTTP {response.status_code}"
                        log(f"  [错误] 尝试{attempt + 1}失败: {last_error}")
                        if response.status_code in [429, 500, 502, 503, 504]:
                            continue  # 服务器错误，重试
                        else:
                            break  # 客户端错误，不重试

                except requests.exceptions.Timeout:
                    last_error = "请求超时"
                    log(f"  [错误] 尝试{attempt + 1}失败: 请求超时")
                    continue
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    log(f"  [错误] 尝试{attempt + 1}失败: {e}")
                    continue

            log(f"  [ERROR] 召回失败: {last_error}")
            if response:
                log(f"     响应: {response.text[:200]}")
            return []

        except json.JSONDecodeError as e:
            log(f"  [ERROR] 解析响应失败: {e}")
            return []

    def call_llm_api(self, query: str, context: str = "") -> str:
        """
        调用LLM接口获取回答

        Args:
            query: 用户问题
            context: 上下文（可选）

        Returns:
            str: LLM回答
        """
        if self.llm is None:
            log(f"  [WARN] LLM未初始化，使用空回答")
            return ""

        try:
            # 构建prompt
            if context:
                prompt = f"请根据以下上下文回答用户问题。\n\n上下文：\n{context}\n\n用户问题：{query}"
            else:
                prompt = query

            log(f"  [调用] LLM接口...")
            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = response.content if hasattr(response, 'content') else str(response)
            log(f"  [OK] LLM回答成功，长度: {len(answer)} 字符")
            return answer

        except Exception as e:
            log(f"  [ERROR] LLM调用失败: {e}")
            return ""

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

    def generate_ragas_data(self, testcase_df: pd.DataFrame, output_path: str, use_llm: bool = True) -> bool:
        """
        生成Ragas评估数据

        Args:
            testcase_df: 测试用例DataFrame
            output_path: 输出文件路径
            use_llm: 是否调用LLM接口

        Returns:
            bool: 是否成功
        """
        log("")
        log("=" * 60)
        log("开始生成Ragas评估数据")
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

        # 收集待处理数据
        results = []

        # 并发线程数，根据API限制调整
        MAX_WORKERS = 2  # LLM接口可能有并发限制，建议不超过4

        for idx, row in testcase_df.iterrows():
            query = str(row[query_col]) if pd.notna(row[query_col]) else ""
            reference = str(row[reference_col]) if pd.notna(row[reference_col]) else ""

            if not query:
                continue

            results.append({
                'idx': idx,
                'query': query,
                'reference': reference
            })

        log(f"[INFO] 共 {len(results)} 条数据，使用 {MAX_WORKERS} 个并发线程")

        # 重置中断标志
        self._stop_flag = False
        self._processed_results = []
        processed_count = 0

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 提交所有任务
            future_to_data = {
                executor.submit(self._process_single_sample, item, use_llm): item
                for item in results
            }

            # 收集结果
            for future in as_completed(future_to_data):
                # 检查中断信号
                if self.is_stopped():
                    log("[中断] 正在等待正在处理的任务完成...")
                    break

                item = future_to_data[future]
                processed_count += 1
                try:
                    result = future.result()
                    if result:
                        self._processed_results.append(result)
                        log(f"[{processed_count}/{len(results)}] 完成: {item['query'][:30]}...")
                    else:
                        log(f"[{processed_count}/{len(results)}] 失败: {item['query'][:30]}...")
                except Exception as e:
                    log(f"[{processed_count}/{len(results)}] 异常: {e}")
                    processed_count -= 1

                # 打印进度，每隔5条保存检查点
                if processed_count % self._checkpoint_interval == 0:
                    log(f"[进度] 已完成 {processed_count}/{len(results)}")
                    self.save_checkpoint(self._processed_results, output_path + ".checkpoint.xlsx")

        # 按原始顺序排序
        self._processed_results.sort(key=lambda x: x.get('_original_idx', 0))

        # 如果中断，提前保存检查点
        if self.is_stopped():
            self.save_checkpoint(self._processed_results, output_path + ".checkpoint.xlsx")
            log(f"[中断] 已保存 {len(self._processed_results)} 条数据到检查点文件")
            return False

        # 4. 创建DataFrame并保存
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

    def _process_single_sample(self, item: dict, use_llm: bool) -> Optional[dict]:
        """
        处理单个样本（供并发调用）

        Args:
            item: 包含idx, query, reference的字典
            use_llm: 是否调用LLM

        Returns:
            dict: 处理结果
        """
        idx = item['idx']
        query = item['query']
        reference = item['reference']

        log(f"  [{idx + 1}] 查询: {query[:50]}...")

        # 1. 调用召回接口获取retrieved_contexts
        retrieved_contexts = self.call_retrieval_api(query)

        # 定义块分隔符
        BLOCK_SEPARATOR = "<<<__CONTEXT_BLOCK__>>>"

        # 将检索结果合并为单个字符串
        context_text = BLOCK_SEPARATOR.join(retrieved_contexts) if retrieved_contexts else ""

        # 2. 调用LLM接口获取response
        if use_llm:
            llm_answer = self.call_llm_api(query, context_text)
        else:
            llm_answer = ""

        # 3. 构建结果
        return {
            '_original_idx': idx,
            'user_input': query,
            'retrieved_contexts': context_text,
            'response': llm_answer,
            'reference_contexts': "无标准答案上下文",
            'reference': reference
        }


_generator_instance = None

def signal_handler(signum, frame):
    """信号处理器，用于优雅中断"""
    log("\n[中断] 收到中断信号 (Ctrl+C)，正在保存进度...")
    if _generator_instance is not None:
        _generator_instance.set_stop_flag()

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
    testcase_path = os.path.join(project_dir, "testcase3_num2.xlsx")
    output_path = os.path.join(script_dir, "testcase3_gaixie.xlsx")

    # 检查是否有检查点文件
    checkpoint_path = output_path + ".checkpoint.xlsx"
    resume_from_checkpoint = False

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

    # 检查环境变量
    if not os.getenv("RETRIEVAL_API_KEY"):
        log("[WARN] 未设置 RETRIEVAL_API_KEY 环境变量")
        log("   召回接口可能需要认证")

    # 检查是否使用LLM
    use_llm = False
    if os.getenv("DIFY_URL") and os.getenv("DIFY_API_KEY"):
        use_llm = True
        log("[INFO] 检测到Dify配置，将调用LLM接口")
    else:
        log("[WARN] 未设置 DIFY_URL 或 DIFY_API_KEY")
        log("   将跳过LLM调用，response字段将为空")

    # 创建生成器
    generator = RagasDataGenerator(test_mode=not use_llm)
    _generator_instance = generator

    # 加载测试用例
    testcase_df = generator.load_testcase(testcase_path)
    if testcase_df is None:
        log("[ERROR] 加载测试用例失败，程序退出")
        return

    # 生成数据
    success = generator.generate_ragas_data(
        testcase_df,
        output_path,
        use_llm=use_llm
    )

    if success:
        log("\n[SUCCESS] 数据生成成功!")
        log(f"请查看输出文件: {output_path}")
    else:
        log("\n[ERROR] 数据生成失败，请检查错误信息")


if __name__ == "__main__":
    main()