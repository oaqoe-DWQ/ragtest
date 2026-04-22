# -*- coding: utf-8 -*-
"""
多轮对话标准化数据生成脚本（IT帮助台接口版）
功能：从多轮_有追问.xlsx读取chat_history和query，调用IT帮助台召回接口，
     从接口返回的rewrite_query映射到标准化xlsx文档的user_input字段
实际传给 LLM 的内容
组成部分	内容
系统指令	请根据以下上下文回答用户问题
上下文	    召回接口返回的多条 retrieved_contexts，用 <<<__CONTEXT_BLOCK__>>> 分隔
用户问题	原始 query（注意：不是 rewrite_query！）
python generate_multiturn_retrieval_data.py --retry-count 2 --retry-delay 30
--retry-count：失败重试次数，默认 1，设为 0 则不重试
--retry-delay：重试间隔秒数，默认 30
"""

import os
import sys
import json
import time
import signal
import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import ast
import argparse

# 设置UTF-8编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

# 定义块分隔符
BLOCK_SEPARATOR = "<<<__CONTEXT_BLOCK__>>>"


def log(msg: str):
    """输出日志"""
    print(msg)


class MultiTurnRetrievalGenerator:
    """多轮对话检索数据生成器（IT帮助台接口）"""

    # IT帮助台召回接口配置
    RETRIEVAL_API_URL = "https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve"

    # 检索设置
    DEFAULT_TOP_K = 5
    DEFAULT_SCORE_THRESHOLD = 0.5
    DEFAULT_VECTOR_WEIGHT = 0.8

    # 最终失败重试设置
    DEFAULT_RETRY_COUNT = 1
    DEFAULT_RETRY_DELAY = 20

    def __init__(self, retry_count: int = None, retry_delay: int = None):
        """初始化"""
        self.knowledge_id = os.getenv("KNOWLEDGE_ID", "parsed_files_title_re")

        # 中断保存机制
        self._stop_flag = False
        self._stop_lock = threading.Lock()
        self._processed_results = []
        self._checkpoint_interval = 5

        # 失败重试配置
        self.retry_count = retry_count if retry_count is not None else self.DEFAULT_RETRY_COUNT
        self.retry_delay = retry_delay if retry_delay is not None else self.DEFAULT_RETRY_DELAY

        log("=" * 60)
        log("多轮对话检索数据生成器初始化（IT帮助台接口）")
        log("=" * 60)
        log(f"召回接口URL: {self.RETRIEVAL_API_URL}")
        log(f"知识库ID: {self.knowledge_id}")
        log(f"失败重试: 次数={self.retry_count}, 间隔={self.retry_delay}s")
        log("")

    def set_stop_flag(self):
        """设置停止标志"""
        with self._stop_lock:
            self._stop_flag = True
        log("[中断] 收到停止信号，正在保存进度...")

    def is_stopped(self) -> bool:
        """检查是否收到停止信号"""
        with self._stop_lock:
            return self._stop_flag

    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """删除检查点文件"""
        try:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                log(f"[清理] 检查点文件已删除: {checkpoint_path}")
                return True
            return False
        except Exception as e:
            log(f"[WARN] 删除检查点失败: {e}")
            return False

    def save_checkpoint(self, results: List[dict], output_path: str):
        """保存检查点"""
        if not results:
            return False

        try:
            output_df = pd.DataFrame(results)
            output_df = output_df[['user_input', 'retrieved_contexts', 'response', 'reference_contexts', 'reference']]

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

    def parse_chat_history(self, chat_history_str: str) -> List[Dict]:
        """解析chat_history字符串"""
        if not chat_history_str or pd.isna(chat_history_str):
            return []

        try:
            chat_history = ast.literal_eval(chat_history_str)
            if isinstance(chat_history, list):
                return chat_history
            else:
                log(f"  [WARN] chat_history格式不正确")
                return []
        except (ValueError, SyntaxError) as e:
            log(f"  [WARN] 解析chat_history失败: {e}")
            try:
                fixed_str = chat_history_str.replace("'", '"')
                chat_history = json.loads(fixed_str)
                return chat_history if isinstance(chat_history, list) else []
            except:
                return []

    def call_retrieval_api(self, query: str, chat_history: List[Dict] = None, top_k: int = None, item_num: int = None) -> Dict[str, Any]:
        """
        调用IT帮助台召回接口

        Args:
            query: 查询文本
            chat_history: 聊天历史列表
            top_k: 返回结果数量

        Returns:
            Dict: 包含retrieved_contexts和rewrite_query的字典
        """
        if top_k is None:
            top_k = self.DEFAULT_TOP_K

        if chat_history is None:
            chat_history = []
        
        #接口请求参数
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "knowledge_id": self.knowledge_id,
            "chat_history": chat_history,
            "retrieval_setting": {
                "top_k": top_k,
                "score_threshold": self.DEFAULT_SCORE_THRESHOLD,
                "vector_weight": self.DEFAULT_VECTOR_WEIGHT
            },
            "rewrite": True  # 启用改写，获取rewrite_query
        }

        try:
            last_error = None
            response = None
            for attempt in range(3):
                if attempt > 0:
                    wait_time = 3 ** attempt
                    print(f"  ⚠ 第 {item_num} 条重试第 {attempt} 次，等待 {wait_time} 秒...")
                    time.sleep(wait_time)

                try:
                    response = requests.post(
                        self.RETRIEVAL_API_URL,
                        headers=headers,
                        json=payload,
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()

                        # 提取retrieved_contexts
                        texts = []
                        if "results" in result:
                            for item in result["results"]:
                                if "text" in item:
                                    texts.append(item["text"])

                        # 提取rewrite_query
                        rewrite_query = result.get("rewrite_query", "")

                        # 提取llm_answer映射为response
                        llm_answer = result.get("llm_answer", "")

                        return {
                            "retrieved_contexts": texts,
                            "rewrite_query": rewrite_query,
                            "response": llm_answer,
                            "raw_response": result
                        }
                    else:
                        last_error = f"HTTP {response.status_code}"
                        if response.status_code in [429, 500, 502, 503, 504]:
                            continue
                        else:
                            log(f"  [ERROR] HTTP {response.status_code}")
                            break

                except requests.exceptions.Timeout:
                    last_error = "请求超时"
                    continue
                except requests.exceptions.RequestException as e:
                    last_error = str(e)
                    continue

            log(f"  [ERROR] 召回失败: {last_error}")
            return {"retrieved_contexts": [], "rewrite_query": "", "response": "", "raw_response": None}

        except json.JSONDecodeError as e:
            log(f"  [ERROR] 解析响应失败: {e}")
            return {"retrieved_contexts": [], "rewrite_query": "", "response": "", "raw_response": None}

    def load_multiturn_testcase(self, file_path: str) -> pd.DataFrame:
        """加载多轮对话测试用例"""
        log(f"[读取] 加载多轮对话测试用例: {file_path}")

        try:
            df = pd.read_excel(file_path, engine='openpyxl')
            log(f"[OK] 成功加载 {len(df)} 行数据")
            log(f"[列名] {list(df.columns)}")
            return df
        except Exception as e:
            log(f"[ERROR] 加载失败: {e}")
            return None

    def generate_data(self, testcase_df: pd.DataFrame, output_path: str) -> bool:
        """生成标准化数据

        Args:
            testcase_df: 测试用例DataFrame
            output_path: 输出文件路径
        """
        log("")
        log("=" * 60)
        log("开始生成标准化数据")
        log("=" * 60)

        results = []
        MAX_WORKERS = 1

        for idx, row in testcase_df.iterrows():
            query = str(row['query']) if pd.notna(row['query']) else ""
            chat_history_str = str(row['chat_history']) if pd.notna(row['chat_history']) else ""
            reference = ""
            if '标准答案' in testcase_df.columns:
                reference = str(row['标准答案']) if pd.notna(row['标准答案']) else ""

            if not query:
                log(f"  [跳过] 第{idx+1}行: query为空")
                continue

            chat_history = self.parse_chat_history(chat_history_str)

            results.append({
                'idx': idx,
                'query': query,
                'chat_history': chat_history,
                'reference': reference
            })

        log(f"[INFO] 共 {len(results)} 条数据，使用 {MAX_WORKERS} 个并发线程")

        self._stop_flag = False
        self._processed_results = []
        processed_count = 0
        checkpoint_path = output_path + ".checkpoint.xlsx"

        # 给每个item编号（1-based），用于显示"第几条"
        for i, item in enumerate(results):
            item['_item_num'] = i + 1
        total_count = len(results)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {
                executor.submit(self._process_sample, item, item['_item_num'], total_count): item
                for item in results
            }

            for future in as_completed(future_to_data):
                if self.is_stopped():
                    log("[中断] 正在等待正在处理的任务完成...")
                    break

                item = future_to_data[future]
                processed_count += 1
                try:
                    result = future.result()
                    item['_result'] = result
                    if result:
                        self._processed_results.append(result)
                    else:
                        log(f"[{processed_count}/{len(results)}] 失败: {item['query'][:30]}...")
                except Exception as e:
                    log(f"[{processed_count}/{len(results)}] 异常: {e}")
                    item['_result'] = None
                    processed_count -= 1

                if processed_count % self._checkpoint_interval == 0:
                    log(f"[进度] 已完成 {processed_count}/{len(results)}")
                    self.save_checkpoint(self._processed_results, checkpoint_path)

        self._processed_results.sort(key=lambda x: x.get('_original_idx', 0))

        if self.is_stopped():
            self.save_checkpoint(self._processed_results, checkpoint_path)
            log(f"[中断] 已保存 {len(self._processed_results)} 条数据到检查点文件")
            return False

        # 第一轮未成功处理的项
        failed_items = [item for item in results if item.get('_result') is None]

        # 批量重试逻辑
        for retry_round in range(1, self.retry_count + 1):
            if not failed_items:
                break
            if self.is_stopped():
                log("[中断] 重试轮被中断")
                break

            log("")
            log(f"=" * 60)
            log(f"[重试] 第 {retry_round} 轮重试，等待 {self.retry_delay}s，共 {len(failed_items)} 条失败数据")
            log(f"=" * 60)
            time.sleep(self.retry_delay)

            newly_failed = []
            for item in failed_items:
                if self.is_stopped():
                    break
                item_num = item['_item_num']
                total_count = len(results)
                print(f"\n[重试-{retry_round}][{item_num}/{total_count}] 调用接口...")

                result = self._process_sample(item, item_num, total_count)
                if result:
                    self._processed_results.append(result)
                    log(f"  [OK] 重试成功: {item['query'][:30]}...")
                else:
                    newly_failed.append(item)
                    log(f"  [FAIL] 重试仍失败: {item['query'][:30]}...")

                processed_count += 1
                if processed_count % self._checkpoint_interval == 0:
                    self.save_checkpoint(self._processed_results, checkpoint_path)

            failed_items = newly_failed

        # 最终统计
        self._processed_results.sort(key=lambda x: x.get('_original_idx', 0))

        if failed_items:
            failed_df = pd.DataFrame([{
                'idx': item['idx'],
                'query': item['query'],
                'chat_history': str(item['chat_history']),
                'reference': item.get('reference', '')
            } for item in failed_items])
            failed_path = output_path.replace('.xlsx', '_failed.xlsx')
            failed_df.to_excel(failed_path, index=False, engine='openpyxl')
            log(f"[WARN] 最终仍有 {len(failed_items)} 条数据处理失败，已保存到: {failed_path}")

        if self._processed_results:
            output_df = pd.DataFrame(self._processed_results)
            output_df = output_df[['user_input', 'retrieved_contexts', 'response', 'reference_contexts', 'reference']]

            for col in output_df.columns:
                output_df[col] = output_df[col].apply(
                    lambda x: '' if isinstance(x, float) else x
                )

            output_df.to_excel(output_path, index=False, engine='openpyxl')

            # 任务成功完成，删除检查点文件
            self.delete_checkpoint(checkpoint_path)

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

    def _process_sample(self, item: dict, item_num: int, total_count: int) -> Optional[dict]:
        """处理单个样本"""
        idx = item['idx']
        query = item['query']
        chat_history = item['chat_history']
        reference = item.get('reference', '')

        print(f"\n[{item_num}/{total_count}] 调用接口...")

        api_result = self.call_retrieval_api(query, chat_history, item_num=item_num)

        retrieved_contexts_list = api_result["retrieved_contexts"]
        rewrite_query = api_result["rewrite_query"]
        response = api_result["response"]

        # 接口完全失败（无任何召回结果）时返回 None，便于外部收集
        if not retrieved_contexts_list and not response:
            print(f"  [WARN] 第 {item_num} 条接口返回无效（无召回结果也无response）")
            return None

        context_text = BLOCK_SEPARATOR.join(retrieved_contexts_list) if retrieved_contexts_list else ""
        user_input = rewrite_query if rewrite_query else query

        print(f"  输入: {query}")
        if rewrite_query:
            print(f"  输出: rewrite_query={rewrite_query} | 召回{len(retrieved_contexts_list)}条")
        else:
            print(f"  输出: 召回{len(retrieved_contexts_list)}条")

        return {
            '_original_idx': idx,
            'user_input': user_input,
            'retrieved_contexts': context_text,
            'reference_contexts': "无标准答案上下文",
            'reference': reference,
            'response': response
        }


_generator_instance = None


def signal_handler(signum, frame):
    """信号处理器"""
    log("\n[中断] 收到中断信号，正在保存进度...")
    if _generator_instance is not None:
        _generator_instance.set_stop_flag()


def main():
    """主函数"""
    global _generator_instance

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="多轮对话标准化数据生成工具（IT帮助台接口）")
    parser.add_argument("--retry-count", type=int, default=None,
                        help=f"失败重试次数（默认: {MultiTurnRetrievalGenerator.DEFAULT_RETRY_COUNT}，设为0则不重试）")
    parser.add_argument("--retry-delay", type=int, default=None,
                        help=f"失败重试间隔秒数（默认: {MultiTurnRetrievalGenerator.DEFAULT_RETRY_DELAY}）")
    args = parser.parse_args()

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
    input_path = os.path.join(script_dir, "原始数据", "多轮带改写test.xlsx")
    output_dir = os.path.join(script_dir, "标准数据")
    os.makedirs(output_dir, exist_ok=True)

    # 从输入路径提取文件名，生成"原名_标准化_日期时间戳"格式的输出路径
    input_filename = os.path.splitext(os.path.basename(input_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{input_filename}_标准化_{timestamp}.xlsx"
    output_path = os.path.join(output_dir, output_filename)

    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    log("")
    log("=" * 60)
    log("多轮对话标准化数据生成工具（IT帮助台接口）")
    log("=" * 60)
    log(f"[配置] 输入数据: {input_path}")
    log(f"[配置] 输出文件: {output_path}")
    log("")

    checkpoint_path = output_path + ".checkpoint.xlsx"
    if os.path.exists(checkpoint_path):
        log(f"[提示] 发现检查点文件: {checkpoint_path}")

    log("")

    # 创建生成器
    generator = MultiTurnRetrievalGenerator(retry_count=args.retry_count, retry_delay=args.retry_delay)
    _generator_instance = generator

    # 加载测试用例
    testcase_df = generator.load_multiturn_testcase(input_path)
    if testcase_df is None:
        log("[ERROR] 加载测试用例失败，程序退出")
        return

    # 生成数据
    success = generator.generate_data(testcase_df, output_path)

    if success:
        log("\n[SUCCESS] 数据生成成功!")
        log(f"请查看输出文件: {output_path}")
    else:
        log("\n[ERROR] 数据生成失败，请检查错误信息")


if __name__ == "__main__":
    main()
