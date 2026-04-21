# -*- coding: utf-8 -*-
"""
对接：锐浩，4月15日
新检索接口数据生成脚本
功能：从Excel读取问题，调用新的检索接口，将返回数据处理成标准化xlsx格式
curl --location --request POST 'https://pre-jz.zy.com/it-helpdesk/api/v1/retrieval/test' \
--header 'Content-Type: application/json' \
--data-raw '{
    "question": "希沃一体机修改信息"
}'
"""

import os
import sys
import json
import time
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

# 定义块分隔符（与generate_ragas_data.py保持一致）
BLOCK_SEPARATOR = "<<<__CONTEXT_BLOCK__>>>"


def log(msg: str):
    """输出日志"""
    print(msg)


class NewRetrievalDataGenerator:
    """新检索接口数据生成器"""

    # 新检索接口配置
    RETRIEVAL_API_URL = "https://pre-jz.zy.com/it-helpdesk/api/v1/retrieval/test"

    # 检索设置
    DEFAULT_TOP_K = 5

    def __init__(self, test_mode: bool = False):
        """初始化

        Args:
            test_mode: 测试模式，只测试接口，不生成完整数据
        """
        self.test_mode = test_mode

        # 中断保存机制
        self._stop_flag = False
        self._stop_lock = threading.Lock()
        self._processed_results = []
        self._checkpoint_interval = 5  # 每处理5条保存一次

        log("=" * 60)
        log("新检索接口数据生成器初始化")
        log("=" * 60)
        log(f"检索接口URL: {self.RETRIEVAL_API_URL}")
        log(f"测试模式: {test_mode}")
        log("")

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

    def call_retrieval_api(self, question: str) -> Dict[str, Any]:
        """
        调用新检索接口获取结果

        Args:
            question: 问题文本

        Returns:
            Dict: 接口返回的完整数据
        """
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "question": question
        }

        try:
            log(f"  [调用] 检索接口: question='{question[:50]}...'")

            # 重试机制：最多重试2次，间隔5秒
            last_error = None
            response = None
            for attempt in range(3):  # 1次尝试 + 2次重试
                if attempt > 0:
                    log(f"  [重试] 第{attempt}次重试，等待5秒...")
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
                        log(f"  [OK] 检索成功")
                        return result
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

            log(f"  [ERROR] 检索失败: {last_error}")
            if response:
                log(f"     响应: {response.text[:200]}")
            return {}

        except json.JSONDecodeError as e:
            log(f"  [ERROR] 解析响应失败: {e}")
            return {}
        except Exception as e:
            log(f"  [ERROR] 请求异常: {e}")
            return {}

    def extract_retrieved_contexts(self, api_result: Dict[str, Any]) -> str:
        """
        从接口返回结果中提取检索到的文本内容

        Args:
            api_result: 接口返回的字典

        Returns:
            str: 合并后的检索文本，用分隔符连接
        """
        contexts = []

        # 尝试从data.results中提取content
        if 'data' in api_result and isinstance(api_result['data'], dict):
            data = api_result['data']
            if 'results' in data and isinstance(data['results'], list):
                for item in data['results']:
                    if 'content' in item and item['content']:
                        contexts.append(str(item['content']))

        # 如果没有找到content，尝试text字段（兼容旧接口）
        if not contexts and 'data' in api_result:
            data = api_result['data']
            if 'results' in data:
                for item in data['results']:
                    if 'text' in item and item['text']:
                        contexts.append(str(item['text']))

        # 合并所有上下文
        if contexts:
            return BLOCK_SEPARATOR.join(contexts)
        else:
            return ""

    def extract_llm_answer(self, api_result: Dict[str, Any]) -> str:
        """
        从接口返回结果中提取LLM生成的答案

        Args:
            api_result: 接口返回的字典

        Returns:
            str: LLM答案
        """
        # 尝试从data.llm_answer提取
        if 'data' in api_result and isinstance(api_result['data'], dict):
            data = api_result['data']
            if 'llm_answer' in data and data['llm_answer']:
                return str(data['llm_answer'])

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

    def generate_data(self, testcase_df: pd.DataFrame, output_path: str) -> bool:
        """
        生成标准化数据

        Args:
            testcase_df: 测试用例DataFrame
            output_path: 输出文件路径

        Returns:
            bool: 是否成功
        """
        log("")
        log("=" * 60)
        log("开始生成标准化数据")
        log("=" * 60)

        # 检查必要的列
        question_col = None
        for col in ['question', '问题', 'Question', 'query', 'Query']:
            if col in testcase_df.columns:
                question_col = col
                break

        reference_col = None
        for col in ['标准答案', '标准答', 'reference', '参考答案', '答案']:
            if col in testcase_df.columns:
                reference_col = col
                break

        if not question_col:
            log(f"[ERROR] 缺少问题列，可用列: {list(testcase_df.columns)}")
            return False

        if not reference_col:
            log(f"[ERROR] 缺少标准答案列，可用列: {list(testcase_df.columns)}")
            return False

        log(f"[INFO] 使用列: question='{question_col}', reference='{reference_col}'")

        # 收集待处理数据
        results = []
        for idx, row in testcase_df.iterrows():
            question = str(row[question_col]) if pd.notna(row[question_col]) else ""
            reference = str(row[reference_col]) if pd.notna(row[reference_col]) else ""

            if not question.strip():
                continue

            results.append({
                'idx': idx,
                'question': question,
                'reference': reference
            })

        log(f"[INFO] 共 {len(results)} 条数据，使用 5 个并发线程")

        # 重置中断标志
        self._stop_flag = False
        self._processed_results = []
        processed_count = 0

        # 使用线程池并发处理
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交所有任务
            future_to_data = {
                executor.submit(self._process_single_sample, item): item
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
                        log(f"[{processed_count}/{len(results)}] 完成: {item['question'][:30]}...")
                    else:
                        log(f"[{processed_count}/{len(results)}] 失败: {item['question'][:30]}...")
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

    def _process_single_sample(self, item: dict) -> Optional[dict]:
        """
        处理单个样本（供并发调用）

        Args:
            item: 包含idx, question, reference的字典

        Returns:
            dict: 处理结果
        """
        idx = item['idx']
        question = item['question']
        reference = item['reference']

        log(f"  [{idx + 1}] 问题: {question[:50]}...")

        # 1. 调用检索接口
        api_result = self.call_retrieval_api(question)

        if not api_result:
            log(f"  [ERROR] 接口调用失败，跳过")
            return None

        # 2. 提取检索结果
        retrieved_contexts = self.extract_retrieved_contexts(api_result)

        # 3. 提取LLM答案
        llm_answer = self.extract_llm_answer(api_result)

        # 4. 构建结果
        return {
            '_original_idx': idx,
            'user_input': question,
            'retrieved_contexts': retrieved_contexts,
            'response': llm_answer,
            'reference_contexts': "无标准答案上下文",
            'reference': reference
        }


def main():
    """主函数"""
    global _generator_instance

    # 创建生成器
    generator = NewRetrievalDataGenerator(test_mode=False)
    _generator_instance = generator

    # 配置路径
    # 输入文件：请修改为你的测试用例文件路径
    testcase_path = os.path.join(project_dir, "testcase4_num2.xlsx")

    # 输出文件：标准化数据文件
    output_path = os.path.join(script_dir, "testcase4_ruihao.xlsx")

    # 处理路径（确保是绝对路径）
    testcase_path = os.path.abspath(testcase_path)
    output_path = os.path.abspath(output_path)

    log("")
    log("=" * 60)
    log("新检索接口数据生成工具")
    log("=" * 60)
    log(f"[配置] 测试用例: {testcase_path}")
    log(f"[配置] 输出文件: {output_path}")

    # 检查检查点文件
    checkpoint_path = output_path + ".checkpoint.xlsx"
    if os.path.exists(checkpoint_path):
        log(f"[提示] 发现检查点文件: {checkpoint_path}")
        log("   可以从此文件恢复，或删除后重新开始")

    log("")

    # 加载测试用例
    testcase_df = generator.load_testcase(testcase_path)
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
