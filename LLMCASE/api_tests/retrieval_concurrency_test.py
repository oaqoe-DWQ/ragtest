"""
召回接口并发测试脚本

测试接口：https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime

import requests


@dataclass
class TestResult:
    """测试结果"""
    concurrency: int
    total_requests: int
    success_count: int
    fail_count: int
    success_rate: float
    avg_response_time: float
    max_response_time: float
    min_response_time: float
    qps: float


def load_env():
    """加载环境变量"""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip().strip('"')


class RetrievalConcurrencyTester:
    """召回接口并发测试器"""

    API_URL = "https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve"
    DEFAULT_KNOWLEDGE_ID = "parsed_files_title_re"

    def __init__(self):
        load_env()
        self.api_key = os.getenv("RETRIEVAL_API_KEY", "")
        self.knowledge_id = os.getenv("KNOWLEDGE_ID", self.DEFAULT_KNOWLEDGE_ID)
        self.test_query = "如何安装打印机"

    def call_api(self, query: str = None) -> dict:
        """
        调用召回接口

        Args:
            query: 查询语句（可选）

        Returns:
            {"success": bool, "time": float, "error": str}
        """
        start_time = time.time()

        headers = {
            "Content-Type": "application/json"
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "knowledge_id": self.knowledge_id,
            "query": query or self.test_query,
            "top_k": 5,
            "score_threshold": 0.5
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )
            elapsed = time.time() - start_time

            if response.status_code == 200:
                return {"success": True, "time": elapsed}
            else:
                return {
                    "success": False,
                    "time": elapsed,
                    "error": f"HTTP {response.status_code}: {response.text[:100]}"
                }
        except requests.exceptions.RequestException as e:
            elapsed = time.time() - start_time
            return {"success": False, "time": elapsed, "error": str(e)}

    def test_concurrency(self, concurrency: int, total_requests: int = 50) -> TestResult:
        """
        测试指定并发级别

        Args:
            concurrency: 并发数
            total_requests: 总请求数

        Returns:
            TestResult 测试结果
        """
        print(f"\n{'='*50}")
        print(f"测试并发数: {concurrency}, 总请求数: {total_requests}")
        print(f"{'='*50}")

        success_count = 0
        fail_count = 0
        response_times = []

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self.call_api)
                for _ in range(total_requests)
            ]

            completed = 0
            for future in as_completed(futures):
                completed += 1
                result = future.result()

                if result["success"]:
                    success_count += 1
                else:
                    fail_count += 1
                    if completed <= 3 or fail_count <= 3:
                        print(f"  [FAIL] 请求{completed}: {result.get('error', 'Unknown')}")

                response_times.append(result["time"])

                if completed % 10 == 0:
                    print(f"  进度: {completed}/{total_requests}")

        total_time = time.time() - start_time

        result = TestResult(
            concurrency=concurrency,
            total_requests=total_requests,
            success_count=success_count,
            fail_count=fail_count,
            success_rate=success_count / total_requests * 100,
            avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            qps=total_requests / total_time if total_time > 0 else 0
        )

        return result

    def print_result(self, result: TestResult):
        """打印测试结果"""
        print(f"\n{'='*50}")
        print(f"并发数: {result.concurrency}")
        print(f"{'='*50}")
        print(f"  总请求数:     {result.total_requests}")
        print(f"  成功:        {result.success_count}")
        print(f"  失败:        {result.fail_count}")
        print(f"  成功率:      {result.success_rate:.2f}%")
        print(f"  平均响应:     {result.avg_response_time:.3f}s")
        print(f"  最大响应:     {result.max_response_time:.3f}s")
        print(f"  最小响应:     {result.min_response_time:.3f}s")
        print(f"  QPS:         {result.qps:.2f}")

    def run_test_suite(
        self,
        start_concurrency: int = 5,
        max_concurrency: int = 20,
        step: int = 5,
        total_requests: int = 50
    ):
        """
        运行测试套件，测试多个并发级别

        Args:
            start_concurrency: 起始并发数
            max_concurrency: 最大并发数
            step: 并发递增值
            total_requests: 每个并发级别的总请求数
        """
        print("=" * 60)
        print("召回接口并发压力测试")
        print("=" * 60)
        print(f"接口URL: {self.API_URL}")
        print(f"起始并发: {start_concurrency}, 最大并发: {max_concurrency}, 步长: {step}")
        print(f"每个级别总请求数: {total_requests}")
        print("=" * 60)

        # 先测试单个请求，确认接口可用
        print("\n[1] 测试单个请求...")
        single_result = self.call_api()
        if single_result["success"]:
            print(f"  ✓ 单个请求成功，响应时间: {single_result['time']:.3f}s")
        else:
            print(f"  ✗ 单个请求失败: {single_result.get('error')}")
            print("\n接口可能不可用，请检查配置")
            return

        # 测试多个并发级别
        print("\n[2] 开始并发测试...")
        results = []
        concurrency = start_concurrency

        while concurrency <= max_concurrency:
            result = self.test_concurrency(concurrency, total_requests)
            results.append(result)
            self.print_result(result)
            concurrency += step

        # 汇总
        print("\n" + "=" * 60)
        print("测试结果汇总")
        print("=" * 60)
        print(f"{'并发':>6} | {'成功率':>8} | {'平均响应':>10} | {'QPS':>8}")
        print("-" * 50)

        best_result = None
        for r in results:
            print(f"{r.concurrency:>6} | {r.success_rate:>7.2f}% | {r.avg_response_time:>9.3f}s | {r.qps:>7.2f}")
            if best_result is None or (r.success_rate >= 95 and r.qps > best_result.qps):
                best_result = r

        print("-" * 50)
        if best_result:
            print(f"\n推荐配置: 并发数 = {best_result.concurrency}")
            print(f"  - 成功率: {best_result.success_rate:.2f}%")
            print(f"  - QPS: {best_result.qps:.2f}")

        # 检查是否支持目标并发
        target_concurrency = 10
        target_result = next((r for r in results if r.concurrency == target_concurrency), None)
        if target_result:
            print(f"\n目标并发 {target_concurrency} 测试结果:")
            print(f"  - 成功率: {target_result.success_rate:.2f}%")
            if target_result.success_rate >= 99:
                print(f"  ✓ 支持 {target_concurrency} 并发")
            elif target_result.success_rate >= 90:
                print(f"  ⚠ 支持 {target_concurrency} 并发，但成功率略低")
            else:
                print(f"  ✗ 不支持 {target_concurrency} 并发，成功率过低")


def main():
    parser = argparse.ArgumentParser(description="召回接口并发测试")
    parser.add_argument("-c", "--concurrency", type=int, default=None,
                        help="测试单个并发数")
    parser.add_argument("-n", "--requests", type=int, default=50,
                        help="总请求数 (默认: 50)")
    parser.add_argument("--start", type=int, default=5,
                        help="起始并发数 (默认: 5)")
    parser.add_argument("--max", type=int, default=20,
                        help="最大并发数 (默认: 20)")
    parser.add_argument("--step", type=int, default=5,
                        help="并发递增值 (默认: 5)")

    args = parser.parse_args()

    tester = RetrievalConcurrencyTester()

    if args.concurrency:
        # 测试单个并发数
        result = tester.test_concurrency(args.concurrency, args.requests)
        tester.print_result(result)
    else:
        # 测试多个并发级别
        tester.run_test_suite(
            start_concurrency=args.start,
            max_concurrency=args.max,
            step=args.step,
            total_requests=args.requests
        )


if __name__ == "__main__":
    main()
