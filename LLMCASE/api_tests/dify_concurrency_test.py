"""
Dify LLM 接口并发压力测试

测试接口：POST https://ai.zy.com/v1/chat-messages
测试目标：找出该接口最大可承受的并发数
"""

import asyncio
import aiohttp
import time
import json
import statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConcurrencyResult:
    """并发测试结果"""
    concurrency: int
    total_requests: int
    success_count: int
    failure_count: int
    success_rate: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    std_response_time: float
    requests_per_second: float
    error_types: Dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0


class DifyConcurrencyTester:
    """Dify LLM 接口并发测试器"""

    def __init__(
        self,
        dify_url: str = "https://ai.zy.com/v1/chat-messages",
        api_key: str = "Bearer app-B6dyREi9TJJ1gFW1TKlOWETs",
        timeout: int = 120
    ):
        self.dify_url = dify_url
        self.api_key = api_key
        self.timeout = timeout
        self.results: List[ConcurrencyResult] = []

    async def make_request(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        test_prompt: str
    ) -> Tuple[bool, float, str, int]:
        """
        发送单个 Dify API 请求

        Returns:
            (success, response_time, error_message, status_code)
        """
        start_time = time.time()
        error_msg = ""
        status_code = 0

        headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": {},
            "query": test_prompt,
            "response_mode": "blocking",
            "user": "concurrency_test"
        }

        async with semaphore:
            try:
                async with session.post(
                    self.dify_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    status_code = response.status
                    elapsed = time.time() - start_time

                    if response.status == 200:
                        data = await response.json()
                        if "data" in data and "answer" in data["data"]:
                            return True, elapsed, "", status_code
                        elif "answer" in data:
                            return True, elapsed, "", status_code
                        else:
                            return False, elapsed, "响应格式异常", status_code
                    else:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("message", f"HTTP {status_code}")
                        except:
                            error_msg = f"HTTP {status_code}"
                        return False, elapsed, error_msg, status_code

            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                return False, elapsed, "请求超时", 0
            except aiohttp.ClientError as e:
                elapsed = time.time() - start_time
                return False, elapsed, f"连接错误: {str(e)[:50]}", 0
            except Exception as e:
                elapsed = time.time() - start_time
                return False, elapsed, str(e)[:100], 0

    async def test_concurrency(
        self,
        concurrency: int,
        total_requests: int = 20,
        warmup: int = 2
    ) -> ConcurrencyResult:
        """
        测试指定并发级别

        Args:
            concurrency: 并发数
            total_requests: 总请求数
            warmup: 预热请求数
        """
        print(f"\n{'='*60}")
        print(f"测试并发数: {concurrency}, 总请求: {total_requests}")
        print(f"{'='*60}")

        # 简单的测试提示词
        test_prompt = "请用一句话介绍自己"

        # 预热
        if warmup > 0:
            print(f"预热中 ({warmup} 个请求)...")
            warmup_sem = asyncio.Semaphore(min(warmup, 3))
            async with aiohttp.ClientSession() as warmup_session:
                tasks = [
                    self.make_request(warmup_session, warmup_sem, test_prompt)
                    for _ in range(warmup)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(1)

        # 正式测试
        start_time = time.time()
        semaphore = asyncio.Semaphore(concurrency)

        connector = aiohttp.TCPConnector(limit=concurrency + 10)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                self.make_request(session, semaphore, test_prompt)
                for _ in range(total_requests)
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.time() - start_time

        # 解析结果
        response_times = []
        success_count = 0
        failure_count = 0
        error_types: Dict[str, int] = {}

        for result in results:
            if isinstance(result, Exception):
                failure_count += 1
                error_type = type(result).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
            else:
                success, elapsed, error_msg, status = result
                response_times.append(elapsed)

                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    error_key = error_msg if error_msg else f"HTTP_{status}"
                    error_types[error_key] = error_types.get(error_key, 0) + 1

        # 统计
        avg_time = statistics.mean(response_times) if response_times else 0
        min_time = min(response_times) if response_times else 0
        max_time = max(response_times) if response_times else 0
        std_time = statistics.stdev(response_times) if len(response_times) > 1 else 0

        test_result = ConcurrencyResult(
            concurrency=concurrency,
            total_requests=total_requests,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_count / total_requests,
            avg_response_time=avg_time,
            min_response_time=min_time,
            max_response_time=max_time,
            std_response_time=std_time,
            requests_per_second=total_requests / duration if duration > 0 else 0,
            error_types=error_types,
            duration_seconds=duration
        )

        self.results.append(test_result)

        # 打印结果
        print(f"\n结果:")
        print(f"  成功率: {test_result.success_rate*100:>6.1f}%  ({success_count}/{total_requests})")
        print(f"  平均响应: {test_result.avg_response_time:>6.2f}s")
        print(f"  最小响应: {test_result.min_response_time:>6.2f}s")
        print(f"  最大响应: {test_result.max_response_time:>6.2f}s")
        print(f"  响应标准差: {test_result.std_response_time:>6.2f}s")
        print(f"  QPS: {test_result.requests_per_second:>6.2f}")
        print(f"  总耗时: {test_result.duration_seconds:>6.2f}s")

        if error_types:
            print(f"  错误分布:")
            for err, count in sorted(error_types.items(), key=lambda x: -x[1]):
                print(f"    - {err}: {count}")

        return test_result

    async def run_auto_test(
        self,
        start_concurrency: int = 5,
        max_concurrency: int = 50,
        step: int = 5,
        requests_per_level: int = 20,
        success_threshold: float = 0.95
    ) -> Tuple[int, ConcurrencyResult]:
        """
        自动递增测试，直到找到瓶颈

        Returns:
            (推荐最大并发数, 对应结果)
        """
        print("\n" + "="*70)
        print("开始 Dify LLM 接口并发测试")
        print(f"配置: 起始并发={start_concurrency}, 最大={max_concurrency}")
        print(f"      步进={step}, 每级请求={requests_per_level}, 成功阈值={success_threshold*100}%")
        print("="*70)

        concurrency = start_concurrency
        best_result = None
        best_concurrency = start_concurrency

        while concurrency <= max_concurrency:
            result = await self.test_concurrency(
                concurrency=concurrency,
                total_requests=requests_per_level
            )

            if result.success_rate >= success_threshold:
                best_concurrency = concurrency
                best_result = result
            else:
                # 成功率低于阈值，说明已达到瓶颈
                if best_result is None:
                    # 之前没有成功的情况，尝试降低并发
                    if concurrency > step:
                        concurrency = max(concurrency - step, 1)
                        step = 1  # 改为小步进
                    else:
                        break

            # 等待一段时间
            await asyncio.sleep(2)

            # 调整并发
            if result.success_rate >= 0.9:
                concurrency += step
            elif result.success_rate >= 0.7:
                concurrency += 1
            else:
                break

        return best_concurrency, best_result

    def print_summary(self):
        """打印测试摘要"""
        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)

        header = f"{'并发':>6} | {'成功率':>8} | {'平均响应':>10} | {'最大响应':>10} | {'QPS':>8}"
        print(header)
        print("-" * 60)

        for r in self.results:
            print(
                f"{r.concurrency:>6} | "
                f"{r.success_rate*100:>7.1f}% | "
                f"{r.avg_response_time:>9.2f}s | "
                f"{r.max_response_time:>9.2f}s | "
                f"{r.requests_per_second:>7.2f}"
            )

        # 找出最佳点
        successful = [r for r in self.results if r.success_rate >= 0.95]
        if successful:
            best = max(successful, key=lambda x: x.concurrency)
            print(f"\n结论:")
            print(f"  ✓ 推荐最大并发数: {best.concurrency}")
            print(f"    - 成功率: {best.success_rate*100:.1f}%")
            print(f"    - 平均响应: {best.avg_response_time:.2f}s")
            print(f"    - QPS: {best.requests_per_second:.2f}")

    def save_results(self, filename: str = None):
        """保存结果到文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dify_concurrency_test_{timestamp}.json"

        data = {
            "test_time": datetime.now().isoformat(),
            "api_url": self.dify_url,
            "results": [
                {
                    "concurrency": r.concurrency,
                    "total_requests": r.total_requests,
                    "success_count": r.success_count,
                    "failure_count": r.failure_count,
                    "success_rate": r.success_rate,
                    "avg_response_time": r.avg_response_time,
                    "min_response_time": r.min_response_time,
                    "max_response_time": r.max_response_time,
                    "std_response_time": r.std_response_time,
                    "requests_per_second": r.requests_per_second,
                    "error_types": r.error_types,
                    "duration_seconds": r.duration_seconds
                }
                for r in self.results
            ]
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存: {filename}")


async def quick_check():
    """快速检查接口可用性"""
    print("检查 Dify 接口可用性...")

    url = "https://ai.zy.com/v1/chat-messages"
    headers = {
        "Authorization": "Bearer app-B6dyREi9TJJ1gFW1TKlOWETs",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": {},
        "query": "你好",
        "response_mode": "blocking",
        "user": "test"
    }

    async with aiohttp.ClientSession() as session:
        try:
            start = time.time()
            async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                elapsed = time.time() - start
                if resp.status == 200:
                    data = await resp.json()
                    answer = data.get("data", {}).get("answer", "") or data.get("answer", "")
                    print(f"✓ 接口正常 (响应时间: {elapsed:.2f}s)")
                    print(f"  响应: {answer[:100]}...")
                    return True
                else:
                    print(f"✗ 接口异常: HTTP {resp.status}")
                    return False
        except Exception as e:
            print(f"✗ 连接失败: {e}")
            return False


async def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Dify LLM 接口并发测试")
    parser.add_argument("--check", action="store_true", help="仅检查接口可用性")
    parser.add_argument("--start", type=int, default=5, help="起始并发数 (默认: 5)")
    parser.add_argument("--max", type=int, default=50, help="最大并发数 (默认: 50)")
    parser.add_argument("--step", type=int, default=5, help="递增步进 (默认: 5)")
    parser.add_argument("--requests", type=int, default=20, help="每级请求数 (默认: 20)")

    args = parser.parse_args()

    if args.check:
        await quick_check()
        return

    tester = DifyConcurrencyTester()

    # 先检查接口可用性
    if not await quick_check():
        return

    # 运行自动测试
    max_conc, result = await tester.run_auto_test(
        start_concurrency=args.start,
        max_concurrency=args.max,
        step=args.step,
        requests_per_level=args.requests
    )

    # 打印摘要
    tester.print_summary()
    tester.save_results()


if __name__ == "__main__":
    print("="*70)
    print("Dify LLM 接口并发压力测试")
    print("="*70)
    asyncio.run(main())
