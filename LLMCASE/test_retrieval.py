# -*- coding: utf-8 -*-
"""简单测试召回接口"""

import os
import sys
import requests

# 设置编码
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

RETRIEVAL_API_URL = "https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve"

def test_retrieval():
    """测试召回接口"""
    print("=" * 60)
    print("测试召回接口")
    print("=" * 60)
    print(f"URL: {RETRIEVAL_API_URL}")

    headers = {"Content-Type": "application/json"}

    payload = {
        "query": "如何在Mac上安装零信任",
        "knowledge_id": "parsed_files_title_re",
        "chat_history": [],
        "retrieval_setting": {
            "top_k": 5,
            "score_threshold": 0.5,
            "vector_weight": 0.8
        },
        "rewrite": False
    }

    try:
        print("\n发送请求...")
        print(f"Query: {payload['query']}")

        response = requests.post(
            RETRIEVAL_API_URL,
            headers=headers,
            json=payload,
            timeout=60  # 60秒超时
        )

        print(f"\n响应状态码: {response.status_code}")
        print(f"响应内容长度: {len(response.text)} 字符")

        if response.status_code == 200:
            result = response.json()
            print("\n[OK] 请求成功!")
            print(f"Keys: {list(result.keys())}")

            if "results" in result:
                print(f"\n获取到 {len(result['results'])} 条结果")
                for i, item in enumerate(result["results"][:2]):
                    print(f"\n--- 结果 {i+1} ---")
                    text = item.get("text", "")
                    print(f"Text: {text[:200]}..." if len(text) > 200 else f"Text: {text}")
        else:
            print(f"\n[ERROR] 请求失败")
            print(f"响应: {response.text[:500]}")

    except requests.exceptions.Timeout:
        print("\n[ERROR] 请求超时")
    except requests.exceptions.ConnectionError as e:
        print(f"\n[ERROR] 连接失败: {e}")
    except Exception as e:
        print(f"\n[ERROR] 发生错误: {e}")

if __name__ == "__main__":
    test_retrieval()
