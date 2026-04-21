"""
测试检索接口 - 调用知识库API进行向量检索

接口：POST https://jz.zy.com/kb/api/v1/retrieval
"""

import json
import requests
from typing import List, Dict, Any


def test_retrieval_api(
    query: str = "如何安装打印机",
    knowledge_id: str = "dd1d50a2cfad40819d3c2faa6b3337b5",
    top_k: int = 2,
    score_threshold: float = 0.01,
    bearer_token: str = "kb-6bfe4e5c424e4cd7968b6283170b805e",
    app_token: str = "o1jl1ca28r7gZRJAv0kBd88KHrliQyhQ",
    api_url: str = "https://jz.zy.com/kb/api/v1/retrieval"
) -> Dict[str, Any]:
    """
    调用知识库检索接口

    Args:
        query: 查询问题
        knowledge_id: 知识库ID
        top_k: 返回结果数量
        score_threshold: 相似度阈值
        bearer_token: Bearer令牌
        app_token: App令牌
        api_url: API地址

    Returns:
        检索结果字典
    """
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "token": app_token,
        "Content-Type": "application/json"
    }

    payload = {
        "knowledge_id": knowledge_id,
        "query": query,
        "retrieval_setting": {
            "top_k": top_k,
            "score_threshold": score_threshold
        }
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return {"success": True, "data": result}
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def parse_retrieval_results(result: Dict[str, Any]) -> List[str]:
    """
    解析检索结果，提取内容片段

    Args:
        result: API返回的结果

    Returns:
        检索到的文本内容列表
    """
    if not result.get("success"):
        return []

    data = result.get("data", {})
    records = data.get("records", [])

    contexts = []
    for record in records:
        content = record.get("content", "")
        if content:
            contexts.append(content)

    return contexts


if __name__ == "__main__":
    # 示例调用
    print("正在调用检索接口...")
    result = test_retrieval_api(query="如何安装打印机")

    if result["success"]:
        print("✓ 调用成功")
        contexts = parse_retrieval_results(result)
        print(f"  返回 {len(contexts)} 条检索结果:")
        for i, ctx in enumerate(contexts, 1):
            print(f"\n--- 结果 {i} ---")
            print(ctx)
    else:
        print(f"✗ 调用失败: {result['error']}")
