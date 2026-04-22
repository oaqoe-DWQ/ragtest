import requests

# ========== 你的专属固定配置 ==========
API_BASE_URL = "https://ai.zy.com/v1"
API_KEY = "app-pNhTctugLXxMwkqi7GRPrhWO"
# 从你截图里提取的 准确Embedding模型名
EMBEDDING_MODEL_NAME = "Doubao-embedding"

# Dify 官方文本转向量接口地址
request_url = f"{API_BASE_URL}/embeddings"

# 请求头（鉴权固定格式）
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# ========== 待向量化文本（可自行修改） ==========
input_texts = [
    "如何在笔记本安装海康4200客户端设置？"
]

# 请求体参数（严格遵循Dify官方格式）
payload = {
    "model": EMBEDDING_MODEL_NAME,
    "input": input_texts
}

# 发送接口请求
response = requests.post(url=request_url, headers=headers, json=payload)

# 结果解析与打印
print("请求状态码：", response.status_code)
resp_json = response.json()

# 打印向量信息
embedding_vector = resp_json["data"][0]["embedding"]
print(f"向量维度长度：{len(embedding_vector)}")
print(f"向量前10位数值：{embedding_vector[:10]}")