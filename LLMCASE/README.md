# Ragas评估数据生成工具

## 功能说明

本工具用于从 `testcase.xlsx` 生成标准的 Ragas 评估数据文件 `shuju.xlsx`。

## 数据映射关系

| testcase.xlsx | → | shuju.xlsx (Ragas格式) |
|---------------|---|------------------------|
| query | → | user_input |
| 标准答案 | → | reference |
| 召回接口results.text | → | retrieved_contexts |
| LLM接口回答 | → | response |
| 固定值 | → | reference_contexts = ["无标准答案上下文"] |

## API接口说明

### 1. 召回接口
- **URL**: `https://ytidc.zy.com:32212/rag_it_help/rag_it_help/v1/knowledge/test/retrieve`
- **方法**: POST
- **用途**: 根据query检索相关文档片段

### 2. LLM接口（Dify）
- **URL**: 从环境变量 `DIFY_URL` 读取
- **用途**: 生成回答

## 配置步骤

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
cd LLMCASE
copy .env.example .env
```

编辑 `.env` 文件，填写以下配置：

```env
# 召回接口配置
RETRIEVAL_API_KEY=your-retrieval-api-key

# 知识库ID
KNOWLEDGE_ID=parsed_files_title_re

# Dify LLM配置
DIFY_URL=https://ai.zy.com/v1/chat-messages
DIFY_API_KEY=Bearer app-your-api-key-here
```

### 2. 运行脚本

```bash
cd d:\RAGreport\rag_evaluate_ragas_BM25
.venv\Scripts\activate
python LLMCASE\generate_ragas_data.py
```

## 输出文件

运行成功后，会在 `LLMCASE` 文件夹生成 `shuju.xlsx` 文件。

## 输入文件要求

`testcase.xlsx` 必须包含以下列：
- `query`: 用户问题
- `标准答案`: 标准参考答案

## 注意事项

1. 确保网络可以访问召回接口和Dify API
2. 如果未配置Dify API，response字段将为空
3. retrieved_contexts 会包含多个检索片段（取决于top_k参数）
