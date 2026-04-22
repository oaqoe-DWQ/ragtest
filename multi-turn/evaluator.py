# -*- coding: utf-8 -*-
"""
多轮对话评估 - Ragas 评估引擎

功能：
1. 基于 dify_llm.py 的 DifyLLM 作为评估 LLM
2. 支持 Ragas 0.3.2 的多轮对话评估指标
3. 兼容自定义中文指标（faithfulness、context_recall 等）

多轮对话指标（MultiTurnMetric）：
  - AgentGoalAccuracyWithReference : 任务目标达成率（有参考答案）
  - AgentGoalAccuracyWithoutReference : 任务目标达成率（无参考答案）
  - InstanceRubrics : 基于实例评分准则的多维评估
  - RubricsScore : 基于评分准则的评估
  - ToolCallAccuracy : 工具调用准确性
  - TopicAdherenceScore : 话题一致性评分
  - SimpleCriteriaScore / AspectCritic : 单一/多维评判

单轮指标（SingleTurnMetric，也适用于多轮末尾单轮）：
  - Faithfulness : 回答忠实度
  - AnswerRelevancy : 回答相关性
  - ContextPrecision : 上下文精确度
  - ContextRecall : 上下文召回率
  - AnswerCorrectness : 回答正确性
  - ResponseGroundedness : 回答有据性
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# 跨目录导入
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 日志静默
os.environ['RAGAS_QUIET'] = 'true'
os.environ['DISABLE_PROGRESS_BARS'] = 'true'
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('ragas').setLevel(logging.ERROR)

import pandas as pd
import numpy as np

# Ragas 核心
from ragas import evaluate, EvaluationDataset
from ragas.dataset_schema import MultiTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics.base import SingleTurnMetric, MultiTurnMetric

# 评估指标
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    AnswerCorrectness,
    ResponseGroundedness,
    AgentGoalAccuracyWithReference,
    AgentGoalAccuracyWithoutReference,
    InstanceRubrics,
    RubricsScore,
    SimpleCriteriaScore,
    AspectCritic,
    ToolCallAccuracy,
    TopicAdherenceScore,
)

# 自定义中文指标
from custom_metrics import ChineseContextRelevance, ChineseContextRecall

# LLM / Embedding
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.llms import Ollama

# Dify LLM（父项目接口）
from dify_llm import create_dify_llm

# 配置
from dotenv import load_dotenv
load_dotenv()


# ======================== 配置 ========================

@dataclass
class MultiTurnEvalConfig:
    """多轮对话评估配置"""

    # Dify LLM（主要评估器）
    use_dify: bool = True
    dify_api_key: str = ''
    dify_url: str = ''
    dify_app_id: Optional[str] = None
    dify_streaming: bool = False

    # 采样参数（LLM 输出稳定性）
    temperature: float = 0.0
    top_p: float = 0.1
    max_tokens: int = 2000

    # Embedding（用于 answer_relevancy、answer_correctness 等）
    use_ollama: bool = False
    ollama_url: str = 'http://localhost:11434'
    ollama_embedding_model: str = 'embeddinggemma:300m'

    # 云端 Embedding（Qwen）
    api_key: str = ''
    api_base: str = ''
    embedding_model: str = 'text-embedding-v1'

    # 性能
    max_workers: int = 3
    batch_size: int = 6

    # 多轮指标选择
    enabled_multiturn_metrics: List[str] = None

    def __post_init__(self):
        if self.enabled_multiturn_metrics is None:
            self.enabled_multiturn_metrics = [
                'agent_goal_accuracy_with_reference',
                'topic_adherence',
            ]

        # 自动从环境变量加载 Dify 配置
        if self.use_dify:
            self.dify_api_key = self.dify_api_key or os.getenv('DIFY_API_KEY', '')
            self.dify_url = self.dify_url or os.getenv('DIFY_URL', '')
            self.dify_app_id = self.dify_app_id or os.getenv('DIFY_APP_ID')


# ======================== Embedding 包装器 ========================

class OllamaEmbeddingsWrapper:
    """Ollama Embedding 包装器（带缓存）"""

    def __init__(self, ollama_url: str, model_name: str):
        import requests as _requests
        self._url = ollama_url.rstrip('/')
        self._model = model_name
        self._requests = _requests
        self._cache = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import random
        results = []
        for text in texts:
            key = f"{self._model}:{text[:100]}"
            if key in self._cache:
                results.append(self._cache[key])
                continue
            try:
                resp = self._requests.post(
                    f"{self._url}/api/embeddings",
                    json={"model": self._model, "prompt": text},
                    timeout=30
                )
                if resp.status_code == 200:
                    emb = resp.json()['embedding']
                    self._cache[key] = emb
                    results.append(emb)
                else:
                    results.append([random.random() for _ in range(2048)])
            except Exception:
                results.append([random.random() for _ in range(2048)])
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class QwenEmbeddingsWrapper:
    """Qwen Embedding 包装器（带缓存）"""

    def __init__(self, api_key: str, api_base: str, model_name: str):
        import requests as _requests
        self._api_key = api_key
        self._api_base = api_base.rstrip('/')
        self._model = model_name
        self._requests = _requests
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self._cache = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import random
        to_embed, indices = [], []
        for i, t in enumerate(texts):
            key = f"{self._model}:{t[:100]}"
            if key in self._cache:
                to_embed.append(self._cache[key])
            else:
                indices.append(i)
                to_embed.append(t)

        if not indices:
            return texts[:0]  # 全缓存

        results = [None] * len(texts)
        cached_so_far = 0
        for i, t in enumerate(texts):
            key = f"{self._model}:{t[:100]}"
            if key in self._cache:
                results[i] = self._cache[key]
                cached_so_far += 1

        try:
            resp = self._requests.post(
                f"{self._api_base}/embeddings",
                headers=self._headers,
                json={"input": [texts[i] for i in indices], "model": self._model},
                timeout=60
            )
            if resp.status_code == 200:
                new_embs = resp.json()['data']
                pos = 0
                for i, t in enumerate(texts):
                    key = f"{self._model}:{t[:100]}"
                    if key not in self._cache:
                        self._cache[key] = new_embs[pos]['embedding']
                        results[i] = new_embs[pos]['embedding']
                        pos += 1
                return results
        except Exception:
            pass

        import random
        for i in range(len(texts)):
            if results[i] is None:
                results[i] = [random.random() for _ in range(1536)]
        return results

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# ======================== 评估器 ========================

class MultiTurnEvaluator:
    """多轮对话 Ragas 评估器"""

    METRIC_REGISTRY = {
        # 多轮指标
        'agent_goal_accuracy_with_reference': AgentGoalAccuracyWithReference,
        'agent_goal_accuracy_without_reference': AgentGoalAccuracyWithoutReference,
        'instance_rubrics': InstanceRubrics,
        'rubrics_score': RubricsScore,
        'simple_criteria': SimpleCriteriaScore,
        'aspect_critic': AspectCritic,
        'tool_call_accuracy': ToolCallAccuracy,
        'topic_adherence': TopicAdherenceScore,
        # 单轮指标（适用于末尾轮次）
        'faithfulness': Faithfulness,
        'answer_relevancy': AnswerRelevancy,
        'context_precision': ContextPrecision,
        'context_recall': ChineseContextRecall,
        'answer_correctness': AnswerCorrectness,
        'response_groundedness': ResponseGroundedness,
        'context_relevance': ChineseContextRelevance,
    }

    METRIC_DISPLAY_NAMES = {
        'agent_goal_accuracy_with_reference': 'AgentGoalAccuracy（任务目标达成率-有参考）',
        'agent_goal_accuracy_without_reference': 'AgentGoalAccuracy（任务目标达成率-无参考）',
        'instance_rubrics': 'InstanceRubrics（实例评分准则）',
        'rubrics_score': 'RubricsScore（评分准则得分）',
        'simple_criteria': 'SimpleCriteriaScore（简单准则评分）',
        'aspect_critic': 'AspectCritic（多维评判）',
        'tool_call_accuracy': 'ToolCallAccuracy（工具调用准确性）',
        'topic_adherence': 'TopicAdherenceScore（话题一致性）',
        'faithfulness': 'Faithfulness（回答忠实度）',
        'answer_relevancy': 'AnswerRelevancy（回答相关性）',
        'context_precision': 'ContextPrecision（上下文精确度）',
        'context_recall': 'ContextRecall（上下文召回率）',
        'answer_correctness': 'AnswerCorrectness（回答正确性）',
        'response_groundedness': 'ResponseGroundedness（回答有据性）',
        'context_relevance': 'ContextRelevance（上下文相关性）',
    }

    def __init__(self, config: MultiTurnEvalConfig):
        self.config = config
        self.llm = None
        self.ragas_llm = None
        self.embeddings = None
        self.ragas_embeddings = None

    def setup_environment(self):
        """初始化 LLM 和 Embedding 环境"""
        print("[评估器] 初始化环境...")

        # ---- LLM ----
        if self.config.use_dify:
            self.llm = create_dify_llm(
                api_key=self.config.dify_api_key,
                api_url=self.config.dify_url,
                app_id=self.config.dify_app_id,
                streaming=self.config.dify_streaming,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            print(f"[评估器] 使用 Dify LLM: {self.config.dify_url}")
        else:
            self.llm = ChatOpenAI(
                model='qwen-plus',
                openai_api_key=self.config.api_key or os.getenv('QWEN_API_KEY', ''),
                openai_api_base=self.config.api_base or os.getenv('QWEN_API_BASE', ''),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
            )
            print("[评估器] 使用 Qwen 云端 LLM")

        # ---- Embedding ----
        if self.config.use_ollama:
            self.embeddings = OllamaEmbeddingsWrapper(
                self.config.ollama_url,
                self.config.ollama_embedding_model,
            )
        else:
            self.embeddings = QwenEmbeddingsWrapper(
                self.config.api_key or os.getenv('QWEN_API_KEY', ''),
                self.config.api_base or os.getenv('QWEN_API_BASE', ''),
                self.config.embedding_model,
            )

        self.ragas_llm = LangchainLLMWrapper(self.llm)
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        print("[评估器] 环境初始化完成")

    def create_metrics(self) -> Tuple[List, List[str]]:
        """
        根据配置创建评估指标列表

        Returns:
            (metric_objects, metric_names)
        """
        enabled = self.config.enabled_multiturn_metrics
        metric_objs = []
        metric_names = []

        for key in enabled:
            if key not in self.METRIC_REGISTRY:
                print(f"[警告] 未知指标: {key}，跳过")
                continue

            cls = self.METRIC_REGISTRY[key]

            # 需要 embeddings 的指标
            embedding_metrics = {
                'answer_relevancy', 'answer_correctness',
                'response_groundedness',
            }

            try:
                if key in embedding_metrics:
                    obj = cls(embeddings=self.ragas_embeddings)
                else:
                    obj = cls()
                metric_objs.append(obj)
                metric_names.append(key)
                print(f"[指标] 启用: {self.METRIC_DISPLAY_NAMES.get(key, key)}")
            except Exception as e:
                print(f"[警告] 指标 {key} 初始化失败: {e}")

        if not metric_objs:
            print("[警告] 未启用任何指标，使用默认值")
            metric_objs = [AgentGoalAccuracyWithReference()]
            metric_names = ['agent_goal_accuracy_with_reference']

        return metric_objs, metric_names

    async def evaluate(
        self,
        samples: List[MultiTurnSample],
        meta_info: List[Dict],
    ) -> Dict[str, Any]:
        """
        执行多轮对话评估

        Args:
            samples: MultiTurnSample 列表
            meta_info: 每条样本的元信息

        Returns:
            评估结果字典（含详细分数和统计信息）
        """
        print(f"\n[评估] 开始评估 {len(samples)} 个多轮对话样本...")

        if not samples:
            return {'error': '无有效样本'}

        # 构建数据集
        dataset = EvaluationDataset(samples=samples)
        print(f"[数据集] EvaluationDataset 构建完成，共 {len(dataset)} 个样本")

        # 创建指标
        metrics, metric_names = self.create_metrics()
        print(f"[指标] 共启用 {len(metrics)} 个指标")

        # 运行评估
        try:
            from ragas.run_config import RunConfig
            run_cfg = RunConfig(max_workers=self.config.max_workers, timeout=300)

            print("[评估] 调用 Ragas evaluate...")
            results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                run_config=run_cfg,
            )
            print("[OK] Ragas evaluate 完成")

        except Exception as e:
            print(f"[错误] 评估过程出错: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'samples_count': len(samples)}

        # 提取结果
        analysis = self._extract_results(results, metric_names, meta_info)
        return analysis

    def _extract_results(
        self,
        results: Any,
        metric_names: List[str],
        meta_info: List[Dict],
    ) -> Dict[str, Any]:
        """从 Ragas 结果对象中提取分数"""
        # 获取结果字典
        if hasattr(results, '_repr_dict'):
            res_dict = results._repr_dict
        elif hasattr(results, '__dict__') and '_repr_dict' in results.__dict__:
            res_dict = results.__dict__['_repr_dict']
        elif isinstance(results, dict):
            res_dict = results
        else:
            res_dict = {}

        # 提取各指标的平均分数
        scores = {}
        for name in metric_names:
            val = res_dict.get(name)
            if val is not None:
                try:
                    scores[name] = float(val)
                except (ValueError, TypeError):
                    scores[name] = None

        # 如果有样本级结果，也提取
        per_sample = {}
        if hasattr(results, 'run_tests'):
            try:
                df = results.to_pandas()
                for name in metric_names:
                    if name in df.columns:
                        per_sample[name] = df[name].tolist()
            except Exception:
                pass

        return {
            'scores': scores,
            'per_sample': per_sample,
            'meta_info': meta_info,
            'samples_count': len(meta_info),
            'metrics_count': len(metric_names),
            'raw_results': res_dict,
        }
