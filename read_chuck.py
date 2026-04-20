"""
文件读取、文本处理和分块模块
从rag_evaluator.py中提取的数据处理相关功能

包含功能：
1. DataLoader - 数据加载和解析
2. TextProcessor - 文本处理和分块
"""

import os
import json
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Any, Tuple
from config import debug_print, verbose_print, info_print, error_print, QUIET_MODE
from dataclasses import dataclass

@dataclass
class EvaluationConfig:
    """评估配置类"""
    # API配置
    api_key: str
    api_base: str
    model_name: str = "qwen-plus"
    embedding_model: str = "text-embedding-v1"
    
    # Ollama配置
    use_ollama: bool = False
    ollama_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "embeddinggemma:300m"
    ollama_llm_model: str = "llama3.2:3b"
    
    # Dify配置
    use_dify: bool = False
    dify_url: str = ""
    dify_api_key: str = ""
    dify_app_id: Optional[str] = None
    dify_streaming: bool = False
    
    # 评估配置（LLM 输出稳定性参数）
    temperature: float = 0.0  # 使用 0.0 以获得更稳定的输出，提高 Ragas 解析器成功率
    top_p: float = 0.1  # 降低采样多样性，只从最高概率的 10% token 中选择
    max_tokens: int = 2000000  # 最大生成 token 数
    max_chunk_length: int = 2000000
    
    # 性能优化参数
    max_workers: int = 3  # 最大并发工作线程数，提升评估速度
    batch_size: int = 6  # 批处理大小，减少 API 调用次数
    
    # 文件配置
    excel_file_path: str = None
    required_columns: List[str] = None
    
    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = [
                'user_input', 
                'retrieved_contexts', 
                'response', 
                'reference_contexts', 
                'reference'
            ]
        if self.excel_file_path is None:
            self.excel_file_path = os.getenv("EXCEL_FILE_PATH", "standardDataset/standardDataset.xlsx")
        
        # 如果启用Dify，从环境变量加载Dify配置
        if self.use_dify:
            self.dify_url = self.dify_url or os.getenv("DIFY_URL", "")
            self.dify_api_key = self.dify_api_key or os.getenv("DIFY_API_KEY", "")
            self.dify_app_id = self.dify_app_id or os.getenv("DIFY_APP_ID")
            self.dify_streaming = self.dify_streaming or os.getenv("DIFY_STREAMING", "false").lower() == "true"

class DataLoader:
    """数据加载和解析模块"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def load_excel_data(self) -> Optional[pd.DataFrame]:
        """
        从Excel文件加载数据
        
        Returns:
            pd.DataFrame: 加载的数据，失败时返回None
        """
        info_print(f"📖 读取Excel文件: {self.config.excel_file_path}")
        
        if not os.path.exists(self.config.excel_file_path):
            info_print(f"❌ Excel文件不存在: {self.config.excel_file_path}")
            return None
        
        try:
            # 只读取指定的列
            df = pd.read_excel(
                self.config.excel_file_path, 
                usecols=self.config.required_columns
            )
            info_print(f"✅ 成功读取 {len(df)} 行数据")
            info_print(f"📋 列名: {list(df.columns)}")
            return df
        except Exception as e:
            info_print(f"❌ 读取Excel文件失败: {e}")
            return None
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        
        Args:
            df: 数据DataFrame
            
        Returns:
            bool: 验证是否通过
        """
        missing_fields = [
            field for field in self.config.required_columns 
            if field not in df.columns
        ]
        
        if missing_fields:
            info_print(f"❌ Excel文件缺少必要字段: {missing_fields}")
            info_print(f"当前字段: {list(df.columns)}")
            return False
        
        info_print("✅ 数据验证通过")
        return True

class TextProcessor:
    """文本处理和分块模块"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
    
    def split_text_into_chunks(self, text: str, max_chunk_length: Optional[int] = None) -> List[str]:
        """
        将文本分割成小块段落，按<<<__CONTEXT_BLOCK__>>>分隔符分割
        
        Args:
            text: 要分割的文本
            max_chunk_length: 每个块的最大长度（保留参数以兼容性）
            
        Returns:
            List[str]: 分割后的文本块列表
        """
        if not text or not text.strip():
            return [text] if text else []
        
        # 按<<<__CONTEXT_BLOCK__>>>分隔符分割
        delimiter = "<<<__CONTEXT_BLOCK__>>>"
        chunks = [chunk.strip() for chunk in text.split(delimiter)]
        
        # 过滤掉空块
        chunks = [chunk for chunk in chunks if chunk]
        
        # 如果没有找到分隔符，返回整个文本作为一个块
        if len(chunks) <= 1 and delimiter not in text:
            chunks = [text.strip()] if text.strip() else []
        
        return chunks
    
    def process_contexts(self, contexts_str: Any) -> List[str]:
        """
        处理上下文字段，确保是列表格式
        
        Args:
            contexts_str: 上下文字符串或列表
            
        Returns:
            List[str]: 处理后的上下文列表
        """
        try:
            # 处理pandas Series或numpy数组
            if hasattr(contexts_str, '__iter__') and not isinstance(contexts_str, (str, list)):
                if hasattr(contexts_str, 'iloc'):
                    contexts_str = contexts_str.iloc[0]
                elif hasattr(contexts_str, '__len__') and len(contexts_str) > 0:
                    contexts_str = contexts_str[0]
                else:
                    return []
            
            # 检查是否为numpy数组
            if hasattr(contexts_str, 'dtype'):
                contexts_str = str(contexts_str)
            
            # 安全检查是否为NaN或空
            if contexts_str is None:
                return []
            
            try:
                if pd.isna(contexts_str):
                    return []
            except (TypeError, ValueError):
                pass
            
            if (isinstance(contexts_str, str) and contexts_str.strip() == '') or \
               (isinstance(contexts_str, list) and len(contexts_str) == 0) or \
               (isinstance(contexts_str, list) and all(
                   (isinstance(item, str) and item.strip() == '') or 
                   (hasattr(item, '__iter__') and not isinstance(item, str) and len(str(item).strip()) == 0)
                   for item in contexts_str)):
                return []
            
            if isinstance(contexts_str, str):
                # 如果是字符串，尝试解析为列表
                try:
                    # 如果是JSON格式的字符串
                    parsed = json.loads(contexts_str)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed if str(item).strip()]
                    else:
                        return [str(parsed)]
                except:
                    # 如果不是JSON，使用分块逻辑按空行分割
                    parts = self.split_text_into_chunks(contexts_str)
                    return parts
            elif isinstance(contexts_str, list):
                return [str(item) for item in contexts_str if str(item).strip()]
            else:
                return [str(contexts_str)] if str(contexts_str).strip() else []
        except Exception as e:
            info_print(f"⚠️ 处理上下文时出错: {e}")
            return []
    
    def parse_context_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析上下文列，将字符串转换为列表格式
        
        Args:
            df: 包含上下文数据的DataFrame
            
        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        info_print("🔧 解析上下文列...")
        
        # 处理retrieved_contexts和reference_contexts字段
        df['retrieved_contexts'] = df['retrieved_contexts'].apply(self.process_contexts)
        df['reference_contexts'] = df['reference_contexts'].apply(self.process_contexts)
        
        return df
    
    def is_empty_row_data(self, retrieved_contexts: List[str], reference_contexts: List[str], 
                         user_input: str, response: str) -> bool:
        """
        检查是否为空行数据
        
        Args:
            retrieved_contexts: 检索上下文列表
            reference_contexts: 参考上下文列表
            user_input: 用户输入
            response: 回答
            
        Returns:
            bool: 是否为空行数据
        """
        return (
            not retrieved_contexts or 
            not reference_contexts or 
            not user_input or 
            not response or
            (isinstance(retrieved_contexts, list) and len(retrieved_contexts) == 0) or
            (isinstance(reference_contexts, list) and len(reference_contexts) == 0) or
            (isinstance(user_input, str) and user_input.strip() == '') or
            (isinstance(response, str) and response.strip() == '') or
            pd.isna(user_input) or
            pd.isna(response) or
            (isinstance(retrieved_contexts, list) and all(
                pd.isna(item) or (isinstance(item, str) and item.strip() == '') 
                for item in retrieved_contexts)) or
            (isinstance(reference_contexts, list) and all(
                pd.isna(item) or (isinstance(item, str) and item.strip() == '') 
                for item in reference_contexts))
        )
