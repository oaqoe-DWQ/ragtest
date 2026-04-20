"""
Dify API LLM适配器模块
用于将Dify API适配为LangChain兼容的LLM接口
支持Ragas框架的评估调用
"""

import os
import json
import re
import requests
from typing import Any, Dict, List, Optional, Iterator, Union
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field, field_validator

from config import debug_print, verbose_print, info_print, error_print, verbose_info_print


class DifyAPIError(Exception):
    """Dify API错误异常"""
    pass


class DifyResponseParsingError(DifyAPIError):
    """Dify响应解析错误"""
    pass


class DifyLLM(BaseChatModel):
    """
    Dify API LLM适配器
    
    将Dify的chat-messages接口适配为LangChain兼容的ChatModel接口
    用于Ragas评估框架的LLM调用
    """
    
    api_key: str = ""
    api_url: str = ""
    app_id: Optional[str] = None
    streaming: bool = False
    timeout: int = 120
    temperature: float = 0.0
    max_tokens: int = 2000000
    top_p: float = 0.1
    max_retries: int = 3
    retry_delay: float = 10.0
    
    class Config:
        arbitrary_types_allowed = True
        keep_untouched = (property,)
    
    def _get_session(self) -> requests.Session:
        """获取或创建请求会话（带重试机制）"""
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # 用 object.__getattribute__ 绕过 Pydantic 拦截
        try:
            session = object.__getattribute__(self, '_DifyLLM__session')
            if session is not None:
                return session
        except AttributeError:
            pass
        
        # 创建新session
        new_session = requests.Session()
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=0,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        new_session.mount("http://", adapter)
        new_session.mount("https://", adapter)
        
        object.__setattr__(self, '_DifyLLM__session', new_session)
        return new_session
    
    @property
    def _clean_api_key(self) -> str:
        """清理API Key，移除Bearer前缀"""
        key = self.api_key
        if key.startswith('Bearer '):
            return key[7:]
        elif key.startswith('bearer '):
            return key[7:]
        return key
    
    @property
    def _clean_api_url(self) -> str:
        """清理API URL"""
        url = self.api_url
        if url.endswith('/'):
            return url[:-1]
        return url
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._clean_api_key}"
        }
        if self.app_id:
            headers["Dify-App-Id"] = self.app_id
        return headers
    
    def _build_payload(self, messages: List[BaseMessage], **kwargs) -> Dict[str, Any]:
        """构建请求载荷"""
        # 提取对话历史和当前查询
        query = ""
        conversation_history = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                # 最后一个 HumanMessage 作为 query，其他作为历史
                query = msg.content
            elif isinstance(msg, AIMessage) and query:
                # AI回复作为历史消息
                conversation_history.append({
                    "role": "assistant",
                    "content": msg.content
                })
            elif isinstance(msg, SystemMessage):
                # System消息放入inputs
                pass
        
        # 构建载荷 - Dify chat-messages API 格式
        payload = {
            "query": query,
            "inputs": {},
            "response_mode": "blocking" if not self.streaming else "streaming",
            "user": "rag_evaluator"
        }
        
        # 如果有历史消息，按Dify格式添加
        if conversation_history:
            payload["conversation_id"] = kwargs.get("conversation_id")
            # Dify可能需要历史消息在单独的字段中
        
        # 添加可选参数
        if kwargs.get('temperature') is not None:
            payload["temperature"] = kwargs['temperature']
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
            
        if kwargs.get('max_tokens') is not None:
            payload["max_tokens"] = kwargs['max_tokens']
        elif self.max_tokens:
            payload["max_tokens"] = self.max_tokens
            
        if kwargs.get('top_p') is not None:
            payload["top_p"] = kwargs['top_p']
        elif self.top_p:
            payload["top_p"] = self.top_p
            
        return payload
    
    def _parse_response(self, response: requests.Response) -> str:
        """解析Dify API响应"""
        try:
            data = response.json()
            
            # 检查错误
            if response.status_code != 200:
                error_msg = data.get('message', data.get('error', 'Unknown error'))
                raise DifyAPIError(f"Dify API错误 ({response.status_code}): {error_msg}")
            
            # 获取answer字段
            answer = None
            
            # 情况1: answer 在 data 字段中 (blocking模式)
            if 'data' in data and isinstance(data['data'], dict):
                answer = data['data'].get('answer', '')
            
            # 情况2: answer 直接在顶层 (部分Dify配置)
            if not answer and 'answer' in data:
                answer = data['answer']
            
            if answer:
                answer = answer.strip()
                
                # 清理 markdown 代码块包裹的 JSON，改了
                answer = self._unwrap_markdown_json(answer)
                
                # 检查answer是否是转义的JSON字符串
                if answer.startswith('{') or answer.startswith('['):
                    try:
                        # 解析嵌套的JSON，返回格式化后的字符串
                        parsed = json.loads(answer)
                        return json.dumps(parsed, ensure_ascii=False, indent=None, separators=(',', ':'))
                    except json.JSONDecodeError:
                        # 不是有效JSON，清理后返回
                        pass
                
                # 清理转义字符
                answer = answer.replace('\\n', ' ').replace('\\"', '"').replace('\\r', '')
                answer = ' '.join(answer.split())
                return answer
            
            # 尝试其他可能的字段
            for key in ['result', 'message', 'content', 'text', 'output', 'response']:
                if key in data and data[key]:
                    return str(data[key]).strip()
            
            raise DifyResponseParsingError(f"无法从Dify响应中解析出回答内容: {json.dumps(data, ensure_ascii=False)[:300]}")
            
        except json.JSONDecodeError:
            raise DifyResponseParsingError(f"响应不是有效的JSON: {response.text[:200]}")
    #改了
    def _unwrap_markdown_json(self, text: str) -> str:
        """去除 markdown 代码块包裹的 JSON"""
        # 匹配 ```json\n...\n``` 或 ```\n...\n```
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json\n...\n```
            r'```json(.*?)```',             # ```json...\n            
            r'```\s*\n(.*?)\n```',   # ```\n...\n```
            r'```(.*?)```',                # ```...\n        
        ]
        
        for pattern in patterns:
            match = re.search(pattern,text, re.DOTALL)
            if match:
                unwrapped = match.group(1).strip()
                # 验证去除后是有效JSON
                if unwrapped.startswith('{') or unwrapped.startswith('['):
                    return unwrapped
        
        return text
    
    def _is_rate_limit_error(self, response: requests.Response, response_text: str = "") -> bool:
        """检测是否是限流错误"""
        # 429 状态码
        if response.status_code == 429:
            return True
        # 响应内容包含限流相关关键词
        text = response_text.lower()
        rate_limit_keywords = [
            'rate limit', 'too many requests', '请求过于频繁',
            '请求次数超限', '调用频率', 'quota', 'quota exceeded',
            'rate_limit', '请求受限', '限流'
        ]
        return any(kw in text for kw in rate_limit_keywords)
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """生成聊天回复（LangChain接口，带重试机制）"""
        verbose_info_print(f"🤖 Dify LLM调用: {self._clean_api_url}")
        
        payload = self._build_payload(messages, **kwargs)
        headers = self._get_headers()
        session = self._get_session()
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                verbose_info_print(f"📤 发送请求到Dify API (第 {attempt + 1}/{self.max_retries} 次)...")
                
                response = session.post(
                    self._clean_api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                # 检查限流错误
                if self._is_rate_limit_error(response, response.text):
                    # 计算指数退避延迟
                    wait_time = self.retry_delay * (2 ** attempt)
                    error_print(f"⚠️ 检测到限流，等待 {wait_time:.1f} 秒后重试 (第 {attempt + 1}/{self.max_retries})...")
                    if attempt < self.max_retries - 1:
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        raise DifyAPIError(f"限流重试 {self.max_retries} 次后仍失败")
                
                # 检查响应是否是HTML（可能是限流返回的异常页面）
                if response.text.strip().startswith('<!DOCTYPE') or response.text.strip().startswith('<html'):
                    wait_time = self.retry_delay * (2 ** attempt)
                    error_print(f"⚠️ 收到HTML响应（可能是限流），等待 {wait_time:.1f} 秒后重试 (第 {attempt + 1}/{self.max_retries})...")
                    if attempt < self.max_retries - 1:
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        raise DifyAPIError(f"收到异常HTML响应，重试 {self.max_retries} 次后仍失败")
                
                content = self._parse_response(response)
                
                verbose_info_print(f"📥 收到Dify响应，长度: {len(content)} 字符")
                
                ai_message = AIMessage(content=content)
                chat_generation = ChatGeneration(message=ai_message)
                
                return ChatResult(generations=[chat_generation])
                
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = e
                error_print(f"⚠️ Dify API请求失败 (第 {attempt + 1}/{self.max_retries}): {e}")
                
                if attempt < self.max_retries - 1:
                    import time
                    wait_time = self.retry_delay * (2 ** attempt)
                    verbose_info_print(f"⏳ 等待 {wait_time:.1f} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    error_print(f"❌ 已达到最大重试次数 ({self.max_retries})")
                    
            except DifyAPIError as e:
                last_error = e
                error_print(f"❌ Dify API错误: {e}")
                raise
            except Exception as e:
                last_error = e
                error_print(f"❌ Dify LLM调用失败: {e}")
                raise
        
        raise DifyAPIError(f"重试 {self.max_retries} 次后仍失败: {last_error}")
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成聊天回复"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate(messages, stop, run_manager, **kwargs)
        )
    
    @property
    def _llm_type(self) -> str:
        return "dify"
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "api_url": self.api_url,
            "model": "dify",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming
        }


class DifyChatModel(DifyLLM):
    """
    Dify聊天模型（向后兼容别名）
    """
    pass


def create_dify_llm(
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    app_id: Optional[str] = None,
    streaming: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    top_p: float = 0.1
) -> DifyLLM:
    """
    创建Dify LLM实例的工厂函数
    
    Args:
        api_key: Dify API密钥
        api_url: Dify API地址
        app_id: Dify应用ID（可选）
        streaming: 是否使用流式响应
        temperature: 温度参数
        max_tokens: 最大token数
        top_p: top_p参数
        
    Returns:
        DifyLLM实例
    """
    from dotenv import load_dotenv
    load_dotenv()
    
    # 清理API Key
    key = api_key or os.getenv("DIFY_API_KEY", "")
    if key.startswith('Bearer '):
        key = key[7:]
    
    # 清理API URL
    url = api_url or os.getenv("DIFY_URL", "")
    if url.endswith('/'):
        url = url[:-1]
    
    return DifyLLM(
        api_key=key,
        api_url=url,
        app_id=app_id or os.getenv("DIFY_APP_ID"),
        streaming=streaming or os.getenv("DIFY_STREAMING", "false").lower() == "true",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )


# 直接测试
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("🧪 测试Dify LLM...")
    
    dify = create_dify_llm()
    
    # 测试调用
    messages = [HumanMessage(content="你好，请介绍一下你自己")]
    
    try:
        response = dify.invoke(messages)
        print(f"✅ 测试成功!")
        print(f"回复: {response.content}")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
