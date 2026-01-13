"""
MCP 会话管理模块

负责 MCP 客户端连接、Agent 初始化和会话生命周期管理。
"""

import asyncio
import platform
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

import nest_asyncio
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

from .utils_my import random_uuid


@dataclass
class SessionState:
    """会话状态"""
    initialized: bool = False
    agent: Any = None
    mcp_client: Any = None
    tool_count: int = 0
    thread_id: str = field(default_factory=random_uuid)
    event_loop: Optional[asyncio.AbstractEventLoop] = None


class SessionManager:
    """
    MCP 会话管理器

    负责：
    - 事件循环管理
    - MCP 客户端连接
    - ReAct Agent 初始化
    - 会话清理
    """

    def __init__(self):
        self._state = SessionState()
        self._setup_event_loop()

    def _setup_event_loop(self):
        """设置事件循环"""
        # Windows 特殊处理
        if platform.system() == "Windows":
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

        # 允许嵌套事件循环
        nest_asyncio.apply()

        # 创建或复用事件循环
        if self._state.event_loop is None:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            self._state.event_loop = loop

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop:
        """获取事件循环"""
        return self._state.event_loop

    @property
    def is_initialized(self) -> bool:
        """检查会话是否已初始化"""
        return self._state.initialized

    @property
    def agent(self) -> Any:
        """获取 Agent"""
        return self._state.agent

    @property
    def tool_count(self) -> int:
        """获取工具数量"""
        return self._state.tool_count

    @property
    def thread_id(self) -> str:
        """获取线程 ID"""
        return self._state.thread_id

    async def cleanup(self):
        """
        清理 MCP 客户端资源

        安全地关闭现有连接，释放资源。
        """
        if self._state.mcp_client is not None:
            try:
                await self._state.mcp_client.__aexit__(None, None, None)
            except Exception as e:
                print(f"清理 MCP 客户端时出错: {e}")
            finally:
                self._state.mcp_client = None

    async def initialize(
        self,
        mcp_config: Dict[str, Any],
        model_name: str,
        api_key: str,
        base_url: str,
        system_prompt: str,
        temperature: float = 0.1
    ) -> bool:
        """
        初始化 MCP 会话和 Agent

        Args:
            mcp_config: MCP 工具配置
            model_name: 模型名称
            api_key: API 密钥
            base_url: API 基础 URL
            system_prompt: 系统提示词
            temperature: 模型温度

        Returns:
            初始化是否成功
        """
        try:
            # 先清理现有连接
            await self.cleanup()

            # 创建 MCP 客户端
            client = MultiServerMCPClient(mcp_config)
            tools = await client.get_tools()
            self._state.mcp_client = client
            self._state.tool_count = len(tools)

            # 创建 LLM
            model = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                base_url=base_url,
                api_key=api_key
            )

            # 创建 ReAct Agent
            agent = create_react_agent(
                model,
                tools,
                checkpointer=MemorySaver(),
                prompt=system_prompt,
            )
            self._state.agent = agent
            self._state.initialized = True

            return True

        except Exception as e:
            print(f"初始化会话时出错: {e}")
            self._state.initialized = False
            return False

    def reset_thread(self):
        """重置对话线程"""
        self._state.thread_id = random_uuid()

    def run_async(self, coro):
        """
        在事件循环中运行协程

        Args:
            coro: 要运行的协程

        Returns:
            协程执行结果
        """
        return self._state.event_loop.run_until_complete(coro)


# Streamlit 会话状态适配器
class StreamlitSessionAdapter:
    """
    Streamlit 会话状态适配器

    桥接 SessionManager 和 Streamlit 的 session_state。
    """

    def __init__(self, st_session_state):
        """
        Args:
            st_session_state: Streamlit 的 session_state 对象
        """
        self._st = st_session_state

    def get_or_create_manager(self) -> SessionManager:
        """获取或创建 SessionManager"""
        if "session_manager" not in self._st:
            self._st.session_manager = SessionManager()
        return self._st.session_manager

    def get_history(self) -> list:
        """获取对话历史"""
        if "history" not in self._st:
            self._st.history = []
        return self._st.history

    def add_to_history(self, role: str, content: str):
        """添加消息到历史"""
        history = self.get_history()
        history.append({"role": role, "content": content})

    def clear_history(self):
        """清空对话历史"""
        self._st.history = []

    def get_timeout(self) -> int:
        """获取超时设置"""
        return self._st.get("timeout_seconds", 120)

    def set_timeout(self, seconds: int):
        """设置超时"""
        self._st.timeout_seconds = seconds

    def get_recursion_limit(self) -> int:
        """获取递归限制"""
        return self._st.get("recursion_limit", 100)

    def set_recursion_limit(self, limit: int):
        """设置递归限制"""
        self._st.recursion_limit = limit
