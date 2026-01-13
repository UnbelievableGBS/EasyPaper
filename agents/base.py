"""
Agent 基类定义

提供所有 Agent 的抽象基类，定义标准接口和通用功能。
遵循 ReAct 模式：Reasoning（推理）+ Acting（行动）。
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, TypeVar, Generic
from dataclasses import dataclass, field
from openai import OpenAI


# 泛型状态类型
StateT = TypeVar("StateT", bound=Dict[str, Any])


@dataclass
class AgentConfig:
    """
    Agent 配置类

    定义 Agent 的元信息和运行参数。
    参考 Claude Code 插件的 Agent 配置格式。
    """

    # 元信息
    name: str                                    # Agent 标识符 (如 keyword-extractor)
    description: str                             # 触发条件描述
    version: str = "1.0.0"

    # 模型配置
    model_name: str = "Qwen/Qwen3-32B"
    temperature: float = 0.7
    max_tokens: int = 2048

    # API 配置
    api_base_url: str = "https://api.siliconflow.cn/v1"
    api_key_env: str = "SILICONFLOW_API_KEY"

    # 工具访问（参考 Claude Code 格式）
    tools: List[str] = field(default_factory=list)

    # 运行配置
    max_retries: int = 3
    timeout_seconds: int = 60

    def get_api_key(self) -> str:
        """从环境变量获取 API Key"""
        return os.getenv(self.api_key_env, "")


class BaseAgent(ABC, Generic[StateT]):
    """
    Agent 抽象基类

    所有具体 Agent 必须继承此类并实现：
    - process(): 核心处理逻辑
    - get_system_prompt(): 获取系统提示词

    遵循 ReAct 模式：
    1. Thought (思考): 分析当前状态和目标
    2. Action (行动): 调用 LLM 或工具
    3. Observation (观察): 处理返回结果
    """

    def __init__(self, config: AgentConfig):
        """
        初始化 Agent

        Args:
            config: Agent 配置
        """
        self.config = config
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """懒加载 OpenAI 客户端"""
        if self._client is None:
            self._client = OpenAI(
                api_key=self.config.get_api_key(),
                base_url=self.config.api_base_url
            )
        return self._client

    @property
    def name(self) -> str:
        """Agent 名称"""
        return self.config.name

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示词

        Returns:
            系统提示词字符串
        """
        pass

    @abstractmethod
    def process(self, state: StateT) -> StateT:
        """
        处理状态并返回更新后的状态

        这是 Agent 的核心方法，实现具体的业务逻辑。

        Args:
            state: 输入状态

        Returns:
            更新后的状态
        """
        pass

    def call_llm(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        调用 LLM 获取响应

        Args:
            user_prompt: 用户提示词
            system_prompt: 系统提示词（默认使用 get_system_prompt）
            model: 模型名称（默认使用配置）
            temperature: 温度参数（默认使用配置）

        Returns:
            LLM 响应文本
        """
        messages = []

        # 添加系统提示词
        sys_prompt = system_prompt or self.get_system_prompt()
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        # 添加用户提示词
        messages.append({"role": "user", "content": user_prompt})

        # 调用 API
        completion = self.client.chat.completions.create(
            model=model or self.config.model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        return completion.choices[0].message.content or ""

    def __call__(self, state: StateT) -> StateT:
        """
        使 Agent 可调用，用于 LangGraph 节点

        Args:
            state: 输入状态

        Returns:
            更新后的状态
        """
        return self.process(state)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"


class ReActAgent(BaseAgent[StateT]):
    """
    ReAct 模式 Agent 基类

    实现显式的 Thought-Action-Observation 循环。
    子类需要实现 think(), act(), observe() 方法。
    """

    @abstractmethod
    def think(self, state: StateT) -> str:
        """
        思考阶段：分析当前状态，决定下一步行动

        Args:
            state: 当前状态

        Returns:
            思考结果/推理过程
        """
        pass

    @abstractmethod
    def act(self, state: StateT, thought: str) -> Any:
        """
        行动阶段：执行具体操作

        Args:
            state: 当前状态
            thought: 思考结果

        Returns:
            行动结果
        """
        pass

    @abstractmethod
    def observe(self, state: StateT, action_result: Any) -> StateT:
        """
        观察阶段：处理行动结果，更新状态

        Args:
            state: 当前状态
            action_result: 行动结果

        Returns:
            更新后的状态
        """
        pass

    def process(self, state: StateT) -> StateT:
        """
        执行完整的 ReAct 循环

        Thought → Action → Observation
        """
        # Step 1: Think
        thought = self.think(state)

        # Step 2: Act
        action_result = self.act(state, thought)

        # Step 3: Observe
        new_state = self.observe(state, action_result)

        return new_state
