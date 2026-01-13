"""
EasyPaper Agent 模块

提供模块化、类型安全的 Agent 实现。

使用示例:
    from agents import KeywordExtractorAgent, AgentConfig
    from agents.state import create_initial_state

    # 创建 Agent
    config = AgentConfig(name="keyword-extractor", description="...")
    agent = KeywordExtractorAgent(config)

    # 创建状态并执行
    state = create_initial_state("我需要查找大模型论文", "ArXiv")
    result = agent(state)
"""

from .base import BaseAgent, ReActAgent, AgentConfig
from .state import (
    KeywordExtractionState,
    ValidationState,
    CorrectionState,
    PaperScoringState,
    MultiAgentWorkflowState,
    create_initial_state,
)

# 具体 Agent 类
from .keyword_extractor import KeywordExtractorAgent, create_keyword_extractor
from .validator import ValidatorAgent, CorrectorAgent, create_validator, create_corrector
from .paper_scorer import PaperScorerAgent, create_paper_scorer


__all__ = [
    # 基类
    "BaseAgent",
    "ReActAgent",
    "AgentConfig",
    # 状态类型
    "KeywordExtractionState",
    "ValidationState",
    "CorrectionState",
    "PaperScoringState",
    "MultiAgentWorkflowState",
    "create_initial_state",
    # Agent 类
    "KeywordExtractorAgent",
    "ValidatorAgent",
    "CorrectorAgent",
    "PaperScorerAgent",
    # 工厂函数
    "create_keyword_extractor",
    "create_validator",
    "create_corrector",
    "create_paper_scorer",
]
