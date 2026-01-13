"""
Agent 状态类型定义

使用 TypedDict 提供类型安全的状态管理，替代原有的 dict 类型。
这是 LangGraph 多智能体工作流的核心数据结构。
"""

from typing import TypedDict, Optional, List, Literal
from langchain_core.messages import BaseMessage


class KeywordExtractionState(TypedDict):
    """关键词提取 Agent 的状态"""

    # 输入
    messages: List[BaseMessage]           # 用户消息历史
    paper_source: Literal["ArXiv", "IEEE", "SciHub", "Google Scholar", "ACL"]
    user_keywords: Optional[List[str]]    # 用户预设关键词（可选）

    # 输出
    chinese_keywords: Optional[List[str]] # 提取的中文关键词
    english_keywords: Optional[List[str]] # 提取的英文关键词

    # 状态标记
    error: bool
    error_message: Optional[str]


class ValidationState(TypedDict):
    """格式验证 Agent 的状态"""

    # 输入（继承自 KeywordExtractionState）
    paper_source: Literal["ArXiv", "IEEE", "SciHub", "Google Scholar", "ACL"]
    chinese_keywords: Optional[List[str]]
    english_keywords: Optional[List[str]]

    # 输出
    is_valid: bool
    validation_message: str
    need_correction: bool


class CorrectionState(TypedDict):
    """修正 Agent 的状态"""

    # 输入
    original_input: str
    paper_source: Literal["ArXiv", "IEEE", "SciHub", "Google Scholar", "ACL"]
    validation_message: str
    chinese_keywords: Optional[List[str]]
    english_keywords: Optional[List[str]]

    # 输出
    corrected_chinese: Optional[List[str]]
    corrected_english: Optional[List[str]]


class PaperScoringState(TypedDict):
    """论文评分 Agent 的状态"""

    # 输入
    user_requirement: str                 # 用户需求描述
    paper_abstract: str                   # 论文摘要
    keywords: List[str]                   # 关键词列表

    # 输出
    total_score: int                      # 总评分 (1-20)
    keyword_score: int                    # 关键词匹配分 (1-10)
    semantic_score: int                   # 语义相似分 (1-10)
    reasoning: Optional[str]              # 评分理由


class MultiAgentWorkflowState(TypedDict):
    """
    多智能体工作流的完整状态

    这是 LangGraph StateGraph 的核心状态类型，
    包含整个工作流所需的所有字段。
    """

    # 用户输入
    messages: List[BaseMessage]
    paper_source: Literal["ArXiv", "IEEE", "SciHub", "Google Scholar", "ACL"]
    user_keywords: Optional[List[str]]

    # 关键词提取结果
    extracted_keywords: Optional[str]     # 原始提取结果（兼容旧代码）
    chinese_keywords: Optional[List[str]]
    english_keywords: Optional[List[str]]

    # 验证结果
    validation_result: Optional[str]
    need_correction: bool

    # 状态标记
    error: bool
    error_message: Optional[str]

    # 工作流控制
    current_step: Literal["extract", "validate", "correct", "complete"]
    retry_count: int


# 状态工厂函数
def create_initial_state(
    user_message: str,
    paper_source: str,
    user_keywords: Optional[List[str]] = None
) -> MultiAgentWorkflowState:
    """
    创建初始工作流状态

    Args:
        user_message: 用户输入的查询文本
        paper_source: 文献来源 (ArXiv/IEEE/SciHub 等)
        user_keywords: 用户预设的关键词（可选）

    Returns:
        初始化的工作流状态
    """
    from langchain_core.messages import HumanMessage

    return MultiAgentWorkflowState(
        messages=[HumanMessage(content=user_message)],
        paper_source=paper_source,
        user_keywords=user_keywords,
        extracted_keywords=None,
        chinese_keywords=None,
        english_keywords=None,
        validation_result=None,
        need_correction=False,
        error=False,
        error_message=None,
        current_step="extract",
        retry_count=0
    )
