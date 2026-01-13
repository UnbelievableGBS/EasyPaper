"""
关键词提取 Agent

从用户的研究需求描述中提取中英文学术关键词。
根据不同的文献来源（ArXiv/IEEE/SciHub）返回不同格式的关键词。

触发条件:
- 用户输入研究需求描述，需要提取搜索关键词
- 用户选择了文献来源（ArXiv/IEEE/SciHub 等）

示例:
    用户: "我需要查找与大模型，联邦学习相关的分层联邦学习文章"
    Agent: 提取关键词 → 中文: 大模型, 分层联邦学习; 英文: llm, hfl
"""

import re
from typing import Optional, List
from .base import BaseAgent, AgentConfig
from .state import MultiAgentWorkflowState


# 默认配置
DEFAULT_CONFIG = AgentConfig(
    name="keyword-extractor",
    description="""
    当用户需要从研究查询中提取学术关键词时使用此 Agent。

    <example>
    Context: 用户想搜索论文
    user: "我需要查找与大模型相关的论文"
    assistant: "我将使用 keyword-extractor Agent 来识别搜索词"
    <commentary>用户需要搜索论文，需要先提取关键词</commentary>
    </example>

    <example>
    Context: 用户指定了关键词
    user: "帮我找 FL 相关论文"
    assistant: "检测到用户提供了关键词 FL，直接使用"
    <commentary>用户已提供关键词，无需提取</commentary>
    </example>
    """,
    model_name="Qwen/Qwen3-32B",
    temperature=0.3,  # 较低温度保证输出稳定性
)


class KeywordExtractorAgent(BaseAgent[MultiAgentWorkflowState]):
    """
    关键词提取 Agent

    实现从用户描述中提取中英文学术关键词的逻辑。
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or DEFAULT_CONFIG)

    def get_system_prompt(self) -> str:
        """获取关键词提取的系统提示词"""
        return """你是一个优秀的学术领域专家，能够从用户描述中提取科研关键词。

## 输出规则

根据文献来源不同，输出格式不同：

### ArXiv/IEEE 来源
返回中文关键词和**英文缩写**关键词：
```
中文关键词: 大模型, 分层联邦学习, 模型蒸馏
英文关键词: llm, hfl, md
```

### SciHub 来源
返回中文关键词和**英文全称**关键词：
```
中文关键词: 大模型, 分层联邦学习, 模型蒸馏
英文关键词: Large Language Models, Hierarchical Federated Learning, Model Distillation
```

## 特殊情况

- 如果用户已提供关键词，直接采用，不做额外分析
- 严格按照指定格式输出，不要添加其他内容
- 每个关键词用逗号分隔
"""

    def _build_extraction_prompt(
        self,
        user_input: str,
        paper_source: str,
        user_keywords: Optional[List[str]] = None
    ) -> str:
        """构建提取提示词"""
        if user_keywords:
            # 用户已提供关键词
            en_keywords = ", ".join(user_keywords)
            return f"用户已提供关键词: {en_keywords}。请直接采用这些关键词，按照 {paper_source} 格式输出。"

        return f"""请从以下用户描述中提取学术关键词。

用户描述: "{user_input}"
文献来源: {paper_source}

请按照指定格式输出中文和英文关键词。"""

    def _parse_keywords(self, response: str) -> tuple[List[str], List[str]]:
        """
        解析 LLM 响应，提取关键词列表

        Args:
            response: LLM 响应文本

        Returns:
            (中文关键词列表, 英文关键词列表)
        """
        chinese_keywords = []
        english_keywords = []

        # 提取中文关键词
        cn_match = re.search(r'中文关键词[:：]\s*(.+?)(?:\n|$)', response)
        if cn_match:
            cn_text = cn_match.group(1).strip().rstrip(';；')
            chinese_keywords = [k.strip() for k in re.split(r'[,，;；]', cn_text) if k.strip()]

        # 提取英文关键词
        en_match = re.search(r'英文关键词[:：]\s*(.+?)(?:\n|$)', response)
        if en_match:
            en_text = en_match.group(1).strip().rstrip(';；')
            english_keywords = [k.strip() for k in re.split(r'[,，;；]', en_text) if k.strip()]

        return chinese_keywords, english_keywords

    def process(self, state: MultiAgentWorkflowState) -> MultiAgentWorkflowState:
        """
        执行关键词提取

        Args:
            state: 工作流状态

        Returns:
            更新后的状态，包含提取的关键词
        """
        # 获取输入
        messages = state.get("messages", [])
        if not messages:
            return {
                **state,
                "error": True,
                "error_message": "错误：没有提供用户消息",
                "current_step": "complete"
            }

        user_input = messages[-1].content
        paper_source = state.get("paper_source", "ArXiv")
        user_keywords = state.get("user_keywords")

        # 如果用户已提供关键词，直接使用
        if user_keywords:
            return {
                **state,
                "english_keywords": user_keywords,
                "extracted_keywords": f"英文关键词: {', '.join(user_keywords)}",
                "error": False,
                "current_step": "validate"
            }

        # 构建提示词并调用 LLM
        try:
            prompt = self._build_extraction_prompt(user_input, paper_source, user_keywords)
            response = self.call_llm(prompt)

            # 解析响应
            chinese_kw, english_kw = self._parse_keywords(response)

            return {
                **state,
                "chinese_keywords": chinese_kw,
                "english_keywords": english_kw,
                "extracted_keywords": response,  # 保留原始响应（兼容旧代码）
                "error": False,
                "error_message": None,
                "current_step": "validate"
            }

        except Exception as e:
            return {
                **state,
                "error": True,
                "error_message": f"关键词提取失败: {str(e)}",
                "current_step": "complete"
            }


# 工厂函数
def create_keyword_extractor(
    model_name: str = "Qwen/Qwen3-32B",
    api_key_env: str = "SILICONFLOW_API_KEY"
) -> KeywordExtractorAgent:
    """
    创建关键词提取 Agent

    Args:
        model_name: 使用的模型名称
        api_key_env: API Key 环境变量名

    Returns:
        配置好的 KeywordExtractorAgent 实例
    """
    config = AgentConfig(
        name="keyword-extractor",
        description=DEFAULT_CONFIG.description,
        model_name=model_name,
        api_key_env=api_key_env,
        temperature=0.3
    )
    return KeywordExtractorAgent(config)
