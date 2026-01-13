"""
格式验证 Agent

验证提取的关键词是否符合指定格式要求。
根据文献来源检查英文关键词是否为缩写（ArXiv/IEEE）或全称（SciHub）。

触发条件:
- 关键词提取完成后，需要验证格式
- 作为工作流的第二步自动触发

示例:
    输入: 英文关键词 "llm, hfl, md" + 来源 "ArXiv"
    验证: 通过（使用了缩写）

    输入: 英文关键词 "Large Language Models" + 来源 "ArXiv"
    验证: 失败（ArXiv 应使用缩写）
"""

from typing import Optional, List
from .base import BaseAgent, AgentConfig
from .state import MultiAgentWorkflowState


# 默认配置
DEFAULT_CONFIG = AgentConfig(
    name="keyword-validator",
    description="""
    验证提取的关键词格式是否符合要求。

    <example>
    Context: 关键词提取完成
    input: "英文关键词: llm, hfl" + 来源: ArXiv
    output: "格式正确"
    <commentary>ArXiv 使用缩写，验证通过</commentary>
    </example>
    """,
    model_name="Qwen/Qwen3-32B",  # 验证不需要强大模型
    temperature=0.1,
)


class ValidatorAgent(BaseAgent[MultiAgentWorkflowState]):
    """
    格式验证 Agent

    检查关键词格式是否符合文献来源的要求：
    - ArXiv/IEEE: 英文关键词应为缩写（每个词 <= 5 字符）
    - SciHub: 英文关键词应为全称（完整词组）
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or DEFAULT_CONFIG)

    def get_system_prompt(self) -> str:
        """验证 Agent 不使用 LLM，返回空字符串"""
        return ""

    def _is_abbreviation(self, keyword: str) -> bool:
        """
        判断关键词是否为缩写

        规则：
        - 长度 <= 5 字符
        - 或者全大写
        """
        keyword = keyword.strip()
        return len(keyword) <= 5 or keyword.isupper()

    def _is_full_phrase(self, keyword: str) -> bool:
        """
        判断关键词是否为完整词组

        规则：
        - 包含多个单词（有空格）
        - 或者首字母大写的完整单词
        """
        keyword = keyword.strip()
        words = keyword.split()
        return len(words) > 1 or (len(keyword) > 5 and not keyword.isupper())

    def _validate_keywords(
        self,
        english_keywords: List[str],
        paper_source: str
    ) -> tuple[bool, str]:
        """
        验证关键词格式

        Args:
            english_keywords: 英文关键词列表
            paper_source: 文献来源

        Returns:
            (是否有效, 验证消息)
        """
        if not english_keywords:
            return False, "错误：没有提取到英文关键词"

        if paper_source in ["ArXiv", "IEEE"]:
            # 检查是否都是缩写
            non_abbrev = [kw for kw in english_keywords if not self._is_abbreviation(kw)]
            if non_abbrev:
                return False, f"格式错误：ArXiv/IEEE 应使用缩写形式。以下关键词不是缩写: {', '.join(non_abbrev)}"

        elif paper_source == "SciHub":
            # 检查是否都是完整词组
            non_full = [kw for kw in english_keywords if not self._is_full_phrase(kw)]
            if non_full:
                return False, f"格式错误：SciHub 应使用完整英文词组。以下关键词可能是缩写: {', '.join(non_full)}"

        return True, "格式正确"

    def process(self, state: MultiAgentWorkflowState) -> MultiAgentWorkflowState:
        """
        执行格式验证

        Args:
            state: 工作流状态

        Returns:
            更新后的状态，包含验证结果
        """
        # 获取输入
        english_keywords = state.get("english_keywords", [])
        paper_source = state.get("paper_source", "ArXiv")
        error = state.get("error", False)

        # 如果前一步已出错，跳过验证
        if error:
            return {
                **state,
                "validation_result": "跳过验证：前一步已出错",
                "need_correction": False,
                "current_step": "complete"
            }

        # 从 extracted_keywords 解析（兼容旧格式）
        if not english_keywords and state.get("extracted_keywords"):
            extracted = state["extracted_keywords"]
            if "英文关键词:" in extracted:
                en_part = extracted.split("英文关键词:")[-1]
                en_part = en_part.split("\n")[0].strip().rstrip(";；")
                english_keywords = [k.strip() for k in en_part.split(",") if k.strip()]

        # 执行验证
        is_valid, message = self._validate_keywords(english_keywords, paper_source)

        return {
            **state,
            "english_keywords": english_keywords,
            "validation_result": message,
            "need_correction": not is_valid,
            "current_step": "correct" if not is_valid else "complete"
        }


class CorrectorAgent(BaseAgent[MultiAgentWorkflowState]):
    """
    关键词修正 Agent

    当验证失败时，重新生成符合格式要求的关键词。
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        config = config or AgentConfig(
            name="keyword-corrector",
            description="修正不符合格式要求的关键词",
            model_name="deepseek-ai/DeepSeek-V3",  # 使用更强模型进行修正
            temperature=0.3,
        )
        super().__init__(config)

    def get_system_prompt(self) -> str:
        return """你是一个专业的学术关键词修正专家。
请根据错误信息，将关键词修正为正确格式。

规则：
- ArXiv/IEEE: 使用英文缩写（如 LLM, FL, MD）
- SciHub: 使用完整英文词组（如 Large Language Models）

直接输出修正后的关键词，格式：
中文关键词: xxx, xxx
英文关键词: xxx, xxx
"""

    def process(self, state: MultiAgentWorkflowState) -> MultiAgentWorkflowState:
        """
        执行关键词修正

        Args:
            state: 工作流状态

        Returns:
            更新后的状态，包含修正后的关键词
        """
        # 如果不需要修正，直接返回
        if not state.get("need_correction", False):
            return state

        # 获取输入
        messages = state.get("messages", [])
        original_input = messages[-1].content if messages else ""
        paper_source = state.get("paper_source", "ArXiv")
        error_message = state.get("validation_result", "")
        original_output = state.get("extracted_keywords", "")

        # 构建修正提示词
        prompt = f"""
原始输入: {original_input}
文献来源: {paper_source}
错误信息: {error_message}
原输出: {original_output}

请严格按照 {paper_source} 的格式要求，重新生成关键词。
"""

        try:
            # 调用 LLM 修正
            corrected = self.call_llm(prompt)

            # 解析修正结果
            from .keyword_extractor import KeywordExtractorAgent
            extractor = KeywordExtractorAgent()
            chinese_kw, english_kw = extractor._parse_keywords(corrected)

            # 增加重试计数
            retry_count = state.get("retry_count", 0) + 1

            return {
                **state,
                "chinese_keywords": chinese_kw,
                "english_keywords": english_kw,
                "extracted_keywords": corrected,
                "validation_result": "已修正",
                "need_correction": False,
                "retry_count": retry_count,
                "current_step": "validate"  # 修正后重新验证
            }

        except Exception as e:
            return {
                **state,
                "error": True,
                "error_message": f"关键词修正失败: {str(e)}",
                "current_step": "complete"
            }


# 工厂函数
def create_validator() -> ValidatorAgent:
    """创建验证 Agent"""
    return ValidatorAgent()


def create_corrector(model_name: str = "deepseek-ai/DeepSeek-V3") -> CorrectorAgent:
    """创建修正 Agent"""
    config = AgentConfig(
        name="keyword-corrector",
        description="修正不符合格式要求的关键词",
        model_name=model_name,
        temperature=0.3,
    )
    return CorrectorAgent(config)
