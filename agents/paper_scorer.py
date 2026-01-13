"""
论文评分 Agent

评估用户需求与论文摘要的匹配度。
基于关键词匹配和语义相似度进行双维度评分。

触发条件:
- 搜索到论文后，需要对结果进行排序
- 用户提供了需求描述和论文摘要

示例:
    用户需求: "分层联邦学习"
    论文摘要: "本文提出一种新型分层联邦学习框架..."
    评分: 总分 18/20, 关键词 9/10, 语义 9/10
"""

import re
from typing import Optional, List
from .base import BaseAgent, AgentConfig
from .state import PaperScoringState


# 默认配置
DEFAULT_CONFIG = AgentConfig(
    name="paper-scorer",
    description="""
    评估论文与用户需求的匹配度。

    <example>
    Context: 搜索到多篇论文，需要排序
    user: "帮我对这些论文按相关性排序"
    assistant: "我将使用 paper-scorer Agent 评估每篇论文的匹配度"
    </example>
    """,
    model_name="Qwen/Qwen3-32B",
    temperature=0.3,
)


class PaperScorerAgent(BaseAgent[PaperScoringState]):
    """
    论文评分 Agent

    实现双维度评分：
    1. 关键词匹配度 (1-10分): 统计关键词出现频率
    2. 语义相似度 (1-10分): 评估主题一致性
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        super().__init__(config or DEFAULT_CONFIG)

    def get_system_prompt(self) -> str:
        """获取评分系统提示词"""
        return """你是一个专业的科研论文匹配专家，请严格按以下规则评估用户需求与论文摘要的匹配度：

## 评分规则

### 关键词匹配度 (1-10分)
- 统计用户提供关键词在摘要中的出现频率
- 完全匹配得2分/词，部分匹配得1分/词
- 总分按比例换算到10分制

### 语义相似度 (1-10分)
- 评估用户需求描述与摘要内容的主题一致性
- 考虑研究目标、方法、结论的匹配程度

## 输出格式（必须严格遵循）

```
总评分: [1-20的整数]
关键词得分: [1-10的整数]
语义得分: [1-10的整数]
理由: [简短说明]
```

示例输出：
```
总评分: 15
关键词得分: 8
语义得分: 7
理由: 摘要涉及联邦学习核心概念，但未提及分层架构
```
"""

    def _build_scoring_prompt(
        self,
        user_requirement: str,
        paper_abstract: str,
        keywords: List[str]
    ) -> str:
        """构建评分提示词"""
        keywords_str = ", ".join(keywords) if keywords else "无"

        return f"""请评估以下内容的匹配度：

## 用户需求
{user_requirement}

## 关键词
{keywords_str}

## 论文摘要
{paper_abstract}

请按照指定格式输出评分结果。"""

    def _parse_scores(self, response: str) -> tuple[int, int, int, str]:
        """
        解析评分响应

        Args:
            response: LLM 响应文本

        Returns:
            (总分, 关键词得分, 语义得分, 理由)
        """
        total_score = 10  # 默认值
        keyword_score = 5
        semantic_score = 5
        reasoning = ""

        # 提取总评分
        total_match = re.search(r'总评分[:：]\s*(\d+)', response)
        if total_match:
            total_score = min(20, max(1, int(total_match.group(1))))

        # 提取关键词得分
        keyword_match = re.search(r'关键词得分[:：]\s*(\d+)', response)
        if keyword_match:
            keyword_score = min(10, max(1, int(keyword_match.group(1))))

        # 提取语义得分
        semantic_match = re.search(r'语义得分[:：]\s*(\d+)', response)
        if semantic_match:
            semantic_score = min(10, max(1, int(semantic_match.group(1))))

        # 提取理由
        reason_match = re.search(r'理由[:：]\s*(.+?)(?:\n|$)', response)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        # 如果总分与分项不一致，以分项为准
        calculated_total = keyword_score + semantic_score
        if abs(total_score - calculated_total) > 2:
            total_score = calculated_total

        return total_score, keyword_score, semantic_score, reasoning

    def process(self, state: PaperScoringState) -> PaperScoringState:
        """
        执行论文评分

        Args:
            state: 评分状态

        Returns:
            更新后的状态，包含评分结果
        """
        # 获取输入
        user_requirement = state.get("user_requirement", "")
        paper_abstract = state.get("paper_abstract", "")
        keywords = state.get("keywords", [])

        if not paper_abstract:
            return {
                **state,
                "total_score": 0,
                "keyword_score": 0,
                "semantic_score": 0,
                "reasoning": "错误：没有提供论文摘要"
            }

        try:
            # 构建提示词并调用 LLM
            prompt = self._build_scoring_prompt(user_requirement, paper_abstract, keywords)
            response = self.call_llm(prompt)

            # 解析响应
            total, keyword, semantic, reason = self._parse_scores(response)

            return {
                **state,
                "total_score": total,
                "keyword_score": keyword,
                "semantic_score": semantic,
                "reasoning": reason
            }

        except Exception as e:
            return {
                **state,
                "total_score": 0,
                "keyword_score": 0,
                "semantic_score": 0,
                "reasoning": f"评分失败: {str(e)}"
            }

    def score_papers(
        self,
        user_requirement: str,
        papers: List[dict],
        keywords: List[str]
    ) -> List[dict]:
        """
        批量评分论文并排序

        Args:
            user_requirement: 用户需求描述
            papers: 论文列表，每个论文需包含 'abstract' 字段
            keywords: 关键词列表

        Returns:
            按评分排序后的论文列表，每个论文增加 'score' 字段
        """
        scored_papers = []

        for paper in papers:
            abstract = paper.get("abstract", paper.get("summary", ""))

            state = PaperScoringState(
                user_requirement=user_requirement,
                paper_abstract=abstract,
                keywords=keywords,
                total_score=0,
                keyword_score=0,
                semantic_score=0,
                reasoning=None
            )

            result = self.process(state)

            paper_with_score = {
                **paper,
                "score": result["total_score"],
                "keyword_score": result["keyword_score"],
                "semantic_score": result["semantic_score"],
                "score_reasoning": result.get("reasoning", "")
            }
            scored_papers.append(paper_with_score)

        # 按总分降序排序
        scored_papers.sort(key=lambda x: x["score"], reverse=True)

        return scored_papers


# 工厂函数
def create_paper_scorer(model_name: str = "Qwen/Qwen3-32B") -> PaperScorerAgent:
    """
    创建论文评分 Agent

    Args:
        model_name: 使用的模型名称

    Returns:
        配置好的 PaperScorerAgent 实例
    """
    config = AgentConfig(
        name="paper-scorer",
        description=DEFAULT_CONFIG.description,
        model_name=model_name,
        temperature=0.3
    )
    return PaperScorerAgent(config)
