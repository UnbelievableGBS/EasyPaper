"""
流式响应处理模块

负责处理 LLM 流式响应的回调和展示。
支持文本响应和工具调用信息的分离展示。
"""

from typing import Any, Callable, List, Optional, Tuple
from dataclasses import dataclass, field

from langchain_core.messages.ai import AIMessageChunk
from langchain_core.messages.tool import ToolMessage


@dataclass
class StreamingAccumulator:
    """流式响应累加器"""
    text: List[str] = field(default_factory=list)
    tool: List[str] = field(default_factory=list)

    def append_text(self, content: str):
        """添加文本内容"""
        self.text.append(content)

    def append_tool(self, content: str):
        """添加工具调用信息"""
        self.tool.append(content)

    def get_text(self) -> str:
        """获取累积的文本"""
        return "".join(self.text)

    def get_tool_info(self) -> str:
        """获取累积的工具信息"""
        return "".join(self.tool)

    def clear(self):
        """清空累积内容"""
        self.text.clear()
        self.tool.clear()


class StreamingHandler:
    """
    流式响应处理器

    处理来自 LangChain/LangGraph 的流式消息，
    分离文本响应和工具调用信息。
    """

    def __init__(
        self,
        text_callback: Optional[Callable[[str], None]] = None,
        tool_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Args:
            text_callback: 文本更新回调函数
            tool_callback: 工具信息更新回调函数
        """
        self._text_callback = text_callback
        self._tool_callback = tool_callback
        self._accumulator = StreamingAccumulator()

    @property
    def accumulated_text(self) -> str:
        """获取累积的文本"""
        return self._accumulator.get_text()

    @property
    def accumulated_tool_info(self) -> str:
        """获取累积的工具信息"""
        return self._accumulator.get_tool_info()

    def _handle_ai_message_chunk(self, chunk: AIMessageChunk):
        """处理 AI 消息块"""
        content = chunk.content

        # 处理列表形式的内容（主要是 Claude 模型）
        if isinstance(content, list) and len(content) > 0:
            message_chunk = content[0]

            if message_chunk.get("type") == "text":
                # 文本内容
                text = message_chunk.get("text", "")
                self._accumulator.append_text(text)
                if self._text_callback:
                    self._text_callback(self._accumulator.get_text())

            elif message_chunk.get("type") == "tool_use":
                # 工具调用
                if "partial_json" in message_chunk:
                    self._accumulator.append_tool(message_chunk["partial_json"])
                elif hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                    tool_info = f"\n```json\n{chunk.tool_call_chunks[0]}\n```\n"
                    self._accumulator.append_tool(tool_info)

                if self._tool_callback:
                    self._tool_callback(self._accumulator.get_tool_info())

        # 处理字符串形式的内容
        elif isinstance(content, str):
            self._accumulator.append_text(content)
            if self._text_callback:
                self._text_callback(self._accumulator.get_text())

        # 处理工具调用（OpenAI 模型）
        elif hasattr(chunk, "tool_calls") and chunk.tool_calls:
            if len(chunk.tool_calls[0].get("name", "")) > 0:
                tool_info = f"\n```json\n{chunk.tool_calls[0]}\n```\n"
                self._accumulator.append_tool(tool_info)
                if self._tool_callback:
                    self._tool_callback(self._accumulator.get_tool_info())

        # 处理无效工具调用
        elif hasattr(chunk, "invalid_tool_calls") and chunk.invalid_tool_calls:
            tool_info = f"\n```json\n{chunk.invalid_tool_calls[0]}\n```\n"
            self._accumulator.append_tool(tool_info)
            if self._tool_callback:
                self._tool_callback(self._accumulator.get_tool_info())

        # 处理 tool_call_chunks
        elif hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
            tool_info = f"\n```json\n{chunk.tool_call_chunks[0]}\n```\n"
            self._accumulator.append_tool(tool_info)
            if self._tool_callback:
                self._tool_callback(self._accumulator.get_tool_info())

        # 处理 additional_kwargs 中的工具调用
        elif hasattr(chunk, "additional_kwargs") and "tool_calls" in chunk.additional_kwargs:
            tool_info = f"\n```json\n{chunk.additional_kwargs['tool_calls'][0]}\n```\n"
            self._accumulator.append_tool(tool_info)
            if self._tool_callback:
                self._tool_callback(self._accumulator.get_tool_info())

    def _handle_tool_message(self, message: ToolMessage):
        """处理工具消息（工具返回结果）"""
        tool_info = f"\n```json\n{message.content}\n```\n"
        self._accumulator.append_tool(tool_info)
        if self._tool_callback:
            self._tool_callback(self._accumulator.get_tool_info())

    def handle_message(self, message: dict) -> None:
        """
        处理流式消息

        Args:
            message: 包含 'content' 键的消息字典
        """
        content = message.get("content")

        if isinstance(content, AIMessageChunk):
            self._handle_ai_message_chunk(content)
        elif isinstance(content, ToolMessage):
            self._handle_tool_message(content)

    def get_callback(self) -> Callable[[dict], None]:
        """获取回调函数（用于 astream_graph）"""
        return self.handle_message

    def reset(self):
        """重置处理器状态"""
        self._accumulator.clear()


def create_streamlit_streaming_handler(
    text_placeholder,
    tool_placeholder
) -> Tuple[StreamingHandler, Callable[[dict], None]]:
    """
    创建 Streamlit 流式处理器

    Args:
        text_placeholder: Streamlit 文本占位符
        tool_placeholder: Streamlit 工具信息占位符

    Returns:
        (StreamingHandler 实例, 回调函数)
    """
    def text_callback(text: str):
        text_placeholder.markdown(text)

    def tool_callback(tool_info: str):
        with tool_placeholder.expander("🔧 工具调用信息", expanded=True):
            import streamlit as st
            st.markdown(tool_info)

    handler = StreamingHandler(
        text_callback=text_callback,
        tool_callback=tool_callback
    )

    return handler, handler.get_callback()


# 兼容旧接口
def get_streaming_callback(text_placeholder, tool_placeholder):
    """
    创建流式回调函数（兼容旧接口）

    Args:
        text_placeholder: Streamlit 文本占位符
        tool_placeholder: Streamlit 工具信息占位符

    Returns:
        (回调函数, 累积文本列表, 累积工具信息列表)
    """
    handler, callback = create_streamlit_streaming_handler(
        text_placeholder, tool_placeholder
    )

    # 返回兼容旧接口的对象
    return callback, handler._accumulator.text, handler._accumulator.tool
