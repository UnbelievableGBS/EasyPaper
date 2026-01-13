"""
MCP 配置管理模块

负责 MCP 服务器配置的加载、保存和验证。
将配置管理逻辑从主应用中分离，遵循单一职责原则。
"""

import os
import json
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MCPServerConfig:
    """单个 MCP 服务器配置"""
    command: Optional[str] = None
    args: list = field(default_factory=list)
    transport: str = "stdio"
    url: Optional[str] = None  # SSE 模式使用


@dataclass
class MCPConfig:
    """MCP 配置管理器"""

    config_file_path: str = "config.json"
    _config: Dict[str, Any] = field(default_factory=dict)

    # 默认配置
    DEFAULT_CONFIG: Dict[str, Any] = field(default_factory=lambda: {
        "get_current_time": {
            "command": "python",
            "args": ["./mcp_server_time.py"],
            "transport": "stdio"
        },
        "get_current_weather": {
            "command": "python",
            "args": ["./mcp_server_local_weather.py"],
            "transport": "stdio"
        }
    })

    def __post_init__(self):
        """初始化后加载配置"""
        self._config = self.load()

    @property
    def config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config

    def load(self) -> Dict[str, Any]:
        """
        从 JSON 文件加载配置

        Returns:
            配置字典，如果文件不存在则返回默认配置
        """
        try:
            if os.path.exists(self.config_file_path):
                with open(self.config_file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                # 文件不存在，创建默认配置
                self.save(self.DEFAULT_CONFIG)
                return self.DEFAULT_CONFIG.copy()
        except json.JSONDecodeError as e:
            print(f"配置文件 JSON 解析错误: {e}")
            return self.DEFAULT_CONFIG.copy()
        except Exception as e:
            print(f"加载配置文件时出错: {e}")
            return self.DEFAULT_CONFIG.copy()

    def save(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存配置到 JSON 文件

        Args:
            config: 要保存的配置，为 None 则保存当前配置

        Returns:
            保存是否成功
        """
        config_to_save = config if config is not None else self._config
        try:
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
            if config is not None:
                self._config = config_to_save
            return True
        except Exception as e:
            print(f"保存配置文件时出错: {e}")
            return False

    def add_tool(self, name: str, tool_config: Dict[str, Any]) -> tuple[bool, str]:
        """
        添加工具配置

        Args:
            name: 工具名称
            tool_config: 工具配置

        Returns:
            (成功标志, 消息)
        """
        # 验证配置
        is_valid, message = self.validate_tool_config(name, tool_config)
        if not is_valid:
            return False, message

        # 处理 URL 模式
        if "url" in tool_config:
            tool_config["transport"] = "sse"
        elif "transport" not in tool_config:
            tool_config["transport"] = "stdio"

        # 添加到配置
        self._config[name] = tool_config
        return True, f"工具 '{name}' 添加成功"

    def remove_tool(self, name: str) -> tuple[bool, str]:
        """
        移除工具配置

        Args:
            name: 工具名称

        Returns:
            (成功标志, 消息)
        """
        if name in self._config:
            del self._config[name]
            return True, f"工具 '{name}' 已移除"
        return False, f"工具 '{name}' 不存在"

    def validate_tool_config(self, name: str, config: Dict[str, Any]) -> tuple[bool, str]:
        """
        验证工具配置

        Args:
            name: 工具名称
            config: 工具配置

        Returns:
            (有效标志, 验证消息)
        """
        # 必须有 command 或 url
        if "command" not in config and "url" not in config:
            return False, f"工具 '{name}' 配置需要 'command' 或 'url' 字段"

        # 如果有 command，必须有 args
        if "command" in config and "args" not in config:
            return False, f"工具 '{name}' 配置需要 'args' 字段"

        # args 必须是列表
        if "command" in config and not isinstance(config.get("args"), list):
            return False, f"工具 '{name}' 的 'args' 字段必须是数组格式"

        return True, "配置有效"

    def get_tool_names(self) -> list:
        """获取所有工具名称"""
        return list(self._config.keys())

    def get_tool_count(self) -> int:
        """获取工具数量"""
        return len(self._config)

    @staticmethod
    def parse_tool_json(json_str: str) -> tuple[Optional[Dict[str, Any]], str]:
        """
        解析工具 JSON 字符串

        Args:
            json_str: JSON 字符串

        Returns:
            (解析结果, 错误消息)
        """
        json_str = json_str.strip()

        # 验证 JSON 格式
        if not json_str.startswith("{") or not json_str.endswith("}"):
            return None, "JSON 必须以 { 开头并以 } 结尾"

        try:
            parsed = json.loads(json_str)

            # 处理 mcpServers 包装格式
            if "mcpServers" in parsed:
                parsed = parsed["mcpServers"]

            return parsed, ""
        except json.JSONDecodeError as e:
            return None, f"JSON 解析错误: {e}"


# 全局配置实例（单例模式）
_config_manager: Optional[MCPConfig] = None


def get_config_manager(config_file_path: str = "config.json") -> MCPConfig:
    """
    获取配置管理器实例（单例）

    Args:
        config_file_path: 配置文件路径

    Returns:
        MCPConfig 实例
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = MCPConfig(config_file_path=config_file_path)
    return _config_manager
