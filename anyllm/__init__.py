"""
AnyLLM - Unified LLM Client

Provides compatible interfaces for OpenAI and g4f
"""

from .client import Client, Chat, ChatCompletions, AsyncClient, AsyncChat, AsyncChatCompletions
from .result import to_result, ato_result

__version__ = "0.1.0"

__all__ = [
    "Client",
    "Chat", 
    "ChatCompletions",
    "AsyncClient",
    "AsyncChat",
    "AsyncChatCompletions",
    "to_result",
    "ato_result",
]
