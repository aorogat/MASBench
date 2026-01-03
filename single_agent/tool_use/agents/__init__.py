"""
Agent implementations for StableToolBench evaluation.

This package provides a base interface (BaseAgent) and framework-specific implementations.
Each framework has its own subdirectory with agent implementation and helper code.
"""
from .base_agent import BaseAgent

__all__ = ['BaseAgent']

# Framework-specific agents can be imported from their respective subdirectories
# Example: from agents.langgraph import LangGraphAgent
try:
    from .langgraph import LangGraphAgent
    __all__.append('LangGraphAgent')
except ImportError:
    pass  # LangGraph not available

try:
    from .crewai import CrewAIAgent
    __all__.append('CrewAIAgent')
except ImportError:
    pass  # CrewAI not available

try:
    from .autogen import AutoGenAgent
    __all__.append('AutoGenAgent')
except ImportError:
    pass  # AutoGen not available

try:
    from .openai_sdk import OpenAISDKAgent
    __all__.append('OpenAISDKAgent')
except ImportError:
    pass  # OpenAI SDK not available


