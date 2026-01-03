"""
Agno Agent Implementation

Implements BaseAgent interface using Agno framework for tool-calling agents.
Uses Agno's Agent class which automatically converts Python functions to tools.
"""

try:
    from agno.agent import Agent as AgnoAgent
    from agno.models.openai import OpenAIChat
    AGNO_AVAILABLE = True
    AGNO_IMPORT_ERROR = None
except ImportError as e:
    AGNO_AVAILABLE = False
    AGNO_IMPORT_ERROR = e
    # Create dummy classes to allow module import
    class AgnoAgent:
        pass
    class OpenAIChat:
        pass

try:
    from .agent import AgnoAgent as AgnoAgentClass
    __all__ = ['AgnoAgentClass', 'AGNO_AVAILABLE', 'AGNO_IMPORT_ERROR']
except ImportError:
    __all__ = ['AGNO_AVAILABLE', 'AGNO_IMPORT_ERROR']
    if AGNO_AVAILABLE:
        raise

