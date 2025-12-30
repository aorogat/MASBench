"""
FrameworkAdapter: Adapter pattern implementation.

Single Responsibility: Adapts FrameworkInterface to StableToolBench's BaseModelAdapter.
Dependency Inversion: Depends on FrameworkInterface abstraction, not concrete implementations.
"""
import os
import sys
from typing import Dict, Any, Optional, List

# Ensure StableToolBench is importable
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)

try:
    from toolbench.inference.model.model_adapter import BaseModelAdapter
except ImportError:
    # Fallback if BaseModelAdapter is not available
    class BaseModelAdapter:
        def __init__(self):
            pass
        def reset(self):
            pass
        def set_tools(self, tools):
            pass
        def chat(self, query, history=None):
            pass

from .framework_interface import FrameworkInterface


class FrameworkAdapter(BaseModelAdapter):
    """
    Adapter that wraps a FrameworkInterface instance and exposes
    the ToolBench-compatible model API: reset(), set_tools(), chat().
    
    This follows the Adapter pattern and Liskov Substitution Principle:
    - Can be used anywhere BaseModelAdapter is expected
    - Maintains the same interface contract
    """

    def __init__(self, framework_instance: FrameworkInterface):
        """
        Initialize the adapter with a framework instance.
        
        Args:
            framework_instance: An instance implementing FrameworkInterface
        """
        super().__init__()
        if not isinstance(framework_instance, FrameworkInterface):
            raise TypeError(
                "framework_instance must implement FrameworkInterface"
            )
        self.framework = framework_instance

    def reset(self) -> None:
        """Reset the underlying framework."""
        self.framework.reset()

    def set_tools(self, tools: Dict[str, Dict]) -> None:
        """
        Set tools for the underlying framework.
        
        Args:
            tools: Dictionary mapping tool names to tool definition JSON
        """
        self.framework.setup_tools(tools)

    def chat(self, query: str, history: Optional[List] = None) -> Dict[str, Any]:
        """
        Chat interface compatible with StableToolBench.
        
        Args:
            query: The user query string
            history: Optional conversation history (not used in current implementation)
            
        Returns:
            Dictionary with 'response' (final answer) and 'tool_calls' (empty list)
        """
        ans = self.framework.answer(query)
        return {"response": ans, "tool_calls": []}

