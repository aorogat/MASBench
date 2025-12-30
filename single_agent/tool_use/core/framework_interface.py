"""
FrameworkInterface: Abstract base class for framework integration.

Single Responsibility: Defines the contract that frameworks must implement.
"""
from abc import ABC, abstractmethod
from typing import Dict


class FrameworkInterface(ABC):
    """
    A unified interface so any framework (LangGraph, CrewAI, OpenAI_SDK)
    can be evaluated using the StableToolBench evaluation pipeline.
    
    This interface follows the Interface Segregation Principle by providing
    only the essential methods needed for evaluation.
    """

    @abstractmethod
    def setup_tools(self, tools: Dict[str, Dict]) -> None:
        """
        Set up tools for the framework.
        
        Args:
            tools: Dictionary mapping tool names to tool definition JSON
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Reset conversation / memory / internal state.
        Called between queries to ensure clean state.
        """
        raise NotImplementedError

    @abstractmethod
    def answer(self, query: str) -> str:
        """
        Answer the query using the framework.
        
        Args:
            query: The user query string
            
        Returns:
            Only the final answer (string), no intermediate steps
        """
        raise NotImplementedError

