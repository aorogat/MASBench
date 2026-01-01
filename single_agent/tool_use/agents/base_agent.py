"""
Base Agent Interface

All agents must implement this interface to work with run_benchmark.py.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import BaseTool


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement:
    1. bind_tools() - Bind tools to the agent (either from directory or pre-selected tools)
    2. answer() - Generate answer for a query
    
    This ensures all agents can be used with run_benchmark.py in a consistent way.
    
    Architecture:
    - Tool selection is now centralized (via ToolSelector) and runs before bind_tools()
    - Agents receive pre-selected tools, ensuring fairness across frameworks
    - All frameworks use the same tool set for the same query (cached)
    """
    
    @abstractmethod
    def bind_tools(
        self,
        tools: Union[List[BaseTool], str, None] = None,
        tools_dir: Optional[str] = None,
        server_url: str = "http://localhost:8080/virtual"
    ) -> None:
        """
        Bind tools to the agent.
        
        Args:
            tools: Pre-selected list of tools (preferred - from ToolSelector)
            tools_dir: Path to StableToolBench/toolenv/tools/ directory (legacy - loads all tools)
            server_url: URL of the server for tool calls (default: http://localhost:8080/virtual)
        
        This method should:
        1. If tools is provided: Use the pre-selected tools directly
        2. If tools_dir is provided: Load all tools from directory (legacy mode)
        3. Convert tools to framework-specific format
        4. Bind them to the agent so they can be used in answer() calls
        
        Note:
        - Pre-selected tools (tools parameter) is the preferred approach for fairness
        - tools_dir is kept for backward compatibility
        - Should be called before using answer() method
        - Can be called multiple times to reload/update tools
        - Tools should call the server at server_url when invoked
        """
        pass
    
    @abstractmethod
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for a query.
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format:
            {
                "answer": {
                    "final_answer": str,  # The final answer text
                    "answer_details": List[Dict]  # ExecutionGraph format
                },
                "called_apis": Optional[List[List[str]]]  # Optional: List of [tool_name, api_name] pairs
                                                          # If provided, run_benchmark uses this directly
                                                          # Otherwise, it parses from answer_details
            }
            
        The answer_details must be in ExecutionGraph format:
        [
            {
                "role": "tool",
                "message": json.dumps({
                    "name": "tool_name_api_name",  # or "Finish"
                    "arguments": {...},  # Tool arguments or {"return_type": "give_answer", "final_answer": "..."}
                    "response": "..."  # Tool response or "" for Finish
                }),
                "next": []
            },
            ...
        ]
        
        The called_apis (if provided) should be a list of [tool_name, api_name] pairs:
        [
            ["TheClique", "Transfermarkt search"],
            ["TheClique", "Songkick concert"],
            ...
        ]
        
        Note:
        - Tool calls should use format: "tool_name_api_name" in answer_details
        - Must include a "Finish" call at the end in answer_details
        - All tool calls should be recorded in answer_details
        - It's recommended to also provide "called_apis" for easier API tracking
          This avoids parsing answer_details and is more efficient
        - bind_tools() must be called before using answer()
        """
        pass
    
    def __call__(self, query: str) -> Dict[str, Any]:
        """
        Allow agent to be called directly: agent(query)
        
        Args:
            query: The query string
            
        Returns:
            Same as answer() method
        """
        return self.answer(query)

