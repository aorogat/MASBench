"""
LangGraph Agent Implementation

Implements BaseAgent interface using LangChain's create_agent for tool-calling agents.
Uses the recommended create_agent API which automatically handles the ReAct loop.

Important: Tools are pre-selected by MASBench ToolSelector.
LangGraph does not perform additional filtering - it uses whatever tools are provided.

Note on Tool Limits:
- OpenAI has a limit of 128 tools per request
- LangGraph's create_agent sends ALL bound tools to the LLM in each request
- LangGraph has no built-in tool filtering mechanism
- This is why centralized tool selection (ToolSelector) is essential for LangGraph
"""
import os
import json
from typing import Dict, Any, List, Optional, Union
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage

from ..base_agent import BaseAgent
from .tool_loader import load_tools


class LangGraphAgent(BaseAgent):
    """
    LangGraph agent that implements BaseAgent interface.
    
    Uses LangChain's create_agent which automatically:
    1. Creates a graph-based agent runtime using LangGraph
    2. Handles the ReAct loop (reasoning + acting)
    3. Manages tool calls and responses
    4. Returns final answers when appropriate
    
    This agent:
    1. Loads tools from StableToolBench
    2. Uses LLM with tool calling via create_agent
    3. Returns answers in StableToolBench format with ExecutionGraph format
    """
    
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        server_url: str = "http://localhost:8080/virtual",
        temperature: float = 0.0,
        verbose: bool = False,
        max_tools: int = 120
    ):
        """
        Initialize LangGraph agent.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            server_url: URL of the server for tool calls
            temperature: Temperature for LLM (default: 0.0)
            verbose: Whether to print debug information
            max_tools: Maximum number of tools to use per query (default: 120, OpenAI limit is 128)
        """
        self.model = model
        self.server_url = server_url
        self.temperature = temperature
        self.verbose = verbose
        self.max_tools = max_tools
        
        # Initialize LLM
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        
        # Tools are pre-selected by ToolSelector and bound here
        # LangGraph does not perform additional filtering
        self.bound_tools: List[BaseTool] = []  # Pre-selected tools from ToolSelector
        self.agent = None
        self.tools_bound = False
    
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
            server_url: URL of the server for tool calls
        """
        # Update server URL if provided
        self.server_url = server_url
        
        if tools is not None:
            # Use pre-selected tools (preferred approach - from ToolSelector)
            if isinstance(tools, list):
                self.bound_tools = tools
                
                # Fail loudly if tool count exceeds max_tools (benchmark integrity)
                if len(self.bound_tools) > self.max_tools:
                    raise RuntimeError(
                        f"ToolSelector returned {len(self.bound_tools)} tools, "
                        f"exceeds max_tools={self.max_tools}. "
                        f"This indicates a bug in tool selection or configuration."
                    )
                
                if self.verbose:
                    print(f"[LangGraphAgent] Bound {len(self.bound_tools)} pre-selected tools")
            else:
                raise ValueError("tools must be a list of BaseTool objects")
        elif tools_dir is not None:
            # Legacy mode: load all tools from directory (not recommended for benchmarks)
            if self.verbose:
                print(f"[LangGraphAgent] Loading tools from {tools_dir}...")
            loaded_tools = load_tools(tools_dir=tools_dir, server_url=server_url)
            
            # In legacy mode, still enforce max_tools limit
            if len(loaded_tools) > self.max_tools:
                raise RuntimeError(
                    f"Loaded {len(loaded_tools)} tools from {tools_dir}, "
                    f"exceeds max_tools={self.max_tools}. "
                    f"Use ToolSelector for proper tool filtering."
                )
            
            self.bound_tools = loaded_tools
            if self.verbose:
                print(f"[LangGraphAgent] Loaded {len(self.bound_tools)} tools")
        else:
            raise ValueError("Either tools or tools_dir must be provided")
        
        # Note: We don't create the agent here because tools may be filtered per query
        # The agent will be created in answer() with the tools
        self.tools_bound = True
    
    def _get_tools_for_query(self, query: str) -> List[BaseTool]:
        """
        Get tools to use for a query.
        
        Tools are pre-selected by ToolSelector, so we simply return them.
        LangGraph does not perform additional filtering.
        
        Args:
            query: The query string (unused, kept for interface compatibility)
            
        Returns:
            List of pre-selected tools
        """
        # Tools are already pre-selected and validated in bind_tools()
        # LangGraph uses all bound tools - no additional filtering
        return self.bound_tools
    
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for a query.
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format with answer and called_apis
        """
        if not self.tools_bound:
            raise ValueError("Tools must be bound before calling answer(). Call bind_tools() first.")
        
        if self.verbose:
            print(f"[LangGraphAgent] Processing query: {query[:100]}...")
        
        # Get tools for this query (pre-selected tools should already be filtered)
        tools_to_use = self._get_tools_for_query(query)
        
        if self.verbose:
            print(f"[LangGraphAgent] Using {len(tools_to_use)} tools")
        
        # Create agent with tools for this query
        # Note: create_agent is lightweight, so recreating per query is acceptable
        agent = create_agent(
            model=self.llm,
            tools=tools_to_use
        )
        
        # Invoke agent with the query
        # The agent will automatically handle the ReAct loop
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        # Extract messages from result
        messages = result.get("messages", [])
        
        # Extract called_apis and build answer_details
        called_apis: List[List[str]] = []
        answer_details: List[Dict[str, Any]] = []
        final_answer = ""
        
        # Create a mapping of tool_call_id to tool_call info for matching
        tool_call_map: Dict[str, Dict[str, Any]] = {}
        
        # Process messages to extract tool calls and final answer
        for msg in messages:
            if isinstance(msg, AIMessage):
                # Check if this message has tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    # Store tool calls in map for later matching with ToolMessages
                    for tool_call in msg.tool_calls:
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_id = tool_call.get("id", "")
                        
                        # Store in map
                        tool_call_map[tool_id] = {
                            "name": tool_name,
                            "args": tool_args
                        }
                        
                        # Extract tool_name and api_name from metadata
                        # This avoids fragile string parsing and handles edge cases correctly
                        tool_name_part = None
                        api_name_part = None
                        
                        for tool in tools_to_use:
                            if tool.name == tool_name:
                                # Check if tool has structured metadata (preferred)
                                if hasattr(tool, 'metadata') and isinstance(tool.metadata, dict):
                                    # Use structured metadata if available (most reliable)
                                    if 'tool_name' in tool.metadata and 'api_name' in tool.metadata:
                                        tool_name_part = tool.metadata['tool_name']
                                        api_name_part = tool.metadata['api_name']
                                        break
                                    # Fallback to original_name parsing (for backward compatibility)
                                    elif 'original_name' in tool.metadata:
                                        original_name = tool.metadata['original_name']
                                        # Original name format: "tool_name_api_name"
                                        if "_" in original_name:
                                            parts = original_name.split("_", 1)
                                            if len(parts) == 2:
                                                tool_name_part, api_name_part = parts[0], parts[1]
                                                break
                        
                        # Only add if we successfully extracted both parts
                        if tool_name_part and api_name_part:
                            called_apis.append([tool_name_part, api_name_part])
                        elif self.verbose:
                            # Log warning if we couldn't extract API info
                            print(f"[LangGraphAgent] Warning: Could not extract tool/api name for {tool_name}")
                else:
                    # This is a final answer (no tool calls)
                    if msg.content:
                        final_answer = msg.content
            
            elif isinstance(msg, ToolMessage):
                # This is a tool response
                # Match it with the corresponding tool call
                tool_call_id = msg.tool_call_id
                tool_response = msg.content
                
                if tool_call_id in tool_call_map:
                    tool_call_info = tool_call_map[tool_call_id]
                    tool_name = tool_call_info["name"]
                    tool_args = tool_call_info["args"]
                    
                    # Add to answer_details (ExecutionGraph format)
                    # TODO: Current format is minimal - nodes are disconnected and ordering is implicit.
                    # For basic tool-call counting this is sufficient, but for advanced analysis
                    # (planning depth, branching, retries) we may need explicit graph structure with
                    # node IDs and explicit sequencing links.
                    answer_detail = {
                        "role": "tool",
                        "message": json.dumps({
                            "name": tool_name,
                            "arguments": tool_args,
                            "response": str(tool_response)
                        }),
                        "next": []
                    }
                    answer_details.append(answer_detail)
        
        # If no final answer was found, use the last message content
        if not final_answer:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = msg.content
                    break
        
        # If still no answer, use a default
        if not final_answer:
            final_answer = "I processed your query but couldn't generate a final answer."
        
        # Add Finish call to answer_details
        finish_detail = {
            "role": "tool",
            "message": json.dumps({
                "name": "Finish",
                "arguments": {
                    "return_type": "give_answer",
                    "final_answer": final_answer
                },
                "response": ""
            }),
            "next": []
        }
        answer_details.append(finish_detail)
        
        if self.verbose:
            print(f"[LangGraphAgent] Generated answer with {len(called_apis)} tool calls")
        
        # Return in StableToolBench format
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            },
            "called_apis": called_apis
        }
