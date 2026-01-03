"""
AutoGen Agent Implementation

Implements BaseAgent interface using AutoGen's AssistantAgent for tool-calling agents.
Uses AutoGen's AssistantAgent which automatically handles tool calling.

Important: Tools are pre-selected by MASBench ToolSelector.
AutoGen does not perform additional filtering - it uses whatever tools are provided.

Note on Tool Limits:
- OpenAI has a limit of 128 tools per request
- AutoGen's AssistantAgent sends ALL bound tools to the LLM in each request
- AutoGen has no built-in tool filtering mechanism
- This is why centralized tool selection (ToolSelector) is essential for AutoGen
"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable, get_type_hints

# Check for core AutoGen packages
try:
    import autogen_agentchat
    import autogen_core
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.messages import (
        TextMessage,
        ToolCallRequestEvent,
        ToolCallExecutionEvent,
        ToolCallSummaryMessage,
        BaseChatMessage
    )
    AUTOGEN_CORE_AVAILABLE = True
except ImportError as e:
    AUTOGEN_CORE_AVAILABLE = False
    AUTOGEN_CORE_IMPORT_ERROR = e
    # Create dummy classes to allow module import
    class AssistantAgent:
        pass
    TextMessage = None
    ToolCallRequestEvent = None
    ToolCallExecutionEvent = None

# Check for AutoGen extensions (model clients)
try:
    from autogen_ext.models.openai import OpenAIChatCompletionClient
    AUTOGEN_EXT_AVAILABLE = True
    AUTOGEN_EXT_IMPORT_ERROR = None
except ImportError as e:
    AUTOGEN_EXT_AVAILABLE = False
    AUTOGEN_EXT_IMPORT_ERROR = e
    class OpenAIChatCompletionClient:
        pass

# Overall availability requires both core and extensions
AUTOGEN_AVAILABLE = AUTOGEN_CORE_AVAILABLE and AUTOGEN_EXT_AVAILABLE
if not AUTOGEN_AVAILABLE:
    if not AUTOGEN_CORE_AVAILABLE:
        AUTOGEN_IMPORT_ERROR = AUTOGEN_CORE_IMPORT_ERROR
    else:
        AUTOGEN_IMPORT_ERROR = AUTOGEN_EXT_IMPORT_ERROR
else:
    AUTOGEN_IMPORT_ERROR = None

# Check if LangChain tools are being passed (from run_benchmark.py)
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainBaseTool = None
    StructuredTool = None

from ..base_agent import BaseAgent
from utils.tool_utils import create_tool_function


class AutoGenAgent(BaseAgent):
    """
    AutoGen agent that implements BaseAgent interface.
    
    Uses AutoGen's AssistantAgent which:
    1. Uses a language model with tool calling capability
    2. Automatically converts Python functions to FunctionTool
    3. Executes tools and returns results
    4. Supports async execution
    
    This agent:
    1. Loads tools from StableToolBench (via LangChain loader in run_benchmark.py)
    2. Converts LangChain tools to Python functions for AutoGen
    3. Uses LLM with tool calling via AssistantAgent
    4. Returns answers in StableToolBench format with ExecutionGraph format
    
    Note: Requires AutoGen to be installed. If AutoGen is not available,
    this class cannot be instantiated (raises ImportError).
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
        Initialize AutoGen agent.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            server_url: URL of the server for tool calls
            temperature: Temperature for LLM (default: 0.0)
            verbose: Whether to print debug information
            max_tools: Maximum number of tools to use per query (default: 120, OpenAI limit is 128)
        """
        if not AUTOGEN_AVAILABLE:
            if not AUTOGEN_CORE_AVAILABLE:
                error_msg = (
                    "AutoGen AgentChat core packages are not installed. "
                    "Install them with:\n"
                    "  pip install autogen-agentchat autogen-core"
                )
            else:
                error_msg = (
                    "AutoGen extensions are not installed. "
                    "Install them with:\n"
                    "  pip install autogen-ext"
                )
            if AUTOGEN_IMPORT_ERROR is not None:
                raise ImportError(error_msg) from AUTOGEN_IMPORT_ERROR
            else:
                raise ImportError(error_msg)
        self.model = model
        self.server_url = server_url
        self.temperature = temperature
        self.verbose = verbose
        self.max_tools = max_tools
        
        # Tools are pre-selected by ToolSelector and bound here
        # AutoGen does not perform additional filtering
        self.bound_tools: List[Callable] = []  # Python functions for AutoGen
        self.tool_metadata: Dict[str, Dict[str, str]] = {}  # Map tool function name to metadata
        self.tools_bound = False
        
        # Model client will be created when needed (async context)
        self._model_client: Optional[OpenAIChatCompletionClient] = None
    
    def bind_tools(
        self,
        tools: Union[List[LangChainBaseTool], str, None] = None,
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
                # Check if tools are LangChain tools (from run_benchmark.py)
                # If so, convert them to Python functions for AutoGen
                if len(tools) > 0:
                    first_tool = tools[0]
                    if LANGCHAIN_AVAILABLE and isinstance(first_tool, StructuredTool):
                        # Convert LangChain tools to Python functions
                        if self.verbose:
                            print(f"[AutoGenAgent] Converting {len(tools)} LangChain tools to AutoGen functions...")
                        self.bound_tools = self._convert_langchain_to_autogen_tools(tools)
                    else:
                        # Assume they're already Python functions
                        self.bound_tools = tools
                else:
                    self.bound_tools = []
                
                # Fail loudly if tool count exceeds max_tools (benchmark integrity)
                if len(self.bound_tools) > self.max_tools:
                    raise RuntimeError(
                        f"ToolSelector returned {len(self.bound_tools)} tools, "
                        f"exceeds max_tools={self.max_tools}. "
                        f"This violates benchmark integrity."
                    )
                
                if self.verbose:
                    print(f"[AutoGenAgent] Bound {len(self.bound_tools)} pre-selected tools")
            else:
                raise ValueError("tools must be a list of BaseTool objects or Python functions")
        elif tools_dir is not None:
            # Legacy mode: load all tools from directory (not recommended for benchmarks)
            raise NotImplementedError(
                "Direct tool loading from directory is not implemented for AutoGen. "
                "Use the centralized ToolSelector via run_benchmark.py instead."
            )
        else:
            raise ValueError("Either tools or tools_dir must be provided")
        
        self.tools_bound = True
    
    def _convert_langchain_to_autogen_tools(self, langchain_tools: List) -> List[Callable]:
        """
        Convert LangChain tools to Python functions for AutoGen.
        
        AutoGen's AssistantAgent automatically converts Python functions to FunctionTool,
        but it requires functions with proper type annotations (not **kwargs).
        We extract the parameter schema from the LangChain tool and create a properly
        typed function.
        
        Args:
            langchain_tools: List of LangChain StructuredTool objects
            
        Returns:
            List of Python callable functions with proper type annotations
        """
        from inspect import signature, Parameter
        from typing import get_type_hints
        
        autogen_tools = []
        
        for langchain_tool in langchain_tools:
            # Extract metadata from LangChain tool
            tool_name = langchain_tool.name
            description = langchain_tool.description
            
            # Get structured metadata if available
            metadata = getattr(langchain_tool, 'metadata', {}) or {}
            tool_name_original = metadata.get('tool_name', '')
            api_name = metadata.get('api_name', '')
            category = metadata.get('category', '')
            original_name = metadata.get('original_name', tool_name)
            
            # Get the original tool function from LangChain tool
            original_func = langchain_tool.func
            
            # Get the args_schema from LangChain tool to extract parameter types
            args_schema = langchain_tool.args_schema
            
            # Build function signature from args_schema
            # AutoGen needs explicit parameters with type annotations (not **kwargs)
            if args_schema and hasattr(args_schema, 'model_fields'):
                # Import PydanticUndefined to check for it
                try:
                    from pydantic_core import PydanticUndefined
                except ImportError:
                    try:
                        from pydantic import PydanticUndefined
                    except ImportError:
                        # Fallback: use a sentinel object
                        class PydanticUndefined:
                            pass
                
                # Extract parameters from Pydantic model
                param_defs = []
                annotations = {}
                
                for field_name, field_info in args_schema.model_fields.items():
                    # Get field type
                    field_type = field_info.annotation
                    if field_type is None or field_type == Parameter.empty:
                        field_type = str  # Default to str
                    
                    # Check if field has default
                    # Pydantic v2 uses default_factory or default
                    default_value = Parameter.empty
                    if hasattr(field_info, 'default'):
                        default_value = field_info.default
                    elif hasattr(field_info, 'default_factory'):
                        # Skip fields with default_factory (they're computed)
                        default_value = Parameter.empty
                    
                    # Check if default is actually set (not PydanticUndefined)
                    if (default_value is not Parameter.empty and 
                        default_value is not None and
                        not isinstance(default_value, type(PydanticUndefined)) and
                        default_value is not PydanticUndefined):
                        # Optional parameter with default
                        param_defs.append((field_name, field_type, default_value))
                        annotations[field_name] = Optional[field_type] if field_type != str else str
                    else:
                        # Required parameter
                        param_defs.append((field_name, field_type, None))
                        annotations[field_name] = field_type
                
                # Create a function factory that builds functions with explicit parameters
                def make_typed_function(param_defs, func, desc):
                    # Build function signature string
                    param_strs = []
                    for param_name, param_type, default_val in param_defs:
                        type_str = param_type.__name__ if hasattr(param_type, '__name__') else 'str'
                        if default_val is not None and default_val is not Parameter.empty:
                            # Only include default if it's a real value (not PydanticUndefined)
                            # Check if it's PydanticUndefined by comparing types
                            try:
                                from pydantic_core import PydanticUndefined as PUndef
                            except ImportError:
                                try:
                                    from pydantic import PydanticUndefined as PUndef
                                except ImportError:
                                    PUndef = type('PydanticUndefined', (), {})
                            
                            # Skip if it's PydanticUndefined
                            if isinstance(default_val, type(PUndef)) or default_val is PUndef:
                                param_strs.append(f"{param_name}: {type_str}")
                            elif isinstance(default_val, str):
                                # Escape quotes in strings
                                escaped_val = default_val.replace("'", "\\'")
                                param_strs.append(f"{param_name}: {type_str} = '{escaped_val}'")
                            elif isinstance(default_val, (int, float, bool)):
                                param_strs.append(f"{param_name}: {type_str} = {default_val}")
                            elif default_val is None:
                                param_strs.append(f"{param_name}: Optional[{type_str}] = None")
                            else:
                                # For other types, use repr
                                param_strs.append(f"{param_name}: {type_str} = {repr(default_val)}")
                        else:
                            param_strs.append(f"{param_name}: {type_str}")
                    
                    param_str = ', '.join(param_strs)
                    
                    # Build function body
                    param_names = [p[0] for p in param_defs]
                    kwargs_build = ', '.join([f"'{p}': {p}" for p in param_names])
                    
                    func_body = f"""
    try:
        kwargs = {{{kwargs_build}}}
        if asyncio.iscoroutinefunction(func):
            result = await func(**kwargs)
        else:
            result = func(**kwargs)
        if isinstance(result, dict):
            return json.dumps(result)
        return str(result)
    except Exception as e:
        return json.dumps({{"error": str(e), "response": ""}})
"""
                    
                    # Create function dynamically
                    func_code = f"async def autogen_tool_func({param_str}) -> str:\n    \"\"\"{desc}\"\"\"\n{func_body}"
                    local_vars = {
                        'func': func,
                        'asyncio': asyncio,
                        'json': json,
                        'isinstance': isinstance,
                        'str': str,
                        'dict': dict
                    }
                    exec(func_code, globals(), local_vars)
                    return local_vars['autogen_tool_func']
                
                autogen_tool_func = make_typed_function(param_defs, original_func, description)
                autogen_tool_func.__annotations__ = annotations.copy()
                autogen_tool_func.__annotations__['return'] = str
            else:
                # Fallback: use a simple wrapper (may fail if AutoGen can't parse it)
                def create_autogen_tool_func(func):
                    async def autogen_tool_func(*args, **kwargs) -> str:
                        try:
                            if asyncio.iscoroutinefunction(func):
                                result = await func(*args, **kwargs)
                            else:
                                result = func(*args, **kwargs)
                            if isinstance(result, dict):
                                return json.dumps(result)
                            return str(result)
                        except Exception as e:
                            return json.dumps({"error": str(e), "response": ""})
                    return autogen_tool_func
                
                autogen_tool_func = create_autogen_tool_func(original_func)
            
            # Set function name and docstring
            autogen_tool_func.__name__ = tool_name
            autogen_tool_func.__doc__ = description
            
            # Store metadata for later extraction
            self.tool_metadata[tool_name] = {
                'tool_name': tool_name_original or tool_name,
                'api_name': api_name,
                'category': category,
                'original_name': original_name
            }
            
            autogen_tools.append(autogen_tool_func)
        
        return autogen_tools
    
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
            print(f"[AutoGenAgent] Processing query: {query[:100]}...")
        
        # AutoGen uses async, so we need to run the async code
        return asyncio.run(self._answer_async(query))
    
    async def _answer_async(self, query: str) -> Dict[str, Any]:
        """
        Async implementation of answer().
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format with answer and called_apis
        """
        # Create model client if not exists
        if self._model_client is None:
            self._model_client = OpenAIChatCompletionClient(
                model=self.model,
                # api_key is read from environment automatically
            )
        
        # Create agent with bound tools
        # AutoGen's AssistantAgent automatically converts Python functions to FunctionTool
        agent = AssistantAgent(
            name="assistant",
            model_client=self._model_client,
            tools=self.bound_tools if len(self.bound_tools) > 0 else None,
            system_message="Use tools to solve tasks accurately and efficiently.",
        )
        
        if self.verbose:
            print(f"[AutoGenAgent] Using {len(self.bound_tools)} tools")
        
        # Execute agent (async)
        try:
            result = await agent.run(task=query)
        except Exception as e:
            if self.verbose:
                print(f"[AutoGenAgent] Error during execution: {e}")
            raise
        
        # Extract called_apis and build answer_details from messages
        called_apis: List[List[str]] = []
        answer_details: List[Dict[str, Any]] = []
        final_answer = ""
        
        # Parse messages to extract tool calls and final answer
        for message in result.messages:
            if isinstance(message, ToolCallRequestEvent):
                # Tool call request - extract tool name and arguments
                for function_call in message.content:
                    tool_name = function_call.name
                    arguments = json.loads(function_call.arguments) if isinstance(function_call.arguments, str) else function_call.arguments
                    
                    # Get metadata for this tool
                    metadata = self.tool_metadata.get(tool_name, {})
                    tool_name_original = metadata.get('tool_name', tool_name)
                    api_name = metadata.get('api_name', '')
                    
                    # Record API call
                    if tool_name_original and api_name:
                        called_apis.append([tool_name_original, api_name])
                    
                    # Add to answer_details
                    tool_call_detail = {
                        "role": "tool",
                        "message": json.dumps({
                            "name": f"{tool_name_original}_{api_name}" if api_name else tool_name_original,
                            "arguments": arguments,
                            "response": ""  # Will be filled by ToolCallExecutionEvent
                        }),
                        "next": []
                    }
                    answer_details.append(tool_call_detail)
            
            elif isinstance(message, ToolCallExecutionEvent):
                # Tool execution result - update the last tool call detail
                for execution_result in message.content:
                    tool_name = execution_result.name
                    result_content = execution_result.content
                    is_error = execution_result.is_error
                    
                    # Find the corresponding tool call detail and update it
                    for detail in reversed(answer_details):
                        detail_msg = json.loads(detail["message"])
                        if detail_msg["name"].startswith(tool_name) or tool_name in detail_msg["name"]:
                            detail_msg["response"] = result_content if not is_error else f"Error: {result_content}"
                            detail["message"] = json.dumps(detail_msg)
                            break
            
            elif isinstance(message, ToolCallSummaryMessage):
                # Tool call summary - this is the result after tool execution
                # This might be the final answer if no further processing is needed
                if not final_answer:
                    final_answer = message.content
            
            elif isinstance(message, TextMessage) and message.source == 'assistant':
                # Final text message from assistant
                if message.content and not final_answer:
                    final_answer = message.content
        
        # If no final answer was found, use the last message content
        if not final_answer and result.messages:
            last_message = result.messages[-1]
            if hasattr(last_message, 'content'):
                final_answer = str(last_message.content)
        
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
            print(f"[AutoGenAgent] Generated answer (tool calls: {len(called_apis)})")
        
        # Return in StableToolBench format
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            },
            "called_apis": called_apis
        }
    
    def __del__(self):
        """Clean up model client on deletion."""
        # Check if _model_client attribute exists (it might not if __init__ failed)
        if hasattr(self, '_model_client') and self._model_client is not None:
            # Note: AutoGen's model client should be closed with async context
            # This is a best-effort cleanup
            try:
                if hasattr(self._model_client, 'close'):
                    # We can't await in __del__, so this might not work perfectly
                    # But it's better than nothing
                    pass
            except:
                pass

