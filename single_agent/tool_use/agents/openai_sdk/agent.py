"""
OpenAI Agents SDK Agent Implementation

Implements BaseAgent interface using OpenAI Agents SDK for tool-calling agents.
Uses OpenAI's Agent and Runner classes for agent execution.

Important: Tools are pre-selected by MASBench ToolSelector.
OpenAI Agents SDK does not perform additional filtering - it uses whatever tools are provided.

Note on Tool Limits:
- OpenAI has a limit of 128 tools per request
- OpenAI Agents SDK sends ALL bound tools to the LLM in each request
- OpenAI Agents SDK has no built-in tool filtering mechanism
- This is why centralized tool selection (ToolSelector) is essential for OpenAI SDK
"""
import os
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Callable
from typing_extensions import TypedDict

# Import OpenAI Agents SDK
# Note: We need to import from the installed package, not our local 'agents' package
# We use lazy imports to avoid circular import issues - imports happen in __init__ method
OPENAI_SDK_AVAILABLE = None  # Will be determined lazily
OPENAI_SDK_IMPORT_ERROR = None

def _import_openai_sdk():
    """Lazy import function for OpenAI Agents SDK to avoid circular imports."""
    global OPENAI_SDK_AVAILABLE, OPENAI_SDK_IMPORT_ERROR, Agent, Runner, function_tool, FunctionTool, RunResult
    
    if OPENAI_SDK_AVAILABLE is not None:
        # Already tried to import
        return OPENAI_SDK_AVAILABLE
    
    try:
        import importlib
        import sys
        import os
        
        # Get the directory containing our local agents package
        current_file = os.path.abspath(__file__)
        tool_use_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # Go up from agents/openai_sdk/agent.py to tool_use
        
        # Backup and remove our local 'agents' package from sys.modules
        _local_agents_backup = None
        _local_agents_submodules = []
        if 'agents' in sys.modules:
            local_mod = sys.modules['agents']
            local_file = getattr(local_mod, '__file__', '')
            if local_file and 'site-packages' not in local_file:
                _local_agents_backup = sys.modules.pop('agents')
                # Also backup submodules
                _local_agents_submodules = [(k, sys.modules.pop(k)) for k in list(sys.modules.keys()) if k.startswith('agents.')]
        
        # Temporarily remove our local package directory from sys.path
        _removed_paths = []
        original_path = sys.path[:]
        for path in original_path:
            abs_path = os.path.abspath(path) if not os.path.isabs(path) else path
            if tool_use_dir in abs_path or abs_path == tool_use_dir:
                sys.path.remove(path)
                _removed_paths.append(path)
        
        try:
            # Now import the openai-agents package normally
            _openai_agents_pkg = importlib.import_module('agents')
            
            # Verify it's from site-packages, not our local package
            pkg_file = getattr(_openai_agents_pkg, '__file__', '')
            if not pkg_file or 'site-packages' not in pkg_file:
                raise ImportError(f"Imported 'agents' package is not from site-packages: {pkg_file}")
            
            # Verify it has the required attributes
            if not (hasattr(_openai_agents_pkg, 'Agent') and hasattr(_openai_agents_pkg, 'Runner')):
                raise ImportError(f"Imported 'agents' package does not have Agent or Runner. Location: {pkg_file}")
            
            Agent = _openai_agents_pkg.Agent
            Runner = _openai_agents_pkg.Runner
            function_tool = _openai_agents_pkg.function_tool
            FunctionTool = _openai_agents_pkg.FunctionTool
            RunResult = _openai_agents_pkg.result.RunResult
            
            OPENAI_SDK_AVAILABLE = True
            OPENAI_SDK_IMPORT_ERROR = None
            return True
        except ImportError as e:
            # Restore local package and paths on error
            if _local_agents_backup is not None:
                sys.modules['agents'] = _local_agents_backup
            for k, v in _local_agents_submodules:
                sys.modules[k] = v
            # Restore paths
            for path in _removed_paths:
                if path not in sys.path:
                    sys.path.append(path)
            raise
        finally:
            # Always restore paths
            if _removed_paths:
                for path in _removed_paths:
                    if path not in sys.path:
                        sys.path.append(path)
    except (ImportError, AttributeError) as e:
        OPENAI_SDK_AVAILABLE = False
        OPENAI_SDK_IMPORT_ERROR = e
        # Create dummy classes
        class Agent:
            pass
        class Runner:
            pass
        function_tool = None
        FunctionTool = None
        RunResult = None
        return False

# Initialize dummy classes for now (will be replaced by lazy import)
class Agent:
    pass
class Runner:
    pass
function_tool = None
FunctionTool = None
RunResult = None

# Check if LangChain tools are being passed (from run_benchmark.py)
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainBaseTool = None
    StructuredTool = None

from ..base_agent import BaseAgent
from utils.tool_utils import create_tool_function, sanitize_tool_name


class OpenAISDKAgent(BaseAgent):
    """
    OpenAI Agents SDK agent that implements BaseAgent interface.
    
    Uses OpenAI's Agent and Runner which:
    1. Uses a language model with tool calling capability
    2. Automatically converts Python functions to FunctionTool
    3. Executes tools and returns results
    4. Supports async execution
    
    This agent:
    1. Loads tools from StableToolBench (via LangChain loader in run_benchmark.py)
    2. Converts LangChain tools to OpenAI SDK function tools
    3. Uses LLM with tool calling via Agent and Runner
    4. Returns answers in StableToolBench format with ExecutionGraph format
    
    Note: Requires OpenAI Agents SDK to be installed. If OpenAI SDK is not available,
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
        Initialize OpenAI Agents SDK agent.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            server_url: URL of the server for tool calls
            temperature: Temperature for LLM (default: 0.0)
            verbose: Whether to print debug information
            max_tools: Maximum number of tools to use per query (default: 120, OpenAI limit is 128)
        """
        # Lazy import OpenAI SDK (avoids circular import issues)
        if not _import_openai_sdk():
            error_msg = (
                "OpenAI Agents SDK is not installed. "
                "Install it with:\n"
                "  pip install openai-agents"
            )
            if OPENAI_SDK_IMPORT_ERROR is not None:
                raise ImportError(error_msg) from OPENAI_SDK_IMPORT_ERROR
            else:
                raise ImportError(error_msg)
        
        self.model = model
        self.server_url = server_url
        self.temperature = temperature
        self.verbose = verbose
        self.max_tools = max_tools
        
        # Tools are pre-selected by ToolSelector and bound here
        # OpenAI SDK does not perform additional filtering
        self.bound_tools: List[FunctionTool] = []  # OpenAI SDK FunctionTool objects
        self.tool_metadata: Dict[str, Dict[str, str]] = {}  # Map tool name to metadata
        self.tools_bound = False
    
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
                # If so, convert them to OpenAI SDK function tools
                if len(tools) > 0:
                    first_tool = tools[0]
                    if LANGCHAIN_AVAILABLE and isinstance(first_tool, StructuredTool):
                        # Convert LangChain tools to OpenAI SDK function tools
                        if self.verbose:
                            print(f"[OpenAISDKAgent] Converting {len(tools)} LangChain tools to OpenAI SDK tools...")
                        self.bound_tools = self._convert_langchain_to_openai_sdk_tools(tools)
                    else:
                        # Assume they're already OpenAI SDK tools
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
                    print(f"[OpenAISDKAgent] Bound {len(self.bound_tools)} pre-selected tools")
            else:
                raise ValueError("tools must be a list of BaseTool objects or FunctionTool objects")
        elif tools_dir is not None:
            # Legacy mode: load all tools from directory (not recommended for benchmarks)
            raise NotImplementedError(
                "Direct tool loading from directory is not implemented for OpenAI SDK. "
                "Use the centralized ToolSelector via run_benchmark.py instead."
            )
        else:
            raise ValueError("Either tools or tools_dir must be provided")
        
        self.tools_bound = True
    
    def _convert_langchain_to_openai_sdk_tools(self, langchain_tools: List) -> List[FunctionTool]:
        """
        Convert LangChain tools to OpenAI SDK function tools.
        
        OpenAI SDK's @function_tool decorator automatically extracts:
        - Function name from function name
        - Description from docstring
        - Schema from function arguments
        
        We create wrapper functions that call the LangChain tool functions.
        
        Args:
            langchain_tools: List of LangChain StructuredTool objects
            
        Returns:
            List of OpenAI SDK FunctionTool objects
        """
        openai_sdk_tools = []
        
        for langchain_tool in langchain_tools:
            # Extract metadata from LangChain tool
            tool_name_raw = langchain_tool.name
            # Sanitize tool name for Python function name (remove special chars including '-', ensure valid identifier)
            # First sanitize for OpenAI API compliance, then replace '-' with '_' for Python function names
            tool_name = sanitize_tool_name(tool_name_raw, max_length=64).replace('-', '_')
            # Remove any remaining consecutive underscores
            while '__' in tool_name:
                tool_name = tool_name.replace('__', '_')
            description = langchain_tool.description
            
            # Get structured metadata if available
            metadata = getattr(langchain_tool, 'metadata', {}) or {}
            tool_name_original = metadata.get('tool_name', '')
            api_name = metadata.get('api_name', '')
            category = metadata.get('category', '')
            original_name = metadata.get('original_name', tool_name)
            
            # Get the original tool function from LangChain tool
            original_func = langchain_tool.func
            
            # Get args_schema from LangChain tool to extract parameter information
            args_schema = langchain_tool.args_schema
            
            # Create a wrapper function that OpenAI SDK can use
            # OpenAI SDK will automatically extract the signature and docstring
            if args_schema and hasattr(args_schema, 'model_fields'):
                # Extract parameters from Pydantic model to create proper function signature
                from inspect import Parameter
                
                # Build function signature from schema
                param_defs = []
                for field_name, field_info in args_schema.model_fields.items():
                    # Get field type and unwrap Optional/Union
                    field_type = field_info.annotation
                    if field_type is None or field_type == Parameter.empty:
                        field_type = str
                    
                    # Unwrap Optional[T] or Union[T, None] to get the actual type
                    origin = getattr(field_type, '__origin__', None)
                    if origin is Union:
                        # Get the non-None type from Union[T, None] or Union[None, T]
                        args = getattr(field_type, '__args__', ())
                        non_none_types = [arg for arg in args if arg is not type(None)]
                        if non_none_types:
                            field_type = non_none_types[0]  # Use first non-None type
                        else:
                            field_type = str  # Fallback if all are None
                    elif hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        # Handle typing.Union (Python < 3.10)
                        args = getattr(field_type, '__args__', ())
                        non_none_types = [arg for arg in args if arg is not type(None)]
                        if non_none_types:
                            field_type = non_none_types[0]
                        else:
                            field_type = str
                    
                    # Check if field has default
                    default_value = Parameter.empty
                    if hasattr(field_info, 'default'):
                        default_value = field_info.default
                    
                    # Check for PydanticUndefined
                    try:
                        from pydantic_core import PydanticUndefined as PUndef
                    except ImportError:
                        try:
                            from pydantic import PydanticUndefined as PUndef
                        except ImportError:
                            PUndef = type('PydanticUndefined', (), {})
                    
                    if (default_value is not Parameter.empty and 
                        default_value is not None and
                        not isinstance(default_value, type(PUndef)) and
                        default_value is not PUndef):
                        param_defs.append((field_name, field_type, default_value))
                    else:
                        param_defs.append((field_name, field_type, None))
                
                # Create function with proper signature using exec
                # This is necessary because OpenAI SDK needs explicit parameters
                def make_typed_function(param_defs, func, desc, tool_name):
                    # Build function signature string
                    param_strs = []
                    for param_name, param_type, default_val in param_defs:
                        type_str = param_type.__name__ if hasattr(param_type, '__name__') else 'str'
                        if default_val is not None and default_val is not Parameter.empty:
                            # Skip PydanticUndefined
                            try:
                                from pydantic_core import PydanticUndefined as PUndef
                            except ImportError:
                                try:
                                    from pydantic import PydanticUndefined as PUndef
                                except ImportError:
                                    PUndef = type('PydanticUndefined', (), {})
                            
                            if isinstance(default_val, type(PUndef)) or default_val is PUndef:
                                param_strs.append(f"{param_name}: {type_str}")
                            elif isinstance(default_val, str):
                                escaped_val = default_val.replace("'", "\\'")
                                param_strs.append(f"{param_name}: {type_str} = '{escaped_val}'")
                            elif isinstance(default_val, (int, float, bool)):
                                param_strs.append(f"{param_name}: {type_str} = {default_val}")
                            elif default_val is None:
                                # For None defaults, use Any to avoid Pydantic Optional issues
                                # Pydantic will still validate based on the actual type at runtime
                                param_strs.append(f"{param_name}: Any = None")
                            else:
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
                    func_code = f"async def {tool_name}({param_str}) -> str:\n    \"\"\"{desc}\"\"\"\n{func_body}"
                    local_vars = {
                        'func': func,
                        'asyncio': asyncio,
                        'json': json,
                        'isinstance': isinstance,
                        'str': str,
                        'dict': dict,
                        'Any': Any,
                        'Parameter': Parameter
                    }
                    exec(func_code, globals(), local_vars)
                    return local_vars[tool_name]
                
                wrapper_func = make_typed_function(param_defs, original_func, description, tool_name)
            else:
                # Fallback: create simple wrapper
                async def wrapper_func(**kwargs) -> str:
                    """Wrapper function for LangChain tool."""
                    try:
                        if asyncio.iscoroutinefunction(original_func):
                            result = await original_func(**kwargs)
                        else:
                            result = original_func(**kwargs)
                        if isinstance(result, dict):
                            return json.dumps(result)
                        return str(result)
                    except Exception as e:
                        return json.dumps({"error": str(e), "response": ""})
                
                wrapper_func.__name__ = tool_name
                wrapper_func.__doc__ = description
            
            # Use @function_tool decorator to create OpenAI SDK tool
            # This automatically extracts signature and docstring
            openai_sdk_tool = function_tool(wrapper_func)
            
            # Store metadata for later extraction
            self.tool_metadata[tool_name] = {
                'tool_name': tool_name_original or tool_name,
                'api_name': api_name,
                'category': category,
                'original_name': original_name
            }
            
            openai_sdk_tools.append(openai_sdk_tool)
        
        return openai_sdk_tools
    
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
            print(f"[OpenAISDKAgent] Processing query: {query[:100]}...")
        
        # OpenAI SDK uses async, so we need to run the async code
        return asyncio.run(self._answer_async(query))
    
    async def _answer_async(self, query: str) -> Dict[str, Any]:
        """
        Async implementation of answer().
        
        Args:
            query: The query string to answer
            
        Returns:
            Dictionary in StableToolBench format with answer and called_apis
        """
        # Create agent with bound tools
        agent = Agent(
            name="assistant",
            model=self.model,
            instructions="Use tools to solve tasks accurately and efficiently.",
            tools=self.bound_tools if len(self.bound_tools) > 0 else None,
        )
        
        if self.verbose:
            print(f"[OpenAISDKAgent] Using {len(self.bound_tools)} tools")
        
        # Execute agent (async)
        try:
            result = await Runner.run(agent, query)
        except Exception as e:
            if self.verbose:
                print(f"[OpenAISDKAgent] Error during execution: {e}")
            raise
        
        # Extract called_apis and build answer_details from result
        called_apis: List[List[str]] = []
        answer_details: List[Dict[str, Any]] = []
        final_answer = ""
        
        # Extract final answer
        if hasattr(result, 'final_output') and result.final_output:
            final_answer = str(result.final_output)
        elif hasattr(result, 'messages') and result.messages:
            # Try to get last message
            last_message = result.messages[-1]
            if hasattr(last_message, 'content'):
                final_answer = str(last_message.content)
        
        # Extract tool calls from result
        # OpenAI SDK stores tool calls in result.new_items or result.messages
        if hasattr(result, 'new_items'):
            for item in result.new_items:
                # Check if this is a tool call item
                if hasattr(item, 'type'):
                    item_type = getattr(item, 'type', None)
                    if item_type == 'tool_call' or 'tool' in str(item_type).lower():
                        # Extract tool name and arguments
                        tool_name = getattr(item, 'tool_name', None) or getattr(item, 'name', None)
                        arguments = getattr(item, 'arguments', None) or getattr(item, 'input', None)
                        output = getattr(item, 'output', None) or getattr(item, 'result', None)
                        
                        if tool_name:
                            # Get metadata for this tool
                            metadata = self.tool_metadata.get(tool_name, {})
                            tool_name_original = metadata.get('tool_name', tool_name)
                            api_name = metadata.get('api_name', '')
                            
                            # Record API call
                            if tool_name_original and api_name:
                                called_apis.append([tool_name_original, api_name])
                            
                            # Parse arguments if needed
                            if isinstance(arguments, str):
                                try:
                                    arguments = json.loads(arguments)
                                except:
                                    arguments = {"raw": arguments}
                            elif arguments is None:
                                arguments = {}
                            
                            # Add to answer_details
                            tool_call_detail = {
                                "role": "tool",
                                "message": json.dumps({
                                    "name": f"{tool_name_original}_{api_name}" if api_name else tool_name_original,
                                    "arguments": arguments,
                                    "response": str(output) if output else ""
                                }),
                                "next": []
                            }
                            answer_details.append(tool_call_detail)
        
        # If no tool calls found, try parsing from messages
        if not called_apis and hasattr(result, 'messages'):
            for message in result.messages:
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    for tool_call in message.tool_calls:
                        tool_name = getattr(tool_call, 'function', {}).get('name', '') if hasattr(tool_call, 'function') else getattr(tool_call, 'name', '')
                        if tool_name:
                            metadata = self.tool_metadata.get(tool_name, {})
                            tool_name_original = metadata.get('tool_name', tool_name)
                            api_name = metadata.get('api_name', '')
                            if tool_name_original and api_name:
                                called_apis.append([tool_name_original, api_name])
        
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
            print(f"[OpenAISDKAgent] Generated answer (tool calls: {len(called_apis)})")
        
        # Return in StableToolBench format
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            },
            "called_apis": called_apis
        }

