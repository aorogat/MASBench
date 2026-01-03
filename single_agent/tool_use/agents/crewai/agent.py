"""
CrewAI Agent Implementation

Implements BaseAgent interface using CrewAI framework for tool-calling agents.
Uses CrewAI's Agent, Task, and Crew classes for single-agent execution.

Important: Tools are pre-selected by MASBench ToolSelector.
CrewAI does not perform additional filtering - it uses whatever tools are provided.

Note on Tool Limits:
- CrewAI uses the underlying LLM (e.g., OpenAI) which has a limit of 128 tools per request
- CrewAI sends ALL bound tools to the LLM in each request
- CrewAI has no built-in tool filtering mechanism
- This is why centralized tool selection (ToolSelector) is essential for CrewAI
"""
import os
import json
from typing import Dict, Any, List, Optional, Union

try:
    from crewai import Agent, Task, Crew
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create dummy classes to allow module import
    class Agent:
        pass
    class Task:
        pass
    class Crew:
        pass
    class BaseTool:
        pass

# Check if LangChain tools are being passed (from run_benchmark.py)
try:
    from langchain_core.tools import BaseTool as LangChainBaseTool, StructuredTool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LangChainBaseTool = None
    StructuredTool = None

from ..base_agent import BaseAgent
from .tool_loader import load_tools


class CrewAIAgent(BaseAgent):
    """
    CrewAI agent that implements BaseAgent interface.
    
    Uses CrewAI's Agent, Task, and Crew classes:
    1. Creates a single agent with tools
    2. Creates a task for the query
    3. Executes the crew to get the answer
    4. Extracts tool calls and returns in StableToolBench format
    
    This agent:
    1. Loads tools from StableToolBench
    2. Uses LLM with tool calling via CrewAI
    3. Returns answers in StableToolBench format with ExecutionGraph format
    
    Note: Requires CrewAI to be installed. If CrewAI is not available,
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
        Initialize CrewAI agent.
        
        Args:
            model: OpenAI model name (default: gpt-4o-mini)
            server_url: URL of the server for tool calls
            temperature: Temperature for LLM (default: 0.0)
            verbose: Whether to print debug information
            max_tools: Maximum number of tools to use per query (default: 120, OpenAI limit is 128)
        """
        if not CREWAI_AVAILABLE:
            raise ImportError(
                "CrewAI is not installed. Install it with: pip install 'crewai[tools]'"
            )
        self.model = model
        self.server_url = server_url
        self.temperature = temperature
        self.verbose = verbose
        self.max_tools = max_tools
        
        # Tools are pre-selected by ToolSelector and bound here
        # CrewAI does not perform additional filtering
        self.bound_tools: List[BaseTool] = []  # Pre-selected tools from ToolSelector
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
                # Check if tools are LangChain tools (from run_benchmark.py)
                # If so, convert them to CrewAI tools
                # Check by type name to handle import issues
                if len(tools) > 0:
                    first_tool = tools[0]
                    tool_type_name = type(first_tool).__name__
                    tool_module = type(first_tool).__module__
                    
                    # Check if it's a LangChain tool
                    is_langchain_tool = (
                        LANGCHAIN_AVAILABLE and isinstance(first_tool, (LangChainBaseTool, StructuredTool))
                    ) or (
                        'langchain' in tool_module.lower() or 
                        tool_type_name in ['StructuredTool', 'BaseTool']
                    )
                    
                    if is_langchain_tool:
                        if self.verbose:
                            print(f"[CrewAIAgent] Converting {len(tools)} LangChain tools to CrewAI tools...")
                        self.bound_tools = self._convert_langchain_to_crewai_tools(tools)
                    else:
                        # Assume they're already CrewAI tools
                        self.bound_tools = tools
                else:
                    self.bound_tools = tools
                
                # Fail loudly if tool count exceeds max_tools (benchmark integrity)
                if len(self.bound_tools) > self.max_tools:
                    raise RuntimeError(
                        f"ToolSelector returned {len(self.bound_tools)} tools, "
                        f"exceeds max_tools={self.max_tools}. "
                        f"This indicates a bug in tool selection or configuration."
                    )
                
                if self.verbose:
                    print(f"[CrewAIAgent] Bound {len(self.bound_tools)} pre-selected tools")
            else:
                raise ValueError("tools must be a list of BaseTool objects")
        elif tools_dir is not None:
            # Legacy mode: load all tools from directory (not recommended for benchmarks)
            if self.verbose:
                print(f"[CrewAIAgent] Loading tools from {tools_dir}...")
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
                print(f"[CrewAIAgent] Loaded {len(self.bound_tools)} tools")
        else:
            raise ValueError("Either tools or tools_dir must be provided")
        
        self.tools_bound = True
    
    def _convert_langchain_to_crewai_tools(self, langchain_tools: List) -> List[BaseTool]:
        """
        Convert LangChain tools to CrewAI tools.
        
        This is needed because run_benchmark.py loads tools using LangChain's loader,
        but CrewAI needs its own BaseTool instances.
        
        Args:
            langchain_tools: List of LangChain StructuredTool objects
            
        Returns:
            List of CrewAI BaseTool objects
        """
        from typing import Type
        from pydantic import BaseModel, Field, create_model
        from utils.tool_utils import sanitize_param_name
        
        crewai_tools = []
        
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
            # LangChain tools have a func attribute
            original_func = langchain_tool.func
            
            # Get args_schema from LangChain tool
            langchain_schema = langchain_tool.args_schema
            
            # Create CrewAI input schema from LangChain schema
            if langchain_schema:
                # Extract fields from LangChain schema
                fields = {}
                annotations = {}
                
                if hasattr(langchain_schema, 'model_fields'):
                    for field_name, field_info in langchain_schema.model_fields.items():
                        # Get field type and description
                        # Pydantic v2 uses FieldInfo which has different attributes
                        field_type = field_info.annotation
                        
                        # Get description - can be in multiple places
                        field_desc = ""
                        if hasattr(field_info, 'description'):
                            field_desc = field_info.description or ""
                        elif hasattr(field_info, 'json_schema_extra') and isinstance(field_info.json_schema_extra, dict):
                            field_desc = field_info.json_schema_extra.get('description', '')
                        
                        # Get default value - Pydantic v2 FieldInfo structure
                        default_value = ...
                        if hasattr(field_info, 'default'):
                            default_value = field_info.default
                        
                        # Check if field is optional (has default or is Optional type)
                        is_optional = (
                            default_value is not ... and default_value is not None
                        ) or (
                            str(field_type).startswith('Optional') or 
                            'Union[' in str(field_type) or
                            'None' in str(field_type)
                        )
                        
                        if default_value is not ... and default_value is not None:
                            # Field has a default value
                            fields[field_name] = Field(default=default_value, description=field_desc)
                            # Make type Optional if not already
                            if not is_optional:
                                annotations[field_name] = Optional[field_type]
                            else:
                                annotations[field_name] = field_type
                        else:
                            # Required field
                            fields[field_name] = Field(description=field_desc)
                            annotations[field_name] = field_type
                
                # Create Pydantic model for CrewAI
                schema_name = f"{tool_name}Input".replace(" ", "_").replace("-", "_")
                if schema_name.startswith('_'):
                    schema_name = 'Schema' + schema_name
                
                crewai_schema = create_model(schema_name, **annotations, __base__=BaseModel)
            else:
                # No schema - create empty schema
                crewai_schema = create_model(f"{tool_name}Input", __base__=BaseModel)
            
            # Create CrewAI BaseTool subclass
            # Capture variables in closure to avoid NameError
            tool_name_capture = tool_name
            description_capture = description
            original_func_capture = original_func
            
            # Store metadata in a dict that we'll attach after creation
            tool_metadata = {
                'tool_name': tool_name_original or tool_name,
                'api_name': api_name,
                'category': category,
                'original_name': original_name
            }
            
            class CrewAITool(BaseTool):
                name: str = tool_name_capture
                description: str = description_capture
                args_schema: Type[BaseModel] = crewai_schema
                
                def _run(self, **kwargs) -> str:
                    """Execute the tool by calling the original LangChain tool function."""
                    try:
                        result = original_func_capture(**kwargs)
                        # Convert result to string if needed
                        if isinstance(result, dict):
                            return json.dumps(result)
                        return str(result)
                    except Exception as e:
                        return json.dumps({"error": str(e), "response": ""})
            
            # Create instance
            crewai_tool = CrewAITool()
            
            # Store structured metadata as a private attribute (not a Pydantic field)
            # This allows us to access it later without violating Pydantic's model structure
            object.__setattr__(crewai_tool, '_masbench_metadata', tool_metadata)
            
            crewai_tools.append(crewai_tool)
        
        return crewai_tools
    
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
            print(f"[CrewAIAgent] Processing query: {query[:100]}...")
        
        # Validate tools before creating agent
        if len(self.bound_tools) == 0:
            if self.verbose:
                print(f"[CrewAIAgent] Warning: No tools bound, agent will not be able to use tools")
        else:
            # Verify tools are valid CrewAI BaseTool instances
            invalid_tools = [t for t in self.bound_tools if not isinstance(t, BaseTool)]
            if invalid_tools:
                raise ValueError(
                    f"Invalid tools detected: {len(invalid_tools)} tools are not CrewAI BaseTool instances. "
                    f"This may cause 'tool_choice is none' errors."
                )
        
        # Create agent with bound tools
        # CrewAI requires role, goal, and backstory for agents
        # Note: CrewAI internally determines tool_choice based on tools presence
        # If tools are provided but tool_choice="none" is set, this is a CrewAI bug
        agent = Agent(
            role="Problem Solver",
            goal="Answer user queries accurately using available tools",
            backstory="You are an expert problem solver who uses tools effectively to answer questions.",
            tools=self.bound_tools if len(self.bound_tools) > 0 else None,  # Explicitly pass None if empty
            verbose=self.verbose,
            llm=self._get_llm()
        )
        
        # Create task for the query
        task = Task(
            description=query,
            expected_output="A clear and accurate answer to the user's query.",
            agent=agent
        )
        
        # Create crew with single agent and task
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=self.verbose
        )
        
        # Execute crew
        if self.verbose:
            print(f"[CrewAIAgent] Using {len(self.bound_tools)} tools")
        
        # Execute with error handling for CrewAI's tool_choice bug
        try:
            result = crew.kickoff()
        except Exception as e:
            error_str = str(e)
            # Check for the specific "tool_choice is none" error
            if "tool_choice is none" in error_str.lower() or "tool_use_failed" in error_str.lower():
                if self.verbose:
                    print(f"[CrewAIAgent] Warning: CrewAI tool_choice error detected")
                    print(f"[CrewAIAgent] This is a known CrewAI issue when tools are bound but tool_choice is incorrectly set to 'none'")
                    print(f"[CrewAIAgent] Tools bound: {len(self.bound_tools)}, all valid: {all(isinstance(t, BaseTool) for t in self.bound_tools)}")
                # Re-raise with more context
                raise RuntimeError(
                    f"CrewAI tool_choice error: Tools are bound ({len(self.bound_tools)} tools) but CrewAI set tool_choice='none'. "
                    f"This is a CrewAI framework bug. Original error: {error_str}"
                ) from e
            # Re-raise other errors as-is
            raise
        
        # Extract called_apis and build answer_details
        # CrewAI doesn't provide direct access to tool calls in the result
        # We need to parse the result or use CrewAI's execution logs
        called_apis: List[List[str]] = []
        answer_details: List[Dict[str, Any]] = []
        final_answer = ""
        
        # Extract final answer from result
        if isinstance(result, str):
            final_answer = result
        elif hasattr(result, 'raw'):
            final_answer = str(result.raw)
        elif hasattr(result, 'content'):
            final_answer = str(result.content)
        else:
            final_answer = str(result)
        
        # Try to extract tool calls from crew execution
        # CrewAI may store execution details in the crew or task objects
        # Check if crew has execution history
        if hasattr(crew, 'tasks'):
            for task in crew.tasks:
                # Check if task has execution details
                if hasattr(task, 'output'):
                    # Task output might contain tool call information
                    pass
                # Check if task's agent has execution history
                if hasattr(task, 'agent') and hasattr(task.agent, 'last_execution'):
                    # Try to extract from agent's last execution
                    pass
        
        # Alternative: Check if crew has execution logs
        if hasattr(crew, 'execution_logs'):
            # Parse execution logs for tool calls
            pass
        
        # TODO: CrewAI tool call extraction needs enhancement
        # Current limitation: CrewAI doesn't expose tool calls directly in the result
        # Possible solutions:
        # 1. Use CrewAI callbacks to capture tool calls during execution
        # 2. Parse execution logs if available
        # 3. Hook into the underlying LLM's tool call events
        # For now, we return empty called_apis - run_benchmark will parse from answer_details
        # This means API call scoring may be limited until we enhance tool call extraction
        
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
            print(f"[CrewAIAgent] Generated answer (tool calls: {len(called_apis)})")
        
        # Return in StableToolBench format
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            },
            "called_apis": called_apis
        }
    
    def _get_llm(self):
        """
        Get LLM instance for CrewAI.
        
        CrewAI uses different LLM configuration than LangChain.
        """
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=self.model, temperature=self.temperature)
        except ImportError:
            raise ImportError(
                "langchain-openai is required for CrewAI. Install it with: pip install langchain-openai"
            )

