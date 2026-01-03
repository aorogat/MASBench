"""
Tool loader for CrewAI agent.

Converts StableToolBench tool definitions to CrewAI tools.
Uses shared utilities from utils.tool_utils for framework-agnostic operations.
"""
import sys
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, create_model

# Add single_agent/tool_use to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up: agents/crewai -> agents -> single_agent/tool_use
TOOL_USE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if TOOL_USE_DIR not in sys.path:
    sys.path.insert(0, TOOL_USE_DIR)

from utils.tool_utils import (
    load_tool_definitions,
    ToolDefinition,
    create_tool_function,
    convert_param_type,
    sanitize_param_name
)

try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Create dummy class to allow module import
    class BaseTool:
        pass


def load_tools(tools_dir: str, server_url: str = "http://localhost:8080/virtual") -> List[BaseTool]:
    """
    Load tools from StableToolBench and convert to CrewAI tools.
    
    Uses shared utilities for framework-agnostic operations, then converts
    to CrewAI-specific format.
    
    Args:
        tools_dir: Path to StableToolBench/toolenv/tools/ directory
        server_url: URL of the server for tool calls
        
    Returns:
        List of CrewAI tools ready to be bound to agent
    """
    if not CREWAI_AVAILABLE:
        raise ImportError(
            "CrewAI is not installed. Install it with: pip install 'crewai[tools]'"
        )
    # Load tool definitions using shared utilities
    tool_definitions = load_tool_definitions(tools_dir=tools_dir, server_url=server_url)
    
    # Convert each ToolDefinition to CrewAI BaseTool
    tools = []
    for tool_def in tool_definitions:
        # Create tool function using shared utility
        tool_func = create_tool_function(
            tool_name=tool_def.tool_name,
            api_name=tool_def.api_name,
            category=tool_def.category,
            server_url=server_url,
            required_params=tool_def.required_params,
            optional_params=tool_def.optional_params
        )
        
        # Create CrewAI input schema
        input_schema, param_mapping = create_input_schema(
            required_params=tool_def.required_params,
            optional_params=tool_def.optional_params,
            tool_name=tool_def.sanitized_name
        )
        
        # Create wrapper function that maps sanitized parameter names back to original names
        def mapped_tool_func(**kwargs):
            # Map sanitized parameter names back to original names
            mapped_kwargs = {}
            for sanitized_name, value in kwargs.items():
                original_name = param_mapping.get(sanitized_name, sanitized_name)
                mapped_kwargs[original_name] = value
            return tool_func(**mapped_kwargs)
        
        mapped_tool_func.__name__ = tool_func.__name__
        mapped_tool_func.__doc__ = tool_func.__doc__
        
        # Create CrewAI BaseTool subclass
        class CrewAITool(BaseTool):
            name: str = tool_def.sanitized_name
            description: str = tool_def.full_description
            args_schema: Type[BaseModel] = input_schema
            
            def _run(self, **kwargs) -> str:
                """Execute the tool with mapped parameters."""
                return mapped_tool_func(**kwargs)
        
        # Create instance
        crewai_tool = CrewAITool()
        
        # Store structured metadata for reliable API tracking
        # This avoids fragile string parsing and handles edge cases correctly
        if not hasattr(crewai_tool, 'metadata') or crewai_tool.metadata is None:
            crewai_tool.metadata = {}
        
        # Store structured metadata (preferred - most reliable)
        crewai_tool.metadata['tool_name'] = tool_def.tool_name
        crewai_tool.metadata['api_name'] = tool_def.api_name
        crewai_tool.metadata['category'] = tool_def.category
        
        # Also store original_name for backward compatibility
        crewai_tool.metadata['original_name'] = tool_def.original_name
        
        tools.append(crewai_tool)
    
    return tools


def create_input_schema(required_params: List[Dict], optional_params: List[Dict], tool_name: str) -> tuple:
    """
    Create a Pydantic model for tool input schema (CrewAI-specific).
    
    This is CrewAI-specific code that converts parameter definitions
    to Pydantic models. Other frameworks can implement their own version.
    
    Args:
        required_params: List of required parameter definitions
        optional_params: List of optional parameter definitions
        tool_name: Name of the tool (for schema class name)
        
    Returns:
        Tuple of (Pydantic BaseModel class, parameter name mapping dict)
        The mapping dict maps sanitized names to original names
    """
    # Create fields dictionary
    fields = {}
    annotations = {}
    param_name_mapping = {}  # Map sanitized names to original names
    
    for param in required_params + optional_params:
        original_param_name = param.get('name', '')
        if not original_param_name:
            continue
        
        # Sanitize parameter name (Pydantic doesn't allow leading underscores)
        param_name = sanitize_param_name(original_param_name)
        param_name_mapping[param_name] = original_param_name
            
        param_type = param.get('type', 'STRING')
        param_desc = param.get('description', '')
        default_value = param.get('default')
        
        # Convert type string to Python type using shared utility
        python_type = convert_param_type(param_type)
        
        # Create field
        if param in optional_params or default_value is not None:
            # Optional field
            fields[param_name] = Field(
                default=default_value,
                description=param_desc or f"{original_param_name} parameter"
            )
            annotations[param_name] = Optional[python_type]
        else:
            # Required field
            fields[param_name] = Field(
                description=param_desc or f"{original_param_name} parameter"
            )
            annotations[param_name] = python_type
    
    # Create dynamic Pydantic model
    schema_name = f"{tool_name}Input".replace(" ", "_").replace("-", "_")
    # Remove any leading underscores from schema name too
    if schema_name.startswith('_'):
        schema_name = 'Schema' + schema_name
    
    schema = create_model(schema_name, **annotations, __base__=BaseModel)
    
    return schema, param_name_mapping

