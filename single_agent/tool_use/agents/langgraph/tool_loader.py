"""
Tool loader for LangGraph agent.

Converts StableToolBench tool definitions to LangChain tools.
Uses shared utilities from utils.tool_utils for framework-agnostic operations.
"""
import sys
import os
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model

# Add single_agent/tool_use to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up: agents/langgraph -> agents -> single_agent/tool_use
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


def load_tools(tools_dir: str, server_url: str = "http://localhost:8080/virtual") -> List[BaseTool]:
    """
    Load tools from StableToolBench and convert to LangChain tools.
    
    Uses shared utilities for framework-agnostic operations, then converts
    to LangChain-specific format.
    
    Args:
        tools_dir: Path to StableToolBench/toolenv/tools/ directory
        server_url: URL of the server for tool calls
        
    Returns:
        List of LangChain tools ready to be bound to agent
    """
    # Load tool definitions using shared utilities
    tool_definitions = load_tool_definitions(tools_dir=tools_dir, server_url=server_url)
    
    # Convert each ToolDefinition to LangChain StructuredTool
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
        
        # Create LangChain input schema
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
        
        # Create LangChain StructuredTool
        langchain_tool = StructuredTool.from_function(
            func=mapped_tool_func,
            name=tool_def.sanitized_name,  # Use sanitized name for OpenAI
            description=tool_def.full_description,
            args_schema=input_schema
        )
        
        # Store original name in metadata for reference
        if not hasattr(langchain_tool, 'metadata') or langchain_tool.metadata is None:
            langchain_tool.metadata = {}
        langchain_tool.metadata['original_name'] = tool_def.original_name
        
        tools.append(langchain_tool)
    
    return tools


# Note: create_tool_function is now imported from utils.tool_utils


def create_input_schema(required_params: List[Dict], optional_params: List[Dict], tool_name: str) -> tuple:
    """
    Create a Pydantic model for tool input schema (LangChain-specific).
    
    This is LangChain-specific code that converts parameter definitions
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

