"""
Tool loader for LangGraph agent.

Loads tools from StableToolBench and converts them to LangChain tools.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field


def load_tools(tools_dir: str, server_url: str = "http://localhost:8080/virtual") -> List[BaseTool]:
    """
    Load tools from StableToolBench and convert to LangChain tools.
    
    Args:
        tools_dir: Path to StableToolBench/toolenv/tools/ directory
        server_url: URL of the server for tool calls
        
    Returns:
        List of LangChain tools ready to be bound to agent
    """
    tools = []
    
    if not os.path.exists(tools_dir):
        raise FileNotFoundError(f"Tools directory not found: {tools_dir}")
    
    # Load all JSON files from tools_dir
    for filename in os.listdir(tools_dir):
        if not filename.endswith('.json'):
            continue
            
        tool_file = os.path.join(tools_dir, filename)
        try:
            with open(tool_file, 'r', encoding='utf-8') as f:
                tool_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load tool file {filename}: {e}")
            continue
        
        tool_name = tool_data.get('tool_name', '')
        tool_description = tool_data.get('tool_description', '')
        api_list = tool_data.get('api_list', [])
        
        # Extract category (default to "Data" if not found)
        category = tool_data.get('tool_category', 'Data')
        # Normalize category name (replace spaces with underscores)
        category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
        while "__" in category:
            category = category.replace("__", "_")
        
        # Create a LangChain tool for each API
        for api_info in api_list:
            api_name = api_info.get('name', '')
            if not api_name:
                continue
                
            api_description = api_info.get('description', '')
            required_params = api_info.get('required_parameters', [])
            optional_params = api_info.get('optional_parameters', [])
            
            # Create tool function
            tool_func = create_tool_function(
                tool_name=tool_name,
                api_name=api_name,
                category=category,
                server_url=server_url,
                required_params=required_params,
                optional_params=optional_params
            )
            
            # Create tool name in format: tool_name_api_name
            # Sanitize to match OpenAI's pattern: ^[a-zA-Z0-9_-]+$ and max length 64
            # Replace spaces and special chars with underscores
            def sanitize_tool_name(name: str, max_length: int = 64) -> str:
                """
                Sanitize tool name to match OpenAI's pattern and length constraints.
                
                OpenAI requirements:
                - Pattern: ^[a-zA-Z0-9_-]+$
                - Max length: 64 characters
                """
                # Replace spaces and special characters with underscores
                sanitized = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
                # Remove consecutive underscores
                while '__' in sanitized:
                    sanitized = sanitized.replace('__', '_')
                # Remove leading/trailing underscores
                sanitized = sanitized.strip('_')
                # Ensure it starts with alphanumeric
                if sanitized and not sanitized[0].isalnum():
                    sanitized = 'tool_' + sanitized
                # Ensure non-empty
                if not sanitized:
                    sanitized = 'tool'
                # Truncate to max_length if needed
                if len(sanitized) > max_length:
                    sanitized = sanitized[:max_length].rstrip('_')
                    # Ensure it doesn't end with underscore after truncation
                    if sanitized.endswith('_'):
                        sanitized = sanitized[:-1]
                return sanitized
            
            tool_name_full = f"{tool_name}_{api_name}"
            # Sanitize for OpenAI API (must match ^[a-zA-Z0-9_-]+$)
            tool_name_sanitized = sanitize_tool_name(tool_name_full)
            
            # Create description
            full_description = f"{tool_description}\n\n{api_description}".strip()
            if not full_description:
                full_description = f"Call {tool_name} {api_name} API"
            
            # Create LangChain StructuredTool
            # We need to define the input schema based on parameters
            input_schema, param_mapping = create_input_schema(required_params, optional_params, tool_name_sanitized)
            
            # Create wrapper function that maps sanitized parameter names back to original names
            original_tool_func = tool_func
            def mapped_tool_func(**kwargs):
                # Map sanitized parameter names back to original names
                mapped_kwargs = {}
                for sanitized_name, value in kwargs.items():
                    original_name = param_mapping.get(sanitized_name, sanitized_name)
                    mapped_kwargs[original_name] = value
                return original_tool_func(**mapped_kwargs)
            mapped_tool_func.__name__ = tool_func.__name__
            mapped_tool_func.__doc__ = tool_func.__doc__
            
            langchain_tool = StructuredTool.from_function(
                func=mapped_tool_func,
                name=tool_name_sanitized,  # Use sanitized name for OpenAI
                description=full_description,
                args_schema=input_schema
            )
            
            # Store original name in metadata for reference
            # Initialize metadata if it doesn't exist
            if not hasattr(langchain_tool, 'metadata') or langchain_tool.metadata is None:
                langchain_tool.metadata = {}
            langchain_tool.metadata['original_name'] = tool_name_full
            
            tools.append(langchain_tool)
    
    return tools


def create_tool_function(tool_name: str, api_name: str, category: str, 
                        server_url: str, required_params: List[Dict], 
                        optional_params: List[Dict]):
    """
    Create a tool function that calls the server.
    
    Returns a function that can be used with StructuredTool.
    """
    def tool_function(**kwargs):
        # Prepare tool input
        tool_input = {}
        for param in required_params + optional_params:
            param_name = param.get('name', '')
            if param_name in kwargs:
                tool_input[param_name] = kwargs[param_name]
            elif param.get('default') is not None:
                # Use default value if provided
                tool_input[param_name] = param.get('default')
        
        # Call server
        payload = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "tool_input": tool_input,
            "strip": "",
            "toolbench_key": "EMPTY"
        }
        
        try:
            response = requests.post(server_url, json=payload, timeout=60)
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                error = result.get("error", "")
                if error:
                    return f"Error: {error}\nResponse: {response_text}"
                return response_text
            else:
                return f"Error: Server returned status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Set function name and docstring
    tool_function.__name__ = f"{tool_name}_{api_name}".replace(" ", "_")
    tool_function.__doc__ = f"Call {tool_name} {api_name} API"
    
    return tool_function


def create_input_schema(required_params: List[Dict], optional_params: List[Dict], tool_name: str) -> tuple:
    """
    Create a Pydantic model for tool input schema.
    
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
    
    def sanitize_param_name(name: str) -> str:
        """Sanitize parameter name to be valid for Pydantic (no leading underscores)."""
        if name.startswith('_'):
            # Replace leading underscore with 'param_'
            return 'param_' + name[1:] if len(name) > 1 else 'param_'
        return name
    
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
        
        # Convert type string to Python type
        if param_type.upper() in ['STRING', 'STR']:
            python_type = str
        elif param_type.upper() in ['INTEGER', 'INT']:
            python_type = int
        elif param_type.upper() in ['FLOAT', 'DOUBLE', 'NUMBER']:
            python_type = float
        elif param_type.upper() in ['BOOLEAN', 'BOOL']:
            python_type = bool
        else:
            python_type = str  # Default to string
        
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
    
    schema = type(schema_name, (BaseModel,), {
        '__annotations__': annotations,
        **fields
    })
    
    return schema, param_name_mapping

