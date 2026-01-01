"""
Shared tool utilities for loading and processing StableToolBench tools.

This module provides framework-agnostic functions for:
- Loading tool definitions from StableToolBench JSON files
- Sanitizing tool names (OpenAI compliance)
- Creating tool functions that call the server
- Parameter type conversion

Framework-specific code should import these utilities and convert
the common format to their framework's tool format.
"""
import os
import json
import requests
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass


@dataclass
class ToolDefinition:
    """
    Common format for a tool definition.
    
    Frameworks can convert this to their own tool format.
    """
    tool_name: str
    api_name: str
    tool_description: str
    api_description: str
    category: str
    required_params: List[Dict[str, Any]]
    optional_params: List[Dict[str, Any]]
    original_name: str  # Format: "tool_name_api_name"
    sanitized_name: str  # Sanitized for OpenAI compliance
    
    @property
    def full_description(self) -> str:
        """Combined tool and API description."""
        desc = f"{self.tool_description}\n\n{self.api_description}".strip()
        if not desc:
            desc = f"Call {self.tool_name} {self.api_name} API"
        return desc


def sanitize_tool_name(name: str, max_length: int = 64) -> str:
    """
    Sanitize tool name to match OpenAI's pattern and length constraints.
    
    OpenAI requirements:
    - Pattern: ^[a-zA-Z0-9_-]+$
    - Max length: 64 characters
    
    Args:
        name: Original tool name
        max_length: Maximum length (default: 64 for OpenAI)
        
    Returns:
        Sanitized tool name
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


def normalize_category(category: str) -> str:
    """
    Normalize category name (replace spaces/special chars with underscores).
    
    Args:
        category: Original category name
        
    Returns:
        Normalized category name
    """
    normalized = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def create_tool_function(
    tool_name: str,
    api_name: str,
    category: str,
    server_url: str,
    required_params: List[Dict[str, Any]],
    optional_params: List[Dict[str, Any]]
) -> Callable:
    """
    Create a tool function that calls the server.
    
    This function is framework-agnostic and can be used by any framework
    that needs to call the StableToolBench server.
    
    Args:
        tool_name: Name of the tool
        api_name: Name of the API
        category: Tool category
        server_url: URL of the server
        required_params: List of required parameter definitions
        optional_params: List of optional parameter definitions
        
    Returns:
        A callable function that accepts kwargs and calls the server
    """
    def tool_function(**kwargs):
        """
        Tool function that calls the server.
        
        Accepts parameters as defined in required_params and optional_params.
        Returns the server response as a string.
        """
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


def load_tool_definitions(
    tools_dir: str,
    server_url: str = "http://localhost:8080/virtual"
) -> List[ToolDefinition]:
    """
    Load tool definitions from StableToolBench JSON files.
    
    This is a framework-agnostic function that loads raw tool definitions.
    Frameworks should convert these ToolDefinition objects to their own format.
    
    Args:
        tools_dir: Path to StableToolBench/toolenv/tools/ directory
        server_url: URL of the server (used for creating tool functions)
        
    Returns:
        List of ToolDefinition objects
    """
    tool_definitions = []
    
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
        category = normalize_category(category)
        
        # Create a ToolDefinition for each API
        for api_info in api_list:
            api_name = api_info.get('name', '')
            if not api_name:
                continue
                
            api_description = api_info.get('description', '')
            required_params = api_info.get('required_parameters', [])
            optional_params = api_info.get('optional_parameters', [])
            
            # Create tool name in format: tool_name_api_name
            tool_name_full = f"{tool_name}_{api_name}"
            # Sanitize for OpenAI API compliance
            tool_name_sanitized = sanitize_tool_name(tool_name_full)
            
            # Create ToolDefinition
            tool_def = ToolDefinition(
                tool_name=tool_name,
                api_name=api_name,
                tool_description=tool_description,
                api_description=api_description,
                category=category,
                required_params=required_params,
                optional_params=optional_params,
                original_name=tool_name_full,
                sanitized_name=tool_name_sanitized
            )
            
            tool_definitions.append(tool_def)
    
    return tool_definitions


def convert_param_type(param_type: str) -> type:
    """
    Convert parameter type string to Python type.
    
    Args:
        param_type: Type string from StableToolBench (e.g., "STRING", "INTEGER")
        
    Returns:
        Python type (str, int, float, bool)
    """
    param_type_upper = param_type.upper()
    if param_type_upper in ['STRING', 'STR']:
        return str
    elif param_type_upper in ['INTEGER', 'INT']:
        return int
    elif param_type_upper in ['FLOAT', 'DOUBLE', 'NUMBER']:
        return float
    elif param_type_upper in ['BOOLEAN', 'BOOL']:
        return bool
    else:
        return str  # Default to string


def sanitize_param_name(name: str) -> str:
    """
    Sanitize parameter name to be valid for frameworks that don't allow
    leading underscores (e.g., Pydantic).
    
    Args:
        name: Original parameter name
        
    Returns:
        Sanitized parameter name
    """
    if name.startswith('_'):
        # Replace leading underscore with 'param_'
        return 'param_' + name[1:] if len(name) > 1 else 'param_'
    return name

