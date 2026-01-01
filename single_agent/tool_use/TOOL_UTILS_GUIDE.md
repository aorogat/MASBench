# Tool Utilities Guide

## Overview

The `utils/tool_utils.py` module provides **framework-agnostic** utilities for loading and processing StableToolBench tools. These utilities can be reused by any framework implementation (LangGraph, CrewAI, OpenAI SDK, etc.).

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│         Shared Utilities (utils/tool_utils.py)           │
│  Framework-agnostic: can be used by any framework      │
├─────────────────────────────────────────────────────────┤
│  • load_tool_definitions() - Loads raw tool definitions  │
│  • ToolDefinition dataclass - Common tool format         │
│  • sanitize_tool_name() - OpenAI name compliance         │
│  • create_tool_function() - Server-calling functions      │
│  • convert_param_type() - Type conversion                 │
│  • sanitize_param_name() - Parameter name sanitization    │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│      Framework-Specific Converters                      │
│  (e.g., agents/langgraph/tool_loader.py)                │
├─────────────────────────────────────────────────────────┤
│  • Converts ToolDefinition → Framework Tool Format      │
│  • LangGraph: ToolDefinition → LangChain StructuredTool   │
│  • CrewAI: ToolDefinition → CrewAI Tool                 │
│  • OpenAI SDK: ToolDefinition → OpenAI Function           │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. `ToolDefinition` (Dataclass)

Common format for tool definitions that all frameworks can use:

```python
@dataclass
class ToolDefinition:
    tool_name: str
    api_name: str
    tool_description: str
    api_description: str
    category: str
    required_params: List[Dict[str, Any]]
    optional_params: List[Dict[str, Any]]
    original_name: str  # Format: "tool_name_api_name"
    sanitized_name: str  # Sanitized for OpenAI compliance
```

### 2. `load_tool_definitions()`

Loads tool definitions from StableToolBench JSON files:

```python
from utils.tool_utils import load_tool_definitions

# Load all tool definitions
tool_defs = load_tool_definitions(
    tools_dir="StableToolBench/toolenv/tools/",
    server_url="http://localhost:8080/virtual"
)

# Returns List[ToolDefinition]
```

### 3. `create_tool_function()`

Creates a callable function that calls the server:

```python
from utils.tool_utils import create_tool_function

tool_func = create_tool_function(
    tool_name="TheClique",
    api_name="Transfermarkt search",
    category="Data",
    server_url="http://localhost:8080/virtual",
    required_params=[...],
    optional_params=[...]
)

# Call the function
result = tool_func(query="Lionel Messi")
```

### 4. `sanitize_tool_name()`

Sanitizes tool names for OpenAI compliance:

```python
from utils.tool_utils import sanitize_tool_name

original = "SEO API - Get Backlinks_GetTopBacklinks"
sanitized = sanitize_tool_name(original)
# Returns: "SEO_API_-_Get_Backlinks_GetTopBacklinks" (max 64 chars)
```

## Usage Examples

### Example 1: LangGraph Implementation

```python
# agents/langgraph/tool_loader.py
from utils.tool_utils import load_tool_definitions, create_tool_function
from langchain_core.tools import StructuredTool

def load_tools(tools_dir: str, server_url: str) -> List[BaseTool]:
    # Load using shared utilities
    tool_defs = load_tool_definitions(tools_dir, server_url)
    
    # Convert to LangChain format
    tools = []
    for tool_def in tool_defs:
        tool_func = create_tool_function(...)
        langchain_tool = StructuredTool.from_function(
            func=tool_func,
            name=tool_def.sanitized_name,
            description=tool_def.full_description
        )
        tools.append(langchain_tool)
    
    return tools
```

### Example 2: CrewAI Implementation (Future)

```python
# agents/crewai/tool_loader.py
from utils.tool_utils import load_tool_definitions, create_tool_function
from crewai_tools import tool

def load_tools(tools_dir: str, server_url: str) -> List[Tool]:
    # Load using shared utilities
    tool_defs = load_tool_definitions(tools_dir, server_url)
    
    # Convert to CrewAI format
    tools = []
    for tool_def in tool_defs:
        tool_func = create_tool_function(...)
        
        @tool(tool_def.full_description)
        def crewai_tool(**kwargs):
            return tool_func(**kwargs)
        
        tools.append(crewai_tool)
    
    return tools
```

### Example 3: OpenAI SDK Implementation (Future)

```python
# agents/openai/tool_loader.py
from utils.tool_utils import load_tool_definitions, create_tool_function
from openai import Function

def load_tools(tools_dir: str, server_url: str) -> List[Function]:
    # Load using shared utilities
    tool_defs = load_tool_definitions(tools_dir, server_url)
    
    # Convert to OpenAI Function format
    functions = []
    for tool_def in tool_defs:
        # Create OpenAI function definition
        function_def = {
            "name": tool_def.sanitized_name,
            "description": tool_def.full_description,
            "parameters": {
                "type": "object",
                "properties": {...}  # Convert from tool_def.required_params
            }
        }
        functions.append(function_def)
    
    return functions
```

## Benefits

1. **Code Reuse**: Framework-agnostic utilities eliminate duplication
2. **Consistency**: All frameworks use the same tool loading logic
3. **Maintainability**: Bug fixes and improvements benefit all frameworks
4. **OpenAI Compliance**: Tool name sanitization ensures compatibility
5. **Type Safety**: `ToolDefinition` provides a clear interface

## Migration Guide

### For New Framework Implementations

1. **Import shared utilities**:
   ```python
   from utils.tool_utils import (
       load_tool_definitions,
       ToolDefinition,
       create_tool_function,
       sanitize_tool_name
   )
   ```

2. **Load tool definitions**:
   ```python
   tool_defs = load_tool_definitions(tools_dir, server_url)
   ```

3. **Convert to framework format**:
   ```python
   for tool_def in tool_defs:
       # Convert ToolDefinition to your framework's tool format
       framework_tool = convert_to_framework_format(tool_def)
   ```

4. **Use `create_tool_function()`** for server calls:
   ```python
   tool_func = create_tool_function(
       tool_name=tool_def.tool_name,
       api_name=tool_def.api_name,
       category=tool_def.category,
       server_url=server_url,
       required_params=tool_def.required_params,
       optional_params=tool_def.optional_params
   )
   ```

## Files

- **`utils/tool_utils.py`**: Shared utilities (framework-agnostic)
- **`agents/langgraph/tool_loader.py`**: LangGraph-specific converter (uses shared utilities)
- **`agents/{framework}/tool_loader.py`**: Framework-specific converters (future)

## See Also

- **`TOOL_SELECTION_ARCHITECTURE.md`**: Tool selection system
- **`README.md`**: Overall project documentation
- **`agents/base_agent.py`**: BaseAgent interface

