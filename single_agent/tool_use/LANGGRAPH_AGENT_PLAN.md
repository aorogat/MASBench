# LangGraph Agent Implementation Plan

## Overview

Create a general agent interface and a LangGraph agent implementation that:
1. Defines a base interface that all agents must implement
2. Loads tools from StableToolBench (`toolenv/tools/`)
3. Converts them to LangChain tools that call our server (`http://localhost:8080/virtual`)
4. Implements the `answer(query: str) -> Dict[str, Any]` interface for `run_benchmark.py`
5. Returns results in StableToolBench format (ExecutionGraph format for `answer_details`)

---

## File Structure

```
single_agent/tool_use/
├── test_langgraph.py          # Main test file (uses run_benchmark)
└── agents/
    ├── __init__.py            # Package init, exports BaseAgent and framework agents
    ├── base_agent.py          # Base interface for all agents (answer + bind_tools)
    └── langgraph/             # LangGraph-specific implementation
        ├── __init__.py        # Exports LangGraphAgent
        ├── agent.py           # LangGraphAgent implementation
        └── tool_loader.py     # Helper to load and convert tools for LangGraph
```

---

## Implementation Steps

### Step 1: Create Agent Directory Structure and Base Interface

Create `agents/` directory with:
- `agents/__init__.py` - Make it a package, export `BaseAgent` and framework agents
- `agents/base_agent.py` - Abstract base class defining the agent interface
- `agents/langgraph/` - LangGraph-specific implementation folder
  - `agents/langgraph/__init__.py` - Exports `LangGraphAgent`
  - `agents/langgraph/agent.py` - LangGraphAgent implementation (inherits from BaseAgent)
  - `agents/langgraph/tool_loader.py` - Helper to load and convert tools for LangGraph

### Step 2: Base Agent Interface (`base_agent.py`)

**2.1 Abstract Base Class**
- Define `BaseAgent` as an abstract base class (ABC)
- Require implementation of:
  - `answer(query: str) -> Dict[str, Any]` method
  - `bind_tools(tools_dir: str, server_url: str) -> None` method
- Define expected return format
- Add optional helper methods if needed

**2.2 `bind_tools` Method**
- Loads tools from `StableToolBench/toolenv/tools/` directory
- Converts them to framework-specific format
- Binds them to the agent
- Parameters:
  - `tools_dir`: Path to `StableToolBench/toolenv/tools/` directory
  - `server_url`: URL of the server (default: `http://localhost:8080/virtual`)

**2.3 Interface Code**
See "Base Agent Interface Code" section below for full implementation.

### Step 3: LangGraph Tool Loader (`agents/langgraph/tool_loader.py`)

**3.1 Load Tools from StableToolBench**
- Load tool definitions from `StableToolBench/toolenv/tools/*.json`
- Each JSON file contains:
  - `tool_name`: Name of the tool
  - `tool_description`: Description of the tool
  - `api_list`: List of APIs with:
    - `name`: API name
    - `description`: API description
    - `required_parameters`: List of required parameters
    - `optional_parameters`: List of optional parameters

**3.2 Convert to LangChain Tools**
- For each API, create a LangChain `@tool` function
- Tool function should:
  - Accept parameters as defined in `required_parameters` and `optional_parameters`
  - Call the server at `http://localhost:8080/virtual` with:
    ```python
    {
        "category": tool_category,  # Extract from tool file or default
        "tool_name": tool_name,
        "api_name": api_name,
        "tool_input": {param_name: param_value, ...},
        "strip": "",
        "toolbench_key": "EMPTY"
    }
  - Return the response from server

**3.3 Tool Naming**
- Format: `{tool_name}_{api_name}` (e.g., "TheClique_Transfermarkt search")
- This matches the format expected by `run_benchmark.py` for API call score calculation

**3.4 Tool Loader Function**
- Create `load_tools(tools_dir: str, server_url: str) -> List[Tool]` function
- Returns list of LangChain tools ready to be bound to the agent

### Step 4: LangGraph Agent Implementation

**4.1 State Definition**
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    answer_details: List[Dict[str, Any]]  # Track tool calls for ExecutionGraph format
```

**4.2 Graph Nodes**
- **LLM Node**: 
  - Takes messages + tools
  - Calls LLM with tool calling enabled
  - Returns tool calls or final answer
  
- **Tool Node**:
  - Executes tool calls
  - Calls server at `http://localhost:8080/virtual`
  - Records tool call in `answer_details` (ExecutionGraph format)
  - Returns ToolMessage with response

- **Finish Node**:
  - When LLM returns final answer (no tool calls)
  - Adds "Finish" call to `answer_details`
  - Returns final answer

**4.3 Graph Edges**
```
START -> LLM Node
LLM Node -> Tool Node (if tool calls)
LLM Node -> Finish Node (if final answer)
Tool Node -> LLM Node (continue conversation)
Finish Node -> END
```

**4.4 ExecutionGraph Format for `answer_details`**
Each tool call should be recorded as:
```python
{
    "role": "tool",
    "message": json.dumps({
        "name": "tool_name_api_name",
        "arguments": {param_name: param_value, ...},
        "response": server_response.get("response", "")
    }),
    "next": []
}
```

Finish call:
```python
{
    "role": "tool",
    "message": json.dumps({
        "name": "Finish",
        "arguments": {
            "return_type": "give_answer",
            "final_answer": final_answer_text
        },
        "response": ""
    }),
    "next": []
}
```

### Step 5: LangGraph Agent Implementation Details

**5.1 Inherit from BaseAgent**
- `LangGraphAgent(BaseAgent)` - Inherit from base interface
- Implement both `answer()` and `bind_tools()` methods as required by interface

**5.1.1 `bind_tools(tools_dir: str, server_url: str) -> None` Method**
- Call `tool_loader.load_tools()` to load and convert tools
- Store tools in agent instance
- Bind tools to LangGraph graph/LLM

**5.2 `answer(query: str) -> Dict[str, Any]` Method**
- Initialize graph state with user query
- Run graph until Finish node is reached
- Extract final answer from last LLM message
- Track all APIs used during execution (save as list of [tool_name, api_name] pairs)
- Format return value according to BaseAgent interface:
  ```python
  {
      "answer": {
          "final_answer": str,
          "answer_details": List[Dict]  # ExecutionGraph format
      },
      "called_apis": [  # List of APIs used (recommended)
          ["tool_name", "api_name"],
          ...
      ]
  }
  ```

**5.3 Tool Call Tracking**
- Track all tool calls during execution
- Convert LangChain ToolMessage to ExecutionGraph format
- Maintain order of tool calls
- **Save list of APIs used**: Keep a list of `[tool_name, api_name]` pairs for each tool call
- Return `called_apis` in the answer dict for easier API tracking by run_benchmark

### Step 6: Test File (`test_langgraph.py`)

**6.1 Structure**
```python
from run_benchmark import run_benchmark
from agents.langgraph import LangGraphAgent
import os

# Create agent
agent = LangGraphAgent(
    model="gpt-4o-mini",
    server_url="http://localhost:8080/virtual"
)

# Bind tools from StableToolBench
tools_dir = os.path.join("StableToolBench", "toolenv", "tools")
agent.bind_tools(tools_dir=tools_dir, server_url="http://localhost:8080/virtual")

# Run benchmark
results = run_benchmark(
    agent=agent,
    test_set="G1_instruction",
    max_queries=5,
    agent_name="langgraph_agent"
)
```

---

## Key Implementation Details

### 1. Tool Loading
- Load all JSON files from `StableToolBench/toolenv/tools/`
- Parse tool structure
- Extract category (may need to infer from tool name or use default "Data")

### 2. Server Communication
- Use `requests.post()` to call server
- Handle errors gracefully
- Extract `response` field from server response

### 3. LangGraph State Management
- Keep track of conversation messages
- Track tool calls in `answer_details` list
- Convert LangChain messages to ExecutionGraph format

### 4. Finish Detection
- Detect when LLM returns final answer (no tool calls)
- Extract final answer text
- Add Finish call to `answer_details`

### 5. Error Handling
- Handle server errors
- Handle tool call errors
- Return error message in final answer if needed

---

## Dependencies

```python
# Required packages
- langgraph
- langchain
- langchain-openai
- requests
- openai
```

---

## Testing Strategy

1. **Unit Test**: Test tool loading and conversion
2. **Integration Test**: Test single tool call to server
3. **End-to-End Test**: Run `test_langgraph.py` with `run_benchmark`

---

## Potential Challenges

1. **Tool Count**: ~600+ tools - may need to limit or filter
2. **Tool Naming**: Ensure consistent naming format
3. **Parameter Handling**: Handle required vs optional parameters
4. **State Management**: Track tool calls correctly in ExecutionGraph format
5. **Finish Detection**: Reliably detect when agent is done

---

## Next Steps

1. ✅ Create plan (this document)
2. ⬜ Implement `agents/base_agent.py` (interface with `answer()` and `bind_tools()`)
3. ⬜ Implement `agents/langgraph/tool_loader.py` (tool loading helper)
4. ⬜ Implement `agents/langgraph/agent.py` (LangGraphAgent with both methods)
5. ⬜ Implement `agents/langgraph/__init__.py` (exports)
6. ⬜ Update `agents/__init__.py` (exports)
7. ⬜ Implement `test_langgraph.py`
8. ⬜ Test with single query
9. ⬜ Test with `run_benchmark`

---

## Base Agent Interface Code

### `agents/base_agent.py`

```python
"""
Base Agent Interface

All agents must implement this interface to work with run_benchmark.py.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    All agents must implement the answer() method that:
    1. Takes a query string as input
    2. Returns a dictionary in StableToolBench format
    3. Includes answer_details in ExecutionGraph format for API call scoring
    4. Optionally includes called_apis list for easier API tracking
    """
    
    @abstractmethod
    def bind_tools(self, tools_dir: str, server_url: str = "http://localhost:8080/virtual") -> None:
        """
        Load and bind tools from StableToolBench to the agent.
        
        Args:
            tools_dir: Path to StableToolBench/toolenv/tools/ directory
            server_url: URL of the server for tool calls (default: http://localhost:8080/virtual)
        
        This method should:
        1. Load all tool definitions from tools_dir/*.json
        2. Convert them to framework-specific tool format
        3. Bind them to the agent so they can be used in answer() calls
        
        Note:
        - Should be called before using answer() method
        - Can be called multiple times to reload/update tools
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
```

### `agents/__init__.py`

```python
"""
Agent implementations for StableToolBench evaluation.
"""
from .base_agent import BaseAgent
from .langgraph import LangGraphAgent

__all__ = ['BaseAgent', 'LangGraphAgent']
```

### `agents/langgraph/__init__.py`

```python
"""
LangGraph agent implementation for StableToolBench.
"""
from .agent import LangGraphAgent

__all__ = ['LangGraphAgent']
```

### `agents/langgraph/tool_loader.py` (Helper)

```python
"""
Tool loader for LangGraph agent.

Loads tools from StableToolBench and converts them to LangChain tools.
"""
import os
import json
import requests
from typing import List, Dict, Any
from langchain.tools import tool
from langchain_core.tools import BaseTool


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
    
    # Load all JSON files from tools_dir
    for filename in os.listdir(tools_dir):
        if not filename.endswith('.json'):
            continue
            
        tool_file = os.path.join(tools_dir, filename)
        with open(tool_file, 'r') as f:
            tool_data = json.load(f)
        
        tool_name = tool_data.get('tool_name', '')
        tool_description = tool_data.get('tool_description', '')
        api_list = tool_data.get('api_list', [])
        
        # Extract category (default to "Data" if not found)
        category = tool_data.get('tool_category', 'Data')
        
        # Create a LangChain tool for each API
        for api_info in api_list:
            api_name = api_info.get('name', '')
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
            
            # Create LangChain tool
            tool_name_full = f"{tool_name}_{api_name}"
            langchain_tool = tool(tool_func, name=tool_name_full)
            langchain_tool.description = f"{tool_description}\n\n{api_description}"
            
            tools.append(langchain_tool)
    
    return tools


def create_tool_function(tool_name: str, api_name: str, category: str, 
                        server_url: str, required_params: List[Dict], 
                        optional_params: List[Dict]):
    """
    Create a tool function that calls the server.
    
    Returns a function that can be used with @tool decorator.
    """
    def tool_function(**kwargs):
        # Prepare tool input
        tool_input = {}
        for param in required_params + optional_params:
            param_name = param.get('name', '')
            if param_name in kwargs:
                tool_input[param_name] = kwargs[param_name]
        
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
                return result.get("response", "")
            else:
                return f"Error: Server returned status {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Set function name and docstring
    tool_function.__name__ = f"{tool_name}_{api_name}"
    tool_function.__doc__ = f"Call {tool_name} {api_name} API"
    
    return tool_function
```

