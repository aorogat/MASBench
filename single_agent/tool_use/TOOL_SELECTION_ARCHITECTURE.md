# Tool Selection Architecture

## Overview

This document describes the centralized **Tool Selection Layer** that operates before any framework binds tools to an agent. This architecture ensures:

1. **Scalability**: Respects LLM tool-exposure limits (e.g., OpenAI's 128-tool constraint)
2. **Fair Comparison**: All frameworks operate over the same candidate tool set for a given question
3. **Reproducibility**: Tool selections are cached per query, ensuring identical results across runs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Tool Selection Layer                      │
│  (Runs BEFORE framework binding, shared across frameworks)   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Load all tools (~1500 tools)                            │
│  2. For each query:                                          │
│     a. Check cache (query hash)                              │
│     b. If cached: return cached tool list                    │
│     c. If not cached:                                        │
│        - Use LLM to select top-k tools (k ≤ 128)            │
│        - Cache selection for reproducibility                 │
│     d. Return ordered list of selected tools                │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Framework-Specific Agent Binding                │
│  (LangGraph, OpenAI Agents SDK, CrewAI, etc.)               │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  - Receives pre-selected tools (max 120 tools)              │
│  - Binds tools using framework's native mechanism           │
│  - Executes query with constrained tool set                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. ToolSelector (`tool_selection/selector.py`)

**Purpose**: Centralized LLM-based tool selection with caching.

**Features**:
- Uses LLM to select top-k most relevant tools for a query (tool names only for speed)
- Caches selections per query (SHA256 hash of query)
- Ensures all frameworks use the same tool set for the same query
- Respects LLM tool limits (default: 120 tools, under OpenAI's 128 limit)
- Handles truncated JSON responses gracefully
- Falls back to keyword-based selection if LLM parsing fails
- Skips empty selections (doesn't cache failures)

**Usage**:
```python
from tool_selection import ToolSelector
from agents.langgraph.tool_loader import load_tools

# Load all tools once
all_tools = load_tools(tools_dir=tools_dir, server_url=server_url)

# Initialize selector
selector = ToolSelector(
    model="gpt-4o-mini",
    max_tools=120,
    verbose=True
)

# Select tools for a query (cached automatically)
selected_tools = selector.select_tools(query="...", all_tools=all_tools)
```

### 2. Updated BaseAgent Interface

**Changes**:
- `bind_tools()` now accepts pre-selected tools (preferred) or tools_dir (legacy)
- Agents receive filtered tools, not the full registry

**New Signature**:
```python
def bind_tools(
    self,
    tools: Union[List[BaseTool], str, None] = None,  # Pre-selected tools (preferred)
    tools_dir: Optional[str] = None,                 # Legacy mode
    server_url: str = "http://localhost:8080/virtual"
) -> None:
```

### 3. Updated run_benchmark.py

**Changes**:
- Loads all tools once at the beginning
- Initializes ToolSelector
- For each query:
  1. Selects tools using ToolSelector (cached)
  2. Binds selected tools to agent
  3. Calls agent.answer()

**New Parameters**:
```python
run_benchmark(
    agent=agent,
    use_tool_selector=True,        # Enable tool selection (default: True)
    tool_selector_model="gpt-4o-mini",
    max_tools=120,
    ...
)
```

## Caching Mechanism

### Cache Location
- Default: `tool_selection/tool_selection_cache/`
- Files named by query hash: `{query_hash}.json`

### Cache Format
```json
{
  "query": "original query text",
  "selected_tools": ["tool_name_1", "tool_name_2", ...],
  "model": "gpt-4o-mini",
  "max_tools": 120
}
```

### Cache Benefits
1. **Reproducibility**: Same query → same tool selection across runs
2. **Fairness**: All frameworks use identical tool sets for the same query
3. **Efficiency**: Avoids redundant LLM calls for repeated queries

### Cache Handling
- **Empty cache entries**: Treated as cache miss (regenerates selection)
- **Name mapping**: Handles both original and sanitized tool names
- **Invalid cache**: Automatically regenerates if tool names don't match current tools

## Workflow Example

```python
# 1. Create agent (no tools bound yet)
agent = LangGraphAgent(model="gpt-4o-mini")

# 2. Run benchmark (tool selection happens inside)
results = run_benchmark(
    agent=agent,
    use_tool_selector=True,  # Enable centralized selection
    max_queries=3
)

# For each query:
#   a. ToolSelector selects tools (or loads from cache)
#   b. Selected tools are bound to agent
#   c. Agent answers query with constrained tool set
```

## Benefits

### 1. Scalability
- Handles large tool registries (~1500 tools)
- Respects LLM provider limits (128 tools)
- No framework-specific filtering needed

### 2. Fair Comparison
- All frameworks see the same tools for the same query
- Eliminates framework-specific tool selection bias
- Enables true framework comparison

### 3. Architectural Insight
- Reveals which frameworks work well with constrained tool visibility
- Identifies frameworks that rely on full tool exposure
- Helps understand framework scalability

### 4. Reproducibility
- Cached selections ensure identical results
- Same query → same tool set → comparable results
- Enables reliable benchmarking

## Migration Guide

### For New Agents
1. Implement `BaseAgent` interface
2. Accept pre-selected tools in `bind_tools(tools=...)`
3. No need to implement tool filtering

### For Existing Agents
1. Update `bind_tools()` to accept `tools` parameter
2. Keep `tools_dir` for backward compatibility
3. Remove framework-specific tool filtering logic

### For Benchmark Scripts
1. Remove manual tool binding
2. Pass agent to `run_benchmark()` with `use_tool_selector=True`
3. Tool selection happens automatically

## Configuration

### ToolSelector Parameters
- `model`: LLM for tool selection (default: "gpt-4o-mini")
- `temperature`: Temperature for reproducibility (default: 0.0)
- `max_tools`: Maximum tools to select (default: 120)
- `cache_dir`: Cache directory (default: `tool_selection/tool_selection_cache/`)
- `verbose`: Print debug information (default: False)

### run_benchmark Parameters
- `use_tool_selector`: Enable tool selection (default: True)
- `tool_selector_model`: Model for selection (default: "gpt-4o-mini")
- `max_tools`: Maximum tools per query (default: 120)

## Tool Name Sanitization

### OpenAI Requirements
Tool names must comply with OpenAI's constraints:
- **Pattern**: `^[a-zA-Z0-9_-]+$` (only alphanumeric, underscore, hyphen)
- **Max Length**: 64 characters

### Implementation
- Tool names are automatically sanitized when creating LangChain tools
- Original names are preserved in `tool.metadata['original_name']`
- Cache lookup handles both original and sanitized names
- Long names are truncated to 64 characters

### Example Transformations
- `"Website Screenshot or Thumbnail_/capture"` → `"Website_Screenshot_or_Thumbnail_capture"`
- `"SEO API - Get Backlinks_GetTopBacklinks"` → `"SEO_API_-_Get_Backlinks_GetTopBacklinks"`
- Very long names are truncated to 64 chars

## Performance Optimizations

### Tool Selection Speed
- **Names only**: Uses tool names (not descriptions) in LLM prompt for faster selection
- **Caching**: First run uses LLM, subsequent runs use cache (instant)
- **Fallback**: Keyword-based selection if LLM parsing fails

### Error Handling
- **Truncated JSON**: Handles incomplete LLM responses gracefully
- **Empty selections**: Automatically uses keyword fallback
- **Invalid cache**: Regenerates selection if cache is invalid

## Future Improvements

1. **Semantic Search**: Use embeddings for better tool selection
2. **Multi-stage Selection**: Coarse filter → fine selection
3. **Tool Categories**: Pre-filter by category before LLM selection
4. **Query Analysis**: Analyze query type to select relevant tool categories

## Files

- `tool_selection/selector.py`: ToolSelector implementation
- `tool_selection/__init__.py`: Package initialization
- `tool_selection/tool_selection_cache/`: Cache directory (auto-created)
- `agents/base_agent.py`: Updated interface
- `agents/langgraph/agent.py`: Updated implementation
- `run_benchmark.py`: Integration with ToolSelector

