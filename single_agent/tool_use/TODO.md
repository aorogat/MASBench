# TODO: Framework Implementations

This document tracks the implementation status of different agent frameworks for StableToolBench evaluation.

## Completed Frameworks

### ✅ LangGraph
- **Status**: Complete
- **Location**: `agents/langgraph/`
- **Files**:
  - `agent.py`: LangGraphAgent implementation using `langchain.agents.create_agent`
  - `tool_loader.py`: Converts StableToolBench tools to LangChain StructuredTool format
- **Test File**: `test_langgraph.py`
- **Features**:
  - Uses centralized ToolSelector
  - Supports tool name sanitization (OpenAI compliance)
  - Returns `called_apis` for API scoring
  - Full ExecutionGraph format support

## Planned Frameworks

### ✅ CrewAI
- **Status**: Implemented
- **Priority**: High
- **Location**: `agents/crewai/`
- **Files**:
  - `agent.py`: CrewAIAgent implementation using CrewAI's Agent, Task, and Crew
  - `tool_loader.py`: Converts StableToolBench tools to CrewAI BaseTool format
- **Test File**: `test_crewai.py`
- **Notes**:
  - Uses single-agent configuration (one Agent, one Task, one Crew)
  - Uses centralized ToolSelector
  - Supports tool name sanitization (OpenAI compliance)
  - Tool call extraction needs enhancement (CrewAI doesn't expose tool calls directly)
  - Similar to LangGraph: depends on LLM for tool selection, limited by 128-tool constraint

### ✅ AutoGen
- **Status**: Implemented
- **Priority**: High
- **Location**: `agents/autogen/`
- **Files**:
  - `agent.py`: AutoGenAgent implementation using AutoGen's AssistantAgent
  - Tool conversion: Converts LangChain tools to Python functions (AutoGen auto-converts to FunctionTool)
- **Test File**: `test_autogen.py`
- **Notes**:
  - Uses AutoGen's AssistantAgent with async execution
  - Converts LangChain tools to Python functions (AutoGen automatically creates FunctionTool)
  - Uses centralized ToolSelector
  - Extracts tool calls from TaskResult messages (ToolCallRequestEvent, ToolCallExecutionEvent)
  - Supports tool name sanitization (OpenAI compliance)
  - Returns `called_apis` for API scoring
  - Similar to LangGraph/CrewAI: depends on LLM for tool selection, limited by 128-tool constraint

### ✅ OpenAI Agents SDK
- **Status**: Implemented
- **Priority**: High
- **Location**: `agents/openai_sdk/`
- **Files**:
  - `agent.py`: OpenAISDKAgent implementation using OpenAI's Agent and Runner
  - Tool conversion: Converts LangChain tools to OpenAI SDK FunctionTool using @function_tool decorator
- **Test File**: `test_openai_sdk.py`
- **Notes**:
  - Uses OpenAI's Agent and Runner classes with async execution
  - Converts LangChain tools to function tools using @function_tool decorator
  - Uses centralized ToolSelector
  - Extracts tool calls from RunResult (new_items or messages)
  - Supports tool name sanitization (OpenAI compliance)
  - Returns `called_apis` for API scoring
  - Similar to LangGraph/CrewAI/AutoGen: depends on LLM for tool selection, limited by 128-tool constraint

### 🔲 Agno
- **Status**: Not Started
- **Priority**: Medium
- **Notes**:
  - Agent framework with tool calling support
  - Check tool integration approach
  - Test if it has built-in tool filtering
  - Verify scalability characteristics

### ❌ OpenAgents
- **Status**: Excluded from experiments
- **Priority**: N/A (not compatible with evaluation setting)
- **Reason for Exclusion**:
  - OpenAgents uses an event-driven, workflow-oriented execution model
  - Tools are bound to specific agents and invoked within event-driven task handlers
  - Routing decisions are determined by explicit, developer-defined coordination logic rather than by LLM-driven selection over a large tool registry
  - OpenAgents does not expose a single agent to a large flat set of tools
  - It does not require the model to choose among hundreds of candidate APIs under a fixed budget
  - Since StableToolBench is designed to evaluate tool selection and execution behavior under tool overload, OpenAgents' event-driven execution model is not directly compatible with this evaluation setting
- **Reference**: See [OpenAgents documentation](https://github.com/openagents-org/openagents/tree/develop/demos) for details on their workflow-oriented approach

## Implementation Checklist

For each new framework, follow this checklist:

### 1. Create Framework Directory
- [ ] Create `agents/{framework}/` directory
- [ ] Add `__init__.py` with framework agent export
- [ ] Create `agent.py` implementing `BaseAgent` interface
- [ ] Create `tool_loader.py` (if needed) or use shared `utils.tool_utils`

### 2. Implement BaseAgent Interface
- [ ] Implement `bind_tools()` method:
  - Accept `tools: Optional[List[BaseTool]]` (pre-selected tools from ToolSelector)
  - Accept `tools_dir: Optional[str]` (legacy mode)
  - Accept `server_url: str`
- [ ] Implement `answer()` method:
  - Accept `query: str`
  - Return `Dict[str, Any]` with:
    - `"answer"`: Dict with `"final_answer"` and `"answer_details"` (ExecutionGraph format)
    - `"called_apis"`: Optional `List[List[str]]` of `[tool_name, api_name]` pairs

### 3. Tool Integration
- [ ] Use shared utilities from `utils.tool_utils`:
  - `load_tool_definitions()`: Load raw tool definitions
  - `create_tool_function()`: Create server-calling functions
  - `sanitize_tool_name()`: OpenAI name compliance
- [ ] Convert `ToolDefinition` to framework-specific tool format
- [ ] Ensure tools call server at `server_url`
- [ ] Store original tool names in metadata for API scoring

### 4. Test File
- [ ] Create `test_{framework}.py`:
  - Import framework agent
  - Create agent instance
  - Call `run_benchmark()` with appropriate parameters
  - Use `use_tool_selector=True` for fairness
  - Test with small number of queries first (e.g., 3)

### 5. Documentation
- [ ] Update `README.md` with framework-specific notes
- [ ] Add framework to this TODO.md
- [ ] Document any framework-specific quirks or limitations

### 6. Scalability Testing
- [ ] Test with different k values using `run_all.py`:
  - k=8, 16, 32, 64, 128, 256, 512, 1024, 1500
  - Identify where framework fails (if at all)
  - Document tool limit constraints
  - Note if framework has built-in tool filtering

## Framework-Specific Notes

### LangGraph
- Uses `langchain.agents.create_agent` for ReAct loop
- Tool names sanitized to 64 characters (OpenAI limit)
- Original names stored in `tool.metadata['original_name']`
- No built-in tool filtering (relies on ToolSelector)

### CrewAI
- Uses CrewAI's Agent, Task, and Crew classes for single-agent execution
- Tool names sanitized to 64 characters (OpenAI limit)
- Structured metadata stored (tool_name, api_name, category)
- No built-in tool filtering (relies on ToolSelector)
- Similar to LangGraph: depends on LLM for tool selection, limited by 128-tool constraint
- **Limitation**: Tool call extraction needs enhancement (CrewAI doesn't expose tool calls directly)

### AutoGen
- Uses AutoGen's AssistantAgent with async execution
- Converts LangChain tools to Python functions (AutoGen auto-converts to FunctionTool)
- Tool names sanitized to 64 characters (OpenAI limit)
- Structured metadata stored (tool_name, api_name, category)
- No built-in tool filtering (relies on ToolSelector)
- Similar to LangGraph/CrewAI: depends on LLM for tool selection, limited by 128-tool constraint
- Extracts tool calls from TaskResult messages for accurate API tracking

### OpenAI Agents SDK
- Uses OpenAI's Agent and Runner with async execution
- Converts LangChain tools to function tools using @function_tool decorator
- Tool names sanitized to 64 characters (OpenAI limit)
- Structured metadata stored (tool_name, api_name, category)
- No built-in tool filtering (relies on ToolSelector)
- Similar to LangGraph/CrewAI/AutoGen: depends on LLM for tool selection, limited by 128-tool constraint
- Extracts tool calls from RunResult for accurate API tracking

### Agno (Planned)
- Check tool integration approach
- Verify if it has built-in tool filtering
- Test scalability characteristics

### OpenAgents (Excluded)
- **Status**: Excluded from experiments (not compatible with evaluation setting)
- **Reason**: Uses event-driven, workflow-oriented execution model
- **Details**: See main OpenAgents section above for full explanation
- **Reference**: See [OpenAgents documentation](https://github.com/openagents-org/openagents/tree/develop/demos)

## Testing Strategy

### Scalability Testing
Use `run_all.py` to test all frameworks with different k values:
- **Small k (8-32)**: Test basic functionality
- **Medium k (64-128)**: Test OpenAI API limits
- **Large k (256-512)**: Test framework scalability
- **Very large k (1024-1500)**: Test extreme cases

### Expected Behaviors
- **Frameworks without filtering**: Will fail at k=128+ (OpenAI API limit)
- **Frameworks with filtering**: Should handle larger k values
- **ToolSelector**: Always provides k tools, frameworks should handle them

### Success Criteria
- Framework successfully processes queries with k tools
- Framework correctly calls tools via server
- Framework returns proper ExecutionGraph format
- Framework provides `called_apis` for scoring
- Framework handles tool name sanitization

## Next Steps

1. ✅ **Implement CrewAI** (completed - tool call extraction needs enhancement)
2. ✅ **Implement AutoGen** (completed)
3. ✅ **Implement OpenAI Agents SDK** (completed)
4. **Implement Agno** (medium priority)
5. ❌ **OpenAgents** (excluded - not compatible with evaluation setting)
6. **Enhance CrewAI tool call extraction** (medium priority - for accurate API scoring)
7. **Run scalability tests** on all frameworks using `run_all.py`
8. **Document findings** in framework-specific notes

## Resources

- **BaseAgent Interface**: `agents/base_agent.py`
- **Shared Utilities**: `utils/tool_utils.py`
- **Tool Selection**: `tool_selection/selector.py`
- **Benchmark Runner**: `run_benchmark.py`
- **LangGraph Example**: `agents/langgraph/` and `test_langgraph.py`
- **CrewAI Example**: `agents/crewai/` and `test_crewai.py`
- **AutoGen Example**: `agents/autogen/` and `test_autogen.py`
- **OpenAI SDK Example**: `agents/openai_sdk/` and `test_openai_sdk.py`

