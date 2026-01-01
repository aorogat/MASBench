# Multi-Agent Topology Experiments - Contributions

This document outlines the contributions made in the `multi_agent/topology` folder, which wraps and extends the original code from `multi_agent/agentsNetOriginalCode` to provide enhanced experiment control, configuration management, visualization capabilities, and result analysis.

## Overview

The original `agentsNetOriginalCode` provides the core `LiteralMessagePassing` framework and task implementations (Coloring, Matching, Consensus, LeaderElection, VertexCover). Our contributions add a comprehensive experiment infrastructure that enables:

- **Multi-framework support**: Run experiments across different orchestration frameworks (LangGraph, CrewAI, Concordia)
- **Batch experiment execution**: Automated running of large-scale experiment suites
- **Enhanced result tracking**: Token usage, runtime metrics, and comprehensive result aggregation
- **Visualization tools**: Interactive dashboards and analysis scripts for result interpretation
- **Configuration abstraction**: Centralized model providers, task registries, and graph builders

---

## 1. Experiment Control

### 1.1 Master Experiment Runner (`run_all_experiments.py`)

**Purpose**: Orchestrates batch execution of experiments across multiple tasks, frameworks, graph models, and sizes.

**Key Features**:
- **Automated experiment sweeps**: Configurable product of tasks × frameworks × graph models × sizes
- **Asynchronous execution**: Uses `asyncio` for efficient subprocess management
- **Real-time logging**: Streams output from each experiment run
- **Result aggregation**: Automatically collects and aggregates all JSON results into CSV/JSON summaries
- **Framework-aware execution**: Handles framework-specific requirements (e.g., skipping Concordia for certain tasks)

**Usage**:
```bash
python -m multi_agent.topology.run_all_experiments
```

**Configuration**:
- `TASKS`: List of tasks to run (e.g., `["coloring", "matching", "vertex_cover"]`)
- `SIZES`: Graph sizes to test (e.g., `[50, 100]`)
- `FRAMEWORKS`: Framework configurations with graph model mappings
- `MODEL`, `ROUNDS`, `SAMPLES`: Experiment parameters

### 1.2 Unified Runner (`runner.py`)

**Purpose**: Provides a single entry point that dispatches to framework-specific runners via a registry pattern.

**Key Features**:
- **Framework registry**: Centralized mapping of framework names to runner modules
- **Extensible architecture**: New frameworks can be added by implementing `run_framework(args, commit_hash)`
- **Consistent CLI interface**: Unified argument parsing across all frameworks
- **Environment management**: Automatic `.env` loading for API keys

**Supported Frameworks**:
- `literal`: Original LiteralMessagePassing framework
- `langgraph`: LangGraph-based orchestration
- `crewai`: CrewAI sequential/hierarchical patterns
- `concordia`: Concordia relay hub simulation

**Usage**:
```bash
python -m multi_agent.topology.runner \
  --task coloring \
  --model gpt-4o-mini \
  --graph_models ws \
  --graph_size 4 \
  --rounds 4 \
  --framework langgraph
```

### 1.3 Framework-Specific Runners (`frameworks/`)

**Purpose**: Isolated implementations for each orchestration framework, allowing fair comparison across different multi-agent systems.

**Implementations**:

1. **`langgraph_runner.py`**: Executes experiments using the LangGraph-based `LiteralMessagePassing` framework
   - Uses HuggingFace graphs as input
   - Leverages LangGraph's `StateGraph` for agent coordination

2. **`crewai_runner.py`**: Simulates CrewAI's orchestration patterns
   - **Sequential mode**: Rewires graphs into path structures (chain of agents)
   - **Hierarchical mode**: Rewires graphs into balanced 4-ary trees
   - Maintains original node set while changing communication topology

3. **`concordia_runner.py`**: Simulates Concordia's relay hub behavior
   - Converts graphs to fully-connected topologies (all-to-all)
   - Represents centralized communication patterns

**Benefits**:
- **Modularity**: Each framework runner is self-contained
- **Consistency**: All runners follow the same interface (`run_framework`)
- **Comparability**: Same tasks and graphs can be run across frameworks

---

## 2. Configuration Management

### 2.1 Model Provider Registry (`model_providers.py`)

**Purpose**: Centralized mapping of model names to their API providers.

**Key Features**:
- **Unified model interface**: Single dictionary maps model names to providers (OpenAI, Anthropic, Google, Ollama, Together)
- **Easy extension**: New models can be added by updating the dictionary
- **Provider abstraction**: Decouples model selection from provider implementation

**Supported Providers**:
- OpenAI: `gpt-4o-mini`, `gpt-4o`, `o1`, `o3-mini`, etc.
- Anthropic: `claude-3-5-haiku`, `claude-3-opus`, `claude-3-7-sonnet`, etc.
- Google: `gemini-2.0-flash`, `gemini-2.5-pro`, etc.
- Ollama: `llama3.1`, `deepseek-llm:7b`, `qwen:7b`, etc.
- Together: `meta-llama/Llama-4-Scout-17B-16E-Instruct`, etc.

### 2.2 Task Registry (`tasks.py`)

**Purpose**: Maps task names (CLI arguments) to their corresponding `LiteralMessagePassing` subclasses.

**Key Features**:
- **Simple string-to-class mapping**: Enables easy task selection via CLI
- **Type safety**: Direct imports from original codebase
- **Extensibility**: New tasks can be added by importing and registering

**Registered Tasks**:
- `matching`: Maximum matching algorithm
- `consensus`: Byzantine fault-tolerant consensus
- `coloring`: Distributed graph coloring
- `leader_election`: Distributed leader selection
- `vertex_cover`: Minimum vertex cover finding

### 2.3 Graph Builder Abstraction (`graph_builder.py`)

**Purpose**: Provides a unified interface for graph construction from multiple sources.

**Key Features**:
- **Multiple graph sources**: Supports HuggingFace datasets, framework-specific generators, and topology modifications
- **Topology transformations**: Converts base graphs to sequential, hierarchical, or fully-connected structures
- **Consistent interface**: Single `get_graph()` function with source-based dispatch

**Graph Sources**:
- `hf`: Loads graphs from HuggingFace AgentsNet dataset
- `hf_all_connected`: Loads HF graph and enforces full connectivity (for Concordia simulation)
- `hf_sequential`: Loads HF graph and rewires as path (for CrewAI sequential)
- `hf_hierarchical`: Loads HF graph and rewires as 4-ary tree (for CrewAI hierarchical)
- `framework`: Placeholder for future framework-native graph generation

**Benefits**:
- **Reproducibility**: Same graph instances can be used across frameworks
- **Fair comparison**: Framework differences are in orchestration, not graph structure
- **Flexibility**: Easy to add new graph transformation patterns

### 2.4 Utility Functions (`utils.py`)

**Purpose**: Shared helper functions for experiment execution.

**Key Features**:
- **Round determination**: Automatically adjusts rounds based on task type and graph size
- **Git integration**: Captures commit hash for result reproducibility
- **Timing utilities**: Context managers and formatters for runtime measurement

**Functions**:
- `determine_rounds()`: Sets rounds to `2 * diameter + 1` for consensus/leader_election or large graphs
- `get_git_commit_hash()`: Failsafe git commit hash retrieval
- `timer()`: Context manager for elapsed time measurement
- `format_seconds()`: Human-readable time formatting

---

## 3. Result Management

### 3.1 Enhanced Result Saving (`results.py`)

**Purpose**: Extends original result saving with additional metrics and structured output.

**Key Enhancements**:

1. **Token Usage Tracking**:
   - `summarize_tokens()`: Aggregates token usage across all agents
   - Per-agent token breakdowns (input, output, total)
   - Total token usage across the entire experiment

2. **Comprehensive Metadata**:
   - Framework identification
   - Graph metrics (diameter, max degree)
   - Execution metadata (commit hash, runtime, success status)
   - Error tracking and fallback counts

3. **Structured Output**:
   - Timestamped filenames with experiment parameters
   - JSON format with NetworkX graph serialization
   - Consistent schema across all frameworks

**Output Schema**:
```json
{
  "answers": [...],
  "num_nodes": 16,
  "framework": "langgraph",
  "diameter": 3,
  "max_degree": 5,
  "rounds": 8,
  "model_name": "gpt-4o-mini",
  "task": "coloring",
  "score": 0.95,
  "runtime_seconds": 12.3,
  "token_summary": {
    "total_tokens": 50000,
    "per_agent": {...}
  },
  "transcripts": {...},
  "graph": {...}
}
```

### 3.2 Result Aggregation (`run_all_experiments.py`)

**Purpose**: Automatically collects and aggregates results from batch experiments.

**Key Features**:
- **CSV export**: Creates `scalability_summary.csv` with key metrics
- **JSON export**: Creates `scalability_summary.json` for programmatic access
- **Error handling**: Gracefully handles malformed result files
- **Unified schema**: Normalizes results across different frameworks

**Aggregated Fields**:
- Task, framework, graph model
- Number of nodes, rounds executed
- Score, runtime, success status
- Error messages (if any)

---

## 4. Visualization

### 4.1 Interactive Scalability Dashboard (`visualization/scalability_dashboard.py`)

**Purpose**: Generates interactive HTML dashboards for scalability analysis.

**Key Features**:
- **Plotly-based visualizations**: Interactive charts with zoom, pan, and hover
- **Multi-metric views**: Score, runtime, and token usage across different dimensions
- **Task-specific analysis**: Separate views for each task type
- **Scalability trends**: Performance vs. number of agents

**Usage**:
```bash
python -m multi_agent.topology.visualization.scalability_dashboard \
  results/ontology/*.json \
  --output scalability.html
```

### 4.2 Task-Specific Visualization Scripts

**Purpose**: Specialized visualization tools for individual tasks.

**Available Scripts**:
- `color_main.py`: Graph coloring visualization with color assignments
- `consensus_main.py`: Consensus algorithm visualization
- `consensus_scalability.py`: Consensus scalability analysis
- `leader_main.py`: Leader election visualization
- `matching_main.py`: Matching algorithm visualization
- `vertexcover_main.py`: Vertex cover visualization

**Features**:
- NetworkX graph rendering
- Agent state visualization
- Solution quality metrics
- Interactive exploration

### 4.3 Visualization Utilities (`visualization/`)

**Supporting Modules**:
- `loader.py`: Result file loading and graph extraction
- `layouts.py`: Graph layout algorithms for visualization
- `static_plot.py`: Static matplotlib-based plots
- `interactive_plot.py`: Interactive plotly-based plots
- `analysis.py`: Statistical analysis and aggregation
- `utils.py`: Shared visualization helpers

---

## 5. Architecture Improvements

### 5.1 Separation of Concerns

**Original Code Structure**:
- Monolithic `main.py` with inline graph loading, task execution, and result saving
- Hard-coded model providers and task mappings
- Single framework (LiteralMessagePassing)

**New Structure**:
- **Modular design**: Separate modules for graph building, task execution, result saving
- **Framework abstraction**: Registry pattern enables multiple frameworks
- **Configuration separation**: Model providers, tasks, and graph builders in dedicated modules

### 5.2 Extensibility

**Adding New Frameworks**:
1. Create a new runner in `frameworks/` (e.g., `new_framework_runner.py`)
2. Implement `run_framework(args, commit_hash)` function
3. Register in `FRAMEWORK_REGISTRY` in `runner.py`

**Adding New Tasks**:
1. Implement task class in `agentsNetOriginalCode/LiteralMessagePassing.py`
2. Register in `tasks.py` dictionary

**Adding New Graph Sources**:
1. Implement builder function in `graph_builder.py`
2. Add dispatch case in `get_graph()`

### 5.3 Reproducibility

**Enhancements**:
- **Git commit tracking**: All results include commit hash
- **Seed management**: Configurable random seeds for graph generation
- **Deterministic execution**: Consistent graph loading from HuggingFace dataset
- **Comprehensive logging**: Detailed execution logs for debugging

---

## 6. Comparison with Original Code

### Original Code (`agentsNetOriginalCode/`)
- ✅ Core `LiteralMessagePassing` framework
- ✅ Task implementations (Coloring, Matching, etc.)
- ✅ Basic result saving
- ✅ HuggingFace graph loading
- ❌ Single framework support
- ❌ Manual experiment execution
- ❌ Limited result analysis
- ❌ No visualization tools

### New Contributions (`topology/`)
- ✅ Multi-framework support (LangGraph, CrewAI, Concordia)
- ✅ Automated batch experiment execution
- ✅ Enhanced result tracking (tokens, metrics)
- ✅ Comprehensive visualization suite
- ✅ Configuration abstraction
- ✅ Result aggregation and analysis
- ✅ Extensible architecture

---

## 7. Usage Examples

### Running a Single Experiment
```bash
python -m multi_agent.topology.runner \
  --task coloring \
  --model gpt-4o-mini \
  --graph_models ws ba \
  --graph_size 16 \
  --rounds 8 \
  --samples_per_graph_model 3 \
  --framework langgraph
```

### Running Batch Experiments
```bash
# Edit run_all_experiments.py to configure TASKS, SIZES, FRAMEWORKS
python -m multi_agent.topology.run_all_experiments
```

### Analyzing Results
```bash
# Generate interactive dashboard
python -m multi_agent.topology.visualization.scalability_dashboard \
  results/ontology/*.json \
  --output dashboard.html

# View aggregated summary
cat results/ontology/scalability_summary.csv
```

---

## 8. Future Enhancements

Potential areas for further development:

1. **Additional Frameworks**: Integration with AutoGen, Swarm, or custom frameworks
2. **Advanced Graph Generation**: Framework-native graph construction
3. **Real-time Monitoring**: Live experiment tracking and progress visualization
4. **Statistical Analysis**: Automated significance testing and confidence intervals
5. **Distributed Execution**: Parallel experiment execution across multiple machines
6. **Configuration Files**: YAML/JSON-based experiment configuration instead of code

---

## Summary

The contributions in `multi_agent/topology` transform the original AgentsNet codebase from a single-framework research tool into a comprehensive multi-agent experimentation platform. Key achievements include:

- **Experiment Control**: Automated batch execution and multi-framework support
- **Configuration**: Centralized, extensible configuration management
- **Results**: Enhanced tracking and aggregation capabilities
- **Visualization**: Interactive dashboards and analysis tools
- **Architecture**: Modular, extensible design for future enhancements

These contributions enable systematic comparison of multi-agent frameworks while maintaining compatibility with the original AgentsNet benchmark tasks and evaluation methodology.

