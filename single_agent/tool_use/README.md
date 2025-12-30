# 🧰 StableToolBench Evaluation Framework

**A modular evaluation system for testing LLM agent frameworks on tool-use tasks**

This directory provides a complete wrapper around StableToolBench to evaluate any framework (LangGraph, CrewAI, OpenAI SDK, etc.) using a unified interface. All evaluation logic, metrics, and result saving are handled automatically.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Directory Structure](#directory-structure)
3. [Wrapper Components](#wrapper-components)
4. [Original Code Location](#original-code-location)
5. [Prerequisites](#prerequisites)
6. [Setup Instructions](#setup-instructions)
7. [Running Experiments](#running-experiments)
8. [Evaluation Process](#evaluation-process)
9. [Troubleshooting](#troubleshooting)

---

## 🏗️ Architecture Overview

This evaluation framework wraps StableToolBench's original code to provide a clean, reusable interface following SOLID principles. The architecture consists of:

```
┌─────────────────────────────────────────────────────────┐
│              Your Framework (LangGraph, CrewAI, etc.)   │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         FrameworkInterface (Abstract Base Class)         │
│  - setup_tools(tools: dict)                              │
│  - reset()                                                │
│  - answer(query: str) -> str                            │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FrameworkAdapter (Wrapper)                    │
│  Adapts FrameworkInterface to BaseModelAdapter           │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  QAPipeline (Wrapper)                    │
│  Wraps StableToolBench's QAPipeline                     │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         StableToolBenchEvaluator (Modular)               │
│  - Evaluates answers using SoPR                          │
│  - Checks for Finish calls                                │
│  - Calculates API call scores                             │
│  - Saves detailed results to JSON                        │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

The codebase is organized into modular components following SOLID principles:

```
single_agent/tool_use/
├── core/                          # Core interfaces and adapters
│   ├── __init__.py
│   ├── framework_interface.py    # FrameworkInterface (Interface Segregation)
│   └── framework_adapter.py      # FrameworkAdapter (Adapter Pattern, Dependency Inversion)
│
├── evaluation/                     # Evaluation components
│   ├── __init__.py
│   ├── evaluator.py              # Main evaluator (composition, orchestration)
│   ├── evaluator_loader.py       # EvaluatorLoader (Single Responsibility)
│   ├── api_scorer.py             # APIScorer (Single Responsibility)
│   └── heuristic_evaluator.py   # HeuristicEvaluator (Single Responsibility)
│
├── evaluation_helpers/            # Reusable evaluation utilities
│   ├── __init__.py
│   ├── result_formatter.py       # Formats results and calculates statistics
│   ├── result_saver.py            # Saves results to JSON files
│   └── evaluation_printer.py     # Prints evaluation progress and results
│
├── pipeline/                      # Pipeline wrappers
│   ├── __init__.py
│   └── qa_pipeline.py            # QAPipeline (Open/Closed Principle)
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── query_loader.py           # QueryLoader (Single Responsibility)
│   └── answer_validator.py       # AnswerValidator (Single Responsibility)
│
├── tests/                         # Test scripts
│   ├── __init__.py
│   ├── test_benchmark.py         # Main test script
│   ├── fake_answer_generator.py  # FakeAnswerGenerator (Single Responsibility)
│   └── test_*.py                 # Unit tests for each component
│
└── StableToolBench/               # Original StableToolBench code (NOT MODIFIED)
```

### Component Dependencies

```
StableToolBenchEvaluator (orchestrator)
    ├── EvaluatorLoader (loads evaluator)
    ├── APIScorer (calculates API scores)
    ├── HeuristicEvaluator (fallback evaluation)
    │   └── APIScorer (dependency)
    └── AnswerValidator (validates answers)

FrameworkAdapter
    └── FrameworkInterface (depends on abstraction)

QAPipeline
    └── BaseModelAdapter (depends on abstraction)

QueryLoader (independent)
FakeAnswerGenerator (independent)
```

### SOLID Principles Applied

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Open for extension, closed for modification
- **Liskov Substitution**: Subtypes are substitutable for base types
- **Interface Segregation**: Focused, specific interfaces
- **Dependency Inversion**: Depend on abstractions, not concretions

---

## 🔧 Wrapper Components

### 1. `core/framework_interface.py`

**Purpose**: Abstract base class that any framework must implement.

**Methods**:
- `setup_tools(tools: dict)`: Called once before evaluation to set up available tools
- `reset()`: Reset conversation/memory/internal state between queries
- `answer(query: str) -> str`: Answer the query and return only the final answer string

**Example**:
```python
from core import FrameworkInterface

class MyFramework(FrameworkInterface):
    def setup_tools(self, tools: dict):
        self.tools = tools
    
    def reset(self):
        self.conversation_history = []
    
    def answer(self, query: str) -> str:
        # Your framework logic here
        return "Final answer string"
```

### 2. `core/framework_adapter.py`

**Purpose**: Wraps a `FrameworkInterface` instance to conform to StableToolBench's `BaseModelAdapter` interface.

**Usage**:
```python
from core import FrameworkInterface, FrameworkAdapter

my_framework = MyFramework()  # Implements FrameworkInterface
adapter = FrameworkAdapter(my_framework)
# Now adapter can be used with QAPipeline
```

### 3. `pipeline/qa_pipeline.py`

**Purpose**: Wrapper around StableToolBench's `QAPipeline` for easier integration.

**Usage**:
```python
from pipeline import QAPipeline

pipeline = QAPipeline(
    tool_dir="StableToolBench/toolenv/tools",
    query_dir="StableToolBench/solvable_queries",
    model=adapter,  # FrameworkAdapter instance
    use_mirrorapi_cache=True  # Use cache/MirrorAPI (default)
)

results = pipeline.run()
```

### 4. `evaluation/evaluator.py`

**Purpose**: Modular evaluator that encapsulates all evaluation logic.

**Features**:
- Uses `.env` file for OpenAI API key (required)
- Wraps StableToolBench's official evaluator
- Provides fallback heuristic evaluation
- Returns SoPR scores (0.0, 0.5, 1.0) and answer status

**Usage**:
```python
from evaluation import StableToolBenchEvaluator

evaluator = StableToolBenchEvaluator(
    model_name="gpt-4o-mini",  # Model for evaluation
    verbose=True
)

result = evaluator.evaluate_answer(query_dict, answer_dict)
# Returns: {
#     "sopr_score": 1.0,  # 0.0 (Unsolved), 0.5 (Unsure), 1.0 (Solved)
#     "answer_status": "Solved",
#     "has_finish": True,
#     "api_call_score": 0.85,
#     "reason": "...",
#     "evaluation_time": 0.123,
#     "evaluation_method": "official"  # or "heuristic"
# }
```

### 5. `evaluation_helpers/` - Reusable Components

**Purpose**: Common evaluation utilities that can be reused across framework evaluations.

- **`result_formatter.py`**: Formats results and calculates summary statistics
- **`result_saver.py`**: Saves results to JSON files with standardized naming
- **`evaluation_printer.py`**: Prints evaluation progress and results to terminal

**Usage**:
```python
from evaluation_helpers import ResultFormatter, ResultSaver, EvaluationPrinter

formatter = ResultFormatter()
saver = ResultSaver(Path("results/tools"))
printer = EvaluationPrinter()

# Format result
result = formatter.format_result(query, answer, eval_result, gold_apis)

# Calculate summary
summary = formatter.calculate_summary(all_results)

# Save results
output_file = saver.save(results_data, "framework_name_config.json")

# Print summary
printer.print_summary(summary, output_file)
```

---

## 📍 Original Code Location

**All original StableToolBench code is located in**: `single_agent/tool_use/StableToolBench/`

**Key directories**:
- `StableToolBench/toolbench/inference/`: Inference pipeline code
- `StableToolBench/toolbench/tooleval/`: Evaluation metrics and evaluators
- `StableToolBench/solvable_queries/`: Test queries
- `StableToolBench/toolenv/tools/`: Tool specifications (~7k APIs)
- `StableToolBench/server/`: MirrorAPI server (optional)

**What we added**:
- `core/`: Framework interface and adapter
- `evaluation/`: Modular evaluation components
- `evaluation_helpers/`: Reusable evaluation utilities
- `pipeline/`: Pipeline wrapper
- `utils/`: Utility functions
- `tests/`: Test scripts
- `README.md`: This documentation
- `EVALUATION_PROCESS.md`: Detailed evaluation process explanation

**We did NOT modify any files inside `StableToolBench/` directory.**

---

## 🔧 Prerequisites

### Required Downloads

1. **StableToolBench Repository**
   - Already included in `StableToolBench/` directory
   - Contains queries, tools, and evaluation code

2. **Tool Cache (for GPT-based caching, no GPU required)**
   - Download from: [StableToolBench Cache](https://huggingface.co/stabletoolbench/MirrorAPI-Cache)
   - Or use GPT-based caching (no download needed, uses OpenAI API)

3. **MirrorAPI Model (optional, requires GPU)**
   - Download from: [MirrorAPI Model](https://huggingface.co/stabletoolbench/MirrorAPI)
   - Only needed if you want to use MirrorAPI instead of GPT-based caching

### Required API Keys

**OpenAI API Key** (required for evaluation and GPT-based caching):
- **Required**: Set in `.env` file: `OPENAI_API_KEY=your-key`
- Create `.env` file in `single_agent/tool_use/` directory
- The code will automatically load it using `python-dotenv`

**Note**: The code creates a temporary `openai_key.json` file at runtime (required by StableToolBench's original evaluator), but the source of truth is always the `.env` file.

### Python Dependencies

```bash
pip install python-dotenv openai langchain langgraph  # Add your framework dependencies
pip install tenacity  # Required by StableToolBench evaluator
```

---

## 🚀 Setup Instructions

### Step 1: Configure Environment

**Create `.env` file** (required):
```bash
cd single_agent/tool_use
echo "OPENAI_API_KEY=your-key" > .env
```

The `.env` file must be in the `single_agent/tool_use/` directory. The code will automatically load it using `python-dotenv`.

### Step 2: Choose API Simulation Method

You have two options for simulating API calls:

#### Option A: GPT-based Caching (No GPU Required)

**Best for**: Users without GPU access who want quick setup.

**How it works**: Uses OpenAI API with a cache to simulate tool responses. No downloads required.

**Setup**:
1. Ensure `OPENAI_API_KEY` is set in `.env`
2. Set `use_mirrorapi_cache=True` in `QAPipeline` (this uses GPT-based caching, not MirrorAPI)
3. The system will automatically use OpenAI API for simulation

**Example**:
```python
pipeline = QAPipeline(
    tool_dir="StableToolBench/toolenv/tools",
    query_dir="StableToolBench/solvable_queries",
    model=adapter,
    use_mirrorapi_cache=True  # Uses GPT-based caching
)
```

#### Option B: MirrorAPI (Requires GPU)

**Best for**: Users with GPU access who want trained model simulation.

1. **Download MirrorAPI model**:
   - From: https://huggingface.co/stabletoolbench/MirrorAPI
   - Or MirrorAPI-Cache: https://huggingface.co/stabletoolbench/MirrorAPI-Cache

2. **Start vLLM server**:
   ```bash
   vllm serve {model-path} --api-key EMPTY --port 12345 --served-model-name {model-name}
   ```

3. **Configure `StableToolBench/server/config_mirrorapi.yml`**:
   ```yaml
   api_key: "EMPTY"  # or your OpenAI key
   api_base: "http://127.0.0.1:12345/v1"
   model: "{model-name}"
   temperature: 0
   tools_folder: "./tools"
   port: 8080
   ```

4. **Run server**:
   ```bash
   cd single_agent/tool_use/StableToolBench/server
   python main_mirrorapi.py  # or main_mirrorapi_cache.py
   ```

### Step 3: Test the Setup

Run the test script to verify everything works:

```bash
cd single_agent/tool_use
python test_benchmark.py
```

This will:
- Load queries from StableToolBench
- Create fake answers (good, bad, partial)
- Evaluate them using the evaluator
- Save results to `results/tools/Test_benchmark_3queries_G1_instruction.json`

---

## 🧪 Running Experiments

### Template: Running with Your Framework

```python
import os
import sys
from pathlib import Path

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from core import FrameworkInterface, FrameworkAdapter
from pipeline import QAPipeline
from evaluation import StableToolBenchEvaluator
from evaluation_helpers import ResultFormatter, ResultSaver, EvaluationPrinter
from utils import QueryLoader

# 1. Implement your framework
class MyFramework(FrameworkInterface):
    def setup_tools(self, tools: dict):
        # Set up tools in your framework
        self.tools = tools
    
    def reset(self):
        # Reset state
        self.history = []
    
    def answer(self, query: str) -> str:
        # Your framework logic
        # Must return only the final answer string
        return "Final answer"

# 2. Create adapter
framework = MyFramework()
adapter = FrameworkAdapter(framework)

# 3. Run evaluation pipeline
pipeline = QAPipeline(
    tool_dir="StableToolBench/toolenv/tools",
    query_dir="StableToolBench/solvable_queries",
    model=adapter,
    use_mirrorapi_cache=True  # or False for MirrorAPI
)

results = pipeline.run()

# 4. Evaluate results (optional, if you want custom evaluation)
evaluator = StableToolBenchEvaluator(model_name="gpt-4o-mini")
for query, answer in zip(queries, answers):
    eval_result = evaluator.evaluate_answer(query, answer)
    print(f"SoPR: {eval_result['sopr_score']}, Status: {eval_result['answer_status']}")
```

### Example: Test Script

See `test_benchmark.py` for a complete example that:
- Loads queries
- Creates fake answers
- Evaluates them
- Saves results to JSON

---

## 📊 Evaluation Process

For detailed information about the evaluation process, see **[EVALUATION_PROCESS.md](EVALUATION_PROCESS.md)**.

This document covers:
- Test file structure and locations
- Tool selection process (how agents access tools)
- API execution (cache vs real calls)
- Answer comparison (SoPR and API Call Score)
- Evaluation metrics and result format

### Quick Summary

- **Test Files**: Queries loaded from `StableToolBench/solvable_queries/test_instruction/*.json`
- **Tool Selection**: Agents have access to **ALL tools** in the tool directory, not just query-specific ones
- **API Execution**: Uses cache/MirrorAPI by default (no real API calls needed)
- **Evaluation Metrics**:
  - **SoPR Score**: LLM-based semantic evaluation of final answer quality (0.0, 0.5, 1.0)
  - **API Call Score**: Proportion of correctly called APIs (0.0 to 1.0)
- **Results**: Saved to `results/tools/{framework}_{config}.json`

---

## 🐛 Troubleshooting

### Issue: "No OpenAI key found"

**Solution**: 
- Ensure `.env` file exists in `single_agent/tool_use/` directory
- Check that `OPENAI_API_KEY=your-key` is set in `.env`
- Verify the key is valid

### Issue: "Could not import evaluation modules"

**Solution**:
- Check that `StableToolBench/` directory exists
- Verify Python path includes `StableToolBench/` and `StableToolBench/toolbench/`
- The code will fall back to heuristic evaluation if imports fail

### Issue: "Finish Call: ✗ Missing"

**Solution**:
- Ensure your answer includes a "Finish" tool call in `answer_details`
- The Finish call must be in the ExecutionGraph format:
  ```python
  {
      "role": "tool",
      "message": '{"name": "Finish", "arguments": {"final_answer": "..."}, "response": ""}',
      "next": []
  }
  ```

### Issue: All answers marked as "Unsolved"

**Possible causes**:
1. **Missing Finish call**: Add a Finish call to your answer
2. **Generic final_answer**: The evaluator uses semantic judgment - make your `final_answer` specific and informative
3. **Incomplete answer**: For multi-part queries, ensure all parts are addressed

**Solution**:
- Check that `final_answer` is specific and addresses the query
- Verify Finish call exists
- Review the evaluator's `reason` field in results

### Issue: Import errors for modules

**Solution**:
- Use the new import structure:
  ```python
  from core import FrameworkInterface, FrameworkAdapter
  from pipeline import QAPipeline
  from evaluation import StableToolBenchEvaluator
  from utils import QueryLoader, AnswerValidator
  from evaluation_helpers import ResultFormatter, ResultSaver, EvaluationPrinter
  ```

### Issue: Results not saving

**Solution**:
- Check that `results/tools/` directory exists (created automatically)
- Verify write permissions
- Check the file path in your code (should be relative: `results/tools/`)

---

## 📚 Additional Resources

- **Evaluation Process Details**: [EVALUATION_PROCESS.md](EVALUATION_PROCESS.md) - Complete explanation of test files, tool selection, API execution, and answer comparison
- **Original StableToolBench README**: `StableToolBench/README.md`
- **StableToolBench Paper**: [arXiv](https://arxiv.org/pdf/2403.07714.pdf)
- **StableToolBench Project**: [Website](https://zhichengg.github.io/stb.github.io/)

---

## 🤝 Contributing

When adding new frameworks:

1. Implement `FrameworkInterface` in your framework class (from `core/framework_interface.py`)
2. Use `FrameworkAdapter` to wrap it (from `core/framework_adapter.py`)
3. Use `QAPipeline` to run evaluation (from `pipeline/qa_pipeline.py`)
4. Use `StableToolBenchEvaluator` for custom evaluation logic (from `evaluation/evaluator.py`)
5. Use `evaluation_helpers` for result formatting and saving
6. Save results to `results/tools/{framework_name}_{config}.json`

---

## 📝 Notes

- **Original Code**: All StableToolBench code is in `StableToolBench/` directory and is **not modified**
- **Wrapper Files**: Our additions are organized in `core/`, `evaluation/`, `pipeline/`, `utils/`, and `evaluation_helpers/` directories
- **API Keys**: Always use `.env` file, never commit API keys
- **Evaluation**: See [EVALUATION_PROCESS.md](EVALUATION_PROCESS.md) for detailed evaluation process
- **Finish Call**: Required by StableToolBench - must be included in answers

---

## 🙋 Questions?

For issues or questions:
1. Check this README first
2. Review [EVALUATION_PROCESS.md](EVALUATION_PROCESS.md) for evaluation details
3. Review `test_benchmark.py` for examples
4. Check original StableToolBench documentation in `StableToolBench/README.md`
