# 🧰 StableToolBench Evaluation Framework

**A modular evaluation system for testing LLM agent frameworks on tool-use tasks**

This directory provides a complete wrapper around StableToolBench to evaluate any framework (LangGraph, CrewAI, OpenAI SDK, etc.) using a unified interface. All evaluation logic, metrics, and result saving are handled automatically.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [What We Implemented](#what-we-implemented)
3. [Quick Start](#quick-start)
4. [How to Run the Server](#how-to-run-the-server)
5. [How to Run Benchmark](#how-to-run-benchmark)
6. [Evaluation Scores](#evaluation-scores)
7. [Architecture Overview](#architecture-overview)
8. [Directory Structure](#directory-structure)
9. [Troubleshooting](#troubleshooting)

---

## Overview

**Original StableToolBench**: All original code remains unchanged in `StableToolBench/` directory.  
**Our Additions**: Minimal wrapper code in `single_agent/tool_use/` (this directory).

This framework provides:
- ✅ **Benchmark Runner** (`run_benchmark.py`): Simple interface to test any agent
- ✅ **Server Modifications**: CPU-friendly API simulation with GPT fallback
- ✅ **Evaluation Wrapper**: Clean interface to StableToolBench's original evaluator
- ✅ **API Call Score**: Verifies actual tool usage by parsing `answer_details`

---

## What We Implemented

### 1. Server Modifications (`StableToolBench/server/main.py`)

**Changes**:
- ✅ Load OpenAI API key from `.env` file in root folder (`MASBench/.env`)
- ✅ Changed default model to `gpt-4o-mini` (CPU-friendly, cost-effective)
- ✅ Fixed config file path to work from any directory

**What it does**:
- Provides `/virtual` endpoint for API simulation
- Cache-first approach: checks cache, then real API, then GPT fallback
- Uses GPT-4o-mini to generate API responses when cache misses

**Original code location**: `StableToolBench/server/main.py` (modified, but minimal changes)

---

### 2. Benchmark Runner (`run_benchmark.py`)

**Purpose**: Test any framework's ability to use tools to solve problems.

**What it does**:
1. Loads queries from StableToolBench benchmark files
2. Executes gold APIs via server to generate gold answers
3. Runs your agent to generate system answers
4. Compares using StableToolBench's original evaluation
5. Saves results to `results/tools/` folder

**Key Components**:
- `GoldAnswerGenerator`: Executes gold APIs via server
- `extract_called_apis_from_answer_details()`: Parses `answer_details` to extract tool calls
- `calculate_api_call_score()`: Calculates proportion of gold APIs called
- `run_benchmark()`: Main function that orchestrates the evaluation

**Dependencies**: Uses original StableToolBench code for:
- Query loading: `StableToolBench/solvable_queries/test_instruction/*.json`
- Evaluation: `StableToolBench/toolbench/tooleval/` (via our wrapper)
- Server: `StableToolBench/server/main.py` (modified)

---

### 3. Evaluation Wrapper (`evaluation/`)

**Purpose**: Wrapper around StableToolBench's original evaluator.

**Components**:
- `StableToolBenchEvaluator`: Main evaluator class
- `EvaluatorLoader`: Loads original StableToolBench evaluator

**What it does**: Provides a clean interface to StableToolBench's evaluation without modifying original code.

---

## Quick Start

### Prerequisites

1. **Python dependencies**:
   ```bash
   pip install fastapi uvicorn python-dotenv openai pyyaml requests slowapi
   ```

2. **OpenAI API Key**: Create `.env` file in root folder (`MASBench/.env`):
   ```bash
   cd /path/to/MASBench
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```

### Step 1: Start the Server (from a separate terminal)

```bash
python single_agent/tool_use/StableToolBench/server/main.py
```

**Expected output**:
```
Loaded .env file from: /path/to/MASBench/.env
OpenAI API key loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8080
```

**Server endpoint**: `http://localhost:8080/virtual`

### Step 2: Run Benchmark

```python
from run_benchmark import run_benchmark

# Your agent must have an answer() method
class MyAgent:
    def answer(self, query: str):
        # Your agent logic here
        return {
            "answer": {
                "final_answer": "Your answer",
                "answer_details": [...]  # Tool calls in ExecutionGraph format
            }
        }

# Run benchmark
agent = MyAgent()
results = run_benchmark(
    agent=agent,
    test_set="G1_instruction",
    max_queries=10,
    agent_name="my_agent"
)
```

Results are saved to: `results/tools/{agent_name}_{test_set}_{timestamp}.json`

---

## How to Run the Server

### Prerequisites

1. **Python dependencies**:
   ```bash
   pip install fastapi uvicorn python-dotenv openai pyyaml requests slowapi
   ```

2. **OpenAI API Key**: Create `.env` file in root folder (`MASBench/.env`):
   ```bash
   cd /path/to/MASBench
   echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
   ```

### Start the Server

```bash
python single_agent/tool_use/StableToolBench/server/main.py
```

**Expected output**:
```
Loaded .env file from: /path/to/MASBench/.env
OpenAI API key loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8080
```

**Server endpoint**: `http://localhost:8080/virtual`

### How the Server Works

```
API Call Request
        ↓
1) Check cache (JSON file)
        ↓
   Cache hit? → Return cached response (instant)
        ↓
   Cache miss? → Continue
        ↓
2) Try real API call (optional, may fail)
        ↓
   Success? → Save to cache, return response
        ↓
   Failed? → Continue
        ↓
3) GPT fallback (gpt-4o-mini)
        ↓
   Generate simulated response using OpenAI API
        ↓
   Save to cache for future use
        ↓
   Return response
```

**Cache location**: `StableToolBench/server/tool_response_cache/`  
**Cache structure**: `category/tool_name/api_name.json`

For detailed server setup instructions, see: `StableToolBench/server/SERVER_SETUP.md`

---

## How to Run Benchmark

### Step 1: Implement Your Agent

Your agent must have an `answer(query: str) -> Dict[str, Any]` method:

```python
class MyAgent:
    def answer(self, query: str) -> Dict[str, Any]:
        """
        Generate answer for query.
        
        Args:
            query: The query string
            
        Returns:
            Dict with 'answer' key containing:
            {
                "final_answer": str,
                "answer_details": List[Dict]  # ExecutionGraph format
            }
        """
        # Your agent logic here
        # Can use tools, call APIs, etc.
        return {
            "answer": {
                "final_answer": "Your final answer",
                "answer_details": [...]  # Tool calls in ExecutionGraph format
            }
        }
```

### Step 2: Run Benchmark

```python
from run_benchmark import run_benchmark

# Create your agent
agent = MyAgent()

# Run benchmark
results = run_benchmark(
    agent=agent,
    test_set="G1_instruction",  # or other test sets
    max_queries=10,  # None for all queries
    server_url="http://localhost:8080/virtual",
    evaluator_model="gpt-4o-mini",
    agent_name="my_agent"
)
```

### Step 3: Check Results

Results are saved to: `results/tools/{agent_name}_{test_set}_{timestamp}.json`

**Result structure**:
```json
{
  "metadata": {
    "agent_name": "my_agent",
    "test_set": "G1_instruction",
    "num_queries": 10,
    "timestamp": "2024-01-01T12:00:00",
    "overall_time": 123.45
  },
  "summary": {
    "total_queries": 10,
    "solved_count": 7,
    "solved_percentage": 70.0,
    "average_sopr_score": 0.75,
    "average_api_call_score": 0.85,
    ...
  },
  "results": [
    {
      "query_id": "123",
      "query_text": "...",
      "gold_apis": [
        ["TheClique", "Songkick concert"],
        ["TheClique", "Songkick artist"]
      ],
      "scores": {
        "sopr_score": 1.0,
        "api_call_score": 0.5,
        "answer_status": "Solved"
      },
      "called_apis": [
        ["TheClique", "Songkick concert"]
      ],
      "timing": {
        "gold_answer_time": 0.5,
        "system_answer_time": 2.3,
        "evaluation_time": 1.2
      },
      ...
    },
    ...
  ]
}
```

---

## Evaluation Scores

### SoPR Score (Solvable Pass Rate)

**What it measures**: Whether the agent successfully solved the query.

**Values**:
- `1.0` = **Solved**: Agent successfully addressed the query
- `0.5` = **Unsure**: Agent partially addressed the query or evaluator is uncertain
- `0.0` = **Unsolved**: Agent failed to address the query

**How it's computed**: Uses StableToolBench's original evaluator from `StableToolBench/toolbench/tooleval/evaluators/registered_cls/tooleval.py`

**Original code**: `StableToolBench/toolbench/tooleval/evaluators/registered_cls/tooleval.py::OpenAINormalizedEvaluator.check_solve_query()`

The evaluator:
1. Takes query and final answer
2. Uses GPT to determine if the answer solves the query
3. Returns "Solved", "Unsure", or "Unsolved"
4. Converted to scores: Solved=1.0, Unsure=0.5, Unsolved=0.0

**Reference**: See `StableToolBench/toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/template.txt` for evaluation prompt.

---

### API Call Score

**What it measures**: Proportion of correctly called APIs (gold APIs vs system-called APIs).

**Values**: `0.0` to `1.0`

**How it's computed**:
1. Extract called APIs from system answer: Parse `answer_details` (ExecutionGraph format) to find all tool calls
2. Extract tool name and API name from each tool call's `message` field (format: `tool_name_api_name`)
3. Compare to gold APIs: Gold APIs come from `query["relevant APIs"]` or `query["relevant_apis"]`
4. Calculate intersection: How many gold APIs were actually called
5. Score: `correct_calls / total_gold_apis`

**Example**:
- Gold APIs: `[["TheClique", "Songkick concert"], ["TheClique", "Songkick artist"]]`
- System called: `[["TheClique", "Songkick concert"]]`
- Score: `1 / 2 = 0.5`

**Implementation**: 
- Function `extract_called_apis_from_answer_details()` in `run_benchmark.py` recursively parses the ExecutionGraph structure
- Function `calculate_api_call_score()` compares called APIs with gold APIs
- This verifies that the system actually used tools, not just guessed the answer

---

### Answer Status

**Values**: `"Solved"`, `"Unsure"`, `"Unsolved"`, or `"Error"`

**Source**: Same as SoPR score, from StableToolBench's evaluator.

---

### Finish Call Check

**What it measures**: Whether the answer contains a Finish call (required by StableToolBench).

**Values**: `True` or `False`

**How it's computed**: Checks if `answer_details` contains a node with `name="Finish"`.

**Reference**: `utils/answer_validator.py::check_has_finish()`

---

## Architecture Overview

This evaluation framework wraps StableToolBench's original code to provide a clean, reusable interface. The architecture consists of:

```
┌─────────────────────────────────────────────────────────┐
│              Your Agent (with answer() method)         │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              run_benchmark() Function                    │
│  - Loads queries                                         │
│  - Generates gold answers (via server)                   │
│  - Runs agent to get system answers                     │
│  - Evaluates using StableToolBenchEvaluator             │
│  - Calculates API call scores                           │
│  - Saves results to JSON                                │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         StableToolBenchEvaluator (Wrapper)               │
│  - Evaluates answers using SoPR                          │
│  - Checks for Finish calls                                │
│  - Uses original StableToolBench evaluator              │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure

```
single_agent/tool_use/
├── StableToolBench/          # Original code (mostly unchanged)
│   ├── server/
│   │   ├── main.py          # Modified: .env loading, gpt-4o-mini
│   │   ├── config.yml        # Modified: gpt-4o-mini
│   │   └── SERVER_SETUP.md   # Server setup guide
│   ├── solvable_queries/    # Original benchmark queries
│   └── toolbench/           # Original evaluation code
│
├── run_benchmark.py          # Our benchmark runner
├── utils/                    # Our utilities
│   └── query_loader.py       # Query loading
└── evaluation/               # Our evaluation wrapper
    └── evaluator.py         # Main evaluator
```

---

## Example: Running with LLM Agent

```python
from run_benchmark import run_benchmark
from openai import OpenAI
import os
import json
import requests

class LLMAgent:
    """Simple LLM agent that makes one API call for testing."""
    
    def __init__(self, model="gpt-4o-mini", server_url="http://localhost:8080/virtual"):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.server_url = server_url
    
    def answer(self, query: str):
        answer_details = []
        
        # Make one API call for testing
        try:
            api_payload = {
                "category": "Data",
                "tool_name": "TheClique",
                "api_name": "Transfermarkt search",
                "tool_input": {"query": "Lionel Messi"},
                "strip": "",
                "toolbench_key": "EMPTY"
            }
            api_response = requests.post(self.server_url, json=api_payload, timeout=10)
            if api_response.status_code == 200:
                api_result = api_response.json()
                answer_details.append({
                    "role": "tool",
                    "message": json.dumps({
                        "name": "TheClique_Transfermarkt search",
                        "arguments": {"query": "Lionel Messi"},
                        "response": api_result.get("response", "")
                    }),
                    "next": []
                })
        except Exception:
            pass
        
        # Generate final answer with LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer the question based on the information provided."},
                {"role": "user", "content": query}
            ]
        )
        
        final_answer = response.choices[0].message.content
        
        # Add Finish call
        answer_details.append({
            "role": "tool",
            "message": json.dumps({
                "name": "Finish",
                "arguments": {
                    "return_type": "give_answer",
                    "final_answer": final_answer
                },
                "response": ""
            }),
            "next": []
        })
        
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            }
        }

# Run benchmark
agent = LLMAgent()
results = run_benchmark(
    agent=agent,
    test_set="G1_instruction",
    max_queries=5,
    agent_name="llm_agent"
)
```

---

## Integration with Frameworks

Your framework agent should:
1. Accept query as input
2. Use tools (which call the server at `http://localhost:8080/virtual`)
3. Return answer in StableToolBench format

**Example**:
```python
class FrameworkAgent:
    def __init__(self):
        # Setup your framework
        self.framework = YourFramework()
        # Set SERVICE_URL so framework uses our server
        os.environ['SERVICE_URL'] = 'http://localhost:8080/virtual'
    
    def answer(self, query: str):
        # Your framework processes query
        result = self.framework.process(query)
        
        # Convert to StableToolBench format
        return {
            "answer": {
                "final_answer": result.final_answer,
                "answer_details": result.answer_details  # ExecutionGraph format
            }
        }
```

---

## Troubleshooting

### Server Issues

**Problem**: Server not starting  
**Solution**: 
- Check `.env` file exists in root folder with `OPENAI_API_KEY`
- Check port 8080 is not in use
- Check Python dependencies are installed

**Problem**: "Cache miss" errors  
**Solution**: 
- This is normal for first run
- Server will generate responses using GPT and cache them
- Subsequent runs will use cache

### Benchmark Issues

**Problem**: Agent returns wrong format  
**Solution**: 
- Ensure `answer()` returns dict with `"answer"` key
- Ensure `answer_details` is in ExecutionGraph format
- Ensure Finish call is included

**Problem**: Evaluation fails  
**Solution**: 
- Check that evaluator can access OpenAI API
- Check that answer format matches StableToolBench format
- Check server logs for API call errors

**Problem**: API call score is always 0  
**Solution**: 
- Check that `answer_details` contains tool calls (not just Finish)
- Verify tool call format: `"name": "tool_name_api_name"` (with underscore)
- Ensure tool calls are in ExecutionGraph format with `"role": "tool"`

---

## Summary

**What we added**:
- ✅ Server modifications (`.env` loading, `gpt-4o-mini`)
- ✅ Benchmark runner (`run_benchmark.py`)
- ✅ Evaluation wrapper (clean interface to original evaluator)
- ✅ API call score calculation (extracts tool calls from `answer_details`)

**What we use from original**:
- ✅ Query files (`StableToolBench/solvable_queries/`)
- ✅ Evaluation logic (`StableToolBench/toolbench/tooleval/`)
- ✅ Server infrastructure (`StableToolBench/server/main.py`)

**Scores computed**:
- ✅ **SoPR Score**: From original evaluator (`StableToolBench/toolbench/tooleval/`)
- ✅ **API Call Score**: Proportion of gold APIs called (extracted from `answer_details`)
- ✅ **Answer Status**: Solved/Unsure/Unsolved from evaluator

**Note**: API Call Score is computed by parsing `answer_details` (ExecutionGraph format) to extract tool calls. This verifies that the system actually used tools, not just guessed the answer based on the query.

**Results saved to**: `results/tools/{agent_name}_{test_set}_{timestamp}.json`

---

## References

- **Original StableToolBench**: See `StableToolBench/README.md`
- **Server Setup**: See `StableToolBench/server/SERVER_SETUP.md`
- **Evaluation Details**: See `StableToolBench/toolbench/tooleval/README.md`
- **Server Code**: `StableToolBench/server/main.py`
- **Benchmark Runner**: `run_benchmark.py` (this directory)
