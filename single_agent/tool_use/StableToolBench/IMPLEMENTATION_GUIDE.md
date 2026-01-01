# StableToolBench Implementation Guide

## Overview

This document explains what we implemented on top of the original StableToolBench code, how to run the server, and how to use the benchmark runner to test framework ability for tool-use tasks.

**Original StableToolBench**: All original code remains unchanged in this directory.  
**Our Additions**: Minimal wrapper code in `single_agent/tool_use/` (outside this directory).

---

## What We Implemented

### 1. Server Modifications (`server/main.py`)

**Changes**:
- ✅ Load OpenAI API key from `.env` file in root folder (`MASBench/.env`)
- ✅ Changed default model to `gpt-4o-mini` (CPU-friendly, cost-effective)
- ✅ Fixed config file path to work from any directory

**What it does**:
- Provides `/virtual` endpoint for API simulation
- Cache-first approach: checks cache, then real API, then GPT fallback
- Uses GPT-4o-mini to generate API responses when cache misses

**Original code location**: `server/main.py` (modified, but minimal changes)

---

### 2. Benchmark Runner (`single_agent/tool_use/run_benchmark.py`)

**Purpose**: Test any framework's ability to use tools to solve problems.

**What it does**:
1. Loads queries from StableToolBench benchmark files
2. Executes gold APIs via server to generate gold answers
3. Runs your agent to generate system answers
4. Compares using StableToolBench's original evaluation
5. Saves results to `results/tools/` folder

**Key Components**:
- `GoldAnswerGenerator`: Executes gold APIs via server
- `run_benchmark()`: Main function that orchestrates the evaluation

**Dependencies**: Uses original StableToolBench code for:
- Query loading: `solvable_queries/test_instruction/*.json`
- Evaluation: `toolbench/tooleval/` (via our wrapper)
- Server: `server/main.py` (modified)

---

### 3. Evaluation Wrapper (`single_agent/tool_use/evaluation/`)

**Purpose**: Wrapper around StableToolBench's original evaluator.

**Components**:
- `StableToolBenchEvaluator`: Main evaluator class
- `APIScorer`: Calculates API call scores
- `EvaluatorLoader`: Loads original StableToolBench evaluator

**What it does**: Provides a clean interface to StableToolBench's evaluation without modifying original code.

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
   Generate fake response using OpenAI API
        ↓
   Save to cache for future use
        ↓
   Return response
```

**Cache location**: `StableToolBench/server/tool_response_cache/`  
**Cache structure**: `category/tool_name/api_name.json`

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
from single_agent.tool_use.run_benchmark import run_benchmark

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
      "scores": {
        "sopr_score": 1.0,
        "api_call_score": 0.9,
        "answer_status": "Solved"
      },
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

### 1. SoPR Score (Solvable Pass Rate)

**What it measures**: Whether the agent successfully solved the query.

**Values**:
- `1.0` = **Solved**: Agent successfully addressed the query
- `0.5` = **Unsure**: Agent partially addressed the query or evaluator is uncertain
- `0.0` = **Unsolved**: Agent failed to address the query

**How it's computed**: Uses StableToolBench's original evaluator from `toolbench/tooleval/evaluators/registered_cls/tooleval.py`

**Original code**: `toolbench/tooleval/evaluators/registered_cls/tooleval.py::OpenAINormalizedEvaluator.check_solve_query()`

The evaluator:
1. Takes query and final answer
2. Uses GPT to determine if the answer solves the query
3. Returns "Solved", "Unsure", or "Unsolved"
4. Converted to scores: Solved=1.0, Unsure=0.5, Unsolved=0.0

**Reference**: See `toolbench/tooleval/evaluators/tooleval_gpt-3.5-turbo_default/template.txt` for evaluation prompt.

---

### 2. API Call Score

**What it measures**: Proportion of correctly called APIs (gold APIs vs system-called APIs).

**Values**: `0.0` to `1.0`

**How it's computed**:
```python
# From evaluation/api_scorer.py
correct_calls = len(called_apis.intersection(gold_apis))
api_call_score = correct_calls / len(gold_apis)
```

**Steps**:
1. Extract gold APIs from query: `query["relevant APIs"]` or `query["relevant_apis"]`
2. Extract called APIs from system answer: Parse `answer_details` (ExecutionGraph format)
3. Calculate intersection: How many gold APIs were actually called
4. Score: `correct_calls / total_gold_apis`

**Example**:
- Gold APIs: `[["TheClique", "Songkick concert"], ["TheClique", "Songkick artist"]]`
- System called: `[["TheClique", "Songkick concert"]]`
- Score: `1 / 2 = 0.5`

---

### 3. Answer Status

**Values**: `"Solved"`, `"Unsure"`, `"Unsolved"`, or `"Error"`

**Source**: Same as SoPR score, from StableToolBench's evaluator.

---

### 4. Finish Call Check

**What it measures**: Whether the answer contains a Finish call (required by StableToolBench).

**Values**: `True` or `False`

**How it's computed**: Checks if `answer_details` contains a node with `name="Finish"`.

**Reference**: `utils/answer_validator.py::check_has_finish()`

---

## How Results Are Saved

### Output Location

Results are saved to: `results/tools/{agent_name}_{test_set}_{timestamp}.json`

**Example**: `results/tools/my_agent_G1_instruction_20240101_120000.json`

### Result Structure

```json
{
  "metadata": {
    "agent_name": "my_agent",
    "test_set": "G1_instruction",
    "num_queries": 10,
    "evaluator_model": "gpt-4o-mini",
    "server_url": "http://localhost:8080/virtual",
    "timestamp": "2024-01-01T12:00:00",
    "overall_time": 123.45
  },
  "summary": {
    "total_queries": 10,
    "solved_count": 7,
    "solved_percentage": 70.0,
    "unsure_count": 1,
    "unsure_percentage": 10.0,
    "unsolved_count": 2,
    "unsolved_percentage": 20.0,
    "finish_count": 9,
    "finish_percentage": 90.0,
    "average_sopr_score": 0.75,
    "average_api_call_score": 0.85,
    "average_gold_answer_time": 0.5,
    "average_system_answer_time": 2.3,
    "average_evaluation_time": 1.2,
    "overall_time": 123.45
  },
  "results": [
    {
      "query_id": "123",
      "query_text": "Find concert info for Arctic Monkeys",
      "gold_apis": [
        ["TheClique", "Songkick concert"],
        ["TheClique", "Songkick artist"]
      ],
      "gold_answer": {
        "final_answer": "...",
        "has_finish": true
      },
      "system_answer": {
        "final_answer": "...",
        "has_finish": true
      },
      "scores": {
        "sopr_score": 1.0,
        "api_call_score": 0.9,
        "answer_status": "Solved"
      },
      "evaluation": {
        "has_finish": true,
        "evaluation_method": "official",
        "reason": "..."
      },
      "timing": {
        "gold_answer_time": 0.5,
        "system_answer_time": 2.3,
        "evaluation_time": 1.2,
        "total_time": 4.0
      }
    },
    ...
  ]
}
```

### Timing Information

- **gold_answer_time**: Time to execute gold APIs via server
- **system_answer_time**: Time for agent to generate answer
- **evaluation_time**: Time for StableToolBench evaluator to score
- **total_time**: Total time per query
- **overall_time**: Total time for entire benchmark

---

## Example: Running with LLM Agent (No Tools)

```python
from single_agent.tool_use.run_benchmark import run_benchmark
from openai import OpenAI
import os
import json

class LLMAgent:
    """Simple LLM agent that answers without tools."""
    
    def __init__(self, model="gpt-4o-mini"):
        api_key = os.getenv('OPENAI_API_KEY')
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def answer(self, query: str):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "Answer the question directly."},
                {"role": "user", "content": query}
            ]
        )
        
        final_answer = response.choices[0].message.content
        
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": [
                    {
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
                    }
                ]
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

### Using with LangGraph, CrewAI, etc.

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

---

## File Structure

```
single_agent/tool_use/
├── StableToolBench/          # Original code (mostly unchanged)
│   ├── server/
│   │   ├── main.py          # Modified: .env loading, gpt-4o-mini
│   │   └── config.yml        # Modified: gpt-4o-mini
│   ├── solvable_queries/    # Original benchmark queries
│   └── toolbench/           # Original evaluation code
│
├── run_benchmark.py          # Our benchmark runner
├── utils/                    # Our utilities
│   └── query_loader.py       # Query loading
└── evaluation/               # Our evaluation wrapper
    ├── evaluator.py         # Main evaluator
    └── api_scorer.py        # API call scoring
```

---

## Summary

**What we added**:
- ✅ Server modifications (`.env` loading, `gpt-4o-mini`)
- ✅ Benchmark runner (`run_benchmark.py`)
- ✅ Evaluation wrapper (clean interface to original evaluator)

**What we use from original**:
- ✅ Query files (`solvable_queries/`)
- ✅ Evaluation logic (`toolbench/tooleval/`)
- ✅ Server infrastructure (`server/main.py`)

**Scores computed**:
- ✅ **SoPR Score**: From original evaluator (`toolbench/tooleval/`)
- ✅ **API Call Score**: Proportion of gold APIs called
- ✅ **Answer Status**: Solved/Unsure/Unsolved from evaluator

**Results saved to**: `results/tools/{agent_name}_{test_set}_{timestamp}.json`

---

## References

- **Original StableToolBench**: See `README.md` in this directory
- **Evaluation Details**: See `toolbench/tooleval/README.md`
- **Server Code**: `server/main.py`
- **Benchmark Runner**: `../run_benchmark.py` (parent directory)

