# Evaluation Process and Test Files Explanation

This document explains the complete evaluation process for StableToolBench, including test file structure, tool selection, API execution (cache vs real calls), and how system answers are compared to gold answers.

## Table of Contents

1. [Test Files Overview](#test-files-overview)
2. [Test File Structure](#test-file-structure)
3. [Tool Selection Process](#tool-selection-process)
4. [API Execution: Cache vs Real Calls](#api-execution-cache-vs-real-calls)
5. [Answer Comparison](#answer-comparison)
6. [Evaluation Metrics](#evaluation-metrics)

---

## Test Files Overview

### Test File Locations

The evaluation uses test files from StableToolBench located in:
```
single_agent/tool_use/StableToolBench/solvable_queries/
├── test_instruction/          # Full query data with all information
│   ├── G1_instruction.json    # Test set G1 queries
│   ├── G2_instruction.json    # Test set G2 queries
│   └── G3_instruction.json    # Test set G3 queries
└── test_query_ids/            # Query ID filters (optional)
    ├── G1_instruction.json    # List of query IDs to evaluate
    ├── G2_instruction.json
    └── G3_instruction.json
```

### Test Sets

- **G1_instruction**: General instruction-following queries
- **G2_instruction**: Category-specific queries
- **G3_instruction**: Tool-specific queries

Each test set contains queries that require the agent to use specific tools/APIs to answer correctly.

---

## Test File Structure

### Query Structure (from `test_instruction/*.json`)

Each query in the JSON file has the following structure:

```json
{
  "query_id": "588",
  "query": "I'm a football enthusiast and I want to know more about Lionel Messi's career...",
  "relevant APIs": [
    ["TheClique", "Transfermarkt details"],
    ["TheClique", "Songkick concert"]
  ],
  "api_list": [
    {
      "category_name": "Data",
      "tool_name": "TheClique",
      "api_name": "Transfermarkt details",
      "api_description": "Player career information",
      "required_parameters": [...],
      "optional_parameters": [...],
      "method": "GET",
      "template_response": {...}
    },
    {
      "category_name": "Data",
      "tool_name": "TheClique",
      "api_name": "Songkick concert",
      "api_description": "Concert info",
      ...
    }
  ]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `query_id` | Unique identifier for the query |
| `query` | The user's question/instruction |
| `relevant APIs` | List of `[tool_name, api_name]` pairs that should be called to answer the query correctly (gold standard) |
| `api_list` | Detailed information about each relevant API, including parameters, descriptions, and response templates |

### Query ID Filter Files (from `test_query_ids/*.json`)

These files contain lists of query IDs to evaluate. They can be:
- A list of query IDs: `["588", "608", "1073"]`
- A dictionary mapping query IDs to metadata: `{"588": {...}, "608": {...}}`

If provided, only queries with IDs in these files will be evaluated.

---

## Tool Selection Process

### How Tools Are Made Available

#### 1. **Tool Loading from Tool Directory**

In a real evaluation (not using fake answers), tools are loaded from the StableToolBench tool directory:

```
StableToolBench/tools/
├── category1/
│   └── tool_name/
│       └── api_name.json
├── category2/
│   └── tool_name/
│       └── api_name.json
...
```

The `QAPipeline` loads all tools from this directory and makes them available to the agent.

#### 2. **Tool Availability per Query**

**Important:** The agent has access to **ALL tools** in the tool directory, not just the tools listed in the query's `api_list`.

- The `api_list` in each query shows the **relevant APIs** (gold standard) that should be called
- The agent can call **any tool** from the entire tool directory
- This tests whether the agent can:
  - Select the correct tools from a large pool
  - Understand which tools are relevant to the query
  - Avoid calling irrelevant tools

#### 3. **Tool Selection in Our Code**

In our evaluation setup:

```python
# Tools are loaded from the tool directory
tool_dir = "StableToolBench/tools"

# The framework receives ALL tools via setup_tools()
framework.setup_tools(all_tools)  # all_tools is a dict: {tool_name: tool_definition}

# For each query, the agent must select which tools to use
answer = framework.answer(query_text)
```

The framework's `answer()` method receives only the query text and must:
1. Select relevant tools from all available tools
2. Call the appropriate APIs
3. Process the responses
4. Return a final answer

#### 4. **Available Tools in Results**

In the evaluation results, `available_tools` is populated from the query's `api_list`:

```python
# From utils/query_loader.py
available_tools = QueryLoader.extract_available_tools(query)
```

This extracts tool information from the query's `api_list` to show which tools were relevant to this specific query. However, the agent had access to many more tools during execution.

---

## API Execution: Cache vs Real Calls

### Overview

When an agent calls a tool/API during evaluation, StableToolBench provides multiple execution modes to avoid making real API calls (which can be expensive, rate-limited, or require authentication). This section explains how API calls are executed.

### Execution Modes

#### 1. **MirrorAPI Cache Mode (Default)**

**Configuration:**
```python
# In pipeline/qa_pipeline.py
pipeline = QAPipeline(
    tool_dir=tool_dir,
    query_dir=query_dir,
    model=model,
    use_mirrorapi_cache=True  # Enable MirrorAPI cache
)
```

**How it works:**
- **MirrorAPI**: A trained model that simulates API responses without making real calls
- **Cache**: Previously executed API calls are cached for reproducibility
- When an agent calls an API:
  1. First checks if the call exists in cache
  2. If cached, returns the cached response immediately
  3. If not cached, uses MirrorAPI to generate a simulated response
  4. Caches the response for future use

**Benefits:**
- ✅ No real API calls needed
- ✅ Reproducible results (same inputs = same outputs)
- ✅ Fast execution
- ✅ No API keys or authentication required
- ✅ No rate limiting issues

**Cache Location:**
```
StableToolBench/
└── cache/
    └── <category>/
        └── <tool_name>/
            └── <api_name>.json
```

**Example:**
```python
# Agent calls: TheClique.Transfermarkt_details({"player_id": "messi"})
# 1. Check cache: cache/Data/TheClique/Transfermarkt_details.json
# 2. If found: Return cached response
# 3. If not: Use MirrorAPI to simulate response, then cache it
```

#### 2. **Real API Calls Mode**

**Configuration:**
```python
pipeline = QAPipeline(
    tool_dir=tool_dir,
    query_dir=query_dir,
    model=model,
    use_mirrorapi_cache=False  # Disable cache, use real APIs
)
```

**How it works:**
- Makes actual HTTP requests to real API endpoints
- Requires:
  - API keys/authentication
  - Network connectivity
  - Handling rate limits
  - Cost considerations

**When to use:**
- Testing with real-world API behavior
- Validating against actual API responses
- Production-like evaluation

**Note:** This mode is rarely used in evaluation due to cost and reproducibility concerns.

#### 3. **GPT-Based Caching (Alternative)**

For environments without GPU access (needed for MirrorAPI), you can use GPT-based caching:

**How it works:**
- Uses OpenAI API to simulate API responses
- Caches responses for reproducibility
- Similar to MirrorAPI but uses GPT instead of a fine-tuned model

**Configuration:**
- Set up OpenAI API key in `.env` file
- The system will use GPT to generate cached responses

### API Call Flow Diagram

```
Agent calls API
    │
    ├─→ Check cache exists?
    │   │
    │   ├─→ YES → Return cached response ✅
    │   │
    │   └─→ NO → Generate response
    │       │
    │       ├─→ use_mirrorapi_cache=True
    │       │   └─→ MirrorAPI generates response
    │       │       └─→ Cache response
    │       │           └─→ Return response
    │       │
    │       └─→ use_mirrorapi_cache=False
    │           └─→ Make real API call
    │               └─→ Cache response (optional)
    │                   └─→ Return response
```

### Implementation Details

#### In Our Code

**Pipeline Setup:**
```python
# From pipeline/qa_pipeline.py
class QAPipeline:
    def __init__(
        self,
        tool_dir: str,
        query_dir: str,
        model: Any,
        use_mirrorapi_cache: bool = True  # Default: use cache
    ):
        self.pipeline = STBQAPipeline(
            tool_root_dir=tool_dir,
            query_dir=query_dir,
            model=model,
            use_mirrorapi_cache=use_mirrorapi_cache,  # Pass to StableToolBench
        )
```

**Framework Execution:**
```python
# When framework calls a tool
framework.setup_tools(all_tools)  # Tools are set up
answer = framework.answer(query)  # Agent processes query

# During answer(), when agent calls a tool:
# 1. Framework makes tool call
# 2. StableToolBench intercepts the call
# 3. Checks cache or uses MirrorAPI
# 4. Returns response to framework
# 5. Framework continues processing
```

#### Cache Structure

Cached responses are stored as JSON files:

```json
{
  "{'player_id': 'messi'}": {
    "error": "",
    "response": {
      "name": "Lionel Messi",
      "clubs": [...],
      "stats": {...}
    }
  },
  "{'player_id': 'ronaldo'}": {
    "error": "",
    "response": {...}
  }
}
```

The key is the stringified tool input parameters, and the value is the API response.

### Advantages of Cache Mode

1. **Reproducibility**: Same query always produces same results
2. **Speed**: No network latency
3. **Cost**: No API usage fees
4. **Reliability**: No rate limits or API downtime
5. **Testing**: Can test evaluation logic without real APIs

### Cache Management

**Cache Location:**
- Default: `StableToolBench/cache/`
- Can be configured via environment variables

**Cache Invalidation:**
- Cache persists across runs
- To refresh: Delete cache files or use `use_mirrorapi_cache=False`

**Cache Size:**
- Grows as more API calls are made
- Can become large with many tools/queries
- Safe to delete and regenerate

### MirrorAPI Model

**What is MirrorAPI?**
- A neural model trained to simulate API responses
- Trained on 7k+ tools from ToolBench
- Available from HuggingFace
- Requires GPU for inference (or use GPT-based alternative)

**Download:**
```bash
# MirrorAPI model can be downloaded from HuggingFace
# See StableToolBench/README.md for details
```

**For CPU-only environments:**
- Use GPT-based caching (set OpenAI API key)
- Or use pre-populated cache files

---

## Answer Comparison

### Gold Answer Structure

The **gold answer** (expected answer) is derived from the query:

```json
{
  "gold_answer": {
    "relevant_apis": [
      ["TheClique", "Transfermarkt details"],
      ["TheClique", "Songkick concert"]
    ],
    "available_tools": [
      {
        "category": "Data",
        "tool_name": "TheClique",
        "apis": [
          {"api_name": "Transfermarkt details", ...},
          {"api_name": "Songkick concert", ...}
        ]
      }
    ]
  }
}
```

### System Answer Structure

The **system answer** (agent's response) has this structure:

```json
{
  "system_answer": {
    "final_answer": "I have gathered information about Lionel Messi...",
    "answer_details": [
      {
        "role": "system",
        "message": "",
        "next": [
          {
            "role": "tool",
            "message": "{\"name\": \"TheClique_Transfermarkt details\", \"arguments\": {...}, \"response\": \"...\"}",
            "next": [...]
          }
        ]
      }
    ]
  }
}
```

### Comparison Process

The evaluation compares system answers to gold answers using two main metrics:

#### 1. **SoPR Score (Solvable Pass Rate)**

Evaluates whether the final answer correctly solves the query:

- **Score: 1.0 (Solved)** - The answer correctly addresses the query
- **Score: 0.5 (Unsure)** - The answer partially addresses the query
- **Score: 0.0 (Unsolved)** - The answer does not solve the query

**How it works:**
- Uses an LLM-as-a-Judge (OpenAI GPT model) to semantically evaluate the `final_answer`
- The evaluator compares the final answer text to the query requirements
- Considers whether the answer is complete, accurate, and addresses all aspects of the query

**Code location:** `evaluation/evaluator.py` → `_evaluate_with_official()`

#### 2. **API Call Score**

Measures the proportion of correctly called APIs:

```
API Call Score = (Number of correctly called APIs) / (Total number of gold APIs)
```

**How it works:**
1. Extract called APIs from `answer_details`:
   - Parse the execution graph structure
   - Find all tool calls (excluding "Finish")
   - Extract `[tool_name, api_name]` pairs

2. Compare to gold APIs:
   - Gold APIs come from `query["relevant APIs"]`
   - Match called APIs to gold APIs
   - Calculate intersection

3. Score calculation:
   ```python
   correct_calls = len(called_apis.intersection(gold_apis))
   score = correct_calls / len(gold_apis) if gold_apis else 0.0
   ```

**Code location:** `evaluation/api_scorer.py`

### Example Comparison

**Query:** "Tell me about Lionel Messi's career"

**Gold APIs:**
- `["TheClique", "Transfermarkt details"]`
- `["TheClique", "Songkick concert"]`

**System Called APIs:**
- `["TheClique", "Transfermarkt details"]` ✓
- `["TheClique", "Songkick concert"]` ✓

**Result:**
- API Call Score: 2/2 = 1.0 (all APIs called correctly)
- SoPR Score: Depends on final answer quality (evaluated by LLM)

### Additional Checks

#### Finish Call Detection

The evaluator also checks if the answer contains a "Finish" call:

```python
# From utils/answer_validator.py
has_finish = AnswerValidator.check_has_finish(answer)
```

A "Finish" call indicates the agent has completed its reasoning and is providing the final answer. This is required by StableToolBench's format.

---

## Evaluation Metrics

### Summary Statistics

After evaluating all queries, the following statistics are calculated:

| Metric | Description |
|--------|-------------|
| `total_queries` | Number of unique queries evaluated |
| `total_evaluations` | Total number of query-answer pairs evaluated |
| `solved_count` | Number of answers with SoPR = 1.0 |
| `solved_percentage` | Percentage of solved answers |
| `finish_count` | Number of answers with Finish call |
| `average_sopr_score` | Average SoPR score across all evaluations |
| `average_api_call_score` | Average API call score across all evaluations |
| `total_evaluation_time` | Total time spent on evaluation |
| `average_evaluation_time` | Average time per evaluation |

### Result File Structure

Results are saved to `results/tools/<framework>_<config>.json`:

```json
{
  "metadata": {
    "framework": "Test_benchmark",
    "config": {
      "test_set": "G1_instruction",
      "num_queries": 3,
      "evaluator_model": "gpt-4o-mini",
      "evaluation_method": "official"
    },
    "overall_time": 15.58,
    "timestamp": "2024-01-15 10:30:45"
  },
  "summary": {
    "total_queries": 3,
    "total_evaluations": 9,
    "solved_count": 0,
    "solved_percentage": 0.0,
    "average_sopr_score": 0.056,
    "average_api_call_score": 0.500,
    ...
  },
  "results": [
    {
      "query_id": "588",
      "query_text": "...",
      "gold_answer": {...},
      "system_answer": {...},
      "scores": {
        "sopr_score": 0.0,
        "api_call_score": 1.0
      },
      "evaluation": {
        "answer_status": "Unsolved",
        "has_finish": false,
        "evaluation_method": "official"
      },
      "timing": {
        "evaluation_time": 1.42
      }
    },
    ...
  ]
}
```

---

## Key Points Summary

1. **Test Files**: Queries are loaded from `test_instruction/*.json` files, optionally filtered by `test_query_ids/*.json`

2. **Tool Selection**: 
   - Agents have access to **ALL tools** in the tool directory
   - Each query's `api_list` shows the **relevant APIs** (gold standard)
   - Agents must select the correct tools from the entire pool

3. **Answer Comparison**:
   - **SoPR Score**: LLM-based semantic evaluation of final answer quality
   - **API Call Score**: Exact match of called APIs to gold APIs
   - Both metrics are combined to assess overall performance

4. **Evaluation Process**:
   - Load queries and tools
   - For each query, agent generates answer using available tools
   - Compare system answer to gold answer using both metrics
   - Aggregate results and save to JSON

---

## Code References

- **Query Loading**: `utils/query_loader.py`
- **Tool Loading**: `pipeline/qa_pipeline.py` (uses StableToolBench's QAPipeline)
- **API Execution**: `pipeline/qa_pipeline.py` → `use_mirrorapi_cache` parameter
- **Answer Evaluation**: `evaluation/evaluator.py`
- **API Scoring**: `evaluation/api_scorer.py`
- **Result Formatting**: `evaluation_helpers/result_formatter.py`
- **Result Saving**: `evaluation_helpers/result_saver.py`

## Summary

### Evaluation Flow

1. **Setup**: Load queries and tools
2. **Tool Selection**: Agent selects tools from entire tool pool
3. **API Execution**: 
   - Check cache → Use cached response OR
   - Generate with MirrorAPI → Cache and return OR
   - Make real API call (if cache disabled)
4. **Answer Generation**: Agent processes API responses and generates final answer
5. **Evaluation**: Compare system answer to gold answer using SoPR and API Call Score
6. **Results**: Save formatted results with metrics

### Key Points

- ✅ Agents have access to **ALL tools**, not just query-specific ones
- ✅ API calls use **cache/MirrorAPI by default** (no real API calls)
- ✅ Evaluation uses **two metrics**: SoPR (semantic) and API Call Score (exact match)
- ✅ Results are **reproducible** due to caching
- ✅ No API keys needed for evaluation (when using cache)

