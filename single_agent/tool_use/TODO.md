# TODO: API Execution Based on Gold and System Answers

## Goal

Create a component that can execute API calls based on comparing gold answers (expected APIs) and system answers (called APIs), returning cached responses. This will allow us to:
- Validate that system's API calls would actually work
- Get actual API responses for analysis
- Compare expected vs actual API responses
- Test API execution without running full framework evaluation

---

## Current State

### What We Have Now

1. **`test_benchmark.py`**: Creates fake answers and evaluates them (no actual API calls)
2. **`QAPipeline`**: Runs full framework evaluation with cached API responses
3. **`StableToolBenchEvaluator`**: Evaluates answers using SoPR and API Call Score
4. **Cache System**: StableToolBench's cache/MirrorAPI system for API responses

### What's Missing

- Component to execute specific API calls based on gold/system answer comparison
- Component to retrieve cached responses for specific API calls
- Integration between answer comparison and API execution

---

## Implementation Plan

### Phase 1: API Call Executor Component

**File**: `utils/api_executor.py`

**Purpose**: Execute API calls and retrieve cached responses

**Responsibilities**:
- Load tool definitions from tool directory
- Execute API calls with given parameters
- Retrieve cached responses (or generate new ones via MirrorAPI)
- Return API responses in structured format

**Methods**:
```python
class APIExecutor:
    def __init__(self, tool_dir: str, use_cache: bool = True):
        """Initialize with tool directory and cache settings"""
    
    def execute_api_call(
        self, 
        tool_name: str, 
        api_name: str, 
        parameters: dict
    ) -> dict:
        """Execute a single API call and return cached response"""
    
    def execute_api_calls(
        self, 
        api_calls: List[Tuple[str, str, dict]]
    ) -> List[dict]:
        """Execute multiple API calls and return responses"""
    
    def get_cached_response(
        self, 
        tool_name: str, 
        api_name: str, 
        parameters: dict
    ) -> Optional[dict]:
        """Check if cached response exists and return it"""
```

**Dependencies**:
- StableToolBench's cache system
- Tool definitions from tool directory
- MirrorAPI or GPT-based caching

---

### Phase 2: Answer Comparison and API Extraction

**File**: `utils/answer_comparator.py`

**Purpose**: Compare gold and system answers, extract API calls to execute

**Responsibilities**:
- Extract API calls from system answer
- Compare with gold APIs
- Identify which APIs to execute
- Prepare API call parameters

**Methods**:
```python
class AnswerComparator:
    def extract_api_calls_to_execute(
        self,
        query: dict,
        system_answer: dict
    ) -> List[dict]:
        """
        Extract API calls from system answer that should be executed.
        Returns list of {
            "tool_name": str,
            "api_name": str,
            "parameters": dict,
            "is_gold": bool,  # Whether this is in gold APIs
            "is_called": bool  # Whether system actually called it
        }
        """
    
    def compare_gold_vs_system(
        self,
        query: dict,
        system_answer: dict
    ) -> dict:
        """
        Compare gold APIs with system's called APIs.
        Returns comparison results with execution plan.
        """
```

---

### Phase 3: API Response Retriever

**File**: `utils/api_response_retriever.py`

**Purpose**: Retrieve cached API responses for gold and system API calls

**Responsibilities**:
- Execute gold API calls and get responses
- Execute system API calls and get responses
- Compare responses
- Return structured results

**Methods**:
```python
class APIResponseRetriever:
    def __init__(
        self, 
        api_executor: APIExecutor,
        answer_comparator: AnswerComparator
    ):
        """Initialize with executor and comparator"""
    
    def retrieve_responses(
        self,
        query: dict,
        system_answer: dict
    ) -> dict:
        """
        Retrieve cached responses for both gold and system API calls.
        Returns:
        {
            "gold_responses": [
                {
                    "tool_name": str,
                    "api_name": str,
                    "parameters": dict,
                    "response": dict,
                    "cached": bool
                }
            ],
            "system_responses": [
                {
                    "tool_name": str,
                    "api_name": str,
                    "parameters": dict,
                    "response": dict,
                    "cached": bool,
                    "matches_gold": bool
                }
            ],
            "comparison": {
                "gold_only": [...],
                "system_only": [...],
                "common": [...],
                "response_differences": [...]
            }
        }
        """
```

---

### Phase 4: Integration with Evaluation

**File**: `evaluation/api_response_evaluator.py`

**Purpose**: Evaluate API responses in addition to answer evaluation

**Responsibilities**:
- Integrate API response retrieval with evaluation
- Add API response analysis to results
- Compare expected vs actual API responses

**Methods**:
```python
class APIResponseEvaluator:
    def __init__(
        self,
        api_response_retriever: APIResponseRetriever,
        base_evaluator: StableToolBenchEvaluator
    ):
        """Initialize with retriever and base evaluator"""
    
    def evaluate_with_api_responses(
        self,
        query: dict,
        system_answer: dict
    ) -> dict:
        """
        Evaluate answer AND retrieve API responses.
        Returns evaluation results + API response data.
        """
```

---

### Phase 5: Test Script

**File**: `tests/test_api_execution.py`

**Purpose**: Test API execution with cached responses

**Test Cases**:
1. Execute gold API calls and verify cached responses
2. Execute system API calls and compare with gold
3. Test cache hit/miss scenarios
4. Test with different query types
5. Verify response structure

---

## Implementation Steps (Tomorrow)

### Step 1: Create API Executor (2-3 hours)

1. Create `utils/api_executor.py`
2. Implement tool loading from tool directory
3. Implement cache checking logic
4. Integrate with StableToolBench's cache system
5. Test with single API call

**Dependencies to understand**:
- How StableToolBench's cache system works
- Tool definition format
- Cache file structure
- MirrorAPI integration

**Files to review**:
- `StableToolBench/toolbench/inference/Downstream_tasks/rapidapi.py`
- `StableToolBench/server/main_mirrorapi_cache.py`
- Cache files in `StableToolBench/cache/`

---

### Step 2: Create Answer Comparator (1-2 hours)

1. Create `utils/answer_comparator.py`
2. Implement API extraction from system answer
3. Implement comparison logic
4. Test with various answer formats

**Dependencies**:
- `utils/query_loader.py` (for gold API extraction)
- `evaluation/api_scorer.py` (for API call extraction)

---

### Step 3: Create API Response Retriever (2-3 hours)

1. Create `utils/api_response_retriever.py`
2. Integrate APIExecutor and AnswerComparator
3. Implement response comparison
4. Add caching status tracking
5. Test end-to-end

---

### Step 4: Integration and Testing (2-3 hours)

1. Create test script
2. Test with real queries
3. Verify cache usage
4. Compare results
5. Document usage

---

## Technical Considerations

### Cache Location

- Default: `StableToolBench/cache/<category>/<tool_name>/<api_name>.json`
- Need to understand cache key format (parameter stringification)
- May need to handle cache misses gracefully

### Tool Definition Format

- Tools are in: `StableToolBench/toolenv/tools/<category>/<tool_name>.json`
- Need to parse tool definitions to get API specifications
- Extract required/optional parameters

### API Call Format

- System answers have API calls in `answer_details` structure
- Need to parse ExecutionGraph format
- Extract tool name, API name, and parameters

### Error Handling

- Handle missing tools
- Handle missing cache entries
- Handle invalid API calls
- Handle cache generation failures

---

## Expected Output

### New Components

```
utils/
├── api_executor.py          # Execute API calls, get cached responses
├── answer_comparator.py      # Compare gold vs system APIs
└── api_response_retriever.py # Retrieve and compare API responses

evaluation/
└── api_response_evaluator.py # Evaluate with API responses

tests/
└── test_api_execution.py     # Test API execution
```

### Usage Example

```python
from utils import APIExecutor, AnswerComparator, APIResponseRetriever

# Initialize components
executor = APIExecutor(tool_dir="StableToolBench/toolenv/tools")
comparator = AnswerComparator()
retriever = APIResponseRetriever(executor, comparator)

# Retrieve API responses
responses = retriever.retrieve_responses(query, system_answer)

# responses contains:
# - gold_responses: Cached responses for gold APIs
# - system_responses: Cached responses for system APIs
# - comparison: Detailed comparison results
```

---

## Questions to Resolve

1. **Cache Key Format**: How are parameters stringified for cache keys?
   - Need to check StableToolBench's cache key generation
   - May need to match exact format

2. **Tool Loading**: How to load tool definitions efficiently?
   - Load all tools upfront vs lazy loading
   - Cache tool definitions

3. **Parameter Extraction**: How to extract parameters from system answer?
   - System answer has parameters in `answer_details`
   - Need to parse ExecutionGraph format correctly

4. **Cache Miss Handling**: What to do when cache doesn't exist?
   - Use MirrorAPI to generate?
   - Use GPT-based caching?
   - Skip and mark as missing?

5. **Integration Point**: Where to integrate this in evaluation flow?
   - As part of `StableToolBenchEvaluator`?
   - Separate component?
   - Optional feature?

---

## Success Criteria

- ✅ Can execute gold API calls and retrieve cached responses
- ✅ Can execute system API calls and retrieve cached responses
- ✅ Can compare gold vs system API responses
- ✅ Handles cache hits and misses gracefully
- ✅ Integrates with existing evaluation system
- ✅ Tested with real queries
- ✅ Documented with examples

---

## Notes

- This feature is **optional** - existing evaluation works without it
- Focus on **cached responses** first (no real API calls)
- Can extend later to support real API calls if needed
- Should not modify StableToolBench original code
- Follow SOLID principles in implementation

---

## References

- StableToolBench cache system: `StableToolBench/cache/`
- Tool definitions: `StableToolBench/toolenv/tools/`
- API execution: `StableToolBench/toolbench/inference/Downstream_tasks/rapidapi.py`
- MirrorAPI server: `StableToolBench/server/main_mirrorapi_cache.py`

