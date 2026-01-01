# File Usage Analysis for `run_benchmark.py`

## Files USED by `run_benchmark.py` (KEEP):

### Direct imports:
1. `run_benchmark.py` - Main file
2. `utils/query_loader.py` - QueryLoader (directly imported)
3. `evaluation/evaluator.py` - StableToolBenchEvaluator (directly imported)

### Indirect dependencies (used by evaluator):
4. `utils/answer_validator.py` - AnswerValidator (used by evaluator)
5. `evaluation/evaluator_loader.py` - EvaluatorLoader (used by evaluator)
6. `evaluation/api_scorer.py` - APIScorer (used by evaluator internally)
7. `evaluation/heuristic_evaluator.py` - HeuristicEvaluator (used by evaluator)

### Required `__init__.py` files:
8. `utils/__init__.py`
9. `evaluation/__init__.py`

---

## Files NOT used by `run_benchmark.py` (CAN DELETE):

### Old framework integration approach (not used):
- `core/` - Framework interface/adapter (old approach, not used by run_benchmark.py)
  - `core/framework_interface.py`
  - `core/framework_adapter.py`
  - `core/__init__.py`

- `pipeline/` - QAPipeline wrapper (old approach, not used by run_benchmark.py)
  - `pipeline/qa_pipeline.py`
  - `pipeline/__init__.py`

### Evaluation helpers (used by test_benchmark.py, not run_benchmark.py):
- `evaluation_helpers/` - Result formatter/saver/printer
  - `evaluation_helpers/result_formatter.py`
  - `evaluation_helpers/result_saver.py`
  - `evaluation_helpers/evaluation_printer.py`
  - `evaluation_helpers/__init__.py`

### Test scripts (separate scripts, not dependencies):
- `test_benchmark.py` - Separate test script
- `langgraph_test.py` - LangGraph test script
- `crewai_test.py` - CrewAI test script
- `tests/` - Test directory
  - `tests/fake_answer_generator.py`
  - `tests/test_*.py` (all test files)

### Deprecated/analysis code:
- `GAIA_deprecated/` - Deprecated GAIA code
- `analysis/` - Analysis scripts

---

## Summary:

**KEEP (8 files + __init__.py):**
- `run_benchmark.py`
- `utils/query_loader.py`
- `utils/answer_validator.py`
- `evaluation/evaluator.py`
- `evaluation/evaluator_loader.py`
- `evaluation/api_scorer.py`
- `evaluation/heuristic_evaluator.py`
- `utils/__init__.py`
- `evaluation/__init__.py`

**CAN DELETE:**
- `core/` directory (3 files)
- `pipeline/` directory (2 files)
- `evaluation_helpers/` directory (4 files)
- `test_benchmark.py`
- `langgraph_test.py`
- `crewai_test.py`
- `tests/` directory (all files)
- `GAIA_deprecated/` directory
- `analysis/` directory

