# Deprecated Code

This folder contains code that is no longer used by `run_benchmark.py`.

## Contents

### Old Framework Integration Approach
- `core/` - Framework interface/adapter (old approach, replaced by direct agent interface)
- `pipeline/` - QAPipeline wrapper (old approach, not used by run_benchmark.py)

### Evaluation Helpers
- `evaluation_helpers/` - Result formatter/saver/printer (used by test_benchmark.py, not run_benchmark.py)

### Test Scripts
- `test_benchmark.py` - Separate test script (not a dependency of run_benchmark.py)
- `langgraph_test.py` - LangGraph test script
- `crewai_test.py` - CrewAI test script
- `tests/` - Test directory with unit tests

### Deprecated/Analysis Code
- `GAIA_deprecated/` - Deprecated GAIA code
- `analysis/` - Analysis scripts

### Documentation
- `FILES_ANALYSIS.md` - Analysis of which files are used/unused

## Current Active Code

The active codebase now consists of:
- `run_benchmark.py` - Main benchmark runner
- `utils/` - QueryLoader and AnswerValidator
- `evaluation/` - Evaluation components (evaluator, evaluator_loader, api_scorer, heuristic_evaluator)
- `StableToolBench/` - Original StableToolBench code

See `../README.md` for current usage.

