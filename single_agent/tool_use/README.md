# 🧰 Multi-Agent Tool-Use Benchmarking

**Evaluate multi-agent frameworks on tool-use tasks with StableToolBench**

This repository provides a complete experimental environment for benchmarking multi-agent systems—including **LangGraph**, **CrewAI**, **Concordia MAS**, and **MASGEN**—on the StableToolBench tool-use benchmark. Everything you need is included: queries, tool specifications, evaluation pipelines, and analysis utilities.

---

## 🎯 Overview

Multi-agent systems must coordinate to solve complex tasks involving external tools. This project measures how well different architectures can:

- Understand user instructions
- Select appropriate tools from ~7,000 APIs
- Construct valid tool calls
- Interpret responses
- Generate correct final answers

**Key Features:**
- ✅ Self-contained: No external downloads required
- ✅ Reproducible: Standardized evaluation metrics
- ✅ Extensible: Easy integration of new frameworks
- ✅ Research-ready: Publication-quality analysis tools

---

## 📁 Repository Structure

```
tool-use/
├── analysis/
│   └── analyze_stabletoolbench.py    # Statistical analysis & visualizations
│
├── StableToolBench/
│   ├── solvable_queries/              # Benchmark test set
│   ├── toolenv/tools/                 # ~7k API specifications
│   ├── toolbench/
│   │   ├── inference/                 # Inference runners
│   │   └── tooleval/                  # Evaluation metrics (SoPR, SoWR, FAC)
│   └── server/                        # Optional MirrorAPI server
│
├── langgraph_test.py                  # LangGraph evaluation
├── crewai_test.py                     # CrewAI evaluation
├── concordia_test.py                  # Concordia MAS evaluation
└── README.md
```

---

## 🧪 StableToolBench Benchmark

Built on ToolBench, StableToolBench improves stability and reproducibility through:

### Solvable Query Set
Curated queries where success is achievable by competent tool-using agents, eliminating noise from impossible tasks.

### Comprehensive Tool Universe
~7,000 tools with complete specifications:
- Tool names and descriptions
- Input/output schemas
- Example usage patterns

### Robust Evaluation Metrics
- **SoPR** (Solvable Pass Rate): Success rate on answerable queries
- **SoWR** (Solvable Win Rate): Comparative performance ranking
- **FAC** (Final Answer Correctness): Semantic accuracy via MirrorAPI

### Optional Virtual Environment
Includes MirrorAPI simulation server for reproducible testing without external dependencies.

---

## 🚀 Quick Start

### 1. Run an Evaluation

```bash
python langgraph_test.py
```

This script will:
1. Load benchmark queries
2. Initialize the tool environment
3. Create an LLM-powered agent
4. Execute queries and log results

### 2. Compute Metrics

```bash
cd StableToolBench/toolbench/tooleval

python eval_pass_rate.py \
    --converted_answer_path ../../results/model_predictions \
    --save_path ../../results/pass_rate \
    --reference_model your_agent \
    --test_set G1_instruction
```

### 3. Generate Analysis

```bash
python analysis/analyze_stabletoolbench.py
```

Produces statistics, distributions, and publication-ready figures.

---

## 🔬 Evaluation Pipeline

Each framework test follows this pattern:

### Load Benchmark Data
```python
import json
from pathlib import Path

# Load queries
queries = json.load(open("StableToolBench/solvable_queries/G1_instruction.json"))

# Load tool definitions
tool_dir = Path("StableToolBench/toolenv/tools")
tools = {f.name: json.load(f.open()) for f in tool_dir.iterdir()}
```

### Initialize Agent
```python
# Framework-specific setup (LangGraph example)
from langgraph import ToolCallingAgent

agent = ToolCallingAgent(
    llm=your_llm,
    tools=tools,
    memory=ConversationMemory()
)
```

### Execute and Evaluate
```python
for query in queries:
    response = agent.run(query["instruction"])
    
    # Log tool calls and compare results
    evaluate_response(response, query["expected_answer"])
```

### Generate Metrics
Run the evaluation scripts to compute SoPR, SoWR, and FAC scores across all frameworks.

---

## 📊 Research Applications

This repository supports research in:

- Multi-agent LLM architectures
- Tool selection and routing strategies
- Planning with external APIs
- Memory-augmented agent systems
- Framework benchmarking and comparison

**Publication Support:**
- Ready for SIGMOD, VLDB, KDD, NeurIPS submissions
- Includes analysis tools for generating tables and figures
- Reproducible experimental setup

---

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- Additional multi-agent framework integrations
- Enhanced evaluation metrics
- Improved analysis visualizations
- Documentation improvements

Please open an issue or submit a pull request.

---

## 📄 License

This repository incorporates components from StableToolBench (MIT License). All additional code is released under the MIT License.

---

## 📚 Citation

If you use this repository in your research, please cite:

```bibtex

```

---

## 🙋 Questions?

For questions, issues, or feature requests, please open a GitHub issue.