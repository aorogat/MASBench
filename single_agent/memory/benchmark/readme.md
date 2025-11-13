# MemoryAgentBench – Evaluation Framework for Memory-Centric Agents

This folder contains the [**MemoryAgentBench**](https://huggingface.co/datasets/ai-hyz/MemoryAgentBench) evaluation pipeline, adapted for multi-agent frameworks such as **CrewAI**, **OpenAI Agent SDK**, and other LLM-driven systems that incorporate **short-term**, **long-term**, or **entity memory**.

The benchmark evaluates how well agent frameworks can **retain**, **retrieve**, **forget**, and **summarize** information across multiple structured tasks.

---

## 📦 Folder Purpose

The directory `single_agent/memory/benchmark/` includes everything needed to run MemoryAgentBench on any agent that exposes the following methods:

- `reset()`
- `ingest(context: str)`
- `query(question: str)`

The main evaluator (`memory_agent_bench.py`) loads dataset splits, runs the agent on every session, and computes **LLM-based semantic metrics**.

---

## 🧪 MemoryAgentBench Overview

MemoryAgentBench is a large-scale benchmark created to measure **memory capabilities** of LLM agents across four major competencies:

### **1. Accurate Retrieval (AR)**
Tests exact recall and multi-hop reasoning over encoded memory.

**Subtasks include:**
- **SH-QA** (single-document QA)
- **MH-QA** (multi-document QA)
- **EventQA**
- **LongMemEval (LME)**

### **2. Test-Time Learning (TTL)**
Tests whether the agent can **learn new information** inside the session.

**Subtasks:**
- **MCC** (in-context classification)
- **MovieRec** (ReDial recommendation)

### **3. Long-Range Understanding (LRU)**
Evaluates the ability to **summarize**, **track events**, and **understand long context**.

**Subtasks:**
- Document summarization
- **DetectiveQA** (multi-event reasoning)

### **4. Selective Forgetting (SF)**
Tests the ability to:
- Incorporate new knowledge
- Override outdated facts
- Retain *only* the most recent correct information

**Subtasks:**
- **FC-SH** (single-hop fact consolidation)
- **FC-MH** (multi-hop fact consolidation)

---

## 📂 Dataset Source

The benchmark uses the official HuggingFace dataset:

> **https://huggingface.co/datasets/ai-hyz/MemoryAgentBench**

Each split contains multiple **sessions**, each with:
- A long **context**
- A list of **questions**
- A list of **gold answers**
- Metadata indicating the **subtask** and **category**

---

## 🧠 What `memory_agent_bench.py` Does

This file contains the **MemoryAgentBench class**, which is responsible for:

### **1. Loading the Benchmark Dataset**

```python
bench = MemoryAgentBench(split="Accurate_Retrieval")
```

Loads a specific split (AR, TTL, LRU, SF), extracts:
- Session context
- Questions
- Gold answers
- Subtask
- Category

Automatically maps each session into:
- Accurate Retrieval
- Test-Time Learning
- Long-Range Understanding
- Selective Forgetting

### **2. Running an Agent Through All Sessions**

The evaluator expects an agent object with:

```python
agent.reset()
agent.ingest(context)
agent.query(question)
```

For each session, it:
1. Resets the agent
2. Ingests the session context
3. Queries every question
4. Collects the system answers

### **3. Applying LLM-Based Semantic Evaluation**

This benchmark does **not** use string matching. Instead, it uses **GPT-4o-mini** to compute:

- ✔ Semantic Exact Match
- ✔ Summary Fact-Level F1 Score
- ✔ Semantic Recall@5
- ✔ Fact-consolidation scoring

Using functions from: `metric_eval_gpt.py`

Batch sizes and evaluation model are configurable via: `single_agent/memory/config.py`

### **4. Limiting Sessions Per Subtask**

You can restrict evaluation cost with:

```python
max_sessions_per_subtask = 10
```

(Defined in `config.py`)

This prevents unnecessary GPT calls.

### **5. Saving Evaluation Results**

Results are written as JSON to: `results/memory/`

Each full evaluation includes:
- Per-question scores
- Per-session averages
- Category averages (AR, TTL, LRU, SF)
- Overall system score
- Runtime statistics

---

## 📝 Example Usage

```python
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from my_agent import MyAgent

# Initialize benchmark with desired split
bench = MemoryAgentBench(split="Accurate_Retrieval")

# Create your agent instance
agent = MyAgent()

# Run evaluation
results = bench.evaluate_agent(
    agent, 
    system_name="my_memory_agent", 
    verbose=True
)

print(results)
```

---

## 🔧 Configuration

Edit evaluation parameters in: `single_agent/memory/config.py`

**Key options:**

```python
max_sessions_per_subtask = 10
eval_llm_model = "gpt-4o-mini"
eval_small_batch_size = 10
eval_summary_batch_size = 1
```

---

## 📊 Outputs

All results are stored in: `results/memory/<system>_<split>_<timestamp>.json`

Each file includes:
- Per-question semantic scores
- Session-level averages
- Category-level averages
- Overall benchmark score
- Timing statistics

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install datasets transformers openai
```

### Running the Benchmark

```bash
python -m single_agent.memory.benchmark.memory_agent_bench
```

### Available Splits

- `Accurate_Retrieval`
- `Test_Time_Learning`
- `Long_Range_Understanding`
- `Selective_Forgetting`


---

## 📄 License

Please refer to the original dataset repository for licensing information.

---

## 🤝 Contributing

Contributions are welcome! Please submit issues or pull requests for:
- New agent adapters
- Evaluation metric improvements
- Documentation enhancements

---

## 📧 Contact

For questions or issues, please open an issue on the repository.