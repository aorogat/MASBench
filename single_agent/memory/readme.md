# Memory Evaluation Framework (single_agent/memory)

This directory contains the **unified memory evaluation layer** used to test different agent frameworks (CrewAI, OpenAI Agents SDK, Agno, custom agents, etc.) against **MemoryAgentBench**, a benchmark that measures how well agents handle:

- Short-term memory  
- Long-term memory  
- Entity-level memory  
- Retrieval and consolidation  
- Summarization across long contexts  
- Forgetting and updating facts over time  

Every agent framework implements its own memory system, so this folder provides:

- 🔹 A **common interface** for memory ingestion and querying  
- 🔹 A **shared configuration module**  
- 🔹 A **GPT-based metric evaluator**  
- 🔹 A **benchmark runner** (MemoryAgentBench)  
- 🔹 Framework-specific agent wrappers  

---

## ⚠️ Important Notice About Token Usage & Cost

The MemoryAgentBench experiments used in this folder process **very large documents**, with many sessions containing **50–100 lengthy context chunks**, and each question often prompts the LLM with **tens of thousands of tokens**.

As a result:

### 🚨 Running the full benchmark on OpenAI models — even small ones like `gpt-4o-mini` — can easily cost **hundreds or even thousands of dollars** in API usage.

This is because the benchmark evaluates:
- 6–200 sessions per split  
- Each session contains 1–30 long questions  
- Each question may require **50k–120k tokens** of context  
- GPT-based evaluation doubles the calls (answer + scoring)

To make MemoryAgentBench **affordable**, we provide a custom **router server** (`router.py`) which transparently forwards any OpenAI LLM API call to **Groq's extremely inexpensive high-speed model**:

### 💸 `openai/gpt-oss-20b` (Groq)

- ~1000+ tokens/sec  
- Supports **131K context**  
- Tiny fraction of GPT-4o-mini cost  
- Works for memory ingestion, retrieval, and long-context reasoning  
- Fully OpenAI-compatible when passed through our router  

### 🧩 How the Router Works

1. Your code (LangGraph, CrewAI, AGNO, MemoryAgentBench, etc.) continues to call:
   ```
   https://api.openai.com/v1/chat/completions
   ```

2. You set:
   ```bash
   export OPENAI_API_BASE="http://localhost:5001/v1"
   -- Return back
   export OPENAI_API_BASE="https://api.openai.com/v1"
   ```

3. Run the Router server: From the root folder, run
   ```bash
   python -m single_agent.memory.router
   ```
   The router automatically loads API keys from the .env file in the project root directory. Make sure your .env file includes the following entries:
   ```bash
   OPENAI_API_KEY=sk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXXXXXXX
   ```

4. All OpenAI LLM requests are transparently captured by `router.py`.

5. The router rewrites the request and sends it to Groq instead:
   - `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` → mapped to `openai/gpt-oss-20b`.

6. Embeddings are NOT forwarded to Groq → The router sends embeddings to real OpenAI, since Groq does not support embeddings.

7. The router returns a standard OpenAI-compatible JSON chunk, so all frameworks work without modification:
   - LangGraph
   - CrewAI
   - AGNO
   - LangChain
   - MemoryAgentBench
   - OpenAI SDK

This reduces cost dramatically — often **10× cheaper** — while remaining faithful to the OpenAI API schema.

---

## 📁 Folder Overview

```
single_agent/memory/
│
├── benchmark/
│   ├── memory_agent_bench.py    # Main benchmark evaluator
│   ├── metric_eval_gpt.py       # GPT-based evaluation metrics
│   └── README.md                # Benchmark documentation
│
├── crewai_test.py               # CrewAI memory-enabled agent wrapper
├── openaiSDK_test.py            # OpenAI Agent SDK memory-enabled wrapper
├── agno_test.py                 # Agno memory agent implementation
├── etc
│   
├── helpers/
│   ├── common_agent_utils.py   # Utility functions (chunking, summarizing, etc.)
│   └── ...
│
├── config.py                    # Global configuration for all memory agents
└── README.md                    # <--- THIS FILE
```

---

## 🧠 Purpose of This Folder

The goal of this directory is to create a **standardized memory interface** so that any agent framework—CrewAI, OpenAI Agent SDK, Agno, custom LLM wrappers—can be evaluated **fairly and consistently** using the same MemoryAgentBench pipeline.

Every agent in this folder must implement:

```python
def reset(self):           # clears all memories
def ingest(self, context): # stores/learns session context
def query(self, question): # answers a question using its memory
```

These three functions allow completely different systems to be evaluated identically.

---

## ⚙️ Configuration (config.py)

All memory agents share the same configuration parameters in:

```
single_agent/memory/config.py
```

**Key parameters:**

```python
llm_max_tokens = 1500
llm_temperature = 0.1

max_sessions_per_subtask = 10

eval_llm_model = "gpt-4o-mini"
eval_small_batch_size = 10
eval_summary_batch_size = 1
```

**Purpose of configuration:**
- Ensures consistent evaluation across frameworks
- Controls how many sessions are tested (limits cost)
- Controls GPT batch sizes for metrics
- Defines which LLM evaluates answers
- Sets memory ingestion parameters (chunk sizes, overlaps)

Framework-specific code should read from configuration, not hard-code values.

---

## 🏗 Framework Implementations

Each framework has its own directory with an agent class implementing:

### ✔ `reset()`

Resets memory buffers, context windows, vector stores, or persistent memory stores.

### ✔ `ingest(context)`

Stores the long textual context in the framework's preferred memory system:

- **CrewAI** → uses Crew Memory (short-term, long-term, entity memories)
- etc

Chunking strategies are shared through `helpers/common_agent_utils.py`.

### ✔ `query(question)`

Uses the agent's internal mechanisms to answer based on stored memory:

- **CrewAI**: `agent.run(task)`
- **OpenAI Agent SDK**: `client.agents.run(...)`
- **Agno**: `agent.run("question")`
- **Custom agent**: prompt LLM with context + question

---

## 📊 GPT-Based Evaluation (metric_eval_gpt.py)

All evaluation is done using **semantic LLM metrics**, not string matching.

**Metrics include:**

- `evaluate_exact_match()` → semantic correctness
- `evaluate_summary_match()` → fact-level F1
- `evaluate_recall_at_5()` → semantic matching for recommendations
- Batch evaluation per config

These metrics ensure fairness across different frameworks.

---

## 🧪 Running the Benchmark

To evaluate any framework:

```python
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.crewai.crewai_agent import CrewAIMemoryAgent

# Initialize benchmark
bench = MemoryAgentBench(split="Accurate_Retrieval")

# Create agent instance
agent = CrewAIMemoryAgent()

# Run evaluation
bench.evaluate_agent(agent, system_name="crewai", verbose=True)
```

You can replace `CrewAIMemoryAgent` with:

- `OpenAIAgentMemoryWrapper`
- `AgnoMemoryAgent`
- Your own custom agent class

Any agent that implements the three core methods (`reset`, `ingest`, `query`) is compatible.

---

## 🚀 Adding a New Framework

To integrate a new agent framework:

### 1. Create a directory:

```
single_agent/memory/<framework_name>/
```

### 2. Implement an agent class:

```python
class MyFrameworkAgent:
    def reset(self):
        """Clear all memory stores."""
        pass
    
    def ingest(self, context):
        """Store context in memory."""
        pass
    
    def query(self, question):
        """Answer question using stored memory."""
        return answer
```

### 3. Import and evaluate:

```python
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.my_framework.my_agent import MyFrameworkAgent

bench = MemoryAgentBench(split="Accurate_Retrieval")
agent = MyFrameworkAgent()

bench.evaluate_agent(agent, system_name="my_framework")
```

You now have full MemoryAgentBench compatibility.

---

## 🔧 Helper Utilities

The `helpers/` directory contains shared utilities:

- **`common_agent_utils.py`**: Common functions for text chunking, summarization, and preprocessing
- Reusable across all framework implementations
- Ensures consistent preprocessing for fair comparison

---

## 📈 Evaluation Workflow

```
1. Load Benchmark Dataset
   ↓
2. Initialize Agent Framework
   ↓
3. For each session:
   - Reset agent memory
   - Ingest session context
   - Query all questions
   - Collect answers
   ↓
4. Evaluate with GPT-based metrics
   ↓
5. Generate results JSON
```

---

## 📊 Results Output

Results are saved to: `results/memory/<system>_<split>_<timestamp>.json`

Each result file contains:

- Per-question semantic scores
- Per-session averages
- Category averages (AR, TTL, LRU, SF)
- Overall system score
- Runtime statistics

---

## 📘 Summary

The `single_agent/memory/` folder provides:

✅ A standard interface for memory agents  
✅ A uniform evaluation framework  
✅ LLM-based metrics for semantic accuracy  
✅ Framework-specific adapters (CrewAI, OpenAI, Agno, custom)  
✅ A centralized configuration for reproducibility  
✅ A benchmark runner that measures all four memory competencies  

This ensures that memory-enabled agents built on different technologies can be compared **fairly**, **consistently**, and **scalably**.

---

## 🤝 Contributing

To contribute a new framework integration:

1. Fork the repository
2. Create your framework directory under `single_agent/memory/`
3. Implement the required interface
4. Add tests and documentation
5. Submit a pull request

---

## 📧 Support

For questions or issues:
- Open an issue on the repository
- Check the benchmark documentation in `benchmark/README.md`
- Review configuration options in `config.py`

---

## 📄 License

Please refer to the main repository for licensing information.