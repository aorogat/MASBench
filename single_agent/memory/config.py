"""
Common configuration for CrewAI Memory Agent experiments.
Keep all shared parameters here for easy reuse across scripts.
"""

# ---------------------------------------------------------------------
# ⚙️ LLM Configuration
# ---------------------------------------------------------------------
openai_sdk_llm_model = "openai/gpt-4o-mini"
agno_llm_model = "gpt-4o-mini"
crewai_llm_model = "gpt-4o-mini"

llm_max_tokens = 1500
llm_temperature = 0.1

# ---------------------------------------------------------------------
# 🧠 Memory & Storage
# ---------------------------------------------------------------------
storage_directory = "/shared_mnt/crewai_memory"

# Chunking parameters for context ingestion
chunk_max_tokens = 4096
chunk_overlap = 200

# ---------------------------------------------------------------------
# 🧩 Benchmark Configuration
# ---------------------------------------------------------------------
# All splits included in MemoryAgentBench evaluation
splits = [
    # "Accurate_Retrieval",
    # "Test_Time_Learning",
    # "Long_Range_Understanding",
    "Conflict_Resolution",
]
# NEW — Maximum sessions to evaluate per task (per subtask)
max_sessions_per_subtask = 1

# Default name for experiment outputs
system_name = "crewai_memory_agent"

# Directory for saving results
results_directory = "results/memory"

# ---------------------------------------------------------------------
# 📊 Evaluation Model & Batch Sizes (NEW)
# ---------------------------------------------------------------------
# Evaluation LLM used for semantic scoring (exact match, summary F1, recall@5)
eval_llm_model = "gpt-4o-mini"

# Default batch size for small-item evaluations (e.g., exact match)
eval_small_batch_size = 10

# Default batch size for summary/fact-evaluation (usually 1 due to long inputs)
eval_summary_batch_size = 1

# ---------------------------------------------------------------------
# 📋 Verbosity / Debug
# ---------------------------------------------------------------------
verbose = True
