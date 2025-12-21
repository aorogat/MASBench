"""
Common configuration for CrewAI Memory Agent experiments.
Keep all shared parameters here for easy reuse across scripts.
"""

# ---------------------------------------------------------------------
# ⚙️ LLM Configuration
# ---------------------------------------------------------------------
# To use Groq, run this router server first to forward any openai request to groq
# export OPENAI_API_BASE="http://localhost:5001/v1"
# python single_agent/memory/router.py

openai_sdk_llm_model = "openai/gpt-4o-mini"
agno_llm_model = "gpt-4o-mini"
crewai_llm_model = "gpt-4o-mini"
langgraph_llm_model = "gpt-4o-mini"

llm_max_tokens = 1500
llm_temperature = 0.1
KEEP_ANALYSIS = True   # or False, for router_local gpt-oss-20b

# ---------------------------------------------------------------------
# 🧠 Memory & Storage
# ---------------------------------------------------------------------
storage_directory = "/shared_mnt/crewai_memory"

# Chunking parameters for context ingestion
chunk_max_tokens = 4096
chunk_overlap = 200
max_context_tokens = 20000 #Change based on the LLM context window, local router hase 2048 max, we use 1000 of them for memory
RETRIEVAL_LIMIT = max(1, min(3, max_context_tokens // chunk_max_tokens))
max_questions_per_session = None #Keep it None to cover all questions, use small number for debugging
ignore_ingest = False # keep it False, use True for debugging issues only


# ---------------------------------------------------------------------
# 🧩 Benchmark Configuration
# ---------------------------------------------------------------------
# All splits included in MemoryAgentBench evaluation
splits = [
    "Accurate_Retrieval",
    "Test_Time_Learning", #Check the code, benchmark for example "results/memory/Crewai/MCC/session_3.json" like 76 question is weired
    # "Long_Range_Understanding",
    # "Conflict_Resolution",
]
# NEW — Maximum sessions to evaluate per task (per subtask)
max_sessions_per_subtask = 10

# Directory for saving results
results_directory = "results/memory"

# ---------------------------------------------------------------------
# 📊 Evaluation Model & Batch Sizes
# ---------------------------------------------------------------------
# Evaluation LLM used for semantic scoring (exact match, summary F1, recall@5)
eval_llm_model = "gpt-4o-mini" # Use good model for robust evaluation (e.g., gpt-4o with batches of 10 and 2 for eval_small_batch_size and eval_summary_batch_size)

# Default batch size for small-item evaluations (e.g., exact match)
eval_small_batch_size = 1

# Default batch size for summary/fact-evaluation (usually 1 due to long inputs)
eval_summary_batch_size = 1

# ---------------------------------------------------------------------
# 📋 Verbosity / Debug
# ---------------------------------------------------------------------
verbose = True
