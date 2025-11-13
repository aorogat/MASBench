"""
CrewAI Benchmark Runner
-----------------------
Unified runner for GSM8K, CSQA, and MATH benchmarks with CrewAI.
Uses the new benchmarks/ classes for data loading, gold/pred storage,
and evaluation.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.crewai_test
"""

#!/usr/bin/env python
"""
CrewAI Benchmark Runner
-----------------------
Evaluate GSM8K, CSQA, and MATH benchmarks using CrewAI.
Runs in two modes: planning=False and planning=True.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.crewai_test
"""

import os
import time
import json
import re, unicodedata
from pathlib import Path
from single_agent.reasoning.crew_gsm8k import SingleAgentCrewGSM8K
from single_agent.reasoning.crew_csqa import SingleAgentCrewCSQA
from single_agent.reasoning.crew_math import SingleAgentCrewMATH
from single_agent.reasoning.config import CONFIG

# Import benchmark classes
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.csqa import CSQABenchmark
from benchmarks.math import MATHBenchmark







import re, unicodedata
from pathlib import Path



# --- Add this near the top of your script ---
from dotenv import load_dotenv

load_dotenv()  # ensure .env is loaded





# Windows-reserved basenames (case-insensitive)
_RESERVED = { "CON","PRN","AUX","NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)] }

def sanitize_filename_component(s: str, maxlen: int = 80) -> str:
    """Make a string safe to use inside a filename on all major OSes."""
    if s is None:
        return "model"
    # Normalize and drop accents
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    # Remove dangerous/illegal characters and control chars
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    # Collapse whitespace to underscores, dedupe underscores
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    # Trim trailing/leading dots/underscores/spaces (Windows hates trailing dots/spaces)
    s = s.strip("._ ")
    # Avoid path tricks like '..'
    s = s.replace("..", "")
    if not s:
        s = "model"
    if s.upper() in _RESERVED:
        s = f"_{s}_"
    return s[:maxlen]





# ---------------------------------------------------------------------
# Run one benchmark with CrewAI
# ---------------------------------------------------------------------

def run_crewai_on_benchmark(benchmark, crew_cls, log_file, filename):
    """Run CrewAI agent on all questions in a benchmark."""
    # clean log file before run
    if os.path.exists(log_file):
        os.remove(log_file)

    crew = crew_cls().crew()
    total = len(benchmark.questions)

    start = time.perf_counter()
    for idx, q in enumerate(benchmark.questions, 1):
        print(f"🔹 Running {benchmark.name.upper()} Question {idx}/{total}")

        inputs = {"question": q.question, "planning": CONFIG["planning"]}

        q_start = time.perf_counter()
        try:
            result = crew.kickoff(inputs=inputs)
            
            # print(type(result))
            # print(result.tasks_output)

            q_elapsed = time.perf_counter() - q_start   # ⏱ per-question runtime

            task_key = f"{benchmark.name}_task"
            pred = (str(result.get(task_key)) if isinstance(result, dict) else str(result)).strip()

            

            # Approximate output tokens by characters/4 (or replace with model usage stats if available)
            tokens_out = max(1, len(str(result)) // 4)

            # ✅ store prediction, time, and tokens
            benchmark.set_pred(q, pred, time_used=q_elapsed, tokens_out=tokens_out, llm_response=str(result))

        except Exception as e:
            q_elapsed = time.perf_counter() - q_start
            print(f"⚠️  Crew failed on Q{q.qid}: {e}")
            # mark failure with None prediction
            benchmark.set_pred(q, "FAILED", time_used=q_elapsed, tokens_out=0)
            q.correct = False

    elapsed = time.perf_counter() - start
    print(f"\n⏱️ Finished {benchmark.name.upper()} in {elapsed:.2f} sec")

    # save + print results
    benchmark.save_results(CONFIG["results_dir"], filename)
    benchmark.print_summary()



# ---------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------

def run_all_benchmarks():
    # make sure logs/ exists (no-op if it already does)
    Path("logs").mkdir(parents=True, exist_ok=True)

    plan_tag = "planning" if CONFIG.get("planning") else "noplanning"
    llm_tag  = sanitize_filename_component(str(CONFIG.get("llm", "model")))

    
    if "csqa" in CONFIG["benchmarks"]:
        bench = CSQABenchmark(split="validation", n=CONFIG.get("n_csqa"))
        run_crewai_on_benchmark(
            bench, SingleAgentCrewCSQA,
            "logs/SingleAgentCrewCSQA.json",
            f"crewai_csqa_{plan_tag}_{llm_tag}.json"
        )

    if "math" in CONFIG["benchmarks"]:  # Deepseek always fails in planning mode...
        bench = MATHBenchmark(root="data/MATH/test", n=CONFIG.get("n_math"))
        run_crewai_on_benchmark(
            bench, SingleAgentCrewMATH,
            "logs/SingleAgentCrewMATH.json",
            f"crewai_math_{plan_tag}_{llm_tag}.json"
        )
    
    if "gsm8k" in CONFIG["benchmarks"]:
        bench = GSM8KBenchmark(split="test", n=CONFIG.get("n_gsm8k"))
        run_crewai_on_benchmark(
            bench, SingleAgentCrewGSM8K,
            "logs/SingleAgentCrewGSM8K.json",
            f"crewai_gsm8k_{plan_tag}_{llm_tag}.json"
        )

def main():
    for planning in [False, True]:
        CONFIG["planning"] = planning
        print(f"\n=== Running benchmarks with planning={planning} ===")
        run_all_benchmarks()


if __name__ == "__main__":
    main()
