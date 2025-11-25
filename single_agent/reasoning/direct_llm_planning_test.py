#!/usr/bin/env python
"""
Direct LLM Planning Benchmark Runner
------------------------------------
This script replicates the CrewAI planning experiment BUT without CrewAI.

For each question:
  1) Send question to LLM to generate a PLAN
  2) Append PLAN + QUESTION and send again to LLM to generate ANSWER
  3) Store prediction, runtime, tokens usage, and raw LLM response

Output filenames follow the same structure as CrewAI, except:
    direct_planning_<bench>_planning_<llm>.json

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.direct_llm_planning_test
"""

import os
import time
import json
import re
import unicodedata
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# -----------------------
# OpenAI Chat API Import
# -----------------------
from openai import OpenAI
client = OpenAI()

# ------------------------------
# Import Benchmark Implementations
# ------------------------------
from benchmarks.gsm8k import GSM8KBenchmark
from benchmarks.csqa import CSQABenchmark
from benchmarks.math import MATHBenchmark
import requests

# ================================================================
# CONFIG IS EMBEDDED IN THIS FILE (PER YOUR REQUEST)
# ================================================================
CONFIG = {
    "planning": True,       # planning only
    "llm": "gpt-4o-mini",
    "planning_llm": "gpt-4o-mini",
    "math_judge_llm": "gpt-4o-mini",

    "results_dir": "results/planning_direct",
    "benchmarks": ["csqa", "math", "gsm8k"],

    "n_gsm8k": None,
    "n_csqa": None,
    "n_math": 100,
}


# ================================================================
# Filename sanitizing to match CrewAI runner
# ================================================================
_RESERVED = { "CON","PRN","AUX","NUL", *[f"COM{i}" for i in range(1,10)],
              *[f"LPT{i}" for i in range(1,10)] }

def sanitize_filename_component(s: str, maxlen: int = 80) -> str:
    if s is None:
        return "model"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", s)
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    s = s.strip("._ ")
    s = s.replace("..", "")
    if not s:
        s = "model"
    if s.upper() in _RESERVED:
        s = f"_{s}_"
    return s[:maxlen]


# ================================================================
# LLM CALL HELPERS
# ================================================================
import subprocess

def call_llm(model, messages):
    """
    Universal LLM caller:
    - OpenAI models
    - Ollama local models (using subprocess)
    """

    # Convert ChatML-style messages into a plaintext prompt for Ollama
    def messages_to_prompt(msgs):
        prompt = ""
        for m in msgs:
            role = m["role"]
            content = m["content"]
            prompt += f"{role.upper()}:\n{content}\n\n"
        return prompt.strip()

    # -------------------------------
    # 1) OLLAMA LOCAL MODE (subprocess)
    # -------------------------------
    if model.startswith("ollama/"):
        local_model = model.replace("ollama/", "")  # remove prefix

        prompt = messages_to_prompt(messages)

        try:
            result = subprocess.run(
                ["ollama", "run", local_model],
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=200,
            )

            output = result.stdout.decode("utf-8").strip()

            if output:
                return output
            else:
                return f"LLM_ERROR: Empty Ollama output, stderr={result.stderr.decode()}"

        except Exception as e:
            return f"LLM_ERROR: Ollama subprocess failed → {e}"

    # -------------------------------
    # 2) OPENAI MODE (default)
    # -------------------------------
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        return response.choices[0].message["content"]

    except Exception as e:
        return f"LLM_ERROR: OpenAI API error → {e}"


# ================================================================
# DIRECT LLM TWO-STEP PLANNING
# ================================================================
import re

def solve_with_direct_planning(
    question: str,
    model_plan: str,
    model_answer: str,
    answer_mode: str | None = None,
):
    """
    Step 1: Ask LLM to create a plan.
    Step 2: Trim plan to max length if needed.
    Step 3: Append plan + question and ask again for final answer,
            forcing the answer into a benchmark-specific format.
    """

    clean_q = " ".join(question.split()) if question else ""
    print(f"\t - Question: {clean_q[:60]}")

    MAX_PLAN = CONFIG.get("max_plan_chars", 1500)

    # --------- 1) PLAN GENERATION ---------
    plan_prompt = [
        {
            "role": "system",
            "content": (
                "Generate a clear step-by-step solution plan. "
                "Do NOT give the final answer, only the plan. "
                "Keep it concise and structured."
            ),
        },
        {"role": "user", "content": question},
    ]
    plan = call_llm(model_plan, plan_prompt)

    # --------- 1.5) TRIM IF TOO LONG ---------
    if plan and len(plan) > MAX_PLAN:
        original_len = len(plan)
        plan = plan[:MAX_PLAN]

        last_period = plan.rfind(".")
        if last_period > 0:
            plan = plan[: last_period + 1]

        plan += f"\n\n[PLAN_TRIMMED from {original_len} → {len(plan)} characters]"

    clean_plan = " ".join(plan.split()) if plan else ""
    print(f"\t - Plan: {clean_plan[:60]}")

    # --------- 2) FORMAT-CONSTRAINED FINAL ANSWER ---------
    if answer_mode == "csqa_mcq":
        answer_system = (
            "You are solving a multiple-choice question. "
            "Use the provided plan to pick the correct option. "
            "Return ONLY a single uppercase letter: A, B, C, D, or E. "
            "No explanation."
        )
    elif answer_mode == "gsm8k_number":
        answer_system = (
            "You are solving a math word problem. "
            "Use the plan to compute the final numeric answer. "
            "Return ONLY the number (integer or decimal). "
            "No words or explanation."
        )
    elif answer_mode == "math_expression":
        answer_system = (
            "You are solving a math competition problem. "
            "Return ONLY the final numeric or algebraic expression on one line. "
            "No explanation."
        )
    else:
        answer_system = (
            "Use the provided plan to answer the question concisely."
        )

    answer_prompt = [
        {"role": "system", "content": answer_system},
        {
            "role": "user",
            "content": f"QUESTION:\n{question}\n\nPLAN:\n{plan}\n\nNow give the final answer.",
        },
    ]
    final_answer_raw = call_llm(model_answer, answer_prompt)
    final_answer = final_answer_raw.strip() if final_answer_raw else ""

    # --------- 2.5) LIGHT POST-PROCESSING ENFORCEMENT ---------
    original_final = final_answer

    if answer_mode == "csqa_mcq":
        m = re.search(r"[A-E]", final_answer.upper())
        final_answer = m.group(0) if m else final_answer[:1].upper()

    elif answer_mode == "gsm8k_number":
        m = re.search(r"-?\d+(\.\d+)?", final_answer)
        final_answer = m.group(0) if m else final_answer

    elif answer_mode == "math_expression":
        final_answer = final_answer.splitlines()[0].strip()


    # --------- 2.6) FORMAT VALIDATION (your new requirement) ---------
    is_valid = True

    if answer_mode == "csqa_mcq":
        if not re.fullmatch(r"[A-E]", final_answer):
            is_valid = False

    elif answer_mode == "gsm8k_number":
        if not re.fullmatch(r"-?\d+(\.\d+)?", final_answer):
            is_valid = False

    elif answer_mode == "math_expression":
        if not final_answer.strip():
            is_valid = False

    if not is_valid:
        final_answer = f"Failed: {original_final.strip()}"

    clean_ans = " ".join(final_answer.split()) if final_answer else ""
    print(f"\t - Final Answer: {clean_ans[:60]}")

    # --------- 2.7) TOKEN COUNT ----------
    tokens_out = max(1, len(str(plan + final_answer_raw)) // 4)

    return plan, final_answer, tokens_out

# ================================================================
# BENCHMARK RUNNER (DIRECT LLM)
# ================================================================
def run_direct_planning_on_benchmark(benchmark, log_file, output_filename):
    """
    Run the Direct-LLM planning experiment on a given benchmark.
    """
    # clean old logs
    if os.path.exists(log_file):
        os.remove(log_file)

    total = len(benchmark.questions)
    start = time.perf_counter()

    for idx, q in enumerate(benchmark.questions, 1):
        print(f"🔹 Direct-LLM {benchmark.name.upper()} Question {idx}/{total}")

        q_start = time.perf_counter()
        try:
            plan, answer, tokens_out = solve_with_direct_planning(
                q.question,
                CONFIG["planning_llm"],
                CONFIG["llm"]
            )
            q_elapsed = time.perf_counter() - q_start

            benchmark.set_pred(
                q,
                answer,
                time_used=q_elapsed,
                tokens_out=tokens_out,
                llm_response=f"PLAN:\n{plan}\n\nANSWER:\n{answer}"
            )

        except Exception as e:
            q_elapsed = time.perf_counter() - q_start
            print(f"⚠️ Direct LLM failed on Q{q.qid}: {e}")
            benchmark.set_pred(q, "FAILED", time_used=q_elapsed, tokens_out=0)
            q.correct = False

    elapsed = time.perf_counter() - start
    print(f"\n⏱️ Finished {benchmark.name.upper()} in {elapsed:.2f} sec")

    benchmark.save_results(CONFIG["results_dir"], output_filename)
    benchmark.print_summary()


# ================================================================
# MAIN LOGIC
# ================================================================
def run_all_benchmarks_direct():
    Path("logs").mkdir(parents=True, exist_ok=True)
    Path(CONFIG["results_dir"]).mkdir(parents=True, exist_ok=True)

    llm_tag = sanitize_filename_component(str(CONFIG["llm"]))
    plan_tag = "planning"

    # CSQA
    if "csqa" in CONFIG["benchmarks"]:
        bench = CSQABenchmark(split="validation", n=CONFIG["n_csqa"])
        run_direct_planning_on_benchmark(
            bench,
            "logs/DirectLLM_CSQA.json",
            f"direct_planning_csqa_{plan_tag}_{llm_tag}.json"
        )

    # MATH
    if "math" in CONFIG["benchmarks"]:
        bench = MATHBenchmark(root="data/MATH/test", n=CONFIG["n_math"])
        run_direct_planning_on_benchmark(
            bench,
            "logs/DirectLLM_MATH.json",
            f"direct_planning_math_{plan_tag}_{llm_tag}.json"
        )

    # GSM8K
    if "gsm8k" in CONFIG["benchmarks"]:
        bench = GSM8KBenchmark(split="test", n=CONFIG["n_gsm8k"])
        run_direct_planning_on_benchmark(
            bench,
            "logs/DirectLLM_GSM8K.json",
            f"direct_planning_gsm8k_{plan_tag}_{llm_tag}.json"
        )


# ================================================================
# MODEL SWEEP CONFIGURATIONS
# ================================================================
MODEL_CONFIGS = [
    # DeepSeek 7B
    {
        "llm": "ollama/deepseek-llm:7b",
        "planning_llm": "ollama/deepseek-llm:7b",
        "math_judge_llm": "gpt-4o-mini",
    },
    # Llama3.1 8B
    {
        "llm": "ollama/llama3.1:8b",
        "planning_llm": "ollama/llama3.1:8b",
        "math_judge_llm": "gpt-4o-mini",
    },
    # Qwen 7B
    {
        "llm": "ollama/qwen:7b",
        "planning_llm": "ollama/qwen:7b",
        "math_judge_llm": "gpt-4o-mini",
    },
    # Phi-4 14B
    {
        "llm": "ollama/phi4:14b",
        "planning_llm": "ollama/phi4:14b",
        "math_judge_llm": "gpt-4o-mini",
    },
    # GPT-OSS 20B
    {
        "llm": "ollama/gpt-oss:20b",
        "planning_llm": "ollama/gpt-oss:20b",
        "math_judge_llm": "gpt-4o-mini",
    },
    # OpenAI Mini
    {
        "llm": "gpt-4o-mini",
        "planning_llm": "gpt-4o-mini",
        "math_judge_llm": "gpt-4o-mini",
    },
    # OpenAI 4.1
    {
        "llm": "gpt-4.1",
        "planning_llm": "gpt-4.1",
        "math_judge_llm": "gpt-4.1",
    },
]

# ================================================================
# NON-MODEL SETTINGS (same for all runs)
# ================================================================
BASE_CONFIG = {
    "planning": True,  # planning only
    "results_dir": "results/planning_direct",
    "benchmarks": ["csqa", "math", "gsm8k"],
    "n_gsm8k": None,
    "n_csqa": None,
    "n_math": 100,
}


# ================================================================
# MAIN LOOP — RUN ALL MODELS ONE BY ONE
# ================================================================
def main():

    print("\n=== Running Direct-LLM Planning Across All Model Configurations ===")

    for cfg in MODEL_CONFIGS:
        # Merge base config + model config
        CONFIG.update(BASE_CONFIG)
        CONFIG.update(cfg)

        print("\n==============================================")
        print(f"   🔥 Running configuration for LLM = {cfg['llm']}")
        print("==============================================")

        run_all_benchmarks_direct()

    print("\n=== All Model Configurations Completed ===")


if __name__ == "__main__":
    main()
