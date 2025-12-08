"""
Enhanced Analysis Script for MemoryAgentBench
Includes dataset statistics, publication-quality LaTeX table,
and a corrected experiment cost estimation that accounts for:
  - Ingest (context chunks)
  - Per-question calls with question + context window.
"""

# Run: python -m single_agent.memory.benchmark.analysis_benchmark

import os
import numpy as np
from datasets import load_dataset, get_dataset_split_names
from single_agent.memory.benchmark.memory_agent_bench import infer_subtask, category_of


# -------------------------------------------------------------
# Helper formatting
# -------------------------------------------------------------
def fmt_int(x):
    return f"{int(x):,}"


# -------------------------------------------------------------
# Cost model (modify as needed)
# -------------------------------------------------------------
# Approximate tokenization and call structure
TOKEN_PER_WORD = 1.3                 # rough estimate: 1 word ≈ 1.3 tokens
QUESTION_TOKENS = 30                 # average tokens for the question itself
OUTPUT_TOKENS_PER_ANSWER = 50        # average model output length

# Context window of the model used in *queries* (not ingest chunk size)
CTX_WINDOW_TOKENS = 4000
CTX_WINDOW_WORDS = CTX_WINDOW_TOKENS / TOKEN_PER_WORD

# Ingest chunking: your ingest() uses max_tokens=1000 *words* approx.
INGEST_CHUNK_WORDS = 1000

# Pricing (adjust to your real prices)
PRICE_INPUT = 0.0000005              # $ per input token (e.g., gpt-4o-mini)
PRICE_OUTPUT = 0.0000015             # $ per output token


def analyze_memory_agent_bench():
    dataset_name = "ai-hyz/MemoryAgentBench"

    print("\n📊 Starting analysis for MemoryAgentBench...")
    try:
        splits = get_dataset_split_names(dataset_name)
    except Exception as e:
        print("❌ ERROR: Could not load dataset splits:", e)
        return

    if not splits:
        print("❌ ERROR: No splits found. Dataset may not be downloaded.")
        return

    print(f"✔️ Detected {len(splits)} splits:")
    print(splits)

    analysis = []

    # Totals for cost estimation
    global_questions = 0
    global_context_words = 0
    total_ingest_tokens = 0.0
    total_query_input_tokens = 0.0

    for split in splits:
        print(f"\n🔍 Analyzing split: {split}")

        try:
            ds = load_dataset(dataset_name, split=split)
        except Exception as e:
            print(f"❌ ERROR loading split '{split}': {e}")
            continue

        num_sessions = len(ds)
        print(f"   ➤ Sessions: {num_sessions}")

        # -------------------------------------------------------------
        # Core statistics per split
        # -------------------------------------------------------------
        num_questions_arr = np.array([len(row["questions"]) for row in ds])
        num_words_arr = np.array([len(row["context"].split()) for row in ds])

        # Accumulate global raw totals
        split_questions = int(np.sum(num_questions_arr))
        split_context_words = int(np.sum(num_words_arr))
        global_questions += split_questions
        global_context_words += split_context_words

        # Descriptive stats
        avg_q = float(np.mean(num_questions_arr))
        median_q = float(np.median(num_questions_arr))
        min_q, max_q = int(np.min(num_questions_arr)), int(np.max(num_questions_arr))

        avg_ctx = float(np.mean(num_words_arr))
        median_ctx = float(np.median(num_words_arr))
        ctx_std = float(np.std(num_words_arr))
        min_ctx, max_ctx = int(np.min(num_words_arr)), int(np.max(num_words_arr))
        total_ctx = int(np.sum(num_words_arr))

        # -------------------------------------------------------------
        # Cost estimation per session (this is the corrected logic)
        # -------------------------------------------------------------
        split_ingest_tokens = 0.0
        split_query_input_tokens = 0.0

        for ctx_words, q_count in zip(num_words_arr, num_questions_arr):
            # ----- Ingest cost -----
            # You split the context into chunks of size ≈ INGEST_CHUNK_WORDS words.
            # Across all chunks, every word is sent exactly once, so:
            ingest_tokens_session = ctx_words * TOKEN_PER_WORD
            split_ingest_tokens += ingest_tokens_session

            # ----- Query cost -----
            # For each question, you send:
            #   - the question tokens
            #   - PLUS the context, but capped by the model context window.
            effective_ctx_words = min(ctx_words, CTX_WINDOW_WORDS)
            per_query_input_tokens = (
                effective_ctx_words * TOKEN_PER_WORD + QUESTION_TOKENS
            )
            split_query_input_tokens += q_count * per_query_input_tokens

        total_ingest_tokens += split_ingest_tokens
        total_query_input_tokens += split_query_input_tokens

        # -------------------------------------------------------------
        # Subtasks + categories
        # -------------------------------------------------------------
        subtasks = set()
        categories = set()

        for row in ds:
            meta = row.get("metadata", {})
            sub = infer_subtask(meta)
            cat = category_of(sub)
            subtasks.add(sub)
            categories.add(cat)

        categories_str = ", ".join(sorted(categories))

        # Save analysis row
        analysis.append({
            "split": split,
            "sessions": num_sessions,
            "subtasks": len(subtasks),
            "categories": categories_str,
            "avg_q": avg_q,
            "median_q": median_q,
            "min_q": min_q,
            "max_q": max_q,
            "avg_ctx": avg_ctx,
            "median_ctx": median_ctx,
            "ctx_std": ctx_std,
            "min_ctx": min_ctx,
            "max_ctx": max_ctx,
            "total_ctx": total_ctx,
            "split_questions": split_questions,
            "split_ingest_tokens": split_ingest_tokens,
            "split_query_tokens": split_query_input_tokens,
        })

    # -------------------------------------------------------------
    # GLOBAL COST ESTIMATION
    # -------------------------------------------------------------
    print("\n💰 Estimating cost of running the full benchmark...")

    total_input_tokens = total_ingest_tokens + total_query_input_tokens
    total_output_tokens = global_questions * OUTPUT_TOKENS_PER_ANSWER

    cost_input = total_input_tokens * PRICE_INPUT
    cost_output = total_output_tokens * PRICE_OUTPUT
    total_cost = cost_input + cost_output

    print(f"   • Total questions: {fmt_int(global_questions)}")
    print(f"   • Total context words: {fmt_int(global_context_words)}")
    print(f"   • Ingest input tokens: {fmt_int(total_ingest_tokens)}")
    print(f"   • Query input tokens:  {fmt_int(total_query_input_tokens)}")
    print(f"   • TOTAL input tokens:  {fmt_int(total_input_tokens)}")
    print(f"   • Estimated output tokens: {fmt_int(total_output_tokens)}")
    print(f"   • Cost (input):  ${cost_input:.4f}")
    print(f"   • Cost (output): ${cost_output:.4f}")
    print(f"   ➤ Estimated TOTAL experiment cost: **${total_cost:.4f}**")

     # -------------------------------------------------------------
    # LATEX TABLE (statistics per split)
    # -------------------------------------------------------------
    map = {
        "Accurate_Retrieval": "AR",
        "Test_Time_Learning": "TTL",
        "Long_Range_Understanding": "LRU",
        "Selective_Forgetting": "SF",
        "Conflict_Resolution": "CR",
    }

    print("\n\n📄 LaTeX Table Output (statistics):\n")

    print(r"\begin{table}[t]\centering")
    print(r"\caption{Statistics of MemoryAgentBench splits with question distributions and truncated context sizes (in thousands of words).}")
    print(r"\label{tab:memory-bench}")
    print(r"\footnotesize")
    print(r"\begin{tabular}{l r r | rrrr | rrrr}")
    print(r"\toprule")

    # Multi-column headers
    print(
        r" & & & \multicolumn{4}{c|}{\textbf{Questions}} & "
        r"\multicolumn{4}{c}{\textbf{Context (k words)}} \\"
    )

    print(
        r"Split & Sessions & Subtasks & Min & Max & Avg & Total & "
        r"Min & Max & Avg & Total \\"
    )

    # print(r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}")
    print(r"\midrule")

    for a in analysis:

        min_ctx_k   = a['min_ctx'] // 1000
        max_ctx_k   = a['max_ctx'] // 1000
        avg_ctx_k   = int(a['avg_ctx']) // 1000
        total_ctx_k = a['total_ctx'] // 1000

        print(
            f"{map[a['split']]} & "
            f"{fmt_int(a['sessions'])} & "
            f"{a['subtasks']} & "
            f"{a['min_q']} & "
            f"{a['max_q']} & "
            f"{a['avg_q']:.1f} & "
            f"{fmt_int(a['split_questions'])} & "
            f"{fmt_int(min_ctx_k)} & "
            f"{fmt_int(max_ctx_k)} & "
            f"{fmt_int(avg_ctx_k)} & "
            f"{fmt_int(total_ctx_k)} \\\\"
        )

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


    # -------------------------------------------------------------
    # LATEX COST SUMMARY (for the paper)
    # -------------------------------------------------------------
    print("\n📄 LaTeX Cost Summary Snippet:\n")
    print(
        r"\paragraph{Experiment Cost.} "
        r"Running the full MemoryAgentBench evaluation requires processing "
        f"{fmt_int(global_context_words)} context words and {fmt_int(global_questions)} "
        r"questions across all splits. "
        rf"Under our call model, each session first ingests its full context via chunked calls, "
        rf"and each question is answered by sending the question together with the available "
        rf"context (capped at {CTX_WINDOW_TOKENS} tokens). "
        rf"Using pricing assumptions of ${PRICE_INPUT}/\textit{{input}} token and "
        rf"${PRICE_OUTPUT}/\textit{{output}} token, the total estimated cost of the full "
        rf"experiment is \textbf{{${total_cost:.2f}}}, including both ingestion and "
        rf"per-question calls."
    )

    return analysis


if __name__ == "__main__":
    print("🚀 Running MemoryAgentBench analysis...")
    stats = analyze_memory_agent_bench()
    print("\nDone.")
