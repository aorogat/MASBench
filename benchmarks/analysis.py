"""
Benchmark Analysis Module
-------------------------
Run with:

    python -m benchmarks.analysis

Computes descriptive statistics for:
- GSM8K
- CSQA
- MATH

Outputs:
- Console stats
- LaTeX table (SIGMOD multi-column style)
"""

import statistics
from typing import Dict, Any


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def word_count(text: str) -> int:
    """Safe word-count function."""
    if not text:
        return 0
    return len(text.split())


# -------------------------------------------------------------------
# ANALYSIS CORE
# -------------------------------------------------------------------

def analyze_benchmark(bench) -> Dict[str, Any]:
    """Compute descriptive statistics for any loaded Benchmark instance."""

    results = {}

    # ---- Number of questions ----
    q_count = len(bench.questions)
    results["num_questions"] = q_count

    # ---- Question lengths ----
    q_lengths = [word_count(q.question) for q in bench.questions]


    # ⚠ Debug: print any questions with suspiciously small word count
    for idx, q in enumerate(bench.questions):
        wc = word_count(q.question)
        if wc <= 5:   # print extremely short questions
            print("\n🚨 VERY SHORT MATH QUESTION DETECTED")
            print(f"Index: {idx}")
            print(f"QID:   {getattr(q, 'qid', 'N/A')}")
            print(f"Words: {wc}")
            print("TEXT:")
            print(q.question)
            print("-" * 60)



    results["q_min"] = min(q_lengths) if q_lengths else 0
    results["q_max"] = max(q_lengths) if q_lengths else 0
    results["q_avg"] = statistics.mean(q_lengths) if q_lengths else 0
    results["q_total"] = sum(q_lengths) if q_lengths else 0

    # ---- Gold answer lengths ----
    gold_lengths = [word_count(q.gold) for q in bench.questions]
    results["g_min"] = min(gold_lengths) if gold_lengths else 0
    results["g_max"] = max(gold_lengths) if gold_lengths else 0
    results["g_avg"] = statistics.mean(gold_lengths) if gold_lengths else 0
    results["g_total"] = sum(gold_lengths) if gold_lengths else 0

    return results


# -------------------------------------------------------------------
# LATEX TABLE GENERATOR (MULTI-COLUMN, NO TYPES COLUMN)
# -------------------------------------------------------------------

def generate_latex_table(all_results: Dict[str, Dict[str, Any]]) -> str:
    """
    Generate a SIGMOD-style multi-column table:

    Benchmark | #Q | Q(min max avg total) | Gold(min max avg total)
    """

    header = r"""
\begin{table}[ht]
\centering
\footnotesize
\begin{tabular}{l r | rrrr | rrrr}
\toprule
& & \multicolumn{4}{c|}{\textbf{Question Length (words)}} &
    \multicolumn{4}{c}{\textbf{Gold Answer Length (words)}} \\
\cmidrule(lr){3-6} \cmidrule(lr){7-10}
\textbf{Benchmark} & \textbf{\#Q} &
\textbf{Min} & \textbf{Max} & \textbf{Avg} & \textbf{Total} &
\textbf{Min} & \textbf{Max} & \textbf{Avg} & \textbf{Total} \\
\midrule
"""

    rows = []

    for name, r in all_results.items():

        row = (
            f"{name} & {r['num_questions']} & "
            f"{r['q_min']} & {r['q_max']} & {r['q_avg']:.1f} & {r['q_total']:,} & "
            f"{r['g_min']} & {r['g_max']} & {r['g_avg']:.1f} & {r['g_total']:,} \\\\"
        )
        rows.append(row)

    footer = r"""
\bottomrule
\end{tabular}
\caption{Benchmark statistics with multi-column results for question and gold-answer lengths.}
\label{tab:benchmark_stats_multicol}
\end{table}
"""

    return header + "\n".join(rows) + footer


# -------------------------------------------------------------------
# MAIN: RUNNING AS PYTHON MODULE
# -------------------------------------------------------------------

def main():
    """Entry point when running `python -m benchmarks.analysis`."""
    print("📊 Loading benchmarks...")

    from benchmarks.gsm8k import GSM8KBenchmark
    from benchmarks.csqa import CSQABenchmark
    from benchmarks.math import MATHBenchmark

    gsm   = GSM8KBenchmark(n=None)
    csqa  = CSQABenchmark(n=None)
    mathb = MATHBenchmark(n=None, use_remote=False)

    print("✅ Benchmarks loaded.")

    # Compute results
    results = {
        "GSM8K": analyze_benchmark(gsm),
        "CSQA":  analyze_benchmark(csqa),
        "MATH":  analyze_benchmark(mathb),
    }

    print("\n============================")
    print("📄 LATEX TABLE OUTPUT")
    print("============================\n")
    print(generate_latex_table(results))


# -------------------------------------------------------------------
# ENTRYPOINT
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()
