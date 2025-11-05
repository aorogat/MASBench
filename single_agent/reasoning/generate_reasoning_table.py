"""
CrewAI Results to LaTeX Table Generator
---------------------------------------
Aggregates benchmark results for GSM8K, CSQA, and MATH (with and without planning)
from CrewAI JSON outputs and generates a LaTeX summary table.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.generate_reasoning_table
"""

import os
import json
import re


# === CONFIGURATION ===
RESULTS_DIR = "results/planning"
OUTPUT_LATEX = os.path.join(RESULTS_DIR, "reasoning_results_table.tex")


# === HELPERS ===
def parse_filename(filename):
    """
    Extract benchmark, planning flag, and LLM name from filenames like:
    crewai_csqa_noplanning_gpt-4o-mini.json
    """
    pattern = r"crewai_(?P<benchmark>\w+)_(?P<planmode>noplanning|planning)_(?P<llm>.+)\.json"
    match = re.match(pattern, filename)
    if not match:
        return None
    benchmark = match.group("benchmark").lower()
    planning = match.group("planmode") == "planning"
    llm = match.group("llm").replace("ollama_", "").replace("_", ":")
    return benchmark, planning, llm


def extract_failed_percentage(data):
    """Compute percentage of questions where pred == 'FAILED'."""
    if "questions" not in data or not data["questions"]:
        return 0.0
    total = len(data["questions"])
    failed = sum(1 for q in data["questions"] if str(q.get("pred", "")).upper() == "FAILED")
    return (failed / total) * 100


def load_results(filepath):
    """Read JSON file and extract metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)
    m = data["metrics"]
    fail_pct = extract_failed_percentage(data)
    return {
        "accuracy": m["accuracy"] * 100,
        "time_min": m["time_min"],
        "time_max": m["time_max"],
        "time_mean": m["time_mean"],
        "time_total": m["time_total"],
        "tokens_min": m["tokens_min"],
        "tokens_max": m["tokens_max"],
        "tokens_mean": m["tokens_mean"],
        "tokens_total": m["tokens_total"],
        "fail_pct": fail_pct
    }


# === MAIN COLLECTION ===
def collect_results(results_dir):
    """Load all benchmark JSON results and organize by LLM → benchmark → planning flag."""
    results = {}
    for file in os.listdir(results_dir):
        if not file.endswith(".json"):
            continue
        parsed = parse_filename(file)
        if not parsed:
            continue
        benchmark, planning, llm = parsed
        filepath = os.path.join(results_dir, file)
        data = load_results(filepath)
        results.setdefault(llm, {}).setdefault(benchmark, {})[planning] = data
    return results


# === LATEX HELPERS ===
def fmt(x):
    """Format numeric values with at most 2 decimals."""
    if isinstance(x, (float, int)):
        return f"{x:.2f}"
    return str(x)

def latex_minmaxmean(m):
    return f"\\minmaxmean{{{fmt(m['time_min'])}}}{{{fmt(m['time_max'])}}}{{{fmt(m['time_mean'])}}}{{{fmt(m['time_total'])}}}"

def latex_tokens(m):
    return f"\\minmaxmean{{{fmt(m['tokens_min'])}}}{{{fmt(m['tokens_max'])}}}{{{fmt(m['tokens_mean'])}}}{{{fmt(m['tokens_total'])}}}"

def latex_accuracy(no_plan, plan):
    acc_no = no_plan['accuracy']
    acc_plan = plan['accuracy'] if plan else None
    fail_pct = plan['fail_pct'] if plan else 0
    if acc_plan is not None:
        return f"{fmt(acc_no)} & \\accChange{{{fmt(acc_no)}}}{{{fmt(acc_plan)}}} $^{{({fmt(fail_pct)}\\%F)}}$"
    else:
        return f"{fmt(acc_no)} & \\textcolor{{red}}{{X}}"



def build_latex_table(results):
    """Construct LaTeX table string."""
    lines = [
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Reasoning results across benchmarks with different LLMs. "
        "This experiment is designed to measure the effectiveness of CrewAI’s \\emph{reasoning} feature "
        "by comparing performance in two modes: \\xmark = execution without reasoning, "
        "\\checkmark = with predefined reasoning, and F = cases where the model failed to follow CrewAI’s "
        "required reasoning output format. Metrics reported include accuracy, runtime (mean / total), and token usage.}",
        "\\vspace{-3mm}",
        "\\label{tab:reasoning-results}",
        "\\begin{tabular}{llcc|cc|cc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{LLM}} & \\multirow{2}{*}{\\textbf{Metric}} & "
        "\\multicolumn{2}{c|}{\\textbf{GSM8K}} & "
        "\\multicolumn{2}{c|}{\\textbf{CSQA}} & "
        "\\multicolumn{2}{c}{\\textbf{MATH-100}} \\\\",
        "\\cmidrule(lr){3-4} \\cmidrule(lr){5-6} \\cmidrule(lr){7-8}",
        " & & \\xmark & \\checkmark & \\xmark & \\checkmark & \\xmark & \\checkmark \\\\",
        "\\midrule",
    ]

    # --- Split models ---
    local_models = {k: v for k, v in results.items() if any(x in k.lower() for x in ["llama", "deepseek", "qwen", "mistral", "phi"])}
    remote_models = {k: v for k, v in results.items() if any(x in k.lower() for x in ["gpt", "claude", "gemini"])}

    # Helper to write each section
    def write_section(title, model_group):
        if not model_group:
            return
        lines.append(f"\\multicolumn{{8}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        lines.append("\\midrule")
        for llm, data in sorted(model_group.items()):
            lines.append(f"\\multirow{{3}}{{*}}{{{llm}}} ")

            # Runtime row
            row = [" & Runtime (sec) "]
            for bench in ["gsm8k", "csqa", "math"]:
                m_no = data.get(bench, {}).get(False)
                m_pl = data.get(bench, {}).get(True)
                if m_no and m_pl:
                    row.append(latex_minmaxmean(m_no))
                    row.append(latex_minmaxmean(m_pl))
                elif m_no:
                    row.append(latex_minmaxmean(m_no))
                    row.append("\\textcolor{red}{X}")
                else:
                    row.append("\\textcolor{red}{X}")
                    row.append("\\textcolor{red}{X}")
            lines.append(" & ".join(row) + " \\\\")

            # Tokens row
            row = [" & Tokens "]
            for bench in ["gsm8k", "csqa", "math"]:
                m_no = data.get(bench, {}).get(False)
                m_pl = data.get(bench, {}).get(True)
                if m_no and m_pl:
                    row.append(latex_tokens(m_no))
                    row.append(latex_tokens(m_pl))
                elif m_no:
                    row.append(latex_tokens(m_no))
                    row.append("\\textcolor{red}{X}")
                else:
                    row.append("\\textcolor{red}{X}")
                    row.append("\\textcolor{red}{X}")
            lines.append(" & ".join(row) + " \\\\")

            # Accuracy row
            row = [" & Accuracy (\\%) "]
            for bench in ["gsm8k", "csqa", "math"]:
                m_no = data.get(bench, {}).get(False)
                m_pl = data.get(bench, {}).get(True)
                if m_no:
                    row.append(latex_accuracy(m_no, m_pl))
                else:
                    row.append("\\textcolor{red}{X} & \\textcolor{red}{X}")
            lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule")

    # --- Write both groups ---
    write_section("Local Models", local_models)
    write_section("Remote Models", remote_models)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}"
    ])
    return "\n".join(lines)


# === ENTRY POINT ===
def main():
    results = collect_results(RESULTS_DIR)
    latex_content = build_latex_table(results)
    with open(OUTPUT_LATEX, "w") as f:
        f.write(latex_content)
    print(f"✅ LaTeX table generated at: {OUTPUT_LATEX}")


if __name__ == "__main__":
    main()
