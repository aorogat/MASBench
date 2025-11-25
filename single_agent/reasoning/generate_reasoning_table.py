"""
CrewAI + Direct-Planning Results to LaTeX Table Generator
----------------------------------------------------------
Generates TWO separate LaTeX tables:
  1) Accuracy table with failed cases (2 rows per LLM)
  2) Time table (avg per question + total)

LLMs sorted by average accuracy within each category.
Special case: gpt:oss models treated as local.

Usage:
    source mafenv/bin/activate
    python -m single_agent.reasoning.generate_reasoning_table
"""

import os
import json
import re


# === CONFIGURATION ===
CREWAI_DIR = "results/planning"
DIRECT_DIR = "results/planning_direct"
OUTPUT_ACCURACY = os.path.join(CREWAI_DIR, "reasoning_accuracy_table.tex")
OUTPUT_FAILURES = os.path.join(CREWAI_DIR, "reasoning_failures_table.tex")
OUTPUT_TIME = os.path.join(CREWAI_DIR, "reasoning_time_table.tex")

MODEL_NAME_MAP = {
    "Groq:gpt:oss:20b": "GPT-OSS-20B",
    "gpt:oss:20b": "GPT-OSS-20B",
    "phi4:14b": "Phi-4-14B",
    "llama3.1:8b": "Llama-3.1-8B",
    "llama3.1:70b": "Llama-3.1-70B",
    "deepseek-llm:7b": "DeepSeek-7B",
    "qwen:7b": "Qwen-7B",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o-Mini",
    "gpt-4o-mini-high": "GPT-4o-Mini-High",
}


def pretty_name(llm):
    return MODEL_NAME_MAP.get(llm, llm)

# === HELPERS ===
def parse_filename(filename):
    """
    Supports:
      crewai_csqa_planning_gpt4.json
      crewai_csqa_noplanning_gpt4.json
      direct_planning_csqa_planning_ollama_deepseek.json
    """

    # Case 1 — CrewAI results
    cre = r"crewai_(?P<benchmark>\w+)_(?P<planmode>noplanning|planning)_(?P<llm>.+)\.json"
    m = re.match(cre, filename)
    if m:
        return {
            "framework": "crewai",
            "benchmark": m.group("benchmark").lower(),
            "mode": m.group("planmode"),   # 'planning' / 'noplanning'
            "llm": m.group("llm").replace("ollama_", "").replace("_", ":")
        }

    # Case 2 — Direct LLM planning
    direct = r"direct_planning_(?P<benchmark>\w+)_planning_(?P<llm>.+)\.json"
    m = re.match(direct, filename)
    if m:
        return {
            "framework": "direct",
            "benchmark": m.group("benchmark").lower(),
            "mode": "direct",
            "llm": m.group("llm").replace("ollama_", "").replace("_", ":")
        }

    return None


def extract_failed_percentage(data):
    """Compute % of questions where pred == 'FAILED'."""
    if "questions" not in data or not data["questions"]:
        return 0.0
    total = len(data["questions"])
    failed = sum(1 for q in data["questions"] if str(q.get("pred", "")).upper() == "FAILED" or str(q.get("pred", "")) == "")
    return (failed / total) * 100


def load_results(filepath):
    """Read JSON file and extract metrics."""
    with open(filepath, "r") as f:
        data = json.load(f)

    m = data["metrics"]
    fail_pct = extract_failed_percentage(data)
    
    # Calculate avg time per question
    num_questions = len(data.get("questions", [])) or 1
    time_avg_per_q = m["time_total"] / num_questions

    return {
        "accuracy": m["accuracy"] * 100,
        "fail_pct": fail_pct,
        "time_avg_per_q": time_avg_per_q,
        "time_total": m["time_total"],
    }


# === MAIN COLLECTION ===
def collect_results(*dirs):
    """
    Load CrewAI and Direct Planning results.
    Structure:
       results[llm][benchmark] = {
            "noplanning": {...},
            "planning": {...},
            "direct": {...}
       }
    """
    results = {}

    for results_dir in dirs:
        if not os.path.isdir(results_dir):
            continue

        for file in os.listdir(results_dir):
            if not file.endswith(".json"):
                continue

            parsed = parse_filename(file)
            if not parsed:
                continue

            benchmark = parsed["benchmark"]
            mode = parsed["mode"]
            llm = parsed["llm"]

            filepath = os.path.join(results_dir, file)
            data = load_results(filepath)

            results.setdefault(llm, {}).setdefault(benchmark, {})[mode] = data

    return results


# === MODEL CLASSIFICATION ===
def is_local_model(llm_name):
    """Classify as local or remote. Special case: gpt:oss is local."""
    if "gpt:oss" in llm_name.lower():
        return True
    return any(x in llm_name.lower() for x in ["llama", "deepseek", "qwen", "mistral", "phi"])


# === SORTING ===
def compute_avg_accuracy(llm_data):
    """Compute average accuracy across all benchmarks and modes."""
    accs = []
    for bench, modes in llm_data.items():
        for mode, metrics in modes.items():
            if metrics:
                accs.append(metrics["accuracy"])
    return sum(accs) / len(accs) if accs else 0.0


def sort_by_accuracy(model_dict):
    """Sort model dict by average accuracy (descending)."""
    return sorted(model_dict.items(), 
                  key=lambda x: compute_avg_accuracy(x[1]), 
                  reverse=True)


# === LATEX HELPERS ===
def fmt(x):
    """Format numeric values with at most 2 decimals."""
    if isinstance(x, (float, int)):
        return f"{x:.1f}"
    return str(x)


def latex_acc_row(m_no, m_pl, m_dir):
    """Return 3 accuracy cells: NoPlan, \\checkmark (CrewAI)  (Δ), DirectPlan (Δ)."""

    # base accuracy for no-planning
    if not m_no:
        no_str = "\\textcolor{red}{X}"
    else:
        no_str = fmt(m_no["accuracy"])

    # Crew planning change
    pl_str = acc_change(m_no, m_pl)

    # Direct planning change
    dir_str = acc_change(m_no, m_dir)

    return f"{no_str} & {pl_str} & {dir_str}"


def acc_change(old_m, new_m):
    """Return \accChange{old}{new} or X if missing."""
    if not old_m or not new_m:
        return "\\textcolor{red}{X}"

    old = fmt(old_m["accuracy"])
    new = fmt(new_m["accuracy"])
    return f"\\accChange{{{old}}}{{{new}}}"


def latex_time_cells(no, pl, direct):
    """Time with avg per question and total."""
    def time_str(m):
        if not m:
            return "\\textcolor{red}{X} & \\textcolor{red}{X}"
        avg = fmt(m["time_avg_per_q"])
        total = fmt(m["time_total"])
        return f"{avg} & {total}"
    
    return f"{time_str(no)} & {time_str(pl)} & {time_str(direct)}"


# === ACCURACY TABLE ===
def build_accuracy_table(results):
    """Build LaTeX accuracy table."""
    lines = [
        "\\setlength{\\tabcolsep}{1pt}",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Accuracy results (\\%) across CrewAI (no-planning vs planning) and Direct-LLM planning.}",
        "\\label{tab:reasoning-accuracy}",
        "\\begin{tabular}{l|ccc|ccc|ccc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{LLM}} & "
        "\\multicolumn{3}{c}{\\textbf{GSM8K}} & "
        "\\multicolumn{3}{c}{\\textbf{CSQA}} & "
        "\\multicolumn{3}{c}{\\textbf{MATH-100}} \\\\",
        "\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}",
        " & \\xmark & \\checkmark (crew) & \\checkmark (LLM) & \\xmark & \\checkmark (crew) & \\checkmark (LLM)& \\xmark & \\checkmark (crew) & \\checkmark (LLM) \\\\",
        "\\midrule",
    ]

    # Split and sort by model type
    local_models = {k: v for k, v in results.items() if is_local_model(k)}
    remote_models = {k: v for k, v in results.items() if not is_local_model(k)}
    
    local_models = sort_by_accuracy(local_models)
    remote_models = sort_by_accuracy(remote_models)

    def write_section(title, group):
        if not group:
            return

        # lines.append(f"\\multicolumn{{10}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        lines.append(f"\\rowcolor{{gray!20}} \\multicolumn{{10}}{{c}}{{\\textbf{{{title}}}}} \\\\")

        # lines.append("\\midrule")

        for llm, data in group:
            row = [pretty_name(llm)]
            for bench in ["gsm8k", "csqa", "math"]:
                modes = data.get(bench, {})
                m_no = modes.get("noplanning")
                m_pl = modes.get("planning")
                m_dir = modes.get("direct")
                row.append(latex_acc_row(m_no, m_pl, m_dir))
            lines.append(" & ".join(row) + " \\\\")

    write_section("Local Models", local_models)
    write_section("Remote Models", remote_models)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)


# === FAILURES TABLE ===
def build_failures_table(results):
    """Build LaTeX failures table (CrewPlan + Direct only; NoPlan removed)."""
    lines = [
        "\\setlength{\\tabcolsep}{1pt}",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Failed cases (\\%) for CrewAI-planning and Direct-LLM planning (NoPlan removed).}",
        "\\label{tab:reasoning-failures}",
        "\\begin{tabular}{l|cc|cc|cc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{LLM}} & "
        "\\multicolumn{2}{c}{\\textbf{GSM8K}} & "
        "\\multicolumn{2}{c}{\\textbf{CSQA}} & "
        "\\multicolumn{2}{c}{\\textbf{MATH-100}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        " & \\checkmark (CrewAI) & \\checkmark (LLM) & "
        "\\checkmark (CrewAI) & \\checkmark (LLM) & "
        "\\checkmark (CrewAI) & \\checkmark (LLM) \\\\",
        "\\midrule",
    ]

    # Split models
    local_models = {k: v for k, v in results.items() if is_local_model(k)}
    remote_models = {k: v for k, v in results.items() if not is_local_model(k)}
    
    local_models = sort_by_accuracy(local_models)
    remote_models = sort_by_accuracy(remote_models)

    def write_section(title, group):
        if not group:
            return

        # section header spans 7 columns total
        lines.append(f"\\rowcolor{{gray!20}} \\multicolumn{{7}}{{c}}{{\\textbf{{{title}}}}} \\\\")

        for llm, data in group:
            row = [pretty_name(llm)]
            for bench in ["gsm8k", "csqa", "math"]:
                modes = data.get(bench, {})
                m_pl = modes.get("planning")
                m_dir = modes.get("direct")

                def fail_str(m):
                    if not m:
                        return "\\textcolor{red}{X}"
                    return fmt(m['fail_pct']) + "\\%"

                row.append(f"{fail_str(m_pl)} & {fail_str(m_dir)}")

            lines.append(" & ".join(row) + " \\\\")

    write_section("Local Models", local_models)
    write_section("Remote Models", remote_models)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])

    return "\n".join(lines)



# === TIME TABLE ===
def build_time_table(results):
    """
    Build LaTeX table showing runtime multipliers:
       CrewAI multiplier  = T_planning / T_noplanning
       Direct multiplier  = T_direct   / T_noplanning

    Output columns:
        LLM | GSM8K ΔCrew ΔDir | CSQA ΔCrew ΔDir | MATH ΔCrew ΔDir
    """

    def mult(base, new):
        if not base or not new:
            return "\\textcolor{red}{X}"
        t0 = base["time_avg_per_q"]
        t1 = new["time_avg_per_q"]
        if t0 <= 0:
            return "\\textcolor{red}{X}"
        ratio = t1 / t0
        # format: 1 decimal normally, 2 decimals for extreme values
        if ratio < 0.1 or ratio > 10:
            r = f"{ratio:.2f}"
        else:
            r = f"{ratio:.1f}"
        return f"\\times {r}"

    lines = [
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{Runtime multiplier (relative to NoPlan). Lower is faster.}",
        "\\label{tab:reasoning-time-mult}",
        "\\begin{tabular}{l|cc|cc|cc}",
        "\\toprule",
        "\\multirow{2}{*}{\\textbf{LLM}} & "
        "\\multicolumn{2}{c}{\\textbf{GSM8K}} & "
        "\\multicolumn{2}{c}{\\textbf{CSQA}} & "
        "\\multicolumn{2}{c}{\\textbf{MATH}} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        " & \\Delta CrewAI & \\Delta LLM & \\Delta CrewAI & \\Delta LLM & \\Delta CrewAI & \\Delta LLM \\\\",
        "\\midrule",
    ]

    # model groups
    local_models = sort_by_accuracy({k: v for k, v in results.items() if is_local_model(k)})
    remote_models = sort_by_accuracy({k: v for k, v in results.items() if not is_local_model(k)})

    def write_section(title, group):
        if not group:
            return
        lines.append(f"\\rowcolor{{gray!20}} \\multicolumn{{7}}{{c}}{{\\textbf{{{title}}}}} \\\\")
        
        for llm, data in group:
            row = [pretty_name(llm)]
            for bench in ["gsm8k", "csqa", "math"]:
                modes = data.get(bench, {})
                m_no = modes.get("noplanning")
                m_pl = modes.get("planning")
                m_dir = modes.get("direct")
                row.append(mult(m_no, m_pl))
                row.append(mult(m_no, m_dir))
            lines.append(" & ".join(row) + " \\\\")
    
    write_section("Local Models", local_models)
    write_section("Remote Models", remote_models)

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)



# === ENTRY POINT ===
def main():
    results = collect_results(CREWAI_DIR, DIRECT_DIR)

    # Generate accuracy table
    accuracy_latex = build_accuracy_table(results)
    with open(OUTPUT_ACCURACY, "w") as f:
        f.write(accuracy_latex)
    print(f"✅ Accuracy table generated at: {OUTPUT_ACCURACY}")

    # Generate failures table
    failures_latex = build_failures_table(results)
    with open(OUTPUT_FAILURES, "w") as f:
        f.write(failures_latex)
    print(f"✅ Failures table generated at: {OUTPUT_FAILURES}")

    # Generate time table
    time_latex = build_time_table(results)
    with open(OUTPUT_TIME, "w") as f:
        f.write(time_latex)
    print(f"✅ Time table generated at: {OUTPUT_TIME}")


if __name__ == "__main__":
    main()