import os
import json
from glob import glob
from collections import defaultdict

# 1. Subtasks per category
CATEGORY_SUBTASKS = {
    "Accurate Retrieval": ["SH-QA", "MH-QA", "LME(S*)", "EventQA"],
    "Test-Time Learning": ["MovieRec", "MCC"], 
    "Long-Range Understanding": ["Summ.", "DetQA"],
    "Selective Forgetting": ["FC-SH", "FC-MH"],
}


SUBTASK_TO_CATEGORY = {
    st: cat for cat, lst in CATEGORY_SUBTASKS.items() for st in lst
}


def parse_result_file(file_path):
    with open(file_path) as f:
        data = json.load(f)

    print(f"🧾 Parsing: {file_path}")

    system = data.get("system", "Unknown System")
    results = data.get("results_so_far") or data.get("results") or []

    subtasks_scores = defaultdict(list)
    for r in results:
        subtask = r.get("subtask")
        score = r.get("score")
        subtask = r.get("subtask")

        if subtask not in SUBTASK_TO_CATEGORY or score is None:
            continue
        subtasks_scores[subtask].append(score)

    avg_scores = {
        subtask: sum(scores) / len(scores)
        for subtask, scores in subtasks_scores.items()
    }

    if not avg_scores:
        print(f"⚠️ No valid scores found in: {file_path}")

    return system, avg_scores


def format_score(score):
    return f"{score * 100:.1f}" if score is not None else r"\textcolor{red}{--}"



def generate_latex_table(results_dir):
    json_files = [
        f for f in glob(os.path.join(results_dir, "*.json"))
        if "_partial" not in f and "_summary" not in f
    ]

    systems = defaultdict(dict)

    for file_path in json_files:
        system, subtask_scores = parse_result_file(file_path)
        for subtask, score in subtask_scores.items():
            systems[system][subtask] = score

    # Compute category averages + overall
    for system, score_map in systems.items():
        for cat, subtasks in CATEGORY_SUBTASKS.items():
            cat_scores = [score_map.get(st) for st in subtasks if score_map.get(st) is not None]
            if cat_scores:
                score_map[f"{cat} Avg."] = sum(cat_scores) / len(cat_scores)
        all_avgs = [v for k, v in score_map.items() if "Avg." in k]
        if all_avgs:
            score_map["Overall"] = sum(all_avgs) / len(all_avgs)

    # Build LaTeX header
    header = r"""
\setlength{\tabcolsep}{2pt}
\begin{table*}[t]
\centering
\caption{Evaluation of memory competencies on MemoryAgentBench.
For each agent framework, scores are first computed per session, then averaged across all sessions belonging to the same subtask.
For OpenAI SDK variants, $C$ denotes the maximum short-term context window (in tokens) retained during accumulation-based memory.
Category-level scores (AR, TTL, LRU, SF) are obtained by averaging the corresponding subtask scores within each category.
The Overall score is computed as the mean of the category-level averages.
}
\label{tab:memorybench-results}
\footnotesize
\begin{tabular}{
    l|
    *{5}{c}|  % AR (4 subtasks + Avg)
    *{3}{c}|  % TTL (2 subtasks + Avg)
    *{3}{c}|  % LRU (2 subtasks + Avg)
    *{3}{c}|  % SF (2 subtasks + Avg)
    c         % Overall
}
\toprule
\textbf{Agent Framework} &
\multicolumn{5}{c|}{\textbf{Accurate Retrieval (AR)}} &
\multicolumn{3}{c|}{\textbf{Test-Time Learning (TTL)}} &
\multicolumn{3}{c|}{\textbf{Long-Range Understanding (LRU)}} &
\multicolumn{3}{c|}{\textbf{Selective Forgetting (SF)}} &
\textbf{Overall} \\
\midrule
& SH-QA & MH-QA & LME(S*) & EventQA & Avg. &
  MCC & Recom. & Avg. &
  Summ. & DetQA & Avg. &
  FC-SH & FC-MH & Avg. &
  Score \\
\midrule
""".strip()

    body = ""
    for system in sorted(systems.keys()):
        score_map = systems[system]
        row = [
            # AR subtasks + avg
            format_score(score_map.get("SH-QA")),
            format_score(score_map.get("MH-QA")),
            format_score(score_map.get("LME(S*)")),
            format_score(score_map.get("EventQA")),
            format_score(score_map.get("Accurate Retrieval Avg.")),
            # TTL subtasks + avg
            format_score(score_map.get("MCC")),
            format_score(score_map.get("MovieRec")),
            format_score(score_map.get("Test-Time Learning Avg.")),
            # LRU subtasks + avg
            format_score(score_map.get("Summ.")),
            format_score(score_map.get("DetQA")),
            format_score(score_map.get("Long-Range Understanding Avg.")),
            # SF subtasks + avg
            format_score(score_map.get("FC-SH")),
            format_score(score_map.get("FC-MH")),
            format_score(score_map.get("Selective Forgetting Avg.")),
            # Overall
            format_score(score_map.get("Overall")),
        ]
        readable_name = system.replace("_", " ")
        body += f"{readable_name} & " + " & ".join(row) + r" \\" + "\n"

    footer = r"""
\midrule
\multicolumn{15}{p{0.95\linewidth}}{
\footnotesize\textit{Note:} While the Groq-backed \texttt{openai/gpt-oss-20b} model supports context windows up to $\sim$130K tokens, using very large accumulated contexts frequently hits tokens-per-minute (TPM) rate limits (e.g., 250K TPM limit exceeded during benchmark runs). This renders extreme context accumulation impractical for scalable evaluations such as MemoryAgentBench, despite being theoretically supported by the model.
}
\bottomrule
\end{tabular}
\end{table*}"""

    return header + "\n" + body + "\n" + footer


if __name__ == "__main__":
    results_folder = "results/memory"
    latex_code = generate_latex_table(results_folder)

    out_path = os.path.join(results_folder, "memoryagentbench_results_table.tex")
    with open(out_path, "w") as f:
        f.write(latex_code)

    print(f"✅ LaTeX table written to {out_path}")
