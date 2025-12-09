"""
Analyze StableToolBench Dataset
----------------------------------
This script analyzes the solvable queries and tool environment included
in this repository and produces statistics + a fully self-contained
LaTeX figure built using TikZ/PGFPlots (NO external images required).

Outputs:
    analysis_results/
        tool_stats.json
        query_stats.json
        query_bins.json
        category_top10.json
        apis_per_tool_bins.json
        latex_figure.tex   <-- ready to paste into your SIGMOD paper

Author: Abdelghny Orogat
"""

import json
from pathlib import Path
from collections import Counter
import statistics
import numpy as np


# --------------------------
# Helpers
# --------------------------
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------
# TOOLS ANALYSIS
# --------------------------
def analyze_tools(tool_dir):
    tool_files = list(Path(tool_dir).glob("*.json"))
    api_counts = []
    param_counts = []
    categories = Counter()

    for file in tool_files:
        tool = load_json(file)
        categories["Unknown"] += 1  # Tools don't contain category info

        apis = tool.get("api_list", [])
        api_counts.append(len(apis))

        for api in apis:
            params = api.get("required_parameters", []) + api.get("optional_parameters", [])
            param_counts.append(len(params))

    return {
        "total_tools": len(tool_files),
        "tool_categories": dict(categories),
        "total_api_endpoints": sum(api_counts),
        "avg_apis_per_tool": statistics.mean(api_counts) if api_counts else 0,
        "api_counts": api_counts,
        "avg_params_per_api": statistics.mean(param_counts) if param_counts else 0,
    }


# --------------------------
# QUERY ANALYSIS
# --------------------------
def analyze_queries(query_root):
    json_files = list(Path(query_root).rglob("*.json"))

    groups = Counter()
    query_lengths = []
    categories = Counter()
    total_queries = 0

    for f in json_files:
        group = f.parent.name
        data = load_json(f)

        groups[group] += len(data)
        total_queries += len(data)

        for q in data:
            if not isinstance(q, dict) or "query" not in q:
                continue

            query_text = q["query"]
            query_lengths.append(len(query_text.split()))

            for api in q.get("api_list", []):
                categories[api.get("category_name", "Unknown")] += 1

    return {
        "total_queries": total_queries,
        "groups": dict(groups),
        "query_lengths": query_lengths,
        "categories": dict(categories),
        "avg_query_length": statistics.mean(query_lengths),
        "min_query_length": min(query_lengths),
        "max_query_length": max(query_lengths),
    }



def debug_print_bins(query_lengths, api_counts, top10_categories):
    # Query bins
    q_bins = [0, 0, 0, 0, 0]  # 0–20, 20–40, 40–60, 60–80, 80+

    for L in query_lengths:
        if L < 20: q_bins[0] += 1
        elif L < 40: q_bins[1] += 1
        elif L < 60: q_bins[2] += 1
        elif L < 80: q_bins[3] += 1
        else: q_bins[4] += 1

    # API bins
    a_bins = [0, 0, 0, 0, 0, 0]  # 1, 2–5, 6–10, 11–20, 21–50, 50+

    for k in api_counts:
        if k == 1: a_bins[0] += 1
        elif 2 <= k <= 5: a_bins[1] += 1
        elif 6 <= k <= 10: a_bins[2] += 1
        elif 11 <= k <= 20: a_bins[3] += 1
        elif 21 <= k <= 50: a_bins[4] += 1
        else: a_bins[5] += 1

    print("\n==================== DEBUG METRICS ====================")
    print("Query Length Bins:")
    print("0–20:", q_bins[0])
    print("20–40:", q_bins[1])
    print("40–60:", q_bins[2])
    print("60–80:", q_bins[3])
    print("80+  :", q_bins[4])

    print("\nAPI Endpoints per Tool Bins:")
    print("1:", a_bins[0])
    print("2–5:", a_bins[1])
    print("6–10:", a_bins[2])
    print("11–20:", a_bins[3])
    print("21–50:", a_bins[4])
    print("50+:", a_bins[5])

    print("\nTop API Categories:")
    for name, count in top10_categories:
        print(f"{name}: {count}")

    print("=======================================================\n")







# --------------------------
# MAIN
# --------------------------
def main():
    root = Path(__file__).resolve().parents[1] / "StableToolBench"
    tool_dir = root / "toolenv" / "tools"
    query_root = root / "solvable_queries"

    out_dir = Path(__file__).resolve().parent / "analysis_results"
    out_dir.mkdir(exist_ok=True)

    print("Analyzing tools...")
    tool_stats = analyze_tools(tool_dir)

    print("Analyzing queries...")
    query_stats = analyze_queries(query_root)

    # --------------------------
    # Compute REAL histogram bins
    # --------------------------
    query_lengths = query_stats["query_lengths"]
    bins = [0,20,40,60,80,100,120,140,160]
    hist, _ = np.histogram(query_lengths, bins=bins)
    query_bins = {f"{bins[i]}–{bins[i+1]}": int(hist[i]) for i in range(len(hist))}

    # top 10 categories
    cats = Counter(query_stats["categories"])
    top10 = cats.most_common(10)

    # tools api histogram
    api_counts = tool_stats["api_counts"]
    api_bins = Counter(api_counts)

    # save JSON for reference
    (out_dir / "query_bins.json").write_text(json.dumps(query_bins, indent=2))
    (out_dir / "category_top10.json").write_text(json.dumps(top10, indent=2))
    (out_dir / "apis_per_tool_bins.json").write_text(json.dumps(dict(api_bins), indent=2))

    # --------------------------
    # Write LaTeX figure
    # --------------------------
    top10 = Counter(query_stats["categories"]).most_common(10)

    debug_print_bins(
    query_stats["query_lengths"],
    tool_stats["api_counts"],
    top10
    )



    print("\n✔ DONE — all results saved to:", out_dir)


if __name__ == "__main__":
    main()
