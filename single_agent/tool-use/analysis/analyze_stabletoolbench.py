"""
analyze_stabletoolbench.py
------------------------------------
This script analyzes the StableToolBench benchmark, extracts tool and query
statistics, and prepares standard outputs for integration with 
multi-agent frameworks (LangGraph, CrewAI, AutoGen, etc).

"""

import os
import json
from pathlib import Path
from collections import Counter, defaultdict


# -------------------------------------------------------
# 1. Load Tools
# -------------------------------------------------------
def load_tools(tools_dir):
    tools = []
    for category in os.listdir(tools_dir):
        cat_path = os.path.join(tools_dir, category)
        if not os.path.isdir(cat_path):
            continue
        for tool_file in os.listdir(cat_path):
            if not tool_file.endswith(".json"):
                continue
            with open(os.path.join(cat_path, tool_file), "r", encoding="utf8") as f:
                tool_data = json.load(f)
                tool_data["category"] = category
                tools.append(tool_data)
    return tools


def analyze_tools(tools):
    stats = {
        "total_tools": len(tools),
        "categories": Counter([t["category"] for t in tools]),
        "api_count_distribution": Counter([len(t.get("apis", [])) for t in tools]),
    }

    # Flatten API schemas for LangGraph/CrewAI
    flat_schema = []
    for t in tools:
        for api in t.get("apis", []):
            flat_schema.append({
                "tool_name": t["name"],
                "category": t["category"],
                "api_name": api["name"],
                "description": api.get("description", ""),
                "parameters": api.get("parameters", {})
            })

    return stats, flat_schema


# -------------------------------------------------------
# 2. Load Solvable Queries
# -------------------------------------------------------
def load_queries(query_root):
    all_queries = []
    for group in os.listdir(query_root):
        group_path = os.path.join(query_root, group)
        if not group_path.endswith(".json"):
            continue

    # Actually, StableToolBench stores folders:
    for group in os.listdir(query_root):
        group_dir = os.path.join(query_root, group)
        if not os.path.isdir(group_dir):
            continue

        for file in os.listdir(group_dir):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(group_dir, file), "r", encoding="utf8") as f:
                queries = json.load(f)
                for q in queries:
                    q["group"] = group
                all_queries += queries

    return all_queries


def analyze_queries(queries):
    stats = {
        "total_queries": len(queries),
        "groups": Counter([q["group"] for q in queries]),
        "avg_query_length": sum(len(q["instruction"].split()) for q in queries) / len(queries),
        "query_length_dist": Counter([len(q["instruction"].split()) for q in queries]),
    }

    # Tool requirement distribution
    needed_tools = []
    for q in queries:
        if "tool_plan" in q:
            needed_tools += [step["tool_name"] for step in q["tool_plan"]]

    stats["tool_usage_distribution"] = Counter(needed_tools)

    return stats


# -------------------------------------------------------
# 3. Save Summaries
# -------------------------------------------------------
def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def save_markdown(path, tool_stats, query_stats):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        f.write("# StableToolBench Benchmark Analysis\n\n")
        f.write("## Tool Summary\n")
        f.write(f"- Total tools: **{tool_stats['total_tools']}**\n")
        f.write(f"- Categories: `{dict(tool_stats['categories'])}`\n")
        f.write("\n## Query Summary\n")
        f.write(f"- Total solvable queries: **{query_stats['total_queries']}**\n")
        f.write(f"- Groups: `{dict(query_stats['groups'])}`\n")
        f.write(f"- Avg query length: **{query_stats['avg_query_length']:.2f} words**\n")
        f.write("\n---\n")
        f.write("Generated automatically by Abdelghny's StableToolBench Analyzer.\n")


# -------------------------------------------------------
# 4. Entry Point
# -------------------------------------------------------
def main():
    ROOT = Path(".")
    tools_dir = ROOT / "toolenv/tools"
    query_root = ROOT / "solvable_queries"

    print("📌 Loading tools...")
    tools = load_tools(tools_dir)
    tool_stats, flat_schema = analyze_tools(tools)

    print("📌 Loading solvable queries...")
    queries = load_queries(query_root)
    query_stats = analyze_queries(queries)

    print("📌 Saving reports...")
    save_json("stats/tools_summary.json", tool_stats)
    save_json("stats/tools_schema.json", flat_schema)
    save_json("stats/query_summary.json", query_stats)
    save_markdown("stats/for_paper.md", tool_stats, query_stats)

    print("✅ Analysis complete. Outputs saved under /stats.")


if __name__ == "__main__":
    main()
