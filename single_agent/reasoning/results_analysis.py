# Run: python -m single_agent.reasoning.results_analysis

import os
import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = "results/planning"
DIRECT_DIR = "results/planning_direct"
OUTPUT_DIR = "results/planning/analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_filename(fname):
    """
    fname example:
      crewai_csqa_noplanning_gpt-4.1.json
      crewai_gsm8k_planning_Groq_gpt_oss_20b.json
    Returns:
      (framework, benchmark, mode, model)
    """
    name = fname.replace(".json", "")
    parts = name.split("_")

    framework = parts[0]
    benchmark = parts[1]
    mode = parts[2]               # planning / noplanning
    model = "_".join(parts[3:])   # everything after that
    return framework, benchmark, mode, model


def parse_direct_filename(fname):
    """
    direct_planning_csqa_planning_ollama_deepseek-llm_7b.json

    Returns:
      benchmark, model
    """
    name = fname.replace(".json", "")
    parts = name.split("_")

    # direct, planning, csqa, planning, ollama_deepseek-llm_7b
    benchmark = parts[2]
    model = "_".join(parts[4:])
    return benchmark, model


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_files(noplan_file, plan_file, output_file):
    noplan = load_json(noplan_file)
    plan = load_json(plan_file)

    noplan_q = {q["qid"]: q for q in noplan["questions"]}
    plan_q = {q["qid"]: q for q in plan["questions"]}

    diffs = []
    for qid, q_plan in plan_q.items():
        q_noplan = noplan_q.get(qid)
        if q_noplan is None:
            continue

        # ignore failed
        if q_plan["pred"] == "FAILED" or q_noplan["pred"] == "FAILED":
            continue

        if q_plan["correct"] != q_noplan["correct"]:
            diffs.append({
                "qid": qid,
                "question": q_plan["question"],
                "gold": q_plan["gold"],
                "noplan_pred": q_noplan["pred"],
                "noplan_correct": q_noplan["correct"],
                "plan_pred": q_plan["pred"],
                "plan_correct": q_plan["correct"],
                "llm_response_planning": q_plan["llm_response"],
                "llm_response_noplanning": q_noplan["llm_response"]
            })

    result = {
        "benchmark": plan["benchmark"],
        "model": output_file,
        "differences_count": len(diffs),
        "differences": diffs,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Saved analysis → {output_file}   ({len(diffs)} cases)")


def main():
    # ------------------------------------------------
    # PART 1 — existing planning vs. noplanning
    # ------------------------------------------------
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    groups = defaultdict(dict)

    for fname in files:
        fpath = os.path.join(RESULTS_DIR, fname)
        framework, bench, mode, model = parse_filename(fname)
        key = (framework, bench, model)
        groups[key][mode] = fpath

    for (framework, bench, model), modes in groups.items():
        if "planning" in modes and "noplanning" in modes:
            out_name = f"{framework}_{bench}_{model}_analysis.json"
            output_file = os.path.join(OUTPUT_DIR, out_name)
            print(f"\n► Analyzing {framework} {bench} {model} (planning vs noplanning)")
            compare_files(
                modes["noplanning"],
                modes["planning"],
                output_file
            )
        else:
            print(f"Skipping {framework} {bench} {model}, missing planning/noplanning pair.")

    # ------------------------------------------------
    # PART 2 — new: noplanning vs direct LLM planning
    # ------------------------------------------------
    direct_files = [f for f in os.listdir(DIRECT_DIR) if f.endswith(".json")]

    for fname in direct_files:
        direct_path = os.path.join(DIRECT_DIR, fname)
        benchmark, model = parse_direct_filename(fname)

        # find matching noplanning file inside results/planning
        # pattern: ANYFRAMEWORK_<benchmark>_noplanning_<model>.json
        matched_noplan = None
        for f in files:
            if f"_{benchmark}_noplanning_" in f and f.endswith(f"{model}.json"):
                matched_noplan = os.path.join(RESULTS_DIR, f)
                break

        if matched_noplan is None:
            print(f"❌ No matching noplanning file for {fname}")
            continue

        out_name = f"direct_llm_{benchmark}_{model}_analysis.json"
        output_file = os.path.join(OUTPUT_DIR, out_name)

        print(f"\n► Analyzing direct LLM planning for {benchmark} {model}")
        compare_files(
            matched_noplan,
            direct_path,
            output_file
        )


if __name__ == "__main__":
    main()
