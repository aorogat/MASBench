#!/usr/bin/env python3
"""
correct_gsm8k.py
----------------
Re-evaluates all GSM8K result JSON files using the improved normalization logic.
Updates:

    - question['pred_original'] = raw pred (new field)
    - question['pred'] = normalized prediction (overwrite)
    - question['correct']
    - metrics['correct'], metrics['accuracy'], metrics['total']

All other fields remain unchanged.

Usage:
    python -m single_agent.reasoning.correct_gsm8k --dir results/planning
    python -m single_agent.reasoning.correct_gsm8k --dir results/planning_direct
"""

import os
import json
import argparse

from benchmarks.gsm8k import normalize_pred, extract_final_answer


def process_file(path: str):
    """Re-evaluate correctness and update prediction fields."""
    print(f"\n📄 Processing: {path}")

    # Load JSON
    with open(path, "r") as f:
        data = json.load(f)

    if "questions" not in data:
        print("❌ Skipped (no questions field).")
        return

    questions = data["questions"]

    new_correct = 0

    for q in questions:
        gold_raw = q.get("gold", "")
        pred_raw = q.get("pred", "")
        llm_response = q.get("llm_response", "")

        # Some files have: "llm_response": null
        if llm_response is None:
            llm_response = ""


        # --- Option C behavior ---
        # 1. Save original raw prediction
        q["pred_original"] = pred_raw

        # 2. Normalize prediction
        pred_norm = normalize_pred(llm_response)
        q["pred"] = pred_norm  # overwrite

        # 3. Normalize gold value
        gold_extracted = extract_final_answer(gold_raw)
        gold_norm = normalize_pred(gold_extracted)

        # 4. Evaluate correctness
        is_corr = (gold_norm == pred_norm)
        q["correct"] = is_corr

        if is_corr:
            new_correct += 1

    total = len(questions)
    new_acc = new_correct / total if total > 0 else 0.0

    # --- Update metrics ---
    metrics = data.get("metrics", {})
    metrics["correct"] = new_correct
    metrics["total"] = total
    metrics["accuracy"] = new_acc
    data["metrics"] = metrics

    # --- Save file ---
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✔ Updated accuracy: {new_acc:.3f} ({new_correct}/{total})")
    print("💾 File updated.")


def scan_and_process(root_dir: str):
    """Process all JSON files containing 'gsm8k' in name."""
    print(f"🔍 Scanning folder: {root_dir}")

    for root, _, files in os.walk(root_dir):
        for file in files:
            if "gsm8k" in file.lower() and file.lower().endswith(".json"):
                process_file(os.path.join(root, file))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Root directory to scan for GSM8K JSON result files."
    )
    args = parser.parse_args()

    scan_and_process(args.dir)


if __name__ == "__main__":
    main()
