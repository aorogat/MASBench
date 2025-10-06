import os
import json
from collections import defaultdict
from .base import Benchmark, Question
from llms.local_llm import LocalOllamaLLM
from llms.remote_llm import OpenAILLM
from single_agent.reasoning.config import CONFIG

"""
This module loads and benchmarks the MATH dataset.

How to test it:
----------------------------------------
1. Activate your virtual environment.
   Example: (mafenv)

2. From the project root (where `benchmarks/` is a package):
   $ python -m benchmarks.math

The -m flag allows Python to treat `benchmarks/` as a package,
so relative imports like `from .base import Benchmark` work correctly.

Expected output:
----------------------------------------
✅ Loaded 500 questions (target = 500)
📊 Folder-level distribution:
   algebra              → 120 (24.0%)
   geometry             → 135 (27.0%)
   ...

✅ Loaded MATH benchmark
=== Detailed Results ===
Qalgebra-1.json: ...
   Gold: ...
   Result: ✅ Correct
=== MATH ===
Accuracy: ...
"""

def normalize_answer(sol: str) -> str:
    """Extract final boxed answer from LaTeX-style MATH solution strings."""
    text = str(sol).strip()
    start = text.rfind(r"\boxed{")
    if start != -1:
        i = start + len(r"\boxed{")
        depth = 1
        result = []
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            result.append(text[i])
            i += 1
        return "".join(result).strip()

    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else text


class MATHBenchmark(Benchmark):
    """
    Benchmark loader for the MATH dataset.

    Key features:
    - Reads all `.json` questions under the given root directory.
    - Each question includes "problem", "level", "solution".
    - If `n` is provided, it selects exactly `n` questions
      proportionally from each folder, keeping level ratios the same.
    - Deterministic (no randomness) — uses first questions only.
    """

    def __init__(self, root="data/MATH/test", split="test", n=None, use_remote=True):
        self.root = root
        model_name = CONFIG.get("judge_llm", "gpt-4o-mini" if use_remote else "gpt-oss:20b")
        self.llm = OpenAILLM(model_name) if use_remote else LocalOllamaLLM(model_name)
        super().__init__("math")
        self.load_data(split, n)

    def load_data(self, split: str, n=None) -> None:
        """
        Load all JSON problems and select a proportional subset if n is set.
        """
        folder_levels = defaultdict(lambda: defaultdict(list))
        total_questions = 0

        # --- Step 1: Load JSONs grouped by folder + level
        for subdir, _, fs in os.walk(self.root):
            json_files = [f for f in fs if f.endswith(".json")]
            if not json_files:
                continue

            for f in sorted(json_files):  # deterministic order
                path = os.path.join(subdir, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        prob = json.load(file)
                    level = prob.get("level", "Unknown")
                    gold = normalize_answer(prob.get("solution", ""))
                    q = Question(
                        qid=f"{os.path.basename(subdir)}-{f}",
                        question=prob.get("problem", ""),
                        gold=gold,
                    )
                    q.meta = {"folder": os.path.basename(subdir), "level": level}
                    folder_levels[subdir][level].append(q)
                    total_questions += 1
                except Exception as e:
                    print(f"⚠️ Failed to load {path}: {e}")

        # --- Step 2: If n is None or > total, use all
        if not n or n >= total_questions:
            self.questions = [q for fl in folder_levels.values() for lvl in fl.values() for q in lvl]
            print(f"Loaded all {len(self.questions)} questions.")
            return

        # --- Step 3: Proportional selection across folders
        folder_counts = {f: sum(len(v) for v in lv.values()) for f, lv in folder_levels.items()}
        total_folders = sum(folder_counts.values())
        folder_targets = {
            f: max(1, round(n * (folder_counts[f] / total_folders))) for f in folder_counts
        }

        selected_questions = []
        for folder, level_dict in folder_levels.items():
            folder_total = folder_counts[folder]
            folder_target = folder_targets[folder]
            folder_selected = []

            # Maintain level ratios within folder
            level_ratios = {
                lvl: len(qs) / folder_total for lvl, qs in level_dict.items()
            }

            for lvl, lvl_qs in level_dict.items():
                lvl_target = max(1, round(folder_target * level_ratios[lvl]))
                folder_selected.extend(lvl_qs[:lvl_target])  # deterministic

            selected_questions.extend(folder_selected)

        # --- Step 4: Fix rounding mismatch (truncate/extend deterministically)
        if len(selected_questions) > n:
            selected_questions = selected_questions[:n]
        elif len(selected_questions) < n:
            all_questions = [q for fl in folder_levels.values() for lvl in fl.values() for q in lvl]
            remaining = [q for q in all_questions if q not in selected_questions]
            selected_questions.extend(remaining[: (n - len(selected_questions))])

        self.questions = selected_questions

        # --- Print distribution summary
        print(f"✅ Loaded {len(self.questions)} questions (target = {n})")
        print("📊 Folder-level distribution:")
        for folder, count in folder_counts.items():
            target = folder_targets[folder]
            pct = (target / n) * 100
            print(f"   {os.path.basename(folder):<25} → {target:>4} ({pct:.1f}%)")

    def normalize(self, text: str) -> str:
        return str(text).strip()

    def is_equiv(self, gold: str, pred: str) -> bool:
        """Judge if the predicted answer matches the gold mathematically."""
        if not gold or not pred:
            return False

        gold_norm = self.normalize(gold)
        pred_norm = self.normalize(pred)
        if gold_norm == pred_norm:
            return True

        prompt = (
            f"Gold answer: {gold_norm}\n"
            f"Predicted answer: {pred_norm}\n\n"
            "Are these mathematically equivalent?\nReply with exactly one word: YES or NO."
        )

        try:
            resp = self.llm.generate(prompt).strip().lower()
            if resp in {"yes", "y", "equivalent", "true"}:
                return True
            if resp in {"no", "n", "different", "false"}:
                return False
            return "yes" in resp
        except Exception as e:
            print("⚠️ LLM equivalence check failed:", e)
            return False


# ==========================================================
# ✅ Self-test: run this file directly via `python -m`
# ==========================================================
if __name__ == "__main__":
    bench = MATHBenchmark(root="data/MATH/test", n=500, use_remote=True)
    print("✅ Loaded MATH benchmark")

    print("\n=== Detailed Results ===")
    for q in bench.questions[:3]:
        # For testing, set prediction equal to gold answer
        bench.set_pred(q, q.gold)
        q.time_used = 0.0          # avoid NoneType errors
        q.tokens_out = 0           # avoid NoneType errors

        gold_preview = (q.gold[:320] + "...") if len(q.gold) > 120 else q.gold
        status = "✅ Correct" if q.correct else "❌ Incorrect"
        print(f"\nQ{q.qid}: {q.question[:200].replace('\n',' ')}...")
        print(f"   Gold: {gold_preview}")
        print(f"   Result: {status}")

    # Safe summary print (ensure attributes exist)
    for q in bench.questions:
        if getattr(q, "time_used", None) is None:
            q.time_used = 0.0
        if getattr(q, "tokens_out", None) is None:
            q.tokens_out = 0

    bench.print_summary()
