"""
MemoryAgentBench Benchmark Loader and Evaluation Pipeline
==========================================================

Evaluates memory-centric agents on MemoryAgentBench
(https://huggingface.co/datasets/ai-hyz/MemoryAgentBench).

This file is designed to be imported and used by external systems
(e.g., CrewAI, RAG, etc.) that call:

    from benchmarks.memory.memory_agent_bench import MemoryAgentBench

Each benchmark split is evaluated on an agent, and results are saved as JSON
in `results/memory/<system_name>_<split>_<timestamp>.json`.

Metrics per category:
  - Accurate Retrieval (AR): Accuracy (semantic match per question)
  - Test-Time Learning (TTL): Recall@5
  - Long-Range Understanding (LRU): Summary consistency (semantic accuracy)
  - Selective Forgetting (SF): F1-Score (from per-Q Precision/Recall)
"""

import os
import json
import time
from collections import defaultdict
from datasets import load_dataset

from benchmarks.memory.metric_eval_gpt import (
    evaluate_exact_matches_with_gpt,
    evaluate_fact_consistency_with_gpt,
    evaluate_correct_counts_with_gpt,
    get_recall_at_5,
)

# ---------------------------------------------------------------------
# 🔹 Subtask → Category Mapping
# ---------------------------------------------------------------------
def infer_subtask(meta):
    ids = meta.get("qa_pair_ids", [])
    if not ids:
        return "Unknown"
    name = ids[0].lower()
    if "ruler_qa1" in name: return "SH-QA"
    if "ruler_qa2" in name: return "MH-QA"
    if "eventqa" in name: return "EventQA"
    if "longmemeval" in name: return "LME(S*)"
    if "recsys_redial" in name: return "MovieRec"
    if any(x in name for x in ["icl_banking77", "icl_clinic150", "icl_nlu", "trec"]): return "MCC"
    if "infbench_sum" in name: return "Summ."
    if "detective_qa" in name: return "DetQA"
    if "factconsolidation_sh" in name: return "FC-SH"
    if "factconsolidation_mh" in name: return "FC-MH"
    return "Unknown"


def category_of(subtask):
    if subtask in ["SH-QA", "MH-QA", "EventQA", "LME(S*)"]:
        return "AR"   # Accurate Retrieval
    if subtask in ["MCC", "MovieRec"]:
        return "TTL"  # Test-Time Learning
    if subtask in ["Summ.", "DetQA"]:
        return "LRU"  # Long-Range Understanding
    if subtask in ["FC-SH", "FC-MH"]:
        return "SF"   # Selective Forgetting
    return "Other"


# ---------------------------------------------------------------------
# 🧠 Core Benchmark Class
# ---------------------------------------------------------------------
class MemoryAgentBench:
    def __init__(self, split="Accurate_Retrieval", n=None):
        self.split = split
        self.load_data(split, n)
        self.results_dir = os.path.join("results", "memory")
        os.makedirs(self.results_dir, exist_ok=True)

    # --------------------------------------------------------------
    def load_data(self, split, n=None):
        ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
        if n:
            ds = ds.select(range(min(n, len(ds))))
        self.sessions = []
        for i, row in enumerate(ds):
            meta = row.get("metadata", {})
            subtask = infer_subtask(meta)
            category = category_of(subtask)
            self.sessions.append({
                "qid": str(i + 1),
                "context": row.get("context", ""),
                "questions": row.get("questions", []),
                "answers": row.get("answers", []),
                "subtask": subtask,
                "category": category,
                "meta": meta,
            })
        print(f"✅ Loaded {len(self.sessions)} sessions from split '{split}'")
        subtasks = sorted({s["subtask"] for s in self.sessions})
        print("Detected subtasks:", subtasks)

    # --------------------------------------------------------------
    def evaluate_agent(self, agent, system_name="unknown_system", verbose=False, max_sessions_per_task=1):
        """
        Runs the agent and computes metrics per-question using GPT-based functions.
        Evaluates up to `max_sessions_per_task` sessions per subtask to reduce GPT calls.

        Args:
            agent: an object with methods reset(), ingest(context), and query(question)
            system_name (str): name of the system being evaluated (e.g., "crewai", "rag")
            verbose (bool): print session-level results and timings
            max_sessions_per_task (int): max sessions to evaluate per subtask (default=1)
        """
        category_scores = defaultdict(list)
        detailed_results = []
        subtask_session_count = defaultdict(int)
        total_start_time = time.time()

        for sess in self.sessions:
            # ✅ Limit evaluation to max_sessions_per_task per subtask
            subtask = sess["subtask"]
            if subtask_session_count[subtask] >= max_sessions_per_task:
                continue
            subtask_session_count[subtask] += 1

            start_time = time.time()
            agent.reset()
            agent.ingest(sess["context"])
            preds = [agent.query(q) for q in sess["questions"]]

            # Normalize format
            answers_pairs = [
                {
                    "system": [pred] if isinstance(pred, str) else pred,
                    "gold": [g] if isinstance(g, str) else g
                }
                for pred, g in zip(preds, sess["answers"])
            ]

            cat = sess["category"]

            # 🔹 Select metric type
            if cat == "AR":
                correctness = evaluate_exact_matches_with_gpt(answers_pairs)
            elif cat == "TTL":
                correctness = get_recall_at_5(answers_pairs)
            elif cat == "LRU":
                correctness = evaluate_fact_consistency_with_gpt(answers_pairs)
            elif cat == "SF":
                correct_counts = evaluate_correct_counts_with_gpt(answers_pairs)
                correctness = []
                for i, pair in enumerate(answers_pairs):
                    sys_len = len(pair["system"])
                    gold_len = len(pair["gold"])
                    c = correct_counts[i]
                    p = c / sys_len if sys_len else 0
                    r = c / gold_len if gold_len else 0
                    f1 = 2 * p * r / (p + r) if (p + r) else 0
                    correctness.append(f1)
            else:
                correctness = [0.0] * len(answers_pairs)

            # 🔸 Aggregate per-session and per-question
            session_avg = sum(correctness) / len(correctness) if correctness else 0.0
            category_scores[cat].append(session_avg)
            duration = time.time() - start_time  # ⏱️ Runtime per session

            if isinstance(correctness, (int, float)):
                correctness = [correctness] * len(answers_pairs)


            for q, pair, score in zip(sess["questions"], answers_pairs, correctness):
                detailed_results.append({
                    "system": system_name,
                    "split": self.split,
                    "session_id": sess["qid"],
                    "category": cat,
                    "subtask": sess["subtask"],
                    "question": q,
                    "system_answer": pair["system"],
                    "gold_answer": pair["gold"],
                    "score": score,
                    "session_time_sec": round(duration, 2)
                })

            if verbose:
                print(f"[{cat}] Subtask: {sess['subtask']} | Session {sess['qid']} → "
                    f"Avg Score: {session_avg:.3f} | ⏱️ {duration:.2f}s")

            # 🔸 Save incremental progress
            partial_path = os.path.join(
                self.results_dir,
                f"{system_name}_{self.split}_partial.json"
            )
            with open(partial_path, "w") as pf:
                json.dump({
                    "system": system_name,
                    "split": self.split,
                    "progress_session": sess["qid"],
                    "category_avg_so_far": {
                        c: sum(v) / len(v) for c, v in category_scores.items()
                    },
                    "results_so_far": detailed_results,
                }, pf, indent=2, ensure_ascii=False)

        # 🔹 Compute averages
        category_avg = {c: sum(v) / len(v) for c, v in category_scores.items()}
        overall = sum(category_avg.values()) / len(category_avg) if category_avg else 0.0
        total_time = time.time() - total_start_time

        # 🔹 Save detailed JSON
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{system_name}_{self.split}_{timestamp}.json"
        save_path = os.path.join(self.results_dir, filename)
        with open(save_path, "w") as f:
            json.dump({
                "system": system_name,
                "split": self.split,
                "category_avg": category_avg,
                "overall": overall,
                "total_runtime_sec": round(total_time, 2),
                "max_sessions_per_task": max_sessions_per_task,
                "results": detailed_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Saved results to: {save_path}")
        print("\n📊 Category Scores:")
        for c, s in category_avg.items():
            print(f"  {c}: {s:.3f}")
        print(f"⭐ Overall Score: {overall:.3f}")
        print(f"⏱️ Total Runtime: {total_time:.2f} sec")

        return {
            "system": system_name,
            "category_avg": category_avg,
            "overall": overall,
            "total_runtime_sec": round(total_time, 2),
            "path": save_path
        }


# ------------------------------------------------------------------
# 🔍 Example CLI test (for sanity only)
# ------------------------------------------------------------------
if __name__ == "__main__":
    class DummyAgent:
        """Simple mock agent for testing pipeline."""
        def reset(self): pass
        def ingest(self, text): pass
        def query(self, question): return "dummy_answer"

    bench = MemoryAgentBench(split="Accurate_Retrieval", n=2)
    result = bench.evaluate_agent(DummyAgent(), system_name="dummy_system", verbose=True)
    print("\n📈 Final:", result)
