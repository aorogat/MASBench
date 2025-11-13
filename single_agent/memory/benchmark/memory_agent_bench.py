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

from single_agent.memory.benchmark.metric_eval_gpt import (
    evaluate_exact_match,
    evaluate_summary_match,
    evaluate_recall_at_5,
)

from single_agent.memory.config import (
    max_sessions_per_subtask,
    eval_llm_model,
    eval_small_batch_size,
    eval_summary_batch_size,
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
    def __init__(self, split="Accurate_Retrieval"):
        self.split = split
        self.load_data(split)
        self.results_dir = os.path.join("results", "memory")
        os.makedirs(self.results_dir, exist_ok=True)

    # --------------------------------------------------------------
    def load_data(self, split):
        ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
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
    def evaluate_agent(self, agent, system_name="unknown_system", verbose=False):
        """
        Runs the agent and computes metrics per-question using GPT-based functions.
        Uses max_sessions_per_subtask from config.
        """

        category_scores = defaultdict(list)
        detailed_results = []
        subtask_session_count = defaultdict(int)
        total_start_time = time.time()

        for sess in self.sessions:

            subtask = sess["subtask"]

            # ---------------------------------------------------------
            # ✅ Limit evaluation PER SUBTASK (not per split)
            # ---------------------------------------------------------
            if subtask_session_count[subtask] >= max_sessions_per_subtask:
                continue

            subtask_session_count[subtask] += 1

            start_time = time.time()
            agent.reset()
            agent.ingest(sess["context"])

            preds = []
            for idx, q in enumerate(sess["questions"], start=1):
                print(f"🔍 Querying question {idx}/{len(sess['questions'])} ...")
                preds.append(agent.query(q))


            # Normalize pair format
            answers_pairs = [
                {
                    "system": [pred] if isinstance(pred, str) else pred,
                    "gold": [g] if isinstance(g, str) else g
                }
                for pred, g in zip(preds, sess["answers"])
            ]

            cat = sess["category"]

            # ---------------------------------------------------------
            # 🧮 Run the correct evaluation metric according to category
            # ---------------------------------------------------------
            if cat == "AR":
                correctness = evaluate_exact_match(
                    answers_pairs,
                    model=eval_llm_model,
                    batch_size=eval_small_batch_size
                )

            elif cat == "TTL" and subtask == "MovieRec":
                correctness = evaluate_recall_at_5(
                    answers_pairs,
                    model=eval_llm_model
                )

            elif cat == "TTL":
                correctness = evaluate_exact_match(
                    answers_pairs,
                    model=eval_llm_model,
                    batch_size=eval_small_batch_size
                )

            elif cat == "LRU" and subtask == "Summ.":
                correctness = evaluate_summary_match(
                    answers_pairs,
                    model=eval_llm_model,
                    batch_size=eval_summary_batch_size
                )

            elif cat == "LRU":
                correctness = evaluate_exact_match(
                    answers_pairs,
                    model=eval_llm_model,
                    batch_size=eval_small_batch_size
                )

            # ---------------------------------------------------------
            # SF = fact-level F1 score
            # ---------------------------------------------------------
            elif cat == "SF":
                correctness = evaluate_summary_match(
                    answers_pairs,
                    model=eval_llm_model,
                    batch_size=eval_summary_batch_size
                )

            else:
                correctness = [0.0] * len(answers_pairs)

            # ---------------------------------------------------------
            # 📊 Aggregate per-session
            # ---------------------------------------------------------
            session_avg = sum(correctness) / len(correctness) if correctness else 0.0
            category_scores[cat].append(session_avg)

            duration = time.time() - start_time

            # Save global detailed result
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

            # Save individual session file
            session_file_path = self._save_session_result(
                system_name=system_name,
                sess=sess,
                correctness=[
                    {"system": pair["system"], "gold": pair["gold"], "score": s}
                    for pair, s in zip(answers_pairs, correctness)
                ],
                duration=duration
            )

            if verbose:
                print(f"   💾 Session saved: {session_file_path}")


            if verbose:
                print(f"[{cat}] Subtask: {sess['subtask']} | Session {sess['qid']} "
                    f"→ Avg Score: {session_avg:.3f} | ⏱️ {duration:.2f}s")

        # ---------------------------------------------------------
        # 📈 Final summary
        # ---------------------------------------------------------
        category_avg = {c: sum(v) / len(v) for c, v in category_scores.items()}
        overall = sum(category_avg.values()) / len(category_avg) if category_avg else 0.0
        total_time = time.time() - total_start_time

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(
            self.results_dir,
            f"{system_name}_{self.split}_{timestamp}.json"
        )

        with open(save_path, "w") as f:
            json.dump({
                "system": system_name,
                "split": self.split,
                "category_avg": category_avg,
                "overall": overall,
                "total_runtime_sec": round(total_time, 2),
                "max_sessions_per_subtask": max_sessions_per_subtask,
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



    def _save_session_result(self, system_name, sess, correctness, duration):
        """
        Saves an individual session result inside:
        results/memory/<system_name>/<subtask>/session_<id>.json
        """

        subfolder = os.path.join(
            self.results_dir,
            system_name,
            sess["subtask"]
        )
        os.makedirs(subfolder, exist_ok=True)

        path = os.path.join(subfolder, f"session_{sess['qid']}.json")

        # Per-question entries
        per_q = []
        for q, sys_ans, gold_ans, score in zip(
            sess["questions"], 
            [c["system"][0] if isinstance(c["system"], list) else c["system"] for c in correctness],
            [c["gold"][0] if isinstance(c["gold"], list) else c["gold"] for c in correctness],
            [c["score"] if "score" in c else c for c in correctness]
        ):
            per_q.append({
                "question": q,
                "system_answer": sys_ans,
                "gold_answer": gold_ans,
                "score": score
            })

        data = {
            "system": system_name,
            "split": self.split,
            "session_id": sess["qid"],
            "subtask": sess["subtask"],
            "category": sess["category"],
            "runtime_sec": round(duration, 2),
            "questions": per_q
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return path


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
