# To test this file, run: python -m benchmarks.memory.memory_agent_bench


"""
MemoryAgentBench Benchmark Loader and Evaluation Pipeline
==========================================================

This module provides a unified interface to load, inspect, and evaluate
multi-agent frameworks on the **MemoryAgentBench** benchmark
(https://huggingface.co/datasets/ai-hyz/MemoryAgentBench).

Purpose
-------
MemoryAgentBench is a large-scale evaluation suite designed to test
the *memory abilities* of LLM-based agents across four key competencies:

    1. Accurate Retrieval (AR)
    2. Test-Time Learning (TTL)
    3. Long-Range Understanding (LRU)
    4. Selective Forgetting / Conflict Resolution (SF)

Each competency contains several fine-grained subtasks such as:
    - SH-QA, MH-QA, LME(S*), EventQA, Bench-QA (AR)
    - MCC, Recommendation (TTL)
    - Summarization, Detective-QA (LRU)
    - FactConsolidation-SH, FactConsolidation-MH (SF)

This file implements:
---------------------
• A `MemoryAgentBench` class that:
    - Loads each dataset split (AR / TTL / LRU / SF) from Hugging Face.
    - Expands multi-question sessions into evaluable Question objects.
    - Automatically infers the fine-grained subtask and category
      from metadata (using `qa_pair_ids` prefixes such as
      `ruler_qa1_`, `recsys_redial_`, `infbench_sum_`, etc.).
    - Provides a standardized `evaluate_agent()` method
      to test any agent framework (CrewAI, OpenAI Agent SDK, etc.)
      using the same data and scoring logic as in the original paper
      *"Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions"*
      (arXiv:2507.05257).

Evaluation Protocol
-------------------
For each example, the benchmark follows an **incremental multi-turn** setup:
    1. The agent ingests the long context (dialogue or document).
    2. Multiple related questions are asked sequentially
       without resetting memory between them.
    3. The benchmark compares each predicted answer
       with the corresponding gold answer.
    4. Scores are aggregated per subtask, per competency, and overall.

Developers can plug in any agent implementing the following interface:

    class AgentAdapter:
        def reset(self): ...
        def ingest(self, text: str): ...
        def query(self, question: str) -> str: ...
        def is_equiv(self, gold: str, pred: str) -> bool: ...

Usage Example
-------------
>>> from benchmarks.memory.memory_agent_bench import MemoryAgentBench, DummyAgent
>>> bench = MemoryAgentBench(split="Accurate_Retrieval", n=2)
>>> agent = DummyAgent()
>>> results = bench.evaluate_agent(agent, verbose=True)
>>> print(results["task_scores"], results["category_avg"], results["overall"])

File Layout
-----------
- `infer_subtask()` and `category_of()` map dataset metadata to subtasks.
- `MemoryAgentBench` handles dataset loading and evaluation logic.
- `DummyAgent` provides a minimal example agent for smoke testing.
- The `__main__` section runs quick tests across all four splits.

"""

import json, re
from datasets import load_dataset
from collections import defaultdict
from benchmarks.base import Benchmark, Question


def infer_subtask(meta):
    ids = meta.get("qa_pair_ids", [])
    if not ids:
        return "Unknown"
    name = ids[0].lower()
    # Accurate Retrieval
    if "ruler_qa1" in name: return "SH-QA"
    if "ruler_qa2" in name: return "MH-QA"
    if "eventqa" in name: return "EventQA"
    if "longmemeval" in name: return "LME(S*)"
    # Test-Time Learning
    if "recsys_redial" in name: return "Recom."
    if "icl_banking77" in name or "icl_clinic150" in name: return "MCC"
    # Long-Range Understanding
    if "infbench_sum" in name: return "Summ."
    # Conflict Resolution / Selective Forgetting
    if "factconsolidation_sh" in name: return "FC-SH"
    if "factconsolidation_mh" in name: return "FC-MH"
    return "Unknown"

def category_of(subtask):
    if subtask in ["SH-QA", "MH-QA", "EventQA", "LME(S*)"]:
        return "AR"
    if subtask in ["MCC", "Recom."]:
        return "TTL"
    if subtask in ["Summ."]:
        return "LRU"
    if subtask in ["FC-SH", "FC-MH"]:
        return "SF"
    return "Other"

class MemoryAgentBench(Benchmark):
    def __init__(self, split="Accurate_Retrieval", n=None):
        super().__init__("MemoryAgentBench")
        self.load_data(split, n)

    def load_data(self, split: str, n=None):
        ds = load_dataset("ai-hyz/MemoryAgentBench", split=split)
        if n:
            ds = ds.select(range(min(n, len(ds))))
        self.sessions = []  # each session holds context + list of Q&A
        for i, row in enumerate(ds):
            context = row.get("context", "")
            questions = row.get("questions", [])
            answers = row.get("answers", [])
            meta = row.get("metadata", {})
            subtask = infer_subtask(meta)
            category = category_of(subtask)
            session = {
                "qid": f"{i+1}",
                "context": context.strip(),
                "questions": questions,
                "answers": answers,
                "subtask": subtask,
                "category": category,
                "meta": meta
            }
            self.sessions.append(session)
        print(f"✅ Loaded {len(self.sessions)} sessions from split '{split}'")
        subtasks = sorted({s["subtask"] for s in self.sessions})
        print("Detected subtasks:", subtasks)

    def evaluate_agent(self, agent, verbose=False):
        """
        Evaluate the given agent (implements our adapter interface:
         agent.reset(), agent.ingest(text), agent.query(question) -> answer)
        Returns per-session results and aggregates.
        """
        results = []
        for sess in self.sessions:
            agent.reset()
            # feed context
            agent.ingest(sess["context"])
            # now ask each question
            Qs = sess["questions"]
            As = sess["answers"]
            for qi, q in enumerate(Qs):
                pred = agent.query(q)
                gold = As[qi]
                correct = agent.is_equiv(gold, pred)
                results.append({
                    "session_id": sess["qid"],
                    "subtask": sess["subtask"],
                    "category": sess["category"],
                    "question_id": f"{sess['qid']}_{qi+1}",
                    "question": q,
                    "gold": gold,
                    "pred": pred,
                    "correct": correct
                })
                if verbose:
                    print(f"[{sess['qid']}_{qi+1}] {sess['subtask']} | Q: {q[:80]}... | Pred: {pred} | Gold: {gold} | Correct: {correct}")
        # compute aggregates
        task_stats = defaultdict(list)
        for r in results:
            task_stats[r["subtask"]].append(r["correct"])
        task_scores = {t: sum(v)/len(v)*100.0 for t, v in task_stats.items()}
        cat_map = defaultdict(list)
        for t, score in task_scores.items():
            cat = category_of(t)
            cat_map[cat].append(score)
        cat_avg = {cat: sum(v)/len(v) for cat, v in cat_map.items()}
        overall = sum(cat_avg.values()) / len(cat_avg) if cat_avg else 0.0
        return {
            "task_scores": task_scores,
            "category_avg": cat_avg,
            "overall": overall,
            "detailed": results
        }

    def normalize(self, text: str) -> str:
        return text.strip().lower()

    def is_equiv(self, gold: str, pred: str) -> bool:
        return self.normalize(pred) == self.normalize(gold)


# ------------------------------------------------------------------
# 🔧 Simple dummy agent + main for quick testing
# ------------------------------------------------------------------
class DummyAgent:
    """
    Minimal agent adapter for testing the benchmark pipeline.

    - reset(): clears any internal state (we don't really use it here)
    - ingest(text): receives context (we just store it)
    - query(question): returns a fixed string
    - is_equiv(gold, pred): simple string equality after stripping
    """

    def __init__(self):
        self.context = None

    def reset(self):
        self.context = None

    def ingest(self, text: str):
        # In a real agent this would populate memory / context.
        self.context = text

    def query(self, question: str) -> str:
        # For now just return a constant or echo part of the question.
        return "dummy_answer"

    def is_equiv(self, gold, pred) -> bool:
        # Very simple equivalence check for testing.
        return str(gold).strip().lower() == str(pred).strip().lower()


if __name__ == "__main__":
    # Quick smoke test across all four splits with the DummyAgent
    splits = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]

    agent = DummyAgent()

    for split in splits:
        print("=" * 80)
        print(f"🚀 Testing split: {split}")
        bench = MemoryAgentBench(split=split, n=2)  # load only first 2 sessions for speed

        # Use the bench's evaluate_agent with our dummy agent
        results = bench.evaluate_agent(agent, verbose=True)

        print("\n📊 Task-level scores:")
        for task, score in sorted(results["task_scores"].items()):
            print(f"  {task:10s}: {score:5.2f}%")

        print("\n📂 Category averages:")
        for cat, score in sorted(results["category_avg"].items()):
            print(f"  {cat:3s}: {score:5.2f}%")

        print(f"\n⭐ Overall score for {split}: {results['overall']:.2f}%\n")
