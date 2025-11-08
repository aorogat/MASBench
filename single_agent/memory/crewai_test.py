"""
CrewAI Memory Evaluation on MemoryAgentBench
--------------------------------------------
Evaluates CrewAI's built-in memory across all MemoryAgentBench splits.

Each MemoryAgentBench session contains a conversational history and
evaluation questions. This script:
  • Replays prior user–system dialogues as memory context
  • Evaluates CrewAI’s recall, adaptation, and reasoning
  • Saves incremental results after every split

Results are stored under: results/memory/crewai_results_<timestamp>.json
"""

import os
import re
import json
import time
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from single_agent.reasoning.config import CONFIG
from crewai import Agent, Task, Crew, Process, LLM
from benchmarks.memory.memory_agent_bench import MemoryAgentBench


# --------------------------------------------------------------------
# CrewAI Adapter for MemoryAgentBench
# --------------------------------------------------------------------
class CrewAgentAdapter:
    """Adapter wrapping a CrewAI Crew with built-in conversational memory."""

    def __init__(self):
        self._build_crew()

    def _build_crew(self):
        """Initialize a Crew with memory enabled (fresh state)."""
        llm_model = CONFIG.get("llm", "gpt-4o-mini")
        print(f"🔧 Initializing CrewAI conversational agent with LLM = {llm_model}")

        self.agent = Agent(
            role="Conversational QA Assistant",
            goal="Recall previous dialogues and answer follow-up questions accurately.",
            backstory="An assistant trained to maintain long-term conversational memory.",
            llm=LLM(model=llm_model)
        )

        self.task = Task(
            description="Engage in conversation and recall prior facts to answer correctly.",
            agent=self.agent,
            expected_output="A correct, context-aware answer."
        )

        self.crew = Crew(
            agents=[self.agent],
            tasks=[self.task],
            process=Process.sequential,
            memory=True,       # ✅ enables CrewAI’s built-in memory subsystem
            verbose=False
        )

    def reset(self):
        """Rebuild the Crew to clear memory between sessions."""
        self._build_crew()

    def ingest(self, text: str, batch_size: int = 50, max_pairs: int = 500):
        """
        Feed prior user–system dialogue pairs to CrewAI as memory batches.
        Combines several turns per request to reduce API calls.
        Truncates long histories to the most recent 'max_pairs' pairs.
        """
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        paired = []
        user, system = None, None
        for ln in lines:
            if ln.lower().startswith("user:"):
                user = ln[5:].strip()
            elif ln.lower().startswith("system:"):
                system = ln[7:].strip()
            if user and system:
                paired.append((user, system))
                user, system = None, None

        if not paired:
            paired = [(text, "")]

        # Truncate extremely long histories
        if len(paired) > max_pairs:
            print(f"⚠️ Truncating dialogue history from {len(paired)} to {max_pairs} pairs (latest kept).")
            paired = paired[-max_pairs:]

        print(f"💬 Replaying {len(paired)} dialogue pairs (batch={batch_size})...")
        for i in range(0, len(paired), batch_size):
            batch = paired[i:i + batch_size]
            chunk = "\n".join([f"user: {u}\nsystem: {s}" for u, s in batch])
            prompt = f"Remember the following conversation and keep it in your memory:\n{chunk}"
            try:
                _ = self.crew.kickoff(inputs={"question": prompt})
                if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(paired):
                    print(f"   → injected up to {min(i+batch_size, len(paired))}/{len(paired)}", flush=True)
            except Exception as e:
                print(f"[Ingest Error @batch {i//batch_size}] {e}")

        print("✅ Dialogue ingestion complete.\n")

        def query(self, question: str) -> str:
            """Ask the benchmark question; CrewAI recalls from its memory."""
            print(f"❓ Asking: {question[:90]}...")
            try:
                result = self.crew.kickoff(inputs={"question": question})

                # --- handle CrewAI output types safely ---
                if result is None:
                    return "[EMPTY RESULT]"
                if isinstance(result, list):
                    # CrewAI sometimes returns list of TaskOutputs
                    text_blocks = [str(r) for r in result if r]
                    return "\n".join(text_blocks).strip()
                if hasattr(result, "tasks_output"):
                    first_key = next(iter(result.tasks_output.keys()))
                    return str(result.tasks_output[first_key]).strip()
                if isinstance(result, dict):
                    return str(list(result.values())[0]).strip()
                return str(result).strip()

            except Exception as e:
                return f"[ERROR] {e}"


    @staticmethod
    def is_equiv(gold: str, pred: str) -> bool:
        """
        More tolerant equivalence check.
        Matches if either normalized string contains the other.
        """
        g = re.sub(r'[^a-z0-9]', '', str(gold).lower())
        p = re.sub(r'[^a-z0-9]', '', str(pred).lower())
        return bool(g) and (g in p or p in g)


# --------------------------------------------------------------------
# Incremental Save Utility
# --------------------------------------------------------------------
def save_partial_results(out_path, results_obj):
    """Write results to disk incrementally with overwrite-safe flush."""
    tmp_path = out_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(results_obj, f, indent=2)
    os.replace(tmp_path, out_path)


# --------------------------------------------------------------------
# Evaluation Runner
# --------------------------------------------------------------------
def run_evaluation():
    load_dotenv()
    llm_model = CONFIG.get("llm", "gpt-4o-mini")
    plan_tag = "planning" if CONFIG.get("planning") else "noplanning"

    out_dir = Path("results/memory")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"crewai_results_{llm_model}_{plan_tag}_{timestamp}.json"

    print("\n================ MemoryAgentBench Evaluation ================")
    print(f"📅 Timestamp   : {timestamp}")
    print(f"🤖 Model       : {llm_model}")
    print(f"🧠 Memory Mode : Enabled (CrewAI built-in)")
    print(f"💾 Output File : {out_path}")
    print("==============================================================\n")

    agent = CrewAgentAdapter()
    all_results = {"llm_model": llm_model, "timestamp": timestamp, "splits": {}}

    splits = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]

    total_splits = len(splits)
    global_start = time.time()

    for split_idx, split in enumerate(splits, 1):
        print(f"\n{'=' * 80}")
        print(f"🚀 [{split_idx}/{total_splits}] Evaluating CrewAI memory on: {split}")
        bench = MemoryAgentBench(split=split, n=2)  # change n=None for full dataset

        n_sessions = len(bench.sessions)
        print(f"📚 Loaded {n_sessions} sessions. Running evaluation...")

        split_start = time.time()
        pbar = tqdm(total=n_sessions, desc=f"⏳ {split}", ncols=80)

        # Run evaluation manually to print per-question output
        session_results = []
        for sess_idx, session in enumerate(bench.sessions, 1):
            print(f"\n🗂 Session {sess_idx}/{len(bench.sessions)} — Subtask: {session.get('subtask', 'N/A')}")
            agent.reset()  # reset CrewAI memory per session

            # 1️⃣ Re-inject prior dialogue as memory
            dialogue = session.get("dialogue_history", "")
            if dialogue.strip():
                agent.ingest(dialogue)

            # 2️⃣ Iterate through the evaluation questions
            questions = session.get("questions", [])
            gold_answers = session.get("answers", [])
            subtask_results = []

            for q_idx, (question, gold) in enumerate(zip(questions, gold_answers), 1):
                print(f"\n❓ Q{q_idx}: {question}")
                gold_text = gold if isinstance(gold, str) else ", ".join(gold)
                print(f"   🟩 Gold answer: {gold_text}")
                pred = agent.query(question)
                print(f"   🟦 CrewAI answer: {pred}")

                correct = agent.is_equiv(gold_text, pred)
                print(f"   ✅ Correct: {correct}\n")
                subtask_results.append({"question": question, "gold": gold_text, "pred": pred, "correct": correct})

            session_results.append(subtask_results)
            pbar.update(1)

        # Aggregate using benchmark’s scorer for consistency
        results = bench.evaluate_agent(agent, verbose=False)

        pbar.update(n_sessions)
        pbar.close()

        split_end = time.time()
        duration = split_end - split_start

        all_results["splits"][split] = {
            "duration_sec": round(duration, 2),
            **results
        }

        print(f"✅ {split} done. ⏱ {duration:.2f}s | Overall: {results['overall']:.2f}%")
        print(f"📊 Category averages: {results['category_avg']}")
        save_partial_results(out_path, all_results)

    total_time = time.time() - global_start
    print("\n🧠 CrewAI memory evaluation completed!")
    print(f"🕒 Total runtime: {total_time/60:.1f} minutes")
    print(f"📁 Final results saved to: {out_path}")


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
if __name__ == "__main__":
    run_evaluation()
