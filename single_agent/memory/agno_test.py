"""
Run MemoryAgentBench on All Splits using Agno Memory Agent
==========================================================

This script evaluates an Agno agent with built-in memory
(across SQLite-backed automatic memory) on all splits of
MemoryAgentBench:

    - Accurate_Retrieval
    - Test_Time_Learning
    - Long_Range_Understanding
    - Conflict_Resolution

Results are saved automatically in `results/memory/`.
"""

import os
import time
from dotenv import load_dotenv
from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from typing import Iterator
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response
from benchmarks.memory.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)

# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY


# ---------------------------------------------------------------------
# 🧠 Agno Agent Builder
# ---------------------------------------------------------------------
def build_agno_agent(db_path="/shared_mnt/agno_memory.db"):
    """Create an Agno agent with automatic memory enabled."""
    db = SqliteDb(db_file=db_path)
    agent = Agent(
        db=db,
        model=OpenAIChat(id="gpt-4o-mini"),
        enable_user_memories=True,      # automatic memory management
        add_memories_to_context=True,   # include past memories in context
    )
    return agent


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
# ---------------------------------------------------------------------
class AgnoMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() for MemoryAgentBench.
    """

    def __init__(self):
        self.db_path = "/shared_mnt/agno_memory.db"
        self.user_id = "benchmark_user"
        self.agent = build_agno_agent(self.db_path)

    # -------------------------------------------------------------
    def reset(self):
        """Reset memory by recreating the SQLite database."""
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.agent = build_agno_agent(self.db_path)
            print("🧹 Memory reset (new database created).")
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str, max_tokens: int = 1000, overlap: int = 50):
        """Feed benchmark context into Agno’s automatic memory."""
        print("🧠 Ingesting context...")
        chunks = chunk_text(context, max_tokens=max_tokens, overlap=overlap)
        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            try:
                response: RunOutput = self.agent.run(
                    f"Memorize the following context:\n\n{chunk}",
                    user_id=self.user_id,
                )
                print(f"     ✅ Stored chunk {i} ({len(response.content)} chars).")
            except Exception as e:
                print(f"❌ Error storing chunk {i}: {e}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        """Ask a question using remembered context."""
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")
        try:
            response: RunOutput = self.agent.run(
                f"Answer the question based on remembered facts:\n{question}",
                user_id=self.user_id,
            )
            answer = response.content.strip() if response and response.content else ""
            print(f"🧾 Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return answer
        except Exception as e:
            print(f"❌ Query failed: {e}")
            return ""


# ---------------------------------------------------------------------
# 🚀 Run Benchmark for All Splits
# ---------------------------------------------------------------------
def main():
    splits = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]

    overall_summary = {}

    for split in splits:
        print("\n" + "=" * 80)
        print(f"🧩 Running Agno Memory Agent on Split: {split}")
        print("=" * 80)

        bench = MemoryAgentBench(split=split, n=None)
        agent = AgnoMemoryAgent()

        result = bench.evaluate_agent(
            agent,
            system_name="agno_memory_agent",
            verbose=True,
        )
        overall_summary[split] = result["overall"]

    summarize_results("agno_memory_agent", overall_summary)


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
