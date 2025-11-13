"""
Run MemoryAgentBench on All Splits using Agno Memory Agent
==========================================================

This script evaluates an Agno agent with built-in memory
(across SQLite-backed automatic memory) on all splits of
MemoryAgentBench.

To run:
    python -m single_agent.memory.agno_test
"""

import os
import shutil
import time
from dotenv import load_dotenv
from typing import Iterator
from agno.agent import Agent, RunOutput, RunOutputEvent, RunEvent
from agno.db.sqlite import SqliteDb
from agno.models.openai import OpenAIChat
from agno.models.anthropic import Claude
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    agno_llm_model,
    llm_max_tokens,
    llm_temperature,
    storage_directory,
    chunk_max_tokens,
    chunk_overlap,
    splits,
    results_directory,
    verbose,
)

# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY


# ---------------------------------------------------------------------
# 🧠 Agno Agent Builder
# ---------------------------------------------------------------------
def build_agno_agent(db_path: str):
    """Create an Agno agent with automatic memory enabled."""
    db = SqliteDb(db_file=db_path)
    agent = Agent(
        db=db,
        model=OpenAIChat(id=agno_llm_model),
        enable_user_memories=True,       # automatic memory management
        add_memories_to_context=True,    # include past memories in context
    )
    return agent


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
# ---------------------------------------------------------------------
class AgnoMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() for MemoryAgentBench.
    Each run creates its own isolated SQLite DB.
    """

    _session_counter = 0

    def __init__(self):
        os.makedirs(storage_directory, exist_ok=True)
        AgnoMemoryAgent._session_counter += 1
        self.db_path = os.path.join(
            storage_directory, f"agno_memory_session_{AgnoMemoryAgent._session_counter}.db"
        )
        self.user_id = "benchmark_user"
        self.agent = build_agno_agent(self.db_path)
        print(f"🧠 Agno session initialized: {self.db_path}")

    # -------------------------------------------------------------
    def reset(self):
        """Reset memory by creating a new SQLite DB file."""
        try:
            AgnoMemoryAgent._session_counter += 1
            self.db_path = os.path.join(
                storage_directory, f"agno_memory_session_{AgnoMemoryAgent._session_counter}.db"
            )
            self.agent = build_agno_agent(self.db_path)
            print(f"🧹 New Agno session started: {self.db_path}")
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        """Feed benchmark context into Agno’s automatic memory."""
        print("🧠 Ingesting context...")
        chunks = chunk_text(context, max_tokens=chunk_max_tokens, overlap=chunk_overlap)
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
    overall_summary = {}

    try:
        for split in splits:
            print("\n" + "=" * 80)
            print(f"🧩 Running Agno Memory Agent on Split: {split}")
            print("=" * 80)

            bench = MemoryAgentBench(split=split)
            agent = AgnoMemoryAgent()

            result = bench.evaluate_agent(
                agent,
                system_name= "Agno",
                verbose=verbose,
            )
            overall_summary[split] = result["overall"]

        summarize_results("Agno", overall_summary)

    finally:
        # -----------------------------------------------------------------
        # 🧹 Clean up memory database folder after benchmark completion
        # -----------------------------------------------------------------
        try:
            if os.path.exists(storage_directory):
                print(f"\n🗑️  Cleaning up memory DB folder: {storage_directory}")
                shutil.rmtree(storage_directory)
                print("✅ Database folder deleted successfully.")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
