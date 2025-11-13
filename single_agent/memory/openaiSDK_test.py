"""
Run MemoryAgentBench on All Splits using OpenAI SDK Memory Agent
===============================================================

This script evaluates an OpenAI Agents SDK–based agent with built-in
session memory across all splits of MemoryAgentBench:

    - Accurate_Retrieval
    - Test_Time_Learning
    - Long_Range_Understanding
    - Selective_Forgetting

Results are saved automatically in `results/memory/`.
"""

import os
import time
import json
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession
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
# 🗂 Trimming SQLite Session
# ---------------------------------------------------------------------
class TrimmingSQLiteSession(SQLiteSession):
    """Keeps only the most recent N items to prevent overflow."""

    def __init__(self, session_id, db_path, keep_last=30):
        super().__init__(session_id, db_path)
        self.keep_last = keep_last

    async def add_items(self, items):
        await super().add_items(items)
        all_items = await super().get_items()
        if len(all_items) > self.keep_last:
            trimmed = all_items[-self.keep_last:]
            await self.clear_session()
            await super().add_items(trimmed)


# ---------------------------------------------------------------------
# 🧠 OpenAI Agent Builder
# ---------------------------------------------------------------------
def build_openai_agent():
    """Create an OpenAI Agents SDK agent configured for concise factual QA."""
    return Agent(
        name="MemoryQA",
        instructions=(
            "You are a helpful assistant that answers questions accurately "
            "using remembered context."
        ),
        model="gpt-4o-mini",
    )


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
# ---------------------------------------------------------------------
class OpenAIMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() so MemoryAgentBench
    can evaluate the OpenAI Agents SDK with persistent memory.
    """

    def __init__(self):
        self.agent = build_openai_agent()
        self.session_path = "/shared_mnt/openai_memory.db"
        self.session_id = "memory_bench_session"
        self.session = SQLiteSession(self.session_id, self.session_path)

    # -------------------------------------------------------------
    def reset(self):
        """Reset conversation memory between benchmark sessions."""
        try:
            self.session = SQLiteSession(self.session_id, self.session_path)
            asyncio.run(self.session.clear_session())
            print("🧹 Session memory cleared.")
        except Exception as e:
            print(f"⚠️ Session reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str, max_tokens: int = 1000, overlap: int = 50):
        """Feed benchmark context into the agent’s memory."""
        print("🧠 Ingesting context...")
        chunks = chunk_text(context, max_tokens=max_tokens, overlap=overlap)

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            prompt = (
                f"Memorize the following information for future questions:\n\n"
                f"{chunk}\n\n"
                "Acknowledge briefly that it was stored, without repeating it."
            )
            try:
                asyncio.run(Runner.run(self.agent, prompt, session=self.session))
                self.session = TrimmingSQLiteSession(
                    self.session_id, self.session_path, keep_last=50
                )
            except Exception as e:
                print(f"❌ Error storing chunk {i}: {e}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        """Ask a question using the remembered context."""
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")
        try:
            result = asyncio.run(
                Runner.run(self.agent, question, session=self.session)
            )
            self.session = TrimmingSQLiteSession(
                self.session_id, self.session_path, keep_last=50
            )
            answer = result.final_output if hasattr(result, "final_output") else str(result)
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
        print(f"🧩 Running OpenAI SDK Memory Agent on Split: {split}")
        print("=" * 80)

        bench = MemoryAgentBench(split=split, n=None)
        agent = OpenAIMemoryAgent()

        result = bench.evaluate_agent(
            agent,
            system_name="openai_sdk_memory_agent",
            verbose=True,
        )
        overall_summary[split] = result["overall"]

    summarize_results("openai_sdk_memory_agent", overall_summary)


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
