"""
Run MemoryAgentBench on All Splits using OpenAI SDK Memory Agent
===============================================================

This script evaluates an OpenAI Agents SDK–based agent with built-in
session memory across all splits of MemoryAgentBench.

Each benchmark split runs in an isolated SQLite session database
so the full memory is preserved for inspection and analysis.

To run:
    python -m single_agent.memory.openaiSDK_test
"""

import os
import time
import json
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, SQLiteSession
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    openai_sdk_llm_model,
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
        model=openai_sdk_llm_model,
    )


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
# ---------------------------------------------------------------------
class OpenAIMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() so MemoryAgentBench
    can evaluate the OpenAI Agents SDK with persistent memory.
    Each session is stored fully (no trimming).
    """

    _session_counter = 0  # ensures unique DB file per session

    def __init__(self):
        self.agent = build_openai_agent()
        os.makedirs(storage_directory, exist_ok=True)

        # Create unique DB file for this run
        OpenAIMemoryAgent._session_counter += 1
        self.session_id = f"memory_bench_session_{OpenAIMemoryAgent._session_counter}"
        self.session_path = os.path.join(
            storage_directory, f"{self.session_id}.db"
        )
        self.session = SQLiteSession(self.session_id, self.session_path)
        print(f"🧠 Session initialized: {self.session_path}")

    # -------------------------------------------------------------
    def reset(self):
        """Reset conversation memory by starting a new SQLite file."""
        try:
            OpenAIMemoryAgent._session_counter += 1
            self.session_id = f"memory_bench_session_{OpenAIMemoryAgent._session_counter}"
            self.session_path = os.path.join(
                storage_directory, f"{self.session_id}.db"
            )
            self.session = SQLiteSession(self.session_id, self.session_path)
            print(f"🧹 New session started: {self.session_path}")
        except Exception as e:
            print(f"⚠️ Session reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        """Feed benchmark context into the agent’s memory."""
        print("🧠 Ingesting context...")
        chunks = chunk_text(context, max_tokens=chunk_max_tokens, overlap=chunk_overlap)

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            prompt = (
                f"Memorize the following information for future questions:\n\n"
                f"{chunk}\n\n"
                "Acknowledge briefly that it was stored, without repeating it."
            )
            try:
                asyncio.run(Runner.run(self.agent, prompt, session=self.session))
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
            answer = (
                result.final_output if hasattr(result, "final_output") else str(result)
            )
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
            print(f"🧩 Running OpenAI SDK Memory Agent on Split: {split}")
            print("=" * 80)

            bench = MemoryAgentBench(split=split)
            agent = OpenAIMemoryAgent()

            result = bench.evaluate_agent(
                agent,
                system_name="Openai_SDK",
                verbose=verbose,
            )
            overall_summary[split] = result["overall"]

        summarize_results("Openai_SDK", overall_summary)

    finally:
        # -----------------------------------------------------------------
        # 🧹 Clean up database folder after all benchmarks complete
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
