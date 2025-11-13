"""
Run MemoryAgentBench on All Splits using CrewAI Memory Agent
============================================================

This script evaluates a CrewAI agent with built-in memory
(short-term, long-term, entity memory) across all splits of
MemoryAgentBench.

To run:
    python -m single_agent.memory.crewai_test
"""

import os
import shutil
import time
from crewai import Crew, Agent, Task, Process, LLM
from crewai.utilities.paths import db_storage_path
from dotenv import load_dotenv
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    crewai_llm_model,
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
# 🧠 CrewAI Agent Builder
# ---------------------------------------------------------------------
def build_crewai_agent(storage_dir: str):
    """Initialize a CrewAI crew with memory enabled."""
    os.environ["CREWAI_STORAGE_DIR"] = storage_dir
    print(f"🗂 CrewAI memory storage path: {db_storage_path()}")

    llm = LLM(
        model=crewai_llm_model,
        api_key=API_KEY,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens,
    )

    agent_answer = Agent(
        role="MemoryQA",
        goal="Answer questions accurately using remembered context.",
        backstory=(
            "You are a memory-augmented assistant capable of recalling and "
            "reasoning over previously seen information."
        ),
        llm=llm,
    )

    agent_ingest = Agent(
        role="Ingest",
        goal="Read and remember the context for future questions.",
        backstory=(
            "You are a memory-augmented assistant capable of recalling and "
            "reasoning over previously seen information."
        ),
        llm=llm,
    )

    return Crew(
        agents=[agent_answer, agent_ingest],
        tasks=[],
        process=Process.sequential,
        memory=True,
        verbose=verbose,
    )


# ---------------------------------------------------------------------
# 🧩 Wrapper for Benchmark Compatibility
# ---------------------------------------------------------------------
class CrewAIMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() for MemoryAgentBench.
    Each run uses a unique subfolder for its persistent memory DB.
    """

    _session_counter = 0

    def __init__(self):
        self.crew = self._build_new_crewai_agent()

    @classmethod
    def _build_new_crewai_agent(cls):
        cls._session_counter += 1
        subfolder = f"session_{cls._session_counter}"

        memory_path = os.path.join(storage_directory, subfolder)
        shutil.rmtree(memory_path, ignore_errors=True)
        os.makedirs(memory_path, exist_ok=True)

        os.environ["CREWAI_STORAGE_DIR"] = memory_path
        print(f"🧠 New CrewAI session: {memory_path}")

        return build_crewai_agent(memory_path)

    # -------------------------------------------------------------
    def reset(self):
        """Reset memories by starting a new CrewAI instance."""
        try:
            self.crew = self._build_new_crewai_agent()
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        """Feed benchmark context into CrewAI memory."""
        print("🧠 Ingesting context...")
        chunks = chunk_text(context, max_tokens=chunk_max_tokens, overlap=chunk_overlap)
        print(f"🧩 Total Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")
            task = Task(
                description=(
                    f"Part {i}/{len(chunks)}:\n"
                    f"Remember the following context for future questions:\n\n{chunk}\n"
                    f"Return only the message 'Context added to my memory.'"
                ),
                expected_output=f"Context chunk {i} memorized.",
                agent=self.crew.agents[1],
            )
            self.crew.tasks = [task]
            try:
                self.crew.kickoff()
            except Exception as e:
                print(f"❌ Error processing chunk {i}: {e}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        """Ask a question using remembered context."""
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")

        task = Task(
            description=f"Answer this question using remembered context:\n{question}",
            expected_output="A factual and concise answer based on memory.",
            agent=self.crew.agents[0],
        )
        self.crew.tasks = [task]

        try:
            result = self.crew.kickoff()
            answer = str(result.output) if hasattr(result, "output") else str(result)
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
            print(f"🧩 Running CrewAI Memory Agent on Split: {split}")
            print("=" * 80)

            bench = MemoryAgentBench(split=split)
            agent = CrewAIMemoryAgent()

            result = bench.evaluate_agent(
                agent,
                system_name= "Crewai",
                verbose=verbose,
            )
            overall_summary[split] = result["overall"]

        summarize_results("Crewai", overall_summary)

    finally:
        # -----------------------------------------------------------------
        # 🧹 Clean up CrewAI memory folder after benchmark completion
        # -----------------------------------------------------------------
        try:
            if os.path.exists(storage_directory):
                print(f"\n🗑️  Cleaning up CrewAI memory folder: {storage_directory}")
                shutil.rmtree(storage_directory)
                print("✅ Memory folder deleted successfully.")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
