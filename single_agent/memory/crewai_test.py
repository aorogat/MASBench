"""
Run MemoryAgentBench on All Splits using CrewAI Memory Agent
============================================================

This script evaluates a CrewAI agent with built-in memory
(short-term, long-term, entity memory) across all splits of
MemoryAgentBench:

    - Accurate_Retrieval
    - Test_Time_Learning
    - Long_Range_Understanding
    - Selective_Forgetting

Results are saved automatically in `results/memory/`.
"""

import os
import time
import shutil
from crewai import Crew, Agent, Task, Process, LLM
from crewai.utilities.paths import db_storage_path
from dotenv import load_dotenv
from benchmarks.memory.memory_agent_bench import MemoryAgentBench
from single_agent.memory.config import (
    llm_model,
)
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
# 🧠 CrewAI Agent Builder
# ---------------------------------------------------------------------
def build_crewai_agent(storage_dir="/shared_mnt/crewai_memory"):
    os.environ["CREWAI_STORAGE_DIR"] = storage_dir
    print("🗂 CrewAI memory storage path:", db_storage_path())

    llm = LLM(
        model=llm_model,
        api_key=API_KEY,
        max_tokens=3000,
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
        verbose=True,
    )


# ---------------------------------------------------------------------
# 🧩 Wrapper for Benchmark Compatibility
# ---------------------------------------------------------------------
class CrewAIMemoryAgent:
    """
    Adapts CrewAI crew to the interface expected by MemoryAgentBench:
      - reset()
      - ingest(context)
      - query(question)
    """

    _session_counter = 0  # 🔁 Static counter for unique subfolders

    def __init__(self):
        self.crew = self.build_crewai_agent()

    @classmethod
    def build_crewai_agent(cls):
        # Increment and use unique subfolder
        cls._session_counter += 1
        subfolder = f"session_{cls._session_counter}"

        base_path = "/shared_mnt/crewai_memory"
        memory_path = os.path.join(base_path, subfolder)

        # Clean and prepare fresh memory folder
        shutil.rmtree(memory_path, ignore_errors=True)
        os.makedirs(memory_path, exist_ok=True)

        os.environ["CREWAI_STORAGE_DIR"] = memory_path
        print(f"🗂 CrewAI memory storage path: {memory_path}")

        llm = LLM(
            model=llm_model,
            api_key=API_KEY,
            temperature=0,
            max_tokens=3000,
        )

        agent_answer = Agent(
            role="MemoryQA",
            goal="Answer questions accurately using remembered context.",
            backstory="You are a memory-augmented assistant capable of recalling and reasoning over previously seen information.",
            llm=llm,
        )

        agent_ingest = Agent(
            role="Ingest",
            goal="Read and remember the context for future questions.",
            backstory="You are a memory-augmented assistant capable of recalling and reasoning over previously seen information.",
            llm=llm,
        )

        return Crew(
            agents=[agent_answer, agent_ingest],
            tasks=[],
            process=Process.sequential,
            memory=True,
            verbose=True,
        )

    # -------------------------------------------------------------
    def reset(self):
        """Reset memories by creating a new crew with fresh DB."""
        try:
            self.crew = self.build_crewai_agent()
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str, max_tokens: int = 1000, overlap: int = 50):
        """Feed benchmark context into the crew memory."""
        print("🧠 Ingesting current Context")
        chunks = chunk_text(context, max_tokens=max_tokens, overlap=overlap)
        print(f"🧩 Total Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks, 1):
            print(f"\tChunk {i}/{len(chunks)}")
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
                print(f"\t❌ Error processing chunk {i}: {e}")

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
    splits = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Conflict_Resolution",
    ]

    overall_summary = {}

    for split in splits:
        print("\n" + "=" * 80)
        print(f"🧩 Running CrewAI Memory Agent on Split: {split}")
        print("=" * 80)

        bench = MemoryAgentBench(split=split, n=None)
        agent = CrewAIMemoryAgent()

        result = bench.evaluate_agent(
            agent,
            system_name="crewai_memory_agent",
            verbose=True,
        )
        overall_summary[split] = result["overall"]

    summarize_results("crewai_memory_agent", overall_summary)


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
