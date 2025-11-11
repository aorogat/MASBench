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
from crewai import Crew, Agent, Task, Process
from benchmarks.memory.memory_agent_bench import MemoryAgentBench
from crewai import LLM
from dotenv import load_dotenv
from crewai.utilities.paths import db_storage_path

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------------------------------------------------------
# 🧠 CrewAI Agent Builder
# ---------------------------------------------------------

def build_crewai_agent(storage_dir="/shared_mnt/crewai_memory"):
    os.environ["CREWAI_STORAGE_DIR"] = storage_dir
    print("🗂 CrewAI memory storage path:", db_storage_path())


    llm = LLM(
        model="openai/gpt-4o-mini",
        api_key=api_key,
        max_tokens=3000
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

    crew = Crew(
        agents=[agent_answer, agent_ingest],
        tasks=[],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )
    return crew


# ---------------------------------------------------------
# 🧩 Wrapper for Benchmark Compatibility
# ---------------------------------------------------------
class CrewAIMemoryAgent:
    """
    Adapts CrewAI crew to the interface expected by MemoryAgentBench:
      - reset()
      - ingest(context)
      - query(question)
    """

    _session_counter = 0  # 🔁 Static counter to generate unique subfolders

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
            model="openai/gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            max_tokens=3000
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

    def reset(self):
        """Reset memories by creating a new crew with fresh DB."""
        try:
            self.crew = self.build_crewai_agent()
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    def ingest(self, context: str, max_tokens: int = 1000, overlap: int = 50):
        print("Ingesting current Context")

        words = context.split()
        n = len(words)
        chunks = []

        if n <= max_tokens:
            chunks = [context]
        else:
            start = 0
            while start < n:
                end = min(start + max_tokens, n)
                chunks.append(" ".join(words[start:end]))
                if end >= n:
                    break
                start = end - overlap if end - overlap > start else end

        print(f"🧠 Ingesting context in {len(chunks)} chunks (≤ {max_tokens} tokens each)")

        for i, chunk in enumerate(chunks, 1):
            print(f"\tChunk {i}/{len(chunks)}")
            task = Task(
                description=(
                    f"Part {i}/{len(chunks)}:\n"
                    f"Remember the following context for future questions:\n\n{chunk}.\n"
                    f"Return only the message 'Context added to my memory.'"
                ),
                expected_output=f"Context chunk {i} memorized.",
                agent=self.crew.agents[1],
            )
            self.crew.tasks = [task]
            try:
                result = self.crew.kickoff()
            except Exception as e:
                print(f"\t❌ Error processing chunk {i}: {e}")

    def query(self, question: str) -> str:
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




# ---------------------------------------------------------
# 🚀 Run Benchmark for All Splits
# ---------------------------------------------------------
def main():
    splits = [
        "Accurate_Retrieval",
        "Test_Time_Learning",
        "Long_Range_Understanding",
        "Selective_Forgetting",
    ]

    overall_summary = {}
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for split in splits:
        print("\n" + "=" * 80)
        print(f"🧩 Running CrewAI Memory Agent on Split: {split}")
        print("=" * 80)

        bench = MemoryAgentBench(split=split, n=None)  # Full dataset split
        agent = CrewAIMemoryAgent()

        result = bench.evaluate_agent(
            agent,
            system_name="crewai_memory_agent",
            verbose=True,
        )

        overall_summary[split] = result["overall"]

    # -----------------------------------------------------
    # 📊 Summary Report
    # -----------------------------------------------------
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY – CrewAI Memory Agent")
    print("=" * 80)
    for split, score in overall_summary.items():
        print(f"{split:30s} → {score:.3f}")
    print("-" * 80)
    avg_score = sum(overall_summary.values()) / len(overall_summary)
    print(f"⭐ Average Overall Score: {avg_score:.3f}")
    print("=" * 80)

    # Save summary JSON
    results_dir = os.path.join("results", "memory")
    os.makedirs(results_dir, exist_ok=True)
    summary_path = os.path.join(results_dir, f"crewai_memory_summary_{timestamp}.json")

    import json
    with open(summary_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\n💾 Summary saved to {summary_path}")


# ---------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
