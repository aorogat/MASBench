"""
Agno baseline runner for framework overhead experiments.

This runner:
- Uses Agno Agent
- Uses NO memory (no DB)
- Uses NO tools
- Executes a single dummy question
- Measures Agno orchestration + LLM overhead only

Usage:
    python -m single_agent.framework_overhead.agno_runner \
        --model gpt-4o-mini
"""

import time
import os
import argparse
from dotenv import load_dotenv

from agno.agent import Agent, RunOutput
from agno.models.openai import OpenAIChat


QUESTION = "What is 2+2?"


class AgnoRunner:
    """Minimal Agno runner (no memory, no tools)."""

    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")

        self.model_id = model

        # Minimal OpenAI model (no streaming, no tools)
        model = OpenAIChat(
            id=self.model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Minimal agent: no DB → no memory
        self.agent = Agent(
            model=model,
            enable_user_memories=False,
            add_memories_to_context=False,
        )

    def run(self, question: str = QUESTION):
        start = time.perf_counter()

        response: RunOutput = self.agent.run(
            question,
            user_id="overhead_user",
        )

        end = time.perf_counter()

        answer = response.content.strip() if response and response.content else ""

        return answer, (end - start) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model id (e.g., gpt-4o-mini)",
    )
    args = parser.parse_args()

    runner = AgnoRunner(model=args.model)

    print("=== Agno (No-Memory Overhead Test) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
