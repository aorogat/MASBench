"""
OpenAI Agents SDK baseline runner for framework overhead experiments.

This runner:
- Uses OpenAI Agents SDK
- Uses NO memory / NO sessions
- Executes a single dummy question
- Measures SDK + orchestration overhead only

Usage:
    python -m single_agent.framework_overhead.openai_sdk_runner \
        --model gpt-4o-mini
"""

import time
import os
import argparse
import asyncio
from dotenv import load_dotenv

from agents import Agent, Runner


QUESTION = "What is 2+2?"


class OpenAISDKRunner:
    """Minimal OpenAI Agents SDK runner (no memory, no tools)."""

    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")

        self.model = model

        # Minimal agent: no tools, no memory, no planning
        self.agent = Agent(
            name="OverheadAgent",
            instructions="Answer the question briefly and correctly.",
            model=self.model,
        )

    def run(self, question: str = QUESTION):
        """Run a single OpenAI SDK call and measure latency."""
        start = time.perf_counter()

        result = asyncio.run(
            Runner.run(self.agent, question)
        )

        end = time.perf_counter()

        # Extract answer safely
        answer = (
            result.final_output
            if hasattr(result, "final_output")
            else str(result)
        )

        return answer.strip(), (end - start) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (e.g., gpt-4o-mini)",
    )
    args = parser.parse_args()

    runner = OpenAISDKRunner(model=args.model)

    print("=== OpenAI Agents SDK (No-Memory Overhead Test) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
