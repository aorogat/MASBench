"""
AutoGen baseline runner for framework overhead experiments.

This runner:
- Uses AutoGen (AssistantAgent)
- Uses NO memory / NO tools / NO multi-agent coordination
- Executes a single dummy question
- Measures framework + orchestration overhead only

Usage:
    python -m single_agent.framework_overhead.autogen_runner \
        --model gpt-4o-mini
"""

import time
import os
import argparse
from dotenv import load_dotenv

from autogen import AssistantAgent


QUESTION = "What is 2+2?"


class AutoGenRunner:
    """Minimal AutoGen runner (no memory, no tools, single agent)."""

    def __init__(self, model="gpt-4o-mini"):
        load_dotenv()

        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not set")

        self.model = model

        # Minimal AutoGen agent: no tools, no function calls, no memory
        self.agent = AssistantAgent(
            name="OverheadAgent",
            system_message="Answer the question briefly and correctly.",
            llm_config={
                "model": self.model,
                "api_key": os.getenv("OPENAI_API_KEY"),
            },
        )

    def run(self, question: str = QUESTION):
        """Run a single AutoGen call and measure latency."""
        start = time.perf_counter()

        # AutoGen returns the full message history; last assistant message is the answer
        self.agent.reset()
        self.agent.initiate_chat(
            recipient=self.agent,
            message=question,
            max_turns=1,
        )

        end = time.perf_counter()

        # Extract answer safely
        messages = self.agent.chat_messages[self.agent]
        answer = messages[-1]["content"] if messages else ""

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

    runner = AutoGenRunner(model=args.model)

    print("=== AutoGen (No-Memory Overhead Test) ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
