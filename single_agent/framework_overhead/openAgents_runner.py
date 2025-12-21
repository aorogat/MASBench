"""
OpenAgents baseline runner for framework overhead experiments (event-driven).

This implementation follows the official OpenAgents event system and WorkerAgent model.

Usage:
    python -m single_agent.framework_overhead.openAgents_runner --model ollama/deepseek-llm:7b
    python -m single_agent.framework_overhead.openAgents_runner --model openai/gpt-4o-mini
"""

"""
OpenAgents baseline runner for framework overhead experiments.

API-correct, event-style WorkerAgent execution.
"""

"""
OpenAgents baseline runner for framework overhead experiments.

This evaluates OpenAgents at the *agent abstraction level*.
Event routing and network overhead are excluded by design.
"""

import time
import os
import argparse
from dotenv import load_dotenv

from openagents.agents.worker_agent import WorkerAgent
from openai import OpenAI


QUESTION = "What is 2+2?"


class OverheadAgent(WorkerAgent):
    """Minimal OpenAgents WorkerAgent."""

    def __init__(self, model: str):
        super().__init__()
        self.model = model
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def run_llm(self, question: str) -> str:
        """Minimal LLM invocation inside an OpenAgents agent."""
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content


class OpenAgentsRunner:
    """Agent-level OpenAgents overhead runner."""

    def __init__(self, model="openai/gpt-4o-mini"):
        load_dotenv()

        if not model.startswith("openai/"):
            raise ValueError("Only openai/* models supported")

        self.model = model.split("openai/")[1]
        self.agent = OverheadAgent(model=self.model)

    def run(self, question: str = QUESTION):
        start = time.perf_counter()
        answer = self.agent.run_llm(question)
        end = time.perf_counter()

        return answer.strip(), (end - start) * 1000


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use (openai/*)",
    )
    args = parser.parse_args()

    runner = OpenAgentsRunner(model=args.model)

    print("=== OpenAgents (Agent-Abstraction Overhead) Test ===")
    for i in range(3):
        resp, latency = runner.run(QUESTION)
        print(f"Run {i+1}: Q={QUESTION} | A={resp} | ⏱️ {latency:.2f} ms")
