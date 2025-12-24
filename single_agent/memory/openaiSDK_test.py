"""
Run MemoryAgentBench on All Splits using OpenAI SDK Memory Agent
===============================================================

This script evaluates an OpenAI Agents SDK–based agent with built-in
session memory across all splits of MemoryAgentBench.

Each benchmark split runs in an isolated SQLite session database
so the full memory is preserved for inspection and analysis.

To run:
    # This file target Groq WITHOUT using the groq_router
    python -m single_agent.memory.openaiSDK_test
"""


# Engineering safety (fixed)
OPENAI_SDK_INGEST_MAX_TOKENS = 4096

# Experimental variable (sweep this)
OPENAI_SDK_CONTEXT_WINDOW_TOKENS = 16384   # 50 / 512 / 1024 / 2048 / 4096 / 8192 / 16384 / 32768 [big context window hit the TPM]



import os
import time
import asyncio
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# 🔑 OpenAI Agents SDK Imports
# ---------------------------------------------------------------------
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    ModelSettings,
    set_default_openai_api,
    set_tracing_disabled,
    RunConfig,
)

from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    llm_max_tokens,
    llm_temperature,
    storage_directory,
    chunk_max_tokens,
    chunk_overlap,
    splits,
    verbose,
)

# import litellm
# litellm._turn_on_debug()


# ---------------------------------------------------------------------
# ⚠️ Error you may face if you stop the process manually (Ctrl+Z)
# SQLite DB may remain locked.
#
# To solve:
#     rm -rf /PathTo/crewai_memory/*
#     Example
#     rm -rf /shared_mnt/crewai_memory/*
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()

# Required by SDK internals (even when using Groq via LiteLLM)
os.environ["OPENAI_API_KEY"] = API_KEY

# GROQ_API_KEY is loaded from .env
# export GROQ_API_KEY=xxxxxxxxxxxxxxxxxxxx

# Groq does NOT support Responses API
set_default_openai_api("chat_completions")

# Disable tracing to avoid massive hidden payloads
set_tracing_disabled(True)




from agents.run import ModelInputData




def approx_tokens(content) -> int:
    """
    Approximate token count for OpenAI-style message content.

    content can be:
    - str
    - list of {"type": "text", "text": "..."} dicts
    """

    if isinstance(content, str):
        return max(1, len(content.split()))

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
            else:
                text_parts.append(str(part))
        joined = " ".join(text_parts)
        return max(1, len(joined.split()))

    # Fallback (should rarely happen)
    return max(1, len(str(content).split()))



def limit_context_by_tokens(call_model_data):
    """
    call_model_input_filter for OpenAI Agents SDK.

    Input:
      call_model_data : CallModelData
        └── model_data : ModelInputData
              └── input : list[dict]   (model-ready context)

    Output:
      ModelInputData (REQUIRED by Runner)
    """

    model_input = call_model_data.model_data

    if not hasattr(model_input, "input"):
        return model_input  # safety fallback

    items = model_input.input
    if not items:
        return model_input

    kept = []
    total_tokens = 0

    # Walk backwards: most recent context first
    for item in reversed(items):
        content = item.get("content", "")
        tokens = approx_tokens(content)

        if total_tokens + tokens > OPENAI_SDK_CONTEXT_WINDOW_TOKENS:
            break

        kept.append(item)
        total_tokens += tokens

    # Restore chronological order
    model_input.input = list(reversed(kept))

    return model_input


# ---------------------------------------------------------------------
# 🧠 Agent Builders
# ---------------------------------------------------------------------
def build_ingest_agent():
    """
    Agent used ONLY for memory ingestion.

    - Output is irrelevant → capped to 1 token
    - Minimizes request size
    - Prevents Groq context overflow
    """
    return Agent(
        name="MemoryIngest",
        instructions="Store information exactly as provided.",
        model="litellm/groq/openai/gpt-oss-20b",
        model_settings=ModelSettings(
            max_output_tokens=1,
            temperature=0.0,
        ),
    )


def build_query_agent():
    """
    Agent used ONLY for question answering.

    - Normal output length
    - Uses the same persistent SQLite memory
    """
    return Agent(
        name="MemoryQA",
        instructions=(
            "You are a helpful assistant that answers questions accurately "
            "using remembered context."
        ),
        model="litellm/groq/openai/gpt-oss-20b",
        model_settings=ModelSettings(
            max_output_tokens=llm_max_tokens,
            temperature=llm_temperature,
        ),
    )


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
# ---------------------------------------------------------------------
class OpenAIMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() so MemoryAgentBench
    can evaluate the OpenAI Agents SDK with persistent memory.

    Long-term memory: SQLite
    Short-term memory: DISABLED during ingestion
    """

    _session_counter = 0

    def __init__(self):
        self.ingest_agent = build_ingest_agent()
        self.query_agent = build_query_agent()

        self.run_config = RunConfig(
            call_model_input_filter=limit_context_by_tokens
        )


        os.makedirs(storage_directory, exist_ok=True)

        OpenAIMemoryAgent._session_counter += 1
        self.session_id = f"memory_bench_session_{OpenAIMemoryAgent._session_counter}"
        self.session_path = os.path.join(storage_directory, f"{self.session_id}.db")

        self.session = SQLiteSession(self.session_id, self.session_path)
        print(f"🧠 Session initialized: {self.session_path}")

    # -------------------------------------------------------------
    def reset(self):
        """Reset conversation memory by starting a new SQLite file."""
        try:
            OpenAIMemoryAgent._session_counter += 1
            self.session_id = f"memory_bench_session_{OpenAIMemoryAgent._session_counter}"
            self.session_path = os.path.join(storage_directory, f"{self.session_id}.db")
            self.session = SQLiteSession(self.session_id, self.session_path)
            print(f"🧹 New session started: {self.session_path}")
        except Exception as e:
            print(f"⚠️ Session reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        """Feed benchmark context into the agent’s long-term memory."""
        print("🧠 Ingesting context...")

        effective_chunk_tokens = min(chunk_max_tokens, OPENAI_SDK_INGEST_MAX_TOKENS)

        chunks = chunk_text(
            context,
            max_tokens=effective_chunk_tokens,
            overlap=chunk_overlap,
        )

        for i, chunk in enumerate(chunks, 1):
            print(f"  Chunk {i}/{len(chunks)}")

            prompt = (
                "Store the following information for later use:\n\n"
                f"{chunk}"
            )

            try:
                asyncio.run(
                    Runner.run(
                        self.ingest_agent,
                        prompt,
                        session=self.session,
                        run_config=self.run_config,
                    )

                )

                # 🔑 Critical:
                # Reset short-term message state to avoid context accumulation.
                # Long-term memory is already persisted in SQLite.
                self.session = SQLiteSession(self.session_id, self.session_path)

                # TPM throttle
                time.sleep(3.0)

            except Exception as e:
                print(f"❌ Error storing chunk {i}: {e}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        """Ask a question using the remembered context."""
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")

        try:
            result = asyncio.run(
                Runner.run(
                    self.query_agent,
                    question,
                    session=self.session,
                    run_config=self.run_config,
                )
            )

            answer = (
                result.final_output
                if hasattr(result, "final_output")
                else str(result)
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
                system_name=f"Openai_SDK_Groq_CTX{OPENAI_SDK_CONTEXT_WINDOW_TOKENS}",
                verbose=verbose,
            )
            overall_summary[split] = result["overall"]

        summarize_results("Openai_SDK_Groq", overall_summary)

    finally:
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
