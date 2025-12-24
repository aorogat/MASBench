"""
MemoryAgentBench evaluation using AutoGen (new API) + ChromaDBVectorMemory
Model: gpt-4o-mini (OpenAI only)

Stability fixes added:
- Explicit ChromaDB + LLM warm-up (prevents Q1 hang)
- Hard timeout + retries for agent.run()
- Detailed debug/timing logs

NOTE:
- Uses storage_directory from config (NO /tmp usage)
- Does NOT change benchmark semantics
"""

import os
import asyncio
import time
import traceback
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------
# AutoGen imports
# ---------------------------------------------------------------------
from autogen_agentchat.agents import AssistantAgent
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_ext.memory.chromadb import (
    ChromaDBVectorMemory,
    PersistentChromaDBVectorMemoryConfig,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient

# ---------------------------------------------------------------------
# Benchmark + helper utilities
# ---------------------------------------------------------------------
from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    llm_temperature,
    llm_max_tokens,
    storage_directory,   # <-- KEEP YOUR ORIGINAL DIRECTORY
    chunk_max_tokens,
    chunk_overlap,
    splits,
    verbose,
)

# ---------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY

OPENAI_MODEL = "gpt-4o-mini"

# Debug / safety controls
DEBUG = os.environ.get("AUTOGEN_DEBUG", "0") == "1"
LLM_TIMEOUT_S = int(os.environ.get("AUTOGEN_LLM_TIMEOUT_S", "60"))
LLM_RETRIES = int(os.environ.get("AUTOGEN_LLM_RETRIES", "2"))
WARMUP_TIMEOUT_S = int(os.environ.get("AUTOGEN_WARMUP_TIMEOUT_S", "120"))


def dbg(msg: str):
    if DEBUG:
        print(f"[DEBUG] {msg}")


# ---------------------------------------------------------------------
# AutoGen Agent
# ---------------------------------------------------------------------
class AutoGenExtChromaMemoryAgent:
    """
    Retrieval-only memory agent.

    - Long-term memory: ChromaDB (vector retrieval)
    - Short-term memory: ephemeral LLM context only
    - NO accumulation across turns
    """

    _session_counter = 0

    def __init__(self):
        self._new_session()

    # -----------------------------------------------------------------
    def _new_session(self):
        AutoGenExtChromaMemoryAgent._session_counter += 1
        self.session_id = f"autogen_chroma_{AutoGenExtChromaMemoryAgent._session_counter}"

        # ✅ Use YOUR storage_directory (no /tmp)
        self.persist_path = os.path.join(storage_directory, self.session_id)
        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        print(f"🧠 New AutoGen + ChromaDB session: {self.persist_path}")

        self.memory = ChromaDBVectorMemory(
            config=PersistentChromaDBVectorMemoryConfig(
                collection_name="memory_agent_bench",
                persistence_path=self.persist_path,
                k=3,
                score_threshold=None,
            )
        )

        self.agent = AssistantAgent(
            name="AutoGenMemoryQA",
            model_client=OpenAIChatCompletionClient(
                model=OPENAI_MODEL,
                api_key=API_KEY,
                temperature=llm_temperature,
                max_tokens=llm_max_tokens,
            ),
            memory=[self.memory],
        )

        # 🔥 REQUIRED: warm-up to avoid Q1 deadlock
        self._warmup()

    # -----------------------------------------------------------------
    def _warmup(self):
        """
        Forces:
        - ChromaDB embedding + index initialization
        - First OpenAI call (TLS + model warm-up)

        This runs BEFORE the benchmark starts.
        """
        async def _do_warmup():
            t0 = time.time()

            await self.memory.add(
                MemoryContent(
                    content="warmup initialization",
                    mime_type=MemoryMimeType.TEXT,
                )
            )

            try:
                dbg("Warmup: running agent once")
                result = await self.agent.run(task="Reply with OK")
                _ = result.messages[-1].content
            except Exception as e:
                print(f"⚠️ Warmup LLM call failed (continuing): {e}")

            print(f"🔥 Warmup completed in {time.time() - t0:.2f}s")

        try:
            asyncio.run(asyncio.wait_for(_do_warmup(), timeout=WARMUP_TIMEOUT_S))
        except asyncio.TimeoutError:
            print(f"⚠️ Warmup timed out after {WARMUP_TIMEOUT_S}s (continuing)")
        except Exception:
            print("⚠️ Warmup failed (continuing)")
            if DEBUG:
                traceback.print_exc()

    # -----------------------------------------------------------------
    def reset(self):
        self._new_session()

    # -----------------------------------------------------------------
    def ingest(self, context: str):
        print("🧠 Ingesting context into ChromaDB...")
        t0 = time.time()

        chunks = chunk_text(
            context,
            max_tokens=chunk_max_tokens,
            overlap=chunk_overlap,
        )

        async def _ingest():
            total = len(chunks)
            for i, chunk in enumerate(chunks, 1):
                if DEBUG and (i == 1 or i % 10 == 0 or i == total):
                    print(f"  ➜ Ingesting chunk {i}/{total}")
                try:
                    await self.memory.add(
                        MemoryContent(
                            content=chunk,
                            mime_type=MemoryMimeType.TEXT,
                            metadata={"chunk_id": i},
                        )
                    )
                except Exception as e:
                    print(f"❌ Ingest error on chunk {i}: {e}")
                    if DEBUG:
                        traceback.print_exc()


    # -----------------------------------------------------------------
    def query(self, question: str) -> str:
        print(f"🔍 Querying: {question[:80]}{'...' if len(question) > 80 else ''}")

        async def _query_once():
            result = await self.agent.run(task=question)
            return result.messages[-1].content

        for attempt in range(1, LLM_RETRIES + 2):
            try:
                return asyncio.run(
                    asyncio.wait_for(_query_once(), timeout=LLM_TIMEOUT_S)
                )
            except asyncio.TimeoutError:
                print(f"⏱️ LLM timeout ({attempt}/{LLM_RETRIES + 1})")
            except Exception as e:
                print(f"❌ Query failed ({attempt}): {e}")
                if DEBUG:
                    traceback.print_exc()

        print("❌ All query attempts failed")
        return ""


# ---------------------------------------------------------------------
# Run benchmark
# ---------------------------------------------------------------------
def main():
    print(f"📦 Using ChromaDB persistence directory: {storage_directory}")

    overall_summary = {}

    for split in splits:
        print("\n" + "=" * 80)
        print(f"🧩 Running AutoGen + ChromaDB | Split: {split}")
        print("=" * 80)

        bench = MemoryAgentBench(split=split)
        agent = AutoGenExtChromaMemoryAgent()

        result = bench.evaluate_agent(
            agent,
            system_name="AutoGen-ChromaDB-OpenAI",
            verbose=verbose,
        )

        overall_summary[split] = result["overall"]

    summarize_results("AutoGen-ChromaDB-OpenAI", overall_summary)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
