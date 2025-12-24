"""
Run MemoryAgentBench on All Splits using LangGraph
with BOTH long-term (semantic) and short-term (message) memory.

Short-term memory uses an explicit context window truncation strategy
identical in behavior to the OpenAI Agents SDK experiment.
To run:
    python -m single_agent.memory.langgraph_stm_ltm_test
"""

import os
import shutil
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

from single_agent.memory.benchmark.memory_agent_bench import MemoryAgentBench
from single_agent.memory.helpers.common_agent_utils import (
    API_KEY,
    chunk_text,
    summarize_results,
)
from single_agent.memory.config import (
    langgraph_llm_model,
    storage_directory,
    chunk_max_tokens,
    RETRIEVAL_LIMIT,
    chunk_overlap,
    splits,
    verbose,
)

# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY

# 🔑 Experimental variable (match OpenAI SDK behavior)
LANGGRAPH_SHORT_TERM_CONTEXT_TOKENS = 50   # sweep: 50 / 512 / 1024 / 2048 / 4096 / 8192 / 16384


# ---------------------------------------------------------------------
# 🧮 Token Approximation (same philosophy as OpenAI SDK)
# ---------------------------------------------------------------------
def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text.split()))


def trim_messages_by_tokens(messages, max_tokens):
    """
    Walk backwards and keep as many recent messages
    as fit within max_tokens.
    """
    kept = []
    total = 0

    for msg in reversed(messages):
        tokens = approx_tokens(msg.content)
        if total + tokens > max_tokens:
            break
        kept.append(msg)
        total += tokens

    return list(reversed(kept))


# ---------------------------------------------------------------------
# 🧠 LangGraph Agent Builder (STM + LTM)
# ---------------------------------------------------------------------
def build_langgraph_agent():
    embeddings = init_embeddings("openai:text-embedding-3-small")

    # Long-term semantic memory
    store = InMemoryStore(
        index={"embed": embeddings, "dims": 1536}
    )

    # Short-term message memory
    checkpointer = InMemorySaver()

    model = init_chat_model(
        model=langgraph_llm_model,
        streaming=False,
    )

    def chat(state: MessagesState, *, store: BaseStore):
        user_id = "benchmark_user"
        namespace = ("memories", user_id)

        # -------------------------------
        # 1. Long-term memory retrieval
        # -------------------------------
        query = state["messages"][-1].content
        items = store.search(
            namespace,
            query=query,
            limit=RETRIEVAL_LIMIT,
        )

        memories = "\n".join(item.value["text"] for item in items)
        mem_block = f"## Retrieved memories\n{memories}" if memories else ""

        # -------------------------------
        # 2. Short-term memory truncation
        # -------------------------------
        trimmed_messages = trim_messages_by_tokens(
            state["messages"],
            LANGGRAPH_SHORT_TERM_CONTEXT_TOKENS,
        )

        # -------------------------------
        # 3. Final prompt
        # -------------------------------
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant.\n"
                    f"{mem_block}"
                ),
            },
            *trimmed_messages,
        ]

        response = model.invoke(prompt)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")

    graph = builder.compile(
        store=store,
        checkpointer=checkpointer,   # 🔑 activates STM
    )

    return graph, store


# ---------------------------------------------------------------------
# 🧩 Adapter for MemoryAgentBench
# ---------------------------------------------------------------------
class LangGraphSTMLTMAgent:
    _session_counter = 0

    def __init__(self):
        os.makedirs(storage_directory, exist_ok=True)

        LangGraphSTMLTMAgent._session_counter += 1
        self.session_id = f"langgraph_stm_ltm_{LangGraphSTMLTMAgent._session_counter}"

        self.graph, self.store = build_langgraph_agent()
        self.user_id = "benchmark_user"

        print(
            f"🧠 LangGraph STM+LTM session initialized "
            f"(CTX={LANGGRAPH_SHORT_TERM_CONTEXT_TOKENS})"
        )

    # -------------------------------------------------------------
    def reset(self):
        LangGraphSTMLTMAgent._session_counter += 1
        self.session_id = f"langgraph_stm_ltm_{LangGraphSTMLTMAgent._session_counter}"
        self.graph, self.store = build_langgraph_agent()
        print("🧹 New LangGraph STM+LTM session started")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        print("🧠 Ingesting context into long-term semantic memory...")
        chunks = chunk_text(
            context,
            max_tokens=chunk_max_tokens,
            overlap=chunk_overlap,
        )

        namespace = ("memories", self.user_id)

        for i, chunk in enumerate(chunks, 1):
            self.store.put(
                namespace,
                f"chunk_{i}",
                {"text": chunk},
            )
            print(f"  ✅ Stored chunk {i}/{len(chunks)}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        print(f"🔍 Querying: {question[:60]}...")

        result = self.graph.invoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"thread_id": self.session_id},  # 🔑 STM accumulation
            store=self.store,
        )

        msgs = result.get("messages", [])
        if not msgs:
            return ""

        answer = msgs[-1].content.strip()
        print(f"🧾 Answer: {answer[:120]}...")
        return answer


# ---------------------------------------------------------------------
# 🚀 Run Benchmark
# ---------------------------------------------------------------------
def main():
    if os.path.exists(storage_directory):
        shutil.rmtree(storage_directory)
    os.makedirs(storage_directory, exist_ok=True)

    overall_summary = {}

    try:
        for split in splits:
            print("\n" + "=" * 80)
            print(f"🧩 Running LangGraph STM+LTM on Split: {split}")
            print("=" * 80)

            bench = MemoryAgentBench(split=split)
            agent = LangGraphSTMLTMAgent()

            result = bench.evaluate_agent(
                agent,
                system_name=f"LangGraph_STM_LTM_CTX{LANGGRAPH_SHORT_TERM_CONTEXT_TOKENS}",
                verbose=verbose,
            )

            overall_summary[split] = result["overall"]

        summarize_results(
            f"LangGraph_STM_LTM_CTX{LANGGRAPH_SHORT_TERM_CONTEXT_TOKENS}",
            overall_summary,
        )

    finally:
        if os.path.exists(storage_directory):
            shutil.rmtree(storage_directory)


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
