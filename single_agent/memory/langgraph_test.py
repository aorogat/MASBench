"""
Run MemoryAgentBench on All Splits using LangGraph Memory Agent
===============================================================

This script evaluates a LangGraph agent with the
long-term memory, backed by an in-memory semantic store and optional
avoid short-term (checkpointer) becuase each messgae turn is huge and accumulating them with exceed the context window early, on all splits of MemoryAgentBench.

To run:
    python -m single_agent.memory.langgraph_test
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
    max_context_tokens,
    chunk_overlap,
    splits,
    verbose,
)

# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY


# ---------------------------------------------------------------------
# 🧠 LangGraph Agent Builder (NO STREAMING)
# ---------------------------------------------------------------------
def build_langgraph_agent(store_path: str):
    """
    Create a LangGraph instance with:
      - semantic long-term memory (InMemoryStore)
      - short-term message memory (InMemorySaver)
    """

    embeddings = init_embeddings("openai:text-embedding-3-small")

    store = InMemoryStore(
        index={"embed": embeddings, "dims": 1536}
    )

    # Disable streaming entirely
    model = init_chat_model(model=langgraph_llm_model, streaming=False)

    def chat(state: MessagesState, *, store: BaseStore):
        """Retrieve relevant stored facts and generate a response."""
        user_id = "benchmark_user"
        namespace = ("memories", user_id)

        items = store.search(namespace, query=state["messages"][-1].content, limit=RETRIEVAL_LIMIT)
        memories = "\n".join([item.value["text"] for item in items])
        mem_info = f"## Retrieved memories\n{memories}" if memories else ""

        response = model.invoke(
            [
                {"role": "system", "content": f"You are a helpful assistant.\n{mem_info}"},
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    builder.add_edge(START, "chat")

    graph = builder.compile(store=store)
    return graph, store


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility (NO STREAMING)
# ---------------------------------------------------------------------
class LangGraphMemoryAgent:
    """
    Adapter exposing reset(), ingest(), and query() for MemoryAgentBench.
    Uses semantic retrieval for answering and memory storage for ingestion.
    """

    _session_counter = 0

    def __init__(self):
        os.makedirs(storage_directory, exist_ok=True)
        LangGraphMemoryAgent._session_counter += 1

        self.session_id = f"langgraph_session_{LangGraphMemoryAgent._session_counter}"
        self.session_path = os.path.join(storage_directory, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)

        self.graph, self.store = build_langgraph_agent(self.session_path)
        self.user_id = "benchmark_user"

        print(f"🧠 LangGraph session initialized: {self.session_path}")

    # -------------------------------------------------------------
    def reset(self):
        """Reset memory completely."""
        try:
            LangGraphMemoryAgent._session_counter += 1
            self.session_id = f"langgraph_session_{LangGraphMemoryAgent._session_counter}"
            self.session_path = os.path.join(storage_directory, self.session_id)
            os.makedirs(self.session_path, exist_ok=True)

            self.graph, self.store = build_langgraph_agent(self.session_path)
            print(f"🧹 New LangGraph session started: {self.session_path}")
        except Exception as e:
            print(f"⚠️ Memory reset failed: {e}")

    # -------------------------------------------------------------
    def ingest(self, context: str):
        """Store benchmark context in semantic memory."""
        print("🧠 Ingesting context into semantic memory...")
        chunks = chunk_text(context, max_tokens=chunk_max_tokens, overlap=chunk_overlap)
        namespace = ("memories", self.user_id)

        for i, chunk in enumerate(chunks, 1):
            try:
                self.store.put(namespace, f"chunk_{i}", {"text": chunk})
                print(f"  ✅ Stored chunk {i}/{len(chunks)} in memory.")
            except Exception as e:
                print(f"❌ Failed to store chunk {i}: {e}")

    # -------------------------------------------------------------
    def query(self, question: str) -> str:
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")

        try:
            # ------------------------------------------------------
            # 1. Semantic memory search (NO short-term memory)
            # ------------------------------------------------------
            namespace = ("memories", self.user_id)
            items = self.store.search(namespace, query=question, limit=RETRIEVAL_LIMIT)

            memories = "\n".join(item.value["text"] for item in items)
            
            # Cut memory to avoid overflow
            mem_info = f"## Retrieved memories\n{memories}"
            mem_info = mem_info[: max_context_tokens]


            # ------------------------------------------------------
            # 2. Build stateless message list
            # ------------------------------------------------------
            messages = [
                {"role": "system", "content": f"You are a helpful assistant.\n{mem_info}"},
                {"role": "user", "content": question},
            ]


            # ------------------------------------------------------
            # 3. Direct graph invoke (NO accumulation, NO thread_id)
            # ------------------------------------------------------
            result = self.graph.invoke(
                {"messages": messages},
                config={},            # no thread_id → stateless
                store=self.store       # long-term memory only
            )


            # ------------------------------------------------------
            # 4. Extract reply
            # ------------------------------------------------------
            msgs = result.get("messages", [])
            if not msgs:
                print("⚠️ No assistant response returned.")
                return ""

            answer = msgs[-1].content.strip()
            print(f"🧾 Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
            return answer

        except Exception as e:
            print(f"❌ Query failed: {e}")
            return ""


# ---------------------------------------------------------------------
# 🚀 Run Benchmark for All Splits
# ---------------------------------------------------------------------
def main():
    if os.path.exists(storage_directory):
        print(f"🧹 Removing existing memory directory: {storage_directory}")
        shutil.rmtree(storage_directory)
    os.makedirs(storage_directory, exist_ok=True)

    overall_summary = {}

    try:
        for split in splits:
            print("\n" + "=" * 80)
            print(f"🧩 Running LangGraph Memory Agent on Split: {split}")
            print("=" * 80)

            bench = MemoryAgentBench(split=split)
            agent = LangGraphMemoryAgent()

            result = bench.evaluate_agent(
                agent,
                system_name="LangGraph",
                verbose=verbose,
            )
            overall_summary[split] = result["overall"]

        summarize_results("LangGraph", overall_summary)

    finally:
        try:
            if os.path.exists(storage_directory):
                print(f"\n🗑️ Cleaning up LangGraph memory folder: {storage_directory}")
                shutil.rmtree(storage_directory)
                print("✅ Memory folder deleted successfully.")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
