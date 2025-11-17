"""
Run MemoryAgentBench on All Splits using LangGraph Memory Agent
===============================================================

This script evaluates a LangGraph agent with both short-term and
long-term memory, backed by an in-memory semantic store and optional
checkpointer, on all splits of MemoryAgentBench.

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
    llm_max_tokens,
    llm_temperature,
    storage_directory,
    chunk_max_tokens,
    RETRIEVAL_LIMIT,
    chunk_overlap,
    splits,
    results_directory,
    verbose,
)

# ---------------------------------------------------------------------
# 🔧 CONFIGURATION
# ---------------------------------------------------------------------
load_dotenv()
os.environ["OPENAI_API_KEY"] = API_KEY


# ---------------------------------------------------------------------
# 🧠 LangGraph Agent Builder
# ---------------------------------------------------------------------
def build_langgraph_agent(store_path: str):
    """
    Create a LangGraph instance with:
      - semantic long-term memory (InMemoryStore)
      - short-term message memory (InMemorySaver)
    """

    # Semantic embedding-based store
    embeddings = init_embeddings("openai:text-embedding-3-small")
    store = InMemoryStore(
        index={"embed": embeddings, "dims": 1536}
    )

    # Short-term checkpointing memory
    checkpointer = InMemorySaver()

    # LLM model
    model = init_chat_model(model=langgraph_llm_model)

    # Define a simple message-processing node
    def chat(state: MessagesState, *, store: BaseStore):
        """Retrieve relevant stored facts and generate a response."""
        user_id = "benchmark_user"
        namespace = ("memories", user_id)

        # Search relevant context from semantic memory
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

    # Build graph
    builder = StateGraph(MessagesState)
    builder.add_node(chat)
    builder.add_edge(START, "chat")
    graph = builder.compile(store=store, checkpointer=checkpointer)
    return graph, store


# ---------------------------------------------------------------------
# 🧩 Adapter for Benchmark Compatibility
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
        """Reset memory by starting a new semantic store + checkpointer."""
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
        """Embed and store benchmark context in the long-term memory store."""
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
        """Ask a question and retrieve a response using the semantic store."""
        print(f"🔍 Querying: {question[:60]}{'...' if len(question) > 60 else ''}")
        try:
            thread_id = f"{self.session_id}_thread"
            answer_parts = []

            # Stream the assistant’s response (token by token or message by message)
            for msg, _ in self.graph.stream(
                {"messages": [{"role": "user", "content": question}]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="messages",
            ):
                # Accumulate both chunked and full message content
                if hasattr(msg, "content") and msg.content:
                    answer_parts.append(msg.content)

            # Combine everything into the final answer
            if answer_parts:
                answer = "".join(answer_parts).strip()
                print(f"🧾 Answer: {answer[:120]}{'...' if len(answer) > 120 else ''}")
                return answer
            else:
                print("⚠️ No assistant response collected.")
                return ""

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
        # 🧹 Clean up session folders
        try:
            if os.path.exists(storage_directory):
                print(f"\n🗑️  Cleaning up LangGraph memory folder: {storage_directory}")
                shutil.rmtree(storage_directory)
                print("✅ Memory folder deleted successfully.")
        except Exception as e:
            print(f"⚠️ Cleanup failed: {e}")


# ---------------------------------------------------------------------
# 🏁 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
