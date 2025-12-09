"""
Combined LangGraph evaluation script for StableToolBench

This file contains:
  - LangGraphAgent (loads all ~600 tools, builds agent)
  - FrameworkAdapter wrapper
  - Full evaluation pipeline using StableToolBench QAPipeline
  - VERBOSE DEBUGGING OUTPUT

Run: python -m single_agent.tool_use.langgraph_test
"""

import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")

# Make StableToolBench importable
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)

# Alias the internal package "toolbench"
TOOLBENCH_PATH = os.path.join(STB_ROOT, "toolbench")
if TOOLBENCH_PATH not in sys.path:
    sys.path.insert(0, TOOLBENCH_PATH)

# Also alias parent so absolute imports inside package work
sys.modules["toolbench"] = __import__("toolbench")




from typing import List, Dict, Any, TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage
)
from langchain_openai import ChatOpenAI
from langchain.tools import tool

from toolbench.inference.qa_pipeline import QAPipeline
from FrameworkInterface import FrameworkInterface
from FrameworkAdapter import FrameworkAdapter
import os
import sys

# Make StableToolBench importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)



# ====================================================================
# STATE TYPE FOR LANGGRAPH
# ====================================================================
class MessagesState(TypedDict):
    messages: List[BaseMessage]


# ====================================================================
# LANGGRAPH AGENT THAT LOADS ALL 600+ TOOLS AND RUNS TOOLBENCH
# ====================================================================
class LangGraphAgent(FrameworkInterface):
    """
    LangGraph agent that:
      ✔ Loads ALL ~600 StableToolBench tools
      ✔ Wraps each as a LangChain tool for LLM tool-calling
      ✔ Uses LangGraph agent loop (LLM ↔ Tool)
      ✔ Returns final answer to ToolBench for evaluation
      ✔ VERBOSE mode for debugging
    """

    def __init__(self, model_name="gpt-4o-mini", verbose=True):
        self.model_name = model_name
        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.tool_list = []
        self.tool_map = {}
        self.agent = None
        self.verbose = verbose

    # ------------------------------------------------------------
    # FrameworkInterface API
    # ------------------------------------------------------------
    def setup_tools(self, tools: Dict[str, Any]):
        """
        Convert ALL ~600 tools into LangChain dynamic tool functions.
        The tools do NOT call external APIs—ToolBench handles responses
        through MirrorAPI cached evaluation.
        """

        if self.verbose:
            print(f"\n[LangGraphAgent] Loading {len(tools)} tools...")

        self.tool_list = []
        self.tool_map = {}

        for tool_name, tool_data in tools.items():

            if self.verbose:
                print(f"[LangGraphAgent] Registering tool: {tool_name}")

            @tool(name=tool_name)
            def dynamic_tool(*args, __tool_name=tool_name, **kwargs):
                """
                Placeholder tool. ToolBench intercepts and returns cached output.
                """
                if self.verbose:
                    print(f"[LangGraphAgent] Tool executed: {__tool_name}")

                return f"TOOL_CALL({__tool_name})"

            self.tool_list.append(dynamic_tool)
            self.tool_map[tool_name] = dynamic_tool

        # Bind all tools to the LLM
        self.llm_with_tools = self.llm.bind_tools(self.tool_list)

        if self.verbose:
            print("[LangGraphAgent] Building LangGraph agent...")

        # Build the tool-calling agent
        self.agent = self._build_agent()

        if self.verbose:
            print("[LangGraphAgent] Agent ready.\n")

    def reset(self):
        """No persistent memory baseline."""
        if self.verbose:
            print("[LangGraphAgent] Resetting agent state.")
        pass

    def answer(self, query: str) -> str:
        """
        Run the LangGraph tool-calling workflow and return final answer.
        """
        if self.verbose:
            print(f"\n[LangGraphAgent] New query received:")
            print(f"  > {query}\n")
            print("[LangGraphAgent] Beginning agent execution...")

        messages = [HumanMessage(content=query)]
        result = self.agent.invoke({"messages": messages})

        if self.verbose:
            print("[LangGraphAgent] Agent returned messages:")
            for m in result["messages"]:
                msg_type = m.__class__.__name__
                print(f"  [{msg_type}] {m.content[:200]}")

        final_msg = result["messages"][-1]

        if isinstance(final_msg, AIMessage):
            answer = final_msg.content
        else:
            answer = str(final_msg)

        if self.verbose:
            print("\n[LangGraphAgent] FINAL ANSWER:")
            print(answer)
            print("------------------------------------------------------\n")

        return answer

    # ------------------------------------------------------------
    # INTERNAL: AGENT GRAPH (TOOL-CALLING LOOP)
    # ------------------------------------------------------------
    def _build_agent(self):
        """
        Standard LangGraph agent:
            LLM → (maybe calls tool) → tool-node → back to LLM → final answer
        """

        def llm_node(state: MessagesState):
            if self.verbose:
                print("[Node: LLM] Processing LLM call...")

            msgs = state["messages"]
            response = self.llm_with_tools.invoke(msgs)

            if self.verbose:
                print("[Node: LLM] LLM output received:")
                print(f"  Content: {response.content[:150]}")
                if response.tool_calls:
                    print(f"  Tool calls detected: {[c['name'] for c in response.tool_calls]}")

            return {"messages": msgs + [response]}

        def tool_node(state: MessagesState):
            last_msg = state["messages"][-1]

            if self.verbose:
                print("[Node: TOOL] Executing tool calls...")

            outputs = []
            for call in last_msg.tool_calls:
                tool_fn = self.tool_map[call["name"]]
                if self.verbose:
                    print(f"  -> Running tool: {call['name']} with args: {call['args']}")

                observation = tool_fn.invoke(call["args"])
                outputs.append(ToolMessage(content=observation, tool_call_id=call["id"]))

            return {"messages": state["messages"] + outputs}

        def should_continue(state: MessagesState) -> Literal["tool_node", END]:
            last = state["messages"][-1]

            if last.tool_calls:
                if self.verbose:
                    print("[Router] LLM requested a tool call → continue to tool_node")
                return "tool_node"

            if self.verbose:
                print("[Router] No tool calls → END")
            return END

        graph = StateGraph(MessagesState)
        graph.add_node("llm", llm_node)
        graph.add_node("tool_node", tool_node)

        graph.add_edge(START, "llm")
        graph.add_conditional_edges("llm", should_continue, ["tool_node", END])
        graph.add_edge("tool_node", "llm")

        return graph.compile()


# ====================================================================
#           MAIN EVALUATION ENTRY POINT
# ====================================================================
def main():
    tool_dir = "single_agent/tool-use/StableToolBench/toolenv/tools"
    query_dir = "single_agent/tool-use/StableToolBench/solvable_queries"

    print("Loading LangGraph agent...")
    agent = LangGraphAgent(verbose=True)
    adapter = FrameworkAdapter(agent)

    print("Running StableToolBench evaluation...")
    pipeline = QAPipeline(
        tool_dir=tool_dir,
        query_dir=query_dir,
        model=adapter,
        use_mirrorapi_cache=True,   # ensures we do not call real APIs
    )

    results = pipeline.run()

    print("\n===== FINAL RESULTS =====")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
