"""
Test script for LangGraph agent on StableToolBench.

This script:
1. Creates a LangGraphAgent
2. Binds tools from StableToolBench
3. Runs benchmark with 3 queries
4. Saves results to results/tools/

To run: python -m single_agent.tool_use.test_langgraph
"""
import os
import sys
from pathlib import Path

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env from root folder (MASBench/)
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ENV_PATH = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
        print(f"Loaded .env file from: {ENV_PATH}")
    else:
        print(f"Warning: .env file not found at {ENV_PATH}")
except ImportError:
    print("Warning: python-dotenv not installed. Using system environment variables.")

from run_benchmark import run_benchmark
from agents.langgraph import LangGraphAgent


def main():
    """Main test function."""
    print("=" * 80)
    print("LangGraph Agent Test")
    print("=" * 80)
    
    # Check if server is running
    server_url = "http://localhost:8080/virtual"
    print(f"\n[Note] Make sure the server is running at {server_url}")
    print("       Start it with: python single_agent/tool_use/StableToolBench/server/main.py")
    print()
    
    # Paths
    tools_dir = os.path.join(CURRENT_DIR, "StableToolBench", "toolenv", "tools")
    
    if not os.path.exists(tools_dir):
        print(f"Error: Tools directory not found: {tools_dir}")
        print("Please ensure StableToolBench/toolenv/tools/ exists")
        return
    
    # Create agent
    print("[1/2] Creating LangGraphAgent...")
    agent = LangGraphAgent(
        model="gpt-4o-mini",
        server_url=server_url,
        temperature=0.0,
        verbose=True
    )
    print("✓ Agent created")
    print("\n[Note] Tool selection and binding will be handled by run_benchmark")
    print("       This ensures fairness - all frameworks use the same tool set per query")
    
    # Run benchmark (tool selection happens inside run_benchmark)
    print(f"\n[2/2] Running benchmark with 3 queries...")
    try:
        results = run_benchmark(
            agent=agent,
            test_set="G1_instruction",
            max_queries=3,
            server_url=server_url,
            evaluator_model="gpt-4o-mini",
            agent_name="langgraph_agent",
            use_tool_selector=True,  # Use centralized tool selection
            tool_selector_model="gpt-4o-mini",
            max_tools=120
        )
        
        print("\n" + "=" * 80)
        print("Test completed successfully!")
        print("=" * 80)
        print(f"\nResults summary:")
        print(f"  Total queries: {results['summary']['total_queries']}")
        print(f"  Solved: {results['summary']['solved_count']} ({results['summary']['solved_percentage']:.1f}%)")
        print(f"  Average SoPR Score: {results['summary']['average_sopr_score']:.3f}")
        print(f"  Average API Call Score: {results['summary']['average_api_call_score']:.3f}")
        print(f"\nCheck results/tools/ for detailed results JSON file")
        
    except Exception as e:
        print(f"✗ Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

