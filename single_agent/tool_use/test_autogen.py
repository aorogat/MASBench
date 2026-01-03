"""
Test script for AutoGen agent on StableToolBench.

This script:
1. Creates an AutoGenAgent
2. Binds tools from StableToolBench
3. Runs benchmark with 3 queries
4. Saves results to results/tools/

To run: python -m single_agent.tool_use.test_autogen
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
from agents.autogen import AutoGenAgent
from utils.printer import print_header, print_info, print_success, print_warning


def main():
    """Main test function."""
    print_header("AutoGen Agent Test")
    
    # Check if server is running
    server_url = "http://localhost:8080/virtual"
    print_warning(f"Make sure the server is running at {server_url}")
    print_info("Start it with: python single_agent/tool_use/StableToolBench/server/main.py")
    print()
    
    # Paths
    # Note: tools_dir is validated here, but actual tool loading is handled
    # centrally by run_benchmark() via ToolSelector. This path is only used
    # as a sanity check to ensure the benchmark environment is set up correctly.
    tools_dir = os.path.join(CURRENT_DIR, "StableToolBench", "toolenv", "tools")
    
    if not os.path.exists(tools_dir):
        from utils.printer import print_error
        print_error(f"Tools directory not found: {tools_dir}")
        print_error("Please ensure StableToolBench/toolenv/tools/ exists")
        return
    
    # Create agent
    print_info("[1/2] Creating AutoGenAgent...")
    try:
        agent = AutoGenAgent(
            model="gpt-4o-mini",
            server_url=server_url,
            temperature=0.0,
            verbose=True
        )
        print_success("Agent created")
    except ImportError as e:
        from utils.printer import print_error
        print_error(f"AutoGen is not installed: {e}")
        print_error("Install it with: pip install autogen-agentchat autogen-core")
        return
    
    print_info("Tool selection and binding will be handled by run_benchmark")
    print_info("This ensures fairness - all frameworks use the same tool set per query")
    
    # Run benchmark (tool selection happens inside run_benchmark)
    print_info("\n[2/2] Running benchmark with 3 queries...")
    try:
        results = run_benchmark(
            agent=agent,
            test_set="G1_instruction",
            max_queries=3,
            server_url=server_url,
            evaluator_model="gpt-4o-mini",
            agent_name="autogen_agent",
            use_tool_selector=True,  # Use centralized tool selection
            tool_selector_model="gpt-4o-mini",
            max_tools=20,
            verbose=True,
            use_colors=True
        )
        
        print_header("Test Completed Successfully")
        print_info("Check results/tools/ for detailed results JSON file")
        
    except Exception as e:
        from utils.printer import print_error
        print_error(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()

