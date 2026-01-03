"""
Run all frameworks with different k values to test scalability.

This script:
1. Tests all implemented frameworks
2. Runs with different k values (max_tools): 8, 16, 32, 64, 128, 256, 512, 1024, 1500
3. Identifies where frameworks fail due to tool limits
4. Documents scalability characteristics of each framework

Usage:
    python single_agent/tool_use/run_all.py
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

# Load environment variables
try:
    from dotenv import load_dotenv
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ENV_PATH = os.path.join(ROOT_DIR, ".env")
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
        print(f"Loaded .env file from: {ENV_PATH}")
except ImportError:
    pass

from run_benchmark import run_benchmark
from utils.printer import print_header, print_step, print_success, print_error, print_warning, print_info


# K values to test
K_VALUES = [8, 16, 32, 64, 128, 256, 512, 1024, 1500]

# Test configuration
TEST_SET = "G1_instruction"
MAX_QUERIES = 3  # Small number for scalability testing
SERVER_URL = "http://localhost:8080/virtual"
EVALUATOR_MODEL = "gpt-4o-mini"
TOOL_SELECTOR_MODEL = "gpt-4o-mini"


def get_available_frameworks() -> Dict[str, Any]:
    """
    Get list of available framework agents.
    
    Returns:
        Dictionary mapping framework names to agent classes
    """
    frameworks = {}
    
    # LangGraph
    try:
        from agents.langgraph import LangGraphAgent
        frameworks['langgraph'] = {
            'class': LangGraphAgent,
            'name': 'LangGraph',
            'params': {
                'model': 'gpt-4o-mini',
                'server_url': SERVER_URL,
                'temperature': 0.0,
                'verbose': False  # Reduce verbosity for batch runs
            }
        }
    except ImportError as e:
        print_warning(f"LangGraph not available: {e}")
    
    # CrewAI
    try:
        from agents.crewai import CrewAIAgent
        frameworks['crewai'] = {
            'class': CrewAIAgent,
            'name': 'CrewAI',
            'params': {
                'model': 'gpt-4o-mini',
                'server_url': SERVER_URL,
                'temperature': 0.0,
                'verbose': False  # Reduce verbosity for batch runs
            }
        }
    except ImportError as e:
        print_warning(f"CrewAI not available: {e}")
    
    # AutoGen
    try:
        from agents.autogen import AutoGenAgent
        frameworks['autogen'] = {
            'class': AutoGenAgent,
            'name': 'AutoGen',
            'params': {
                'model': 'gpt-4o-mini',
                'server_url': SERVER_URL,
                'temperature': 0.0,
                'verbose': False  # Reduce verbosity for batch runs
            }
        }
    except ImportError as e:
        print_warning(f"AutoGen not available: {e}")
    
    # OpenAI Agents SDK
    try:
        from agents.openai_sdk import OpenAISDKAgent
        frameworks['openai_sdk'] = {
            'class': OpenAISDKAgent,
            'name': 'OpenAI Agents SDK',
            'params': {
                'model': 'gpt-4o-mini',
                'server_url': SERVER_URL,
                'temperature': 0.0,
                'verbose': False  # Reduce verbosity for batch runs
            }
        }
    except ImportError as e:
        print_warning(f"OpenAI Agents SDK not available: {e}")
    
    # OpenAI Agents SDK (when implemented)
    # try:
    #     from agents.openai import OpenAIAgentsAgent
    #     frameworks['openai'] = {
    #         'class': OpenAIAgentsAgent,
    #         'name': 'OpenAI Agents SDK',
    #         'params': {...}
    #     }
    # except ImportError:
    #     pass
    
    # Agno (when implemented)
    # try:
    #     from agents.agno import AgnoAgent
    #     frameworks['agno'] = {
    #         'class': AgnoAgent,
    #         'name': 'Agno',
    #         'params': {...}
    #     }
    # except ImportError:
    #     pass
    
    # OpenAgents (when implemented)
    # try:
    #     from agents.openagents import OpenAgentsAgent
    #     frameworks['openagents'] = {
    #         'class': OpenAgentsAgent,
    #         'name': 'OpenAgents',
    #         'params': {...}
    #     }
    # except ImportError:
    #     pass
    
    return frameworks


def test_framework_scalability(
    framework_name: str,
    framework_info: Dict[str, Any],
    k_values: List[int],
    test_set: str = TEST_SET,
    max_queries: int = MAX_QUERIES
) -> Dict[str, Any]:
    """
    Test a framework with different k values.
    
    Args:
        framework_name: Name of the framework (e.g., 'langgraph')
        framework_info: Framework information dict with 'class' and 'params'
        k_values: List of k values to test
        test_set: Test set name
        max_queries: Maximum number of queries to test
        
    Returns:
        Dictionary with test results for each k value
    """
    results = {
        'framework': framework_name,
        'framework_display_name': framework_info['name'],
        'k_results': {}
    }
    
    print_header(f"Testing {framework_info['name']} Scalability")
    
    for k in k_values:
        print_step(1, len(k_values), f"Testing with k={k}")
        
        try:
            # Create agent instance
            agent_class = framework_info['class']
            agent_params = framework_info['params'].copy()
            agent = agent_class(**agent_params)
            
            # Run benchmark
            start_time = time.time()
            benchmark_results = run_benchmark(
                agent=agent,
                test_set=test_set,
                max_queries=max_queries,
                server_url=SERVER_URL,
                evaluator_model=EVALUATOR_MODEL,
                agent_name=f"{framework_name}_k{k}",
                use_tool_selector=True,
                tool_selector_model=TOOL_SELECTOR_MODEL,
                max_tools=k,
                verbose=False,  # Reduce output for batch runs
                use_colors=True
            )
            elapsed_time = time.time() - start_time
            
            # Store results
            results['k_results'][k] = {
                'status': 'success',
                'total_queries': benchmark_results['summary']['total_queries'],
                'solved_count': benchmark_results['summary']['solved_count'],
                'average_sopr_score': benchmark_results['summary']['average_sopr_score'],
                'average_api_call_score': benchmark_results['summary']['average_api_call_score'],
                'elapsed_time': elapsed_time,
                'error': None
            }
            
            print_success(f"k={k}: Success (SoPR: {benchmark_results['summary']['average_sopr_score']:.3f}, "
                        f"API: {benchmark_results['summary']['average_api_call_score']:.3f})")
            
        except Exception as e:
            error_msg = str(e)
            results['k_results'][k] = {
                'status': 'failed',
                'error': error_msg,
                'elapsed_time': None
            }
            
            print_error(f"k={k}: Failed - {error_msg}")
            
            # Check if it's an OpenAI tool limit error
            if '128' in error_msg or 'too long' in error_msg.lower() or 'BadRequestError' in error_msg:
                print_warning(f"  Likely hit OpenAI tool limit (128 tools)")
                # Don't test larger k values if we hit the limit
                print_info(f"  Skipping larger k values for {framework_name}")
                break
    
    return results


def print_scalability_summary(all_results: Dict[str, Dict[str, Any]]):
    """
    Print summary of scalability test results.
    
    Args:
        all_results: Dictionary mapping framework names to their test results
    """
    print_header("Scalability Test Summary")
    
    for framework_name, results in all_results.items():
        framework_display = results['framework_display_name']
        k_results = results['k_results']
        
        print_info(f"\n{framework_display}:")
        
        # Find max successful k
        successful_ks = [k for k, r in k_results.items() if r['status'] == 'success']
        failed_ks = [k for k, r in k_results.items() if r['status'] == 'failed']
        
        if successful_ks:
            max_k = max(successful_ks)
            print_success(f"  ✓ Max successful k: {max_k}")
            
            # Show performance at different k values
            for k in [8, 64, 128, 256, 512, 1024, 1500]:
                if k in successful_ks:
                    r = k_results[k]
                    print_info(f"    k={k:4d}: SoPR={r['average_sopr_score']:.3f}, "
                              f"API={r['average_api_call_score']:.3f}, "
                              f"Time={r['elapsed_time']:.1f}s")
        else:
            print_error(f"  ✗ No successful runs")
        
        if failed_ks:
            first_failure = min(failed_ks)
            print_error(f"  ✗ First failure at k={first_failure}")
            
            # Show error message
            if first_failure in k_results:
                error = k_results[first_failure]['error']
                if '128' in error or 'too long' in error.lower():
                    print_warning(f"    → Likely OpenAI tool limit (128 tools)")
                else:
                    print_info(f"    → Error: {error[:100]}...")


def save_scalability_results(all_results: Dict[str, Dict[str, Any]], output_dir: Optional[str] = None):
    """
    Save scalability test results to JSON file.
    
    Args:
        all_results: Dictionary mapping framework names to their test results
        output_dir: Output directory (default: results/tools/)
    """
    import json
    
    if output_dir is None:
        output_dir = os.path.join(CURRENT_DIR, "..", "..", "results", "tools")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results_data = {
        'metadata': {
            'test_set': TEST_SET,
            'max_queries': MAX_QUERIES,
            'k_values_tested': K_VALUES,
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')
        },
        'frameworks': all_results
    }
    
    output_file = output_dir / "scalability_test_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print_success(f"Scalability results saved to {output_file}")


def main():
    """Main function to run scalability tests on all frameworks."""
    print_header("Framework Scalability Testing")
    
    print_info(f"Test Configuration:")
    print_info(f"  Test Set: {TEST_SET}")
    print_info(f"  Max Queries: {MAX_QUERIES}")
    print_info(f"  K Values: {K_VALUES}")
    print_info(f"  Server URL: {SERVER_URL}")
    print()
    
    # Check if server is running
    print_warning("Make sure the server is running at http://localhost:8080/virtual")
    print_info("Start it with: python single_agent/tool_use/StableToolBench/server/main.py")
    print()
    
    # Get available frameworks
    frameworks = get_available_frameworks()
    
    if not frameworks:
        print_error("No frameworks available. Please implement at least one framework.")
        return
    
    print_info(f"Found {len(frameworks)} framework(s): {', '.join(frameworks.keys())}")
    print()
    
    # Test each framework
    all_results = {}
    
    for framework_name, framework_info in frameworks.items():
        try:
            results = test_framework_scalability(
                framework_name=framework_name,
                framework_info=framework_info,
                k_values=K_VALUES,
                test_set=TEST_SET,
                max_queries=MAX_QUERIES
            )
            all_results[framework_name] = results
        except Exception as e:
            print_error(f"Failed to test {framework_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if all_results:
        print_scalability_summary(all_results)
        save_scalability_results(all_results)
    else:
        print_error("No results to summarize")


if __name__ == "__main__":
    main()

