"""
Benchmark Runner for StableToolBench

This script:
1. Loads queries from StableToolBench benchmark files
2. Executes gold APIs via server to generate gold answers
3. Runs agent to generate system answers
4. Compares using StableToolBench's evaluation
5. Saves results to results/ folder
"""
import os
import sys
import json
import time
import requests
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional
from datetime import datetime

# Add current directory to path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from utils import QueryLoader
from evaluation import StableToolBenchEvaluator


def extract_called_apis_from_answer_details(answer: Dict[str, Any]) -> List[List[str]]:
    """
    Extract called APIs from answer_details (ExecutionGraph format).
    
    This parses the answer_details structure to find all tool calls
    (excluding "Finish" calls) and extracts [tool_name, api_name] pairs.
    
    Args:
        answer: Answer dictionary with 'answer' containing 'answer_details'
        
    Returns:
        List of [tool_name, api_name] pairs
    """
    called_apis = []
    answer_details = answer.get("answer", {}).get("answer_details", [])
    if not isinstance(answer_details, list):
        answer_details = [answer_details]
    
    def _recursive_extract(nodes: List[Dict[str, Any]]):
        """Recursively extract tool calls from nodes."""
        if not nodes:
            return
        for node in nodes:
            if node.get("role") == "tool":
                try:
                    message_content = node.get("message", "")
                    if isinstance(message_content, str):
                        tool_message = json.loads(message_content)
                        tool_name_full = tool_message.get("name", "")
                        
                        # Skip Finish calls
                        if tool_name_full == "Finish":
                            if node.get("next"):
                                _recursive_extract(node.get("next", []))
                            continue
                        
                        # Parse tool_name_api_name format
                        # Format is typically: "tool_name_api_name" or "tool_name api_name"
                        if "_" in tool_name_full:
                            parts = tool_name_full.split("_", 1)
                            if len(parts) == 2:
                                tool_name, api_name = parts[0], parts[1]
                                called_apis.append([tool_name, api_name])
                        elif " " in tool_name_full:
                            # Handle space-separated format
                            parts = tool_name_full.split(" ", 1)
                            if len(parts) == 2:
                                tool_name, api_name = parts[0], parts[1]
                                called_apis.append([tool_name, api_name])
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass  # Skip invalid nodes
            # Recursively check next nodes
            if node.get("next"):
                _recursive_extract(node.get("next", []))
    
    _recursive_extract(answer_details)
    return called_apis


def calculate_api_call_score(called_apis: List[List[str]], gold_apis: List[List[str]]) -> float:
    """
    Calculate API call score: proportion of gold APIs that were called.
    
    Args:
        called_apis: List of [tool_name, api_name] pairs from system answer
        gold_apis: List of [tool_name, api_name] pairs from query
        
    Returns:
        Score from 0.0 to 1.0
    """
    if not gold_apis:
        return 0.0
    
    # Convert to sets of tuples for comparison
    called_set = set(tuple(api) for api in called_apis)
    gold_set = set(tuple(api) for api in gold_apis)
    
    # Calculate intersection
    correct_calls = len(called_set.intersection(gold_set))
    return correct_calls / len(gold_set)


class GoldAnswerGenerator:
    """Generates gold answers by executing gold APIs via server."""
    
    def __init__(self, server_url: str = "http://localhost:8080/virtual"):
        self.server_url = server_url
    
    def _find_category(self, tool_name: str, available_tools: List[Dict]) -> str:
        """Find category for a tool from available_tools."""
        for tool in available_tools:
            if tool.get("tool_name") == tool_name:
                return tool.get("category", "Data")
        return "Data"  # Fallback
    
    def _call_api(self, category: str, tool_name: str, api_name: str, parameters: dict) -> dict:
        """Call API via server."""
        payload = {
            "category": category,
            "tool_name": tool_name,
            "api_name": api_name,
            "tool_input": parameters,
            "strip": "",
            "toolbench_key": "EMPTY"
        }
        try:
            response = requests.post(self.server_url, json=payload, timeout=60)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Server error: {response.status_code}", "response": ""}
        except Exception as e:
            return {"error": f"Request error: {str(e)}", "response": ""}
    
    def execute_gold_apis(
        self,
        query: Dict[str, Any],
        gold_apis: List[List[str]],
        default_parameters: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute gold APIs and return responses.
        
        Args:
            query: Query dictionary
            gold_apis: List of [tool_name, api_name] pairs
            default_parameters: Optional dict mapping (tool_name, api_name) to parameters
            
        Returns:
            List of API response dicts
        """
        available_tools = QueryLoader.extract_available_tools(query)
        responses = []
        
        for tool_name, api_name in gold_apis:
            # Find category
            category = self._find_category(tool_name, available_tools)
            
            # Get parameters (use defaults if provided, else empty)
            if default_parameters and (tool_name, api_name) in default_parameters:
                parameters = default_parameters[(tool_name, api_name)]
            else:
                # Try to extract default parameters from query's api_list
                parameters = self._extract_default_parameters(query, tool_name, api_name)
            
            # Call API
            response = self._call_api(category, tool_name, api_name, parameters)
            
            responses.append({
                "tool_name": tool_name,
                "api_name": api_name,
                "category": category,
                "parameters": parameters,
                "response": response
            })
        
        return responses
    
    def _extract_default_parameters(self, query: Dict[str, Any], tool_name: str, api_name: str) -> dict:
        """Extract default parameters from query's api_list."""
        api_list = query.get("api_list", [])
        for api_info in api_list:
            if (api_info.get("tool_name") == tool_name and 
                api_info.get("api_name") == api_name):
                # Extract default values from required_parameters
                params = {}
                for param in api_info.get("required_parameters", []):
                    if "default" in param:
                        params[param["name"]] = param["default"]
                return params
        return {}
    
    def generate_gold_answer(
        self,
        query: Dict[str, Any],
        api_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate gold answer in StableToolBench format.
        
        Args:
            query: Query dictionary
            api_responses: List of API responses from execute_gold_apis
            
        Returns:
            Answer dict in StableToolBench format
        """
        answer_details = []
        
        # Add API call nodes
        for api_resp in api_responses:
            tool_name = api_resp["tool_name"]
            api_name = api_resp["api_name"]
            response_data = api_resp["response"]
            
            node = {
                "role": "tool",
                "message": json.dumps({
                    "name": f"{tool_name}_{api_name}",
                    "arguments": api_resp.get("parameters", {}),
                    "response": response_data.get("response", "")
                }),
                "next": []
            }
            answer_details.append(node)
        
        # Combine responses into final answer
        final_answer_parts = []
        for api_resp in api_responses:
            if api_resp["response"].get("error") == "":
                resp_text = str(api_resp["response"].get("response", ""))
                if resp_text:
                    final_answer_parts.append(resp_text)
        
        final_answer = "\n".join(final_answer_parts) if final_answer_parts else "No response received"
        
        # Add Finish call
        finish_node = {
            "role": "tool",
            "message": json.dumps({
                "name": "Finish",
                "arguments": {
                    "return_type": "give_answer",
                    "final_answer": final_answer
                },
                "response": ""
            }),
            "next": []
        }
        answer_details.append(finish_node)
        
        return {
            "answer": {
                "final_answer": final_answer,
                "answer_details": answer_details
            }
        }


def run_benchmark(
    agent: Callable[[str], Dict[str, Any]],
    test_set: str = "G1_instruction",
    max_queries: Optional[int] = None,
    server_url: str = "http://localhost:8080/virtual",
    evaluator_model: str = "gpt-4o-mini",
    output_dir: Optional[str] = None,
    agent_name: str = "agent"
) -> Dict[str, Any]:
    """
    Run benchmark evaluation on an agent.
    
    Args:
        agent: Agent object with an `answer(query: str) -> Dict[str, Any]` method
               The answer dict should have 'answer' key with 'final_answer' and 'answer_details'
        test_set: Test set name (e.g., "G1_instruction")
        max_queries: Maximum number of queries to evaluate (None for all)
        server_url: URL of the running server
        evaluator_model: Model name for evaluation
        output_dir: Output directory for results (default: results/tools/)
        agent_name: Name of the agent (for output file naming)
    
    Returns:
        Dictionary with benchmark results
    """
    print("=" * 80)
    print("StableToolBench Benchmark Runner")
    print("=" * 80)
    
    # Setup paths
    stb_root = os.path.join(CURRENT_DIR, "StableToolBench")
    query_instruction_dir = os.path.join(stb_root, "solvable_queries", "test_instruction")
    
    if output_dir is None:
        output_dir = os.path.join(CURRENT_DIR, "..", "..", "results", "tools")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load queries
    print(f"\n[1/5] Loading queries from {test_set}...")
    query_loader = QueryLoader(query_instruction_dir=query_instruction_dir)
    queries = query_loader.load_queries(test_set=test_set, max_queries=max_queries)
    print(f"Loaded {len(queries)} queries")
    
    # Initialize components
    print(f"\n[2/5] Initializing components...")
    gold_generator = GoldAnswerGenerator(server_url=server_url)
    evaluator = StableToolBenchEvaluator(model_name=evaluator_model, verbose=True)
    
    # Run evaluation
    print(f"\n[3/5] Running benchmark evaluation...")
    all_results = []
    overall_start_time = time.time()
    
    for idx, query in enumerate(queries, 1):
        query_id = query.get("query_id", f"query_{idx}")
        query_text = query.get("query", "")
        gold_apis = QueryLoader.extract_gold_tool_calls(query)
        
        print(f"\n--- Query {idx}/{len(queries)}: {query_id} ---")
        print(f"Query: {query_text[:100]}...")
        print(f"Gold APIs: {len(gold_apis)}")
        
        query_start_time = time.time()
        result = {
            "query_id": query_id,
            "query_text": query_text,
            "gold_apis": gold_apis
        }
        
        try:
            # Generate gold answer
            print("  Generating gold answer...")
            gold_start = time.time()
            api_responses = gold_generator.execute_gold_apis(query, gold_apis)
            gold_answer = gold_generator.generate_gold_answer(query, api_responses)
            gold_time = time.time() - gold_start
            result["gold_answer_time"] = gold_time
            result["gold_api_responses"] = len(api_responses)
            
            # Generate system answer
            print("  Generating system answer...")
            system_start = time.time()
            system_answer = agent.answer(query_text)
            system_time = time.time() - system_start
            result["system_answer_time"] = system_time
            
            # Evaluate system answer
            print("  Evaluating system answer...")
            eval_start = time.time()
            eval_result = evaluator.evaluate_answer(query, system_answer)
            eval_time = time.time() - eval_start
            
            # Calculate API call score
            called_apis = extract_called_apis_from_answer_details(system_answer)
            api_call_score = calculate_api_call_score(called_apis, gold_apis)
            
            # Store results
            result.update({
                "gold_answer": {
                    "final_answer": gold_answer["answer"]["final_answer"] + "..." if len(gold_answer["answer"]["final_answer"]) > 200 else gold_answer["answer"]["final_answer"],
                    "has_finish": True
                },
                "system_answer": {
                    "final_answer": system_answer.get("answer", {}).get("final_answer", "") + "..." if len(system_answer.get("answer", {}).get("final_answer", "")) > 200 else system_answer.get("answer", {}).get("final_answer", ""),
                    "has_finish": eval_result["has_finish"]
                },
                "scores": {
                    "sopr_score": eval_result["sopr_score"],
                    "api_call_score": api_call_score,
                    "answer_status": eval_result["answer_status"]
                },
                "called_apis": called_apis,
                "evaluation": {
                    "has_finish": eval_result["has_finish"],
                    "evaluation_method": eval_result["evaluation_method"],
                    "reason": eval_result.get("reason", "")
                },
                "timing": {
                    "gold_answer_time": gold_time,
                    "system_answer_time": system_time,
                    "evaluation_time": eval_result["evaluation_time"],
                    "total_time": time.time() - query_start_time
                }
            })
            
            # Print result
            status_icon = "✓" if eval_result["sopr_score"] == 1.0 else "?" if eval_result["sopr_score"] == 0.5 else "✗"
            print(f"  Result: {status_icon} {eval_result['answer_status']:8s} | "
                  f"SoPR: {eval_result['sopr_score']:.1f} | "
                  f"API: {api_call_score:.2f} | "
                  f"Time: {result['timing']['total_time']:.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error processing query: {e}")
            import traceback
            traceback.print_exc()
            result["error"] = str(e)
            result["scores"] = {
                "sopr_score": 0.0,
                "api_call_score": 0.0,
                "answer_status": "Error"
            }
            result["called_apis"] = []
        
        all_results.append(result)
    
    overall_time = time.time() - overall_start_time
    
    # Calculate summary statistics
    print(f"\n[4/5] Calculating summary statistics...")
    total_queries = len(queries)
    solved_count = sum(1 for r in all_results if r.get("scores", {}).get("sopr_score", 0) == 1.0)
    unsure_count = sum(1 for r in all_results if r.get("scores", {}).get("sopr_score", 0) == 0.5)
    finish_count = sum(1 for r in all_results if r.get("evaluation", {}).get("has_finish", False))
    
    avg_sopr = sum(r.get("scores", {}).get("sopr_score", 0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    avg_api_score = sum(r.get("scores", {}).get("api_call_score", 0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    avg_gold_time = sum(r.get("timing", {}).get("gold_answer_time", 0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    avg_system_time = sum(r.get("timing", {}).get("system_answer_time", 0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    avg_eval_time = sum(r.get("timing", {}).get("evaluation_time", 0) for r in all_results) / total_queries if total_queries > 0 else 0.0
    
    # Prepare final results
    results_data = {
        "metadata": {
            "agent_name": agent_name,
            "test_set": test_set,
            "num_queries": total_queries,
            "evaluator_model": evaluator_model,
            "server_url": server_url,
            "timestamp": datetime.now().isoformat(),
            "overall_time": overall_time
        },
        "summary": {
            "total_queries": total_queries,
            "solved_count": solved_count,
            "solved_percentage": (solved_count / total_queries * 100) if total_queries > 0 else 0.0,
            "unsure_count": unsure_count,
            "unsure_percentage": (unsure_count / total_queries * 100) if total_queries > 0 else 0.0,
            "unsolved_count": total_queries - solved_count - unsure_count,
            "unsolved_percentage": ((total_queries - solved_count - unsure_count) / total_queries * 100) if total_queries > 0 else 0.0,
            "finish_count": finish_count,
            "finish_percentage": (finish_count / total_queries * 100) if total_queries > 0 else 0.0,
            "average_sopr_score": avg_sopr,
            "average_api_call_score": avg_api_score,
            "average_gold_answer_time": avg_gold_time,
            "average_system_answer_time": avg_system_time,
            "average_evaluation_time": avg_eval_time,
            "overall_time": overall_time
        },
        "results": all_results
    }
    
    # Save results
    print(f"\n[5/5] Saving results...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"{agent_name}_{test_set}_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Print summary
    print(f"\n{'=' * 80}")
    print("Benchmark Results Summary")
    print(f"{'=' * 80}")
    print(f"Agent:                    {agent_name}")
    print(f"Test Set:                 {test_set}")
    print(f"Total Queries:            {total_queries}")
    print(f"Solved:                   {solved_count} ({solved_count/total_queries*100:.1f}%)" if total_queries > 0 else "Solved: 0 (0.0%)")
    print(f"Unsure:                   {unsure_count} ({unsure_count/total_queries*100:.1f}%)" if total_queries > 0 else "Unsure: 0 (0.0%)")
    print(f"Unsolved:                 {total_queries - solved_count - unsure_count} ({(total_queries - solved_count - unsure_count)/total_queries*100:.1f}%)" if total_queries > 0 else "Unsolved: 0 (0.0%)")
    print(f"Has Finish Call:          {finish_count} ({finish_count/total_queries*100:.1f}%)" if total_queries > 0 else "Has Finish Call: 0 (0.0%)")
    print(f"Average SoPR Score:       {avg_sopr:.3f}")
    print(f"Average API Call Score:   {avg_api_score:.3f}")
    print(f"Average Gold Answer Time:  {avg_gold_time:.3f}s")
    print(f"Average System Answer Time: {avg_system_time:.3f}s")
    print(f"Average Evaluation Time:   {avg_eval_time:.3f}s")
    print(f"Overall Time:             {overall_time:.2f}s")
    print(f"\nResults saved to: {output_file}")
    print(f"{'=' * 80}")
    
    return results_data


if __name__ == "__main__":
    # Example usage with LLM-based agent (no tools)
    import os
    from openai import OpenAI
    
    # Load OpenAI API key from environment
    try:
        from dotenv import load_dotenv
        # Load .env from root folder
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ENV_PATH = os.path.join(ROOT_DIR, ".env")
        if os.path.exists(ENV_PATH):
            load_dotenv(ENV_PATH)
    except ImportError:
        pass
    
    class ExampleAgent:
        """Example agent that uses LLM and makes one API call for testing."""
        
        def __init__(self, model: str = "gpt-4o-mini", server_url: str = "http://localhost:8080/virtual"):
            """Initialize the agent with OpenAI client and server URL."""
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment. Please set it in .env file.")
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.server_url = server_url
        
        def answer(self, query: str) -> Dict[str, Any]:
            """
            Generate answer for query using LLM and make one API call for testing.
            
            Args:
                query: The query string
                
            Returns:
                Answer dict in StableToolBench format
            """
            try:
                # Make one API call for testing
                # This will test if API call score calculation works
                answer_details = []
                
                # Try to extract a search term from the query for API call
                # For testing, we'll make a call to TheClique Transfermarkt search
                # which is commonly used in queries about football players
                search_term = None
                if "Messi" in query or "messi" in query.lower():
                    search_term = "Lionel Messi"
                elif "Ronaldo" in query or "ronaldo" in query.lower():
                    search_term = "Cristiano Ronaldo"
                else:
                    # Extract first few words as search term
                    words = query.split()[:3]
                    search_term = " ".join(words) if words else "test"
                
                # Make a test API call to TheClique Transfermarkt search
                try:
                    api_payload = {
                        "category": "Data",
                        "tool_name": "TheClique",
                        "api_name": "Transfermarkt search",
                        "tool_input": {"query": search_term},
                        "strip": "",
                        "toolbench_key": "EMPTY"
                    }
                    api_response = requests.post(self.server_url, json=api_payload, timeout=10)
                    if api_response.status_code == 200:
                        api_result = api_response.json()
                        # Add API call to answer_details
                        answer_details.append({
                            "role": "tool",
                            "message": json.dumps({
                                "name": "TheClique_Transfermarkt search",
                                "arguments": {"query": search_term},
                                "response": api_result.get("response", "")
                            }),
                            "next": []
                        })
                except Exception as api_error:
                    # If API call fails, continue without it
                    pass
                
                # Call LLM to generate answer
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Answer the user's question based on the information provided."
                        },
                        {
                            "role": "user",
                            "content": query
                        }
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                final_answer = response.choices[0].message.content
                
                # Add Finish call
                answer_details.append({
                    "role": "tool",
                    "message": json.dumps({
                        "name": "Finish",
                        "arguments": {
                            "return_type": "give_answer",
                            "final_answer": final_answer
                        },
                        "response": ""
                    }),
                    "next": []
                })
                
                # Format as StableToolBench answer
                return {
                    "answer": {
                        "final_answer": final_answer,
                        "answer_details": answer_details
                    }
                }
            except Exception as e:
                # Return error answer if LLM call fails
                error_answer = f"I encountered an error while processing your query: {str(e)}"
                return {
                    "answer": {
                        "final_answer": error_answer,
                        "answer_details": [
                            {
                                "role": "tool",
                                "message": json.dumps({
                                    "name": "Finish",
                                    "arguments": {
                                        "return_type": "give_answer",
                                        "final_answer": error_answer
                                    },
                                    "response": ""
                                }),
                                "next": []
                            }
                        ]
                    }
                }
    
    # Run benchmark
    print("Initializing example agent...")
    agent = ExampleAgent(model="gpt-4o-mini")
    results = run_benchmark(
        agent=agent,
        test_set="G1_instruction",
        max_queries=3,
        agent_name="example_llm_agent"
    )

