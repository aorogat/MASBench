"""
Test script for StableToolBench evaluation.
Creates fake answers (good, bad, partial) and evaluates them.
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

import sys
import os
# Add tests directory to path for relative imports
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TESTS_DIR)

from evaluation import StableToolBenchEvaluator
from utils import QueryLoader
from fake_answer_generator import FakeAnswerGenerator

# Paths
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
QUERY_INSTRUCTION_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_instruction")
QUERY_IDS_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_query_ids")
RESULTS_DIR = Path(CURRENT_DIR).parents[2] / "results" / "tools"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 80)
    print("StableToolBench Test Evaluation")
    print("=" * 80)
    
    # Load queries using QueryLoader
    print("\n[1/4] Loading queries...")
    query_loader = QueryLoader(
        query_instruction_dir=QUERY_INSTRUCTION_DIR,
        query_ids_dir=QUERY_IDS_DIR
    )
    queries = query_loader.load_queries(test_set="G1_instruction", max_queries=3)
    print(f"Loaded {len(queries)} queries")
    
    # Initialize evaluator
    print("\n[2/4] Initializing evaluator...")
    evaluator = StableToolBenchEvaluator(model_name="gpt-4o-mini", verbose=True)
    
    # Initialize fake answer generator
    answer_generator = FakeAnswerGenerator()
    
    # Evaluate answers
    print("\n[3/4] Creating fake answers and evaluating...")
    all_results = []
    overall_start_time = time.time()
    
    for query in queries:
        query_id = query.get("query_id", "unknown")
        query_text = query.get("query", "")
        relevant_apis = QueryLoader.extract_gold_tool_calls(query)
        
        print(f"\n--- Query ID: {query_id} ---")
        print(f"Query: {query_text[:100]}...")
        print(f"Gold APIs: {len(relevant_apis)}")
        
        # Create three types of answers for testing
        for answer_quality in ["good", "bad", "partial"]:
            # Create fake answer
            fake_answer = answer_generator.create(
                query_id=str(query_id),
                query_text=query_text,
                relevant_apis=relevant_apis,
                include_finish=True,
                answer_quality=answer_quality
            )
            
            # Update fake_answer with available_tools from query
            fake_answer["available_tools"] = query.get("available_tools", [])
            
            # Evaluate
            eval_result = evaluator.evaluate_answer(query, fake_answer)
            
            # Prepare result
            result = {
                "query_id": query_id,
                "query_text": query_text,
                "gold_answer": {
                    "relevant_apis": relevant_apis,
                    "available_tools": query.get("available_tools", [])
                },
                "system_answer": {
                    "final_answer": fake_answer["answer"]["final_answer"],
                    "answer_details": fake_answer["answer"]["answer_details"]
                },
                "scores": {
                    "sopr_score": eval_result["sopr_score"],
                    "api_call_score": eval_result.get("api_call_score", 0.0)
                },
                "evaluation": {
                    "answer_status": eval_result["answer_status"],
                    "has_finish": eval_result["has_finish"],
                    "evaluation_method": eval_result["evaluation_method"],
                    "reason": eval_result.get("reason", "")
                },
                "timing": {
                    "evaluation_time": eval_result["evaluation_time"]
                }
            }
            
            all_results.append(result)
            
            # Print result
            status_icon = "✓" if eval_result["sopr_score"] == 1.0 else "?" if eval_result["sopr_score"] == 0.5 else "✗"
            finish_icon = "✓" if eval_result["has_finish"] else "✗"
            api_score = eval_result.get("api_call_score", 0.0)
            print(f"  [{answer_quality:6s}] Status: {status_icon} {eval_result['answer_status']:8s} | "
                  f"Finish: {finish_icon} | SoPR: {eval_result['sopr_score']:.1f} | "
                  f"API Score: {api_score:.2f} | Time: {eval_result['evaluation_time']:.4f}s")
    
    overall_time = time.time() - overall_start_time
    
    # Calculate summary statistics
    print("\n[4/4] Calculating summary statistics...")
    total_evaluations = len(all_results)
    solved_count = sum(1 for r in all_results if r["scores"]["sopr_score"] == 1.0)
    finish_count = sum(1 for r in all_results if r["evaluation"]["has_finish"])
    avg_sopr = sum(r["scores"]["sopr_score"] for r in all_results) / total_evaluations if total_evaluations > 0 else 0.0
    avg_api_call = sum(r["scores"]["api_call_score"] for r in all_results) / total_evaluations if total_evaluations > 0 else 0.0
    total_eval_time = sum(r["timing"]["evaluation_time"] for r in all_results)
    avg_eval_time = total_eval_time / total_evaluations if total_evaluations > 0 else 0.0
    
    # Prepare final results
    results_data = {
        "metadata": {
            "framework": "Test_benchmark",
            "config": {
                "test_set": "G1_instruction",
                "num_queries": len(queries),
                "evaluator_name": "tooleval_gpt-3.5-turbo_default",
                "evaluator_model": evaluator.model_name,
                "evaluation_method": all_results[0]["evaluation"]["evaluation_method"] if all_results else "unknown"
            },
            "overall_time": overall_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": {
            "total_queries": len(queries),
            "total_evaluations": total_evaluations,
            "solved_count": solved_count,
            "solved_percentage": (solved_count / total_evaluations * 100) if total_evaluations > 0 else 0.0,
            "finish_count": finish_count,
            "finish_percentage": (finish_count / total_evaluations * 100) if total_evaluations > 0 else 0.0,
            "average_sopr_score": avg_sopr,
            "average_api_call_score": avg_api_call,
            "total_evaluation_time": total_eval_time,
            "average_evaluation_time": avg_eval_time
        },
        "results": all_results
    }
    
    # Save results
    output_file = RESULTS_DIR / "Test_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print("Evaluation Summary")
    print(f"{'=' * 80}")
    print(f"Total Queries:        {len(queries)}")
    print(f"Total Evaluations:    {total_evaluations}")
    print(f"Solved:               {solved_count} ({solved_count/total_evaluations*100:.1f}%)" if total_evaluations > 0 else "Solved: 0 (0.0%)")
    print(f"Has Finish Call:      {finish_count} ({finish_count/total_evaluations*100:.1f}%)" if total_evaluations > 0 else "Has Finish Call: 0 (0.0%)")
    print(f"Average SoPR Score:   {avg_sopr:.3f}")
    print(f"Average API Call Score: {avg_api_call:.3f}")
    print(f"Total Time:           {overall_time:.4f}s")
    print(f"Average Eval Time:    {avg_eval_time:.6f}s")
    print(f"\nResults saved to: {output_file}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

