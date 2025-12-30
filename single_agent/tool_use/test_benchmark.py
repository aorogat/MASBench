"""
Test script for StableToolBench evaluation (Updated to use new modular structure).
Creates fake answers (good, bad, partial) and evaluates them.
"""
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Insert at the beginning to ensure our modules are found before StableToolBench's
sys.path.insert(0, CURRENT_DIR)

# Use new modular structure
from evaluation import StableToolBenchEvaluator
from evaluation_helpers import ResultFormatter, ResultSaver, EvaluationPrinter

# Import QueryLoader directly from file to avoid conflict with StableToolBench/toolbench/utils.py
import importlib.util
query_loader_path = os.path.join(CURRENT_DIR, "utils", "query_loader.py")
spec = importlib.util.spec_from_file_location("query_loader", query_loader_path)
query_loader_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(query_loader_module)
QueryLoader = query_loader_module.QueryLoader

from tests.fake_answer_generator import FakeAnswerGenerator

# Paths
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
QUERY_INSTRUCTION_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_instruction")
QUERY_IDS_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_query_ids")
RESULTS_DIR = Path(__file__).parents[2] / "results" / "tools"

# Initialize components
query_loader = QueryLoader(
    query_instruction_dir=QUERY_INSTRUCTION_DIR,
    query_ids_dir=QUERY_IDS_DIR
)
answer_generator = FakeAnswerGenerator()
result_formatter = ResultFormatter()
result_saver = ResultSaver(RESULTS_DIR)
printer = EvaluationPrinter()


def main():
    printer.print_header("StableToolBench Test Evaluation")
    
    # Load queries using QueryLoader - limit to first 3 queries for testing
    printer.print_step(1, 4, "Loading queries...")
    MAX_QUERIES = 3
    queries = query_loader.load_queries(test_set="G1_instruction", max_queries=MAX_QUERIES)
    print(f"Loaded {len(queries)} queries (limited to first {MAX_QUERIES} for testing)")
    
    # Initialize evaluator
    printer.print_step(2, 4, "Initializing evaluator...")
    evaluator = StableToolBenchEvaluator(model_name="gpt-4o-mini", verbose=True)
    
    # Evaluate answers
    printer.print_step(3, 4, "Creating fake answers and evaluating...")
    all_results = []
    overall_start_time = time.time()
    
    for query in queries:
        query_id = query.get("query_id", "unknown")
        query_text = query.get("query", "")
        relevant_apis = QueryLoader.extract_gold_tool_calls(query)
        available_tools = QueryLoader.extract_available_tools(query)
        
        # Add available_tools to query for result formatting
        query["available_tools"] = available_tools
        
        printer.print_query_header(query_id, query_text, len(relevant_apis))
        
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
            fake_answer["available_tools"] = available_tools
            
            # Evaluate
            eval_result = evaluator.evaluate_answer(query, fake_answer)
            
            # Format result using ResultFormatter
            result = result_formatter.format_result(
                query=query,
                answer=fake_answer,
                eval_result=eval_result,
                gold_apis=relevant_apis
            )
            
            all_results.append(result)
            
            # Print result using EvaluationPrinter
            api_score = eval_result.get("api_call_score", 0.0)
            printer.print_evaluation_result(answer_quality, eval_result, api_score)
    
    overall_time = time.time() - overall_start_time
    
    # Calculate summary and format final results
    printer.print_step(4, 4, "Calculating summary statistics...")
    results_data = result_formatter.format_final_results(
        results=all_results,
        framework_name="Test_benchmark",
        config={
            "test_set": "G1_instruction",
            "num_queries": len(queries),
            "evaluator_name": "tooleval_gpt-3.5-turbo_default"
        },
        overall_time=overall_time,
        evaluator_model=evaluator.model_name,
        evaluation_method=all_results[0]["evaluation"]["evaluation_method"] if all_results else "unknown"
    )
    
    # Save results
    filename = ResultSaver.generate_filename(
        framework_name="Test_benchmark",
        test_set="G1_instruction",
        num_queries=len(queries)
    )
    output_file = result_saver.save(results_data, filename)
    
    # Print summary
    printer.print_summary(results_data["summary"], output_file)


if __name__ == "__main__":
    main()
