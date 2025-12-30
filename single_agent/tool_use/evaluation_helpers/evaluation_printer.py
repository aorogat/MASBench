"""
EvaluationPrinter: Prints evaluation progress and results to terminal.

Single Responsibility: Only responsible for printing to terminal.
"""
from pathlib import Path
from typing import Dict, Any, List


class EvaluationPrinter:
    """
    Prints evaluation progress and results to terminal.
    
    This class follows Single Responsibility Principle: it only handles
    terminal output, nothing else.
    """

    @staticmethod
    def print_header(title: str = "StableToolBench Evaluation"):
        """Print evaluation header."""
        print("=" * 80)
        print(title)
        print("=" * 80)

    @staticmethod
    def print_step(step_num: int, total_steps: int, message: str):
        """Print a step message."""
        print(f"\n[{step_num}/{total_steps}] {message}")

    @staticmethod
    def print_query_header(query_id: str, query_text: str, num_gold_apis: int):
        """Print query information header."""
        print(f"\n--- Query ID: {query_id} ---")
        print(f"Query: {query_text[:100]}...")
        print(f"Gold APIs: {num_gold_apis}")

    @staticmethod
    def print_evaluation_result(
        answer_quality: str,
        eval_result: Dict[str, Any],
        api_score: float
    ):
        """
        Print a single evaluation result.
        
        Args:
            answer_quality: Quality label (e.g., "good", "bad", "partial")
            eval_result: Evaluation result dictionary
            api_score: API call score
        """
        status_icon = "✓" if eval_result["sopr_score"] == 1.0 else "?" if eval_result["sopr_score"] == 0.5 else "✗"
        finish_icon = "✓" if eval_result["has_finish"] else "✗"
        print(f"  [{answer_quality:6s}] Status: {status_icon} {eval_result['answer_status']:8s} | "
              f"Finish: {finish_icon} | SoPR: {eval_result['sopr_score']:.1f} | "
              f"API Score: {api_score:.2f} | Time: {eval_result['evaluation_time']:.4f}s")

    @staticmethod
    def print_summary(summary: Dict[str, Any], output_file: Path = None):
        """
        Print evaluation summary.
        
        Args:
            summary: Summary dictionary from ResultFormatter
            output_file: Optional path to output file
        """
        print(f"\n{'=' * 80}")
        print("Evaluation Summary")
        print(f"{'=' * 80}")
        print(f"Total Queries:        {summary['total_queries']}")
        print(f"Total Evaluations:    {summary['total_evaluations']}")
        
        if summary['total_evaluations'] > 0:
            print(f"Solved:               {summary['solved_count']} ({summary['solved_percentage']:.1f}%)")
            print(f"Has Finish Call:      {summary['finish_count']} ({summary['finish_percentage']:.1f}%)")
        else:
            print(f"Solved:               0 (0.0%)")
            print(f"Has Finish Call:      0 (0.0%)")
        
        print(f"Average SoPR Score:   {summary['average_sopr_score']:.3f}")
        print(f"Average API Call Score: {summary['average_api_call_score']:.3f}")
        print(f"Total Time:           {summary.get('total_evaluation_time', 0.0):.4f}s")
        print(f"Average Eval Time:    {summary['average_evaluation_time']:.6f}s")
        
        if output_file:
            print(f"\nResults saved to: {output_file}")
        print(f"{'=' * 80}")

