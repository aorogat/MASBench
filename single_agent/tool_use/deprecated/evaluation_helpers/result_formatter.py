"""
ResultFormatter: Formats evaluation results and calculates summary statistics.

Single Responsibility: Only responsible for formatting results and calculating statistics.
"""
from typing import Dict, Any, List
import time


class ResultFormatter:
    """
    Formats evaluation results and calculates summary statistics.
    
    This class follows Single Responsibility Principle: it only handles
    result formatting and statistics, nothing else.
    """

    @staticmethod
    def format_result(
        query: Dict[str, Any],
        answer: Dict[str, Any],
        eval_result: Dict[str, Any],
        gold_apis: List[List[str]]
    ) -> Dict[str, Any]:
        """
        Format a single evaluation result.
        
        Args:
            query: Query dictionary
            answer: Answer dictionary
            eval_result: Evaluation result dictionary
            gold_apis: List of gold API [tool_name, api_name] pairs
            
        Returns:
            Formatted result dictionary
        """
        return {
            "query_id": query.get("query_id", "unknown"),
            "query_text": query.get("query", ""),
            "gold_answer": {
                "relevant_apis": gold_apis,
                "available_tools": query.get("available_tools", [])  # Will be populated by caller if needed
            },
            "system_answer": {
                "final_answer": answer.get("answer", {}).get("final_answer", ""),
                "answer_details": answer.get("answer", {}).get("answer_details", [])
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

    @staticmethod
    def calculate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate summary statistics from results.
        
        Args:
            results: List of formatted result dictionaries
            
        Returns:
            Dictionary with summary statistics
        """
        total_evaluations = len(results)
        if total_evaluations == 0:
            return {
                "total_queries": 0,
                "total_evaluations": 0,
                "solved_count": 0,
                "solved_percentage": 0.0,
                "finish_count": 0,
                "finish_percentage": 0.0,
                "average_sopr_score": 0.0,
                "average_api_call_score": 0.0,
                "total_evaluation_time": 0.0,
                "average_evaluation_time": 0.0
            }
        
        solved_count = sum(1 for r in results if r["scores"]["sopr_score"] == 1.0)
        finish_count = sum(1 for r in results if r["evaluation"]["has_finish"])
        avg_sopr = sum(r["scores"]["sopr_score"] for r in results) / total_evaluations
        avg_api_call = sum(r["scores"]["api_call_score"] for r in results) / total_evaluations
        total_eval_time = sum(r["timing"]["evaluation_time"] for r in results)
        avg_eval_time = total_eval_time / total_evaluations
        
        # Count unique queries
        unique_query_ids = set(r["query_id"] for r in results)
        total_queries = len(unique_query_ids)
        
        return {
            "total_queries": total_queries,
            "total_evaluations": total_evaluations,
            "solved_count": solved_count,
            "solved_percentage": (solved_count / total_evaluations * 100) if total_evaluations > 0 else 0.0,
            "finish_count": finish_count,
            "finish_percentage": (finish_count / total_evaluations * 100) if total_evaluations > 0 else 0.0,
            "average_sopr_score": avg_sopr,
            "average_api_call_score": avg_api_call,
            "total_evaluation_time": total_eval_time,
            "average_evaluation_time": avg_eval_time
        }

    @staticmethod
    def format_final_results(
        results: List[Dict[str, Any]],
        framework_name: str,
        config: Dict[str, Any],
        overall_time: float,
        evaluator_model: str,
        evaluation_method: str
    ) -> Dict[str, Any]:
        """
        Format final results with metadata and summary.
        
        Args:
            results: List of formatted result dictionaries
            framework_name: Name of the framework being evaluated
            config: Configuration dictionary
            overall_time: Total time taken
            evaluator_model: Model used for evaluation
            evaluation_method: Method used for evaluation
            
        Returns:
            Complete results dictionary ready for saving
        """
        summary = ResultFormatter.calculate_summary(results)
        
        return {
            "metadata": {
                "framework": framework_name,
                "config": {
                    **config,
                    "evaluator_model": evaluator_model,
                    "evaluation_method": evaluation_method
                },
                "overall_time": overall_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "summary": summary,
            "results": results
        }

