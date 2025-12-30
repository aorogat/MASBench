"""
HeuristicEvaluator: Fallback evaluation using simple heuristics.

Single Responsibility: Only responsible for heuristic evaluation when LLM evaluator is unavailable.
"""
from typing import Dict, Any


class HeuristicEvaluator:
    """
    Provides heuristic evaluation as a fallback when the official evaluator is unavailable.
    
    This class follows Single Responsibility Principle: it only handles
    heuristic evaluation logic, nothing else.
    """

    def __init__(self, api_scorer: Any, verbose: bool = True):
        """
        Initialize the heuristic evaluator.
        
        Args:
            api_scorer: APIScorer instance for calculating API call scores
            verbose: Whether to print verbose messages
        """
        self.api_scorer = api_scorer
        self.verbose = verbose

    def evaluate(
        self,
        query: Dict[str, Any],
        answer: Dict[str, Any],
        has_finish: bool,
        api_call_score: float
    ) -> Dict[str, Any]:
        """
        Evaluate an answer using heuristics.
        
        Args:
            query: Query dictionary
            answer: Answer dictionary
            has_finish: Whether answer has Finish call
            api_call_score: Pre-calculated API call score
            
        Returns:
            Dictionary with evaluation results
        """
        final_answer = answer.get("answer", {}).get("final_answer", "")
        query_text = query.get("query", "")
        
        # Simple heuristics
        if not final_answer or final_answer.strip() == "":
            sopr_score = 0.0
            answer_status = "Unsolved"
        elif any(word in final_answer.lower() for word in ["sorry", "cannot", "can't"]):
            sopr_score = 0.0
            answer_status = "Unsolved"
        elif len(final_answer) < 20:  # Very short answer
            sopr_score = 0.0
            answer_status = "Unsolved"
        elif has_finish and len(final_answer) > 50 and api_call_score >= 0.8:
            # Has finish, substantial answer, and most APIs called correctly
            sopr_score = 1.0
            answer_status = "Solved"
        elif has_finish and api_call_score > 0.0:
            # Has finish and some APIs called - use API score to determine partial completion
            # Map API call score to SoPR: 0.0-0.3 -> 0.0, 0.3-0.7 -> 0.5, 0.7-1.0 -> 1.0
            if api_call_score < 0.3:
                sopr_score = 0.0
                answer_status = "Unsolved"
            elif api_call_score < 0.7:
                sopr_score = 0.5
                answer_status = "Unsure"
            else:
                sopr_score = 1.0
                answer_status = "Solved"
        else:
            # Partial or unsure
            sopr_score = 0.5
            answer_status = "Unsure"
        
        return {
            "sopr_score": sopr_score,
            "answer_status": answer_status,
            "has_finish": has_finish,
            "api_call_score": api_call_score,
            "reason": f"Heuristic evaluation (official evaluator not available). API call score: {api_call_score:.2f}",
            "evaluation_method": "heuristic"
        }

