"""
StableToolBenchEvaluator: Main evaluator class using composition.

Single Responsibility: Orchestrates evaluation using composed components.
Dependency Inversion: Depends on abstractions (loader, scorer, heuristic) not concrete implementations.
Open/Closed: Open for extension (can add new evaluation strategies) but closed for modification.
"""
import os
import sys
import time
from typing import Dict, Any

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ENV_PATH = os.path.join(CURRENT_DIR, ".env")
    if os.path.exists(ENV_PATH):
        load_dotenv(ENV_PATH)
except ImportError:
    pass  # dotenv not available, use system environment variables

# Ensure StableToolBench is importable
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)
TOOLBENCH_PATH = os.path.join(STB_ROOT, "toolbench")
if TOOLBENCH_PATH not in sys.path:
    sys.path.insert(0, TOOLBENCH_PATH)

from .evaluator_loader import EvaluatorLoader
from .api_scorer import APIScorer
from .heuristic_evaluator import HeuristicEvaluator

# Import AnswerValidator - handle both relative and absolute imports
try:
    from ..utils.answer_validator import AnswerValidator
except (ImportError, ValueError):
    # Fallback for when running as standalone script
    # Need to be explicit to avoid importing from StableToolBench/toolbench/utils.py
    import os
    import sys
    import importlib.util
    EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(EVAL_DIR)
    # Use direct file path import to avoid conflicts
    answer_validator_path = os.path.join(PARENT_DIR, "utils", "answer_validator.py")
    spec = importlib.util.spec_from_file_location("answer_validator", answer_validator_path)
    answer_validator_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(answer_validator_module)
    AnswerValidator = answer_validator_module.AnswerValidator


class StableToolBenchEvaluator:
    """
    Main evaluator that orchestrates evaluation using composed components.
    
    This class follows:
    - Single Responsibility: Orchestrates evaluation, delegates to specialized components
    - Dependency Inversion: Depends on abstractions (loader, scorer, validator)
    - Open/Closed: Can be extended with new evaluation strategies without modification
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", verbose: bool = True):
        """
        Initialize the evaluator with composed components.
        
        Args:
            model_name: Name of the OpenAI model to use for evaluation
            verbose: Whether to print verbose messages
        """
        self.model_name = model_name
        self.verbose = verbose
        self.stabletoolbench_root = STB_ROOT
        
        # Compose components (Dependency Inversion Principle)
        self.loader = EvaluatorLoader(
            stabletoolbench_root=self.stabletoolbench_root,
            model_name=model_name,
            verbose=verbose
        )
        self.api_scorer = APIScorer()
        self.validator = AnswerValidator()
        self.heuristic_evaluator = HeuristicEvaluator(
            api_scorer=self.api_scorer,
            verbose=verbose
        )
        
        # Load the official evaluator (may be None if unavailable)
        self.evaluator = self.loader.load_evaluator()
        self.evaluation_available = self.evaluator is not None
    
    def check_has_finish(self, answer: Dict[str, Any]) -> bool:
        """
        Check if the answer contains a Finish call.
        
        Args:
            answer: Answer dictionary
            
        Returns:
            True if Finish call is found, False otherwise
        """
        return self.validator.check_has_finish(answer)
    
    def calculate_api_call_score(
        self,
        query: Dict[str, Any],
        answer: Dict[str, Any]
    ) -> float:
        """
        Calculate the proportion of correctly called APIs.
        
        Args:
            query: Query dictionary with gold APIs
            answer: Answer dictionary with answer_details
            
        Returns:
            Score from 0.0 to 1.0
        """
        # Extract gold APIs from query
        gold_apis = query.get("relevant APIs", [])
        if not gold_apis:
            gold_apis = query.get("relevant_apis", [])
        if not gold_apis:
            gold_apis = query.get("api_list", [])
        
        if not gold_apis:
            return 0.0
        
        # Use API scorer to calculate score
        return self.api_scorer.score(answer, gold_apis)
    
    def evaluate_answer(
        self,
        query: Dict[str, Any],
        answer: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate an answer using StableToolBench's evaluator or heuristics.
        
        Args:
            query: Query dictionary with 'query' text and other metadata
            answer: Answer dictionary with 'answer' containing 'final_answer' and 'answer_details'
        
        Returns:
            Dictionary with evaluation results including:
            - sopr_score: 0.0 (Unsolved), 0.5 (Unsure), or 1.0 (Solved)
            - answer_status: "Solved", "Unsure", or "Unsolved"
            - has_finish: Whether the answer contains a Finish call
            - api_call_score: Proportion of correctly called APIs
            - evaluation_time: Time taken for evaluation
            - evaluation_method: "official" or "heuristic"
        """
        start_time = time.time()
        
        # Check for Finish call
        has_finish = self.check_has_finish(answer)
        
        # Calculate API call score (used by both official and heuristic evaluation)
        api_call_score = self.calculate_api_call_score(query, answer)
        
        # Try official evaluation first
        if self.evaluation_available and self.evaluator is not None:
            try:
                return self._evaluate_with_official(
                    query, answer, has_finish, api_call_score, start_time
                )
            except (TypeError, AttributeError, KeyError, IndexError) as e:
                # API response structure issues
                if self.verbose:
                    print(f"[Evaluator] Error in official evaluation (API response issue): {e}")
                    print(f"[Evaluator] Falling back to heuristic evaluation")
            except Exception as e:
                if self.verbose:
                    import traceback
                    print(f"[Evaluator] Error in official evaluation: {e}")
                    print(f"[Evaluator] Traceback: {traceback.format_exc()}")
                    print(f"[Evaluator] Falling back to heuristic evaluation")
        
        # Fall back to heuristic evaluation
        return self._evaluate_with_heuristic(
            query, answer, has_finish, api_call_score, start_time
        )
    
    def _evaluate_with_official(
        self,
        query: Dict[str, Any],
        answer: Dict[str, Any],
        has_finish: bool,
        api_call_score: float,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Evaluate using the official StableToolBench evaluator.
        
        Args:
            query: Query dictionary
            answer: Answer dictionary
            has_finish: Whether answer has Finish call
            api_call_score: Pre-calculated API call score
            start_time: Evaluation start time
            
        Returns:
            Dictionary with evaluation results
        """
        # Prepare task description
        task_description = {
            "query": query.get("query", ""),
            "available_tools": query.get("available_tools", [])
        }
        
        # Prepare answer format
        answer_dict = {
            "final_answer": answer.get("answer", {}).get("final_answer", ""),
            "answer_details": answer.get("answer", {}).get("answer_details", [])
        }
        
        # Call official evaluator
        result = self.evaluator.check_is_solved(
            task_description,
            answer_dict,
            return_reason=True
        )
        
        # Handle both tuple and single value returns
        if isinstance(result, tuple):
            answer_status, reason = result
        else:
            answer_status = result
            reason = ""
        
        # Handle None case
        if answer_status is None:
            raise ValueError("Evaluator returned None for answer_status")
        
        # Convert AnswerStatus enum to string and score
        if isinstance(answer_status, str):
            status_str = answer_status
        else:
            status_str = answer_status.value
        
        if status_str == "Solved":
            sopr_score = 1.0
        elif status_str == "Unsure":
            sopr_score = 0.5
        else:  # Unsolved
            sopr_score = 0.0
        
        evaluation_time = time.time() - start_time
        
        return {
            "sopr_score": sopr_score,
            "answer_status": status_str,
            "has_finish": has_finish,
            "api_call_score": api_call_score,
            "reason": reason if reason else "",
            "evaluation_time": evaluation_time,
            "evaluation_method": "official"
        }
    
    def _evaluate_with_heuristic(
        self,
        query: Dict[str, Any],
        answer: Dict[str, Any],
        has_finish: bool,
        api_call_score: float,
        start_time: float
    ) -> Dict[str, Any]:
        """
        Evaluate using heuristic fallback.
        
        Args:
            query: Query dictionary
            answer: Answer dictionary
            has_finish: Whether answer has Finish call
            api_call_score: Pre-calculated API call score
            start_time: Evaluation start time
            
        Returns:
            Dictionary with evaluation results
        """
        result = self.heuristic_evaluator.evaluate(
            query, answer, has_finish, api_call_score
        )
        result["evaluation_time"] = time.time() - start_time
        return result
