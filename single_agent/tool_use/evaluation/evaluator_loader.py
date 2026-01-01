"""
EvaluatorLoader: Handles loading of StableToolBench evaluator.

Single Responsibility: Only responsible for loading and initializing the evaluator.
Dependency Inversion: Depends on abstractions (environment variables, file system).
"""
import os
import json
import tempfile
from typing import Optional, Any


class EvaluatorLoader:
    """
    Loads and initializes the StableToolBench evaluator.
    
    This class follows Single Responsibility Principle: it only handles
    evaluator loading, not evaluation logic.
    """

    def __init__(
        self,
        stabletoolbench_root: str,
        model_name: str = "gpt-4o-mini",
        verbose: bool = True
    ):
        """
        Initialize the loader.
        
        Args:
            stabletoolbench_root: Root directory of StableToolBench
            model_name: Name of the OpenAI model to use
            verbose: Whether to print verbose messages
        """
        self.stabletoolbench_root = stabletoolbench_root
        self.model_name = model_name
        self.verbose = verbose

    def setup_openai_key(self) -> Optional[str]:
        """
        Set up OpenAI API key from environment variable.
        Creates a temporary file (not in repository) for StableToolBench's evaluator.
        
        Returns:
            Path to the created temporary key file, or None if no key found
        """
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if openai_key:
            if self.verbose:
                print(f"[Evaluator] Found OpenAI key from environment variable (.env file)")
            # Create a temporary key file (not in repository) for the evaluator
            # StableToolBench's evaluator requires API_POOL_FILE to point to a JSON file
            temp_fd, temp_file = tempfile.mkstemp(suffix='.json', prefix='openai_key_', text=True)
            try:
                with os.fdopen(temp_fd, 'w') as f:
                    json.dump([{"api_key": openai_key, "api_base": None}], f)
                os.environ['API_POOL_FILE'] = temp_file
                if self.verbose:
                    print(f"[Evaluator] Created temporary key file (will be cleaned up automatically)")
                return temp_file
            except Exception as e:
                if self.verbose:
                    print(f"[Evaluator] Error creating temporary key file: {e}")
                return None
        else:
            if self.verbose:
                print(f"[Evaluator] Warning: No OpenAI key found in environment variable")
                print(f"[Evaluator] Please set OPENAI_API_KEY in .env file")
            return None

    def setup_model_name(self) -> None:
        """
        Set up model name via environment variable.
        """
        if self.model_name and self.model_name != "gpt-3.5-turbo-16k":
            os.environ['EVAL_MODEL'] = self.model_name
            if self.verbose:
                print(f"[Evaluator] Using model: {self.model_name} (set via EVAL_MODEL env var)")
        elif self.verbose:
            print(f"[Evaluator] Using default model from config.yaml")

    def load_evaluator(self) -> Optional[Any]:
        """
        Load the StableToolBench evaluator.
        
        Returns:
            Evaluator instance if successful, None otherwise
        """
        try:
            # Try to import StableToolBench evaluation modules (lazy import)
            from toolbench.tooleval.evaluators.registered_cls.rtl import (
                ReinforceToolLearningEvaluator
            )
        except ImportError as e:
            if self.verbose:
                print(f"[Evaluator] Could not import evaluation modules: {e}")
                print("[Evaluator] Will use heuristic evaluation")
            return None

        try:
            if self.verbose:
                print("[Evaluator] Loading evaluator...")
            
            evaluators_cfg_path = os.path.join(
                self.stabletoolbench_root, "toolbench", "tooleval", "evaluators"
            )
            
            # Set up OpenAI key
            key_file = self.setup_openai_key()
            if not key_file:
                return None
            
            # Set up model name
            self.setup_model_name()
            
            # Initialize the evaluator
            evaluator_name = "tooleval_gpt-3.5-turbo_default"
            cfg_path = os.path.join(evaluators_cfg_path, evaluator_name)
            
            if os.path.exists(cfg_path) and os.path.isdir(cfg_path):
                evaluator = ReinforceToolLearningEvaluator(cfg_path=cfg_path)
                if self.verbose:
                    print(f"[Evaluator] Loaded evaluator from {cfg_path}")
                    print(f"[Evaluator] Evaluator will call LLM ({self.model_name}) for evaluation")
                return evaluator
            else:
                if self.verbose:
                    print(f"[Evaluator] Warning: Config directory not found at {cfg_path}")
                    print(f"[Evaluator] Will use heuristic evaluation")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"[Evaluator] Error loading evaluator: {e}")
                print(f"[Evaluator] Will use heuristic evaluation")
            return None
