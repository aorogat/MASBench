"""
QAPipeline: Wrapper around StableToolBench's QAPipeline.

Single Responsibility: Provides a simplified interface to StableToolBench's QAPipeline.
Dependency Inversion: Depends on the model adapter abstraction, not concrete implementations.
"""
import os
import sys
from typing import Dict, Any

# Ensure StableToolBench is importable
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
if STB_ROOT not in sys.path:
    sys.path.insert(0, STB_ROOT)
TOOLBENCH_PATH = os.path.join(STB_ROOT, "toolbench")
if TOOLBENCH_PATH not in sys.path:
    sys.path.insert(0, TOOLBENCH_PATH)

try:
    from toolbench.inference.qa_pipeline import QAPipeline as STBQAPipeline
except ImportError:
    # Fallback if the import path is different
    try:
        from inference.qa_pipeline import QAPipeline as STBQAPipeline
    except ImportError:
        STBQAPipeline = None


class QAPipeline:
    """
    A wrapper around StableToolBench's QAPipeline for easier integration.
    
    This class follows the Open/Closed Principle: it's open for extension
    (can be subclassed) but closed for modification (doesn't change StableToolBench code).
    """

    def __init__(
        self,
        tool_dir: str,
        query_dir: str,
        model: Any,
        use_mirrorapi_cache: bool = True
    ):
        """
        Initialize the QA pipeline.
        
        Args:
            tool_dir: Directory containing tool specifications
            query_dir: Directory containing query files
            model: Model adapter instance (e.g., FrameworkAdapter)
            use_mirrorapi_cache: Whether to use MirrorAPI cache for simulation
        """
        if STBQAPipeline is None:
            raise ImportError(
                "Could not import StableToolBench QAPipeline. "
                "Please check your installation."
            )

        self.tool_dir = tool_dir
        self.query_dir = query_dir
        self.model = model
        self.use_mirrorapi_cache = use_mirrorapi_cache
        self.pipeline = STBQAPipeline(
            tool_root_dir=tool_dir,
            query_dir=query_dir,
            model=model,
            use_mirrorapi_cache=use_mirrorapi_cache,
        )

    def run(self) -> Dict[str, Any]:
        """
        Runs the evaluation pipeline.
        
        Returns:
            Dictionary containing evaluation results
        """
        print("Running StableToolBench evaluation...")
        results = self.pipeline.run()
        return results

