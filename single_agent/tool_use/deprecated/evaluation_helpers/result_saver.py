"""
ResultSaver: Saves evaluation results to JSON files.

Single Responsibility: Only responsible for saving results to files.
"""
import json
from pathlib import Path
from typing import Dict, Any


class ResultSaver:
    """
    Saves evaluation results to JSON files.
    
    This class follows Single Responsibility Principle: it only handles
    file saving, nothing else.
    """

    def __init__(self, results_dir: Path):
        """
        Initialize the result saver.
        
        Args:
            results_dir: Directory where results will be saved
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        results_data: Dict[str, Any],
        filename: str
    ) -> Path:
        """
        Save results to a JSON file.
        
        Args:
            results_data: Complete results dictionary
            filename: Name of the output file (will be saved in results_dir)
            
        Returns:
            Path to the saved file
        """
        output_file = self.results_dir / filename
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        return output_file

    @staticmethod
    def generate_filename(
        framework_name: str,
        test_set: str,
        num_tools: int = None,
        num_queries: int = None
    ) -> str:
        """
        Generate a standardized filename for results.
        
        Args:
            framework_name: Name of the framework
            test_set: Name of the test set
            num_tools: Optional number of tools used
            num_queries: Optional number of queries evaluated
            
        Returns:
            Generated filename string
        """
        parts = [framework_name]
        if num_tools is not None:
            parts.append(f"{num_tools}tools")
        if num_queries is not None:
            parts.append(f"{num_queries}queries")
        parts.append(test_set)
        return "_".join(parts) + ".json"

