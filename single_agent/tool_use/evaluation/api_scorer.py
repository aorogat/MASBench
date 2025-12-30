"""
APIScorer: Calculates API call scores.

Single Responsibility: Only responsible for calculating API call scores.
"""
import json
from typing import Dict, Any, List, Set, Tuple


class APIScorer:
    """
    Calculates the proportion of correctly called APIs.
    
    This class follows Single Responsibility Principle: it only handles
    API call scoring, nothing else.
    """

    @staticmethod
    def extract_called_apis(answer_details: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
        """
        Extract called APIs from answer_details structure.
        
        Args:
            answer_details: Answer details structure from StableToolBench format
            
        Returns:
            Set of (tool_name, api_name) tuples
        """
        called_apis = set()
        
        def _extract_from_nodes(nodes: List[Dict[str, Any]]) -> None:
            """Recursively extract API calls from nodes."""
            if not nodes:
                return
            for node in nodes:
                if node.get("role") == "tool":
                    try:
                        message_content = node.get("message", "")
                        if isinstance(message_content, str):
                            tool_message = json.loads(message_content)
                            tool_name = tool_message.get("name", "")
                            if tool_name and tool_name != "Finish":
                                # Parse tool_name format: "ToolName_API_name"
                                parts = tool_name.split("_", 1)
                                if len(parts) == 2:
                                    called_apis.add(tuple(parts))
                    except (json.JSONDecodeError, KeyError):
                        pass
                _extract_from_nodes(node.get("next", []))
        
        _extract_from_nodes(answer_details)
        return called_apis

    @staticmethod
    def normalize_gold_apis(gold_apis: List[List[str]]) -> Set[Tuple[str, str]]:
        """
        Normalize gold APIs to a set of tuples.
        
        Args:
            gold_apis: List of [tool_name, api_name] pairs
            
        Returns:
            Set of (tool_name, api_name) tuples
        """
        gold_apis_set = set()
        for api_info in gold_apis:
            if isinstance(api_info, list) and len(api_info) >= 2:
                gold_apis_set.add(tuple(api_info[:2]))
        return gold_apis_set

    @staticmethod
    def calculate_score(
        called_apis: Set[Tuple[str, str]],
        gold_apis: Set[Tuple[str, str]]
    ) -> float:
        """
        Calculate the proportion of correctly called APIs.
        
        Args:
            called_apis: Set of called API tuples
            gold_apis: Set of gold API tuples
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not gold_apis:
            return 0.0
        
        correct_calls = len(called_apis.intersection(gold_apis))
        return correct_calls / len(gold_apis)

    def score(
        self,
        answer: Dict[str, Any],
        gold_apis: List[List[str]]
    ) -> float:
        """
        Calculate API call score for an answer.
        
        Args:
            answer: Answer dictionary with 'answer_details'
            gold_apis: List of gold API [tool_name, api_name] pairs
            
        Returns:
            Score from 0.0 to 1.0
        """
        if not gold_apis:
            return 0.0
        
        answer_details = answer.get("answer", {}).get("answer_details", [])
        called_apis = self.extract_called_apis(answer_details)
        gold_apis_set = self.normalize_gold_apis(gold_apis)
        
        return self.calculate_score(called_apis, gold_apis_set)

