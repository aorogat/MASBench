"""
AnswerValidator: Validates answer structure and checks for required elements.

Single Responsibility: Only responsible for validating answer structure.
"""
import json
from typing import Dict, Any, List


class AnswerValidator:
    """
    Validates answers according to StableToolBench requirements.
    
    This class follows Single Responsibility Principle: it only handles
    answer validation, nothing else.
    """

    @staticmethod
    def check_has_finish(answer: Dict[str, Any]) -> bool:
        """
        Check if the answer contains a Finish call (required by StableToolBench).
        
        This function recursively searches through the 'answer_details' structure
        to find a tool call named 'Finish'.
        
        Args:
            answer: Answer dictionary with 'answer_details'
            
        Returns:
            True if Finish call is found, False otherwise
        """
        answer_details = answer.get("answer_details", [])
        if not isinstance(answer_details, list):
            answer_details = [answer_details]

        def _recursive_check(nodes: List[Dict[str, Any]]) -> bool:
            """Recursively check nodes for Finish call."""
            if not nodes:
                return False
            for node in nodes:
                if node.get("role") == "tool":
                    try:
                        message_content = node.get("message", "")
                        if isinstance(message_content, str):
                            tool_message = json.loads(message_content)
                            if tool_message.get("name") == "Finish":
                                return True
                    except json.JSONDecodeError:
                        pass  # Not a JSON string, continue
                if _recursive_check(node.get("next", [])):
                    return True
            return False

        return _recursive_check(answer_details)

    @staticmethod
    def validate_answer_structure(answer: Dict[str, Any]) -> bool:
        """
        Validate that answer has the required structure.
        
        Args:
            answer: Answer dictionary to validate
            
        Returns:
            True if structure is valid, False otherwise
        """
        required_keys = ["answer"]
        if not all(key in answer for key in required_keys):
            return False
        
        answer_data = answer.get("answer", {})
        required_answer_keys = ["final_answer", "answer_details"]
        if not all(key in answer_data for key in required_answer_keys):
            return False
        
        return True

