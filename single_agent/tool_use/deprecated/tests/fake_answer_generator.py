"""
FakeAnswerGenerator: Generates fake answers for testing.

Single Responsibility: Only responsible for generating fake test answers.
"""
import json
from typing import Dict, Any, List


class FakeAnswerGenerator:
    """
    Generates fake answers for testing evaluation.
    
    This class follows Single Responsibility Principle: it only handles
    fake answer generation, nothing else.
    """

    @staticmethod
    def create(
        query_id: str,
        query_text: str,
        relevant_apis: List,
        include_finish: bool = True,
        answer_quality: str = "good"  # "good", "bad", "partial"
    ) -> Dict[str, Any]:
        """
        Create a fake answer for testing evaluation.
        
        Args:
            query_id: Query ID string
            query_text: Query text
            relevant_apis: List of [tool_name, api_name] pairs
            include_finish: Whether to include Finish call
            answer_quality: "good", "bad", or "partial"
            
        Returns:
            Answer dictionary in StableToolBench format
        """
        tool_calls = []
        
        if answer_quality == "good":
            # Good answer: use all relevant APIs
            for api_info in relevant_apis:
                if isinstance(api_info, list) and len(api_info) >= 2:
                    tool_name = api_info[0]
                    api_name = api_info[1]
                    tool_calls.append({
                        "name": f"{tool_name}_{api_name}",
                        "arguments": {"query": query_text[:50]},
                        "response": f"Response from {api_name}"
                    })
            final_answer = (
                f"I have gathered the requested information. {query_text[:200]}. "
                "Based on the data retrieved from the relevant APIs, here is a comprehensive "
                "answer that addresses all aspects of your query. The information has been "
                "successfully collected and processed."
            )
        
        elif answer_quality == "bad":
            # Bad answer: no tool calls or refusal
            final_answer = "I'm sorry, I cannot help with that."
        
        else:  # partial
            # Use only one relevant API (incomplete - missing other required APIs)
            if relevant_apis:
                api_info = relevant_apis[0]
                if isinstance(api_info, list) and len(api_info) >= 2:
                    tool_name = api_info[0]
                    api_name = api_info[1]
                    tool_calls.append({
                        "name": f"{tool_name}_{api_name}",
                        "arguments": {"query": query_text[:50]},
                        "response": f"Partial response from {api_name}"
                    })
            # Partial answer - clearly incomplete
            final_answer = (
                f"I was able to retrieve some information, but I could only access one data source. "
                f"I found partial information about: {query_text[:100]}. However, I was unable to "
                "gather complete information for all parts of your request. Some aspects remain "
                "unaddressed due to limited data access."
            )
        
        # Add Finish call if requested
        if include_finish:
            tool_calls.append({
                "name": "Finish",
                "arguments": {"final_answer": final_answer},
                "response": ""
            })
        
        # Create answer details in ExecutionGraph format
        answer_details = {
            "role": "system",
            "message": "",
            "next": []
        }
        
        current_node = answer_details
        for tool_call in tool_calls:
            # All tool calls (including Finish) are stored as tool role with JSON message
            tool_message = {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"],
                "response": tool_call["response"]
            }
            next_node = {
                "role": "tool",
                "message": json.dumps(tool_message),
                "next": []
            }
            current_node["next"] = [next_node]
            current_node = next_node
        
        # Create the full answer structure
        answer = {
            "query": query_text,
            "available_tools": [],
            "answer": {
                "method": "fake_test",
                "total_steps": len(tool_calls),
                "final_answer": final_answer,
                "answer_details": [answer_details]
            }
        }
        return answer

