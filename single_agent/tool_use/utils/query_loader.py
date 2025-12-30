"""
QueryLoader: Handles loading and filtering queries.

Single Responsibility: Only responsible for loading queries from files.
"""
import os
import json
from typing import List, Dict, Any, Optional


class QueryLoader:
    """
    Loads queries from StableToolBench test sets.
    
    This class follows Single Responsibility Principle: it only handles
    query loading and filtering, nothing else.
    """

    def __init__(
        self,
        query_instruction_dir: str,
        query_ids_dir: Optional[str] = None
    ):
        """
        Initialize the query loader.
        
        Args:
            query_instruction_dir: Directory containing full query files
            query_ids_dir: Optional directory containing query ID filter files
        """
        self.query_instruction_dir = query_instruction_dir
        self.query_ids_dir = query_ids_dir

    def load_queries(
        self,
        test_set: str = "G1_instruction",
        max_queries: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load queries from StableToolBench test sets.
        
        Args:
            test_set: Name of the test set (e.g., "G1_instruction")
            max_queries: Maximum number of queries to return (None for all)
            
        Returns:
            List of query dictionaries
        """
        query_file = os.path.join(
            self.query_instruction_dir, f"{test_set}.json"
        )
        
        # Load all queries
        with open(query_file, 'r') as f:
            all_queries = json.load(f)
        
        # Filter by query IDs if available
        if self.query_ids_dir:
            query_ids_file = os.path.join(
                self.query_ids_dir, f"{test_set}.json"
            )
            if os.path.exists(query_ids_file):
                with open(query_ids_file, 'r') as f:
                    query_ids_data = json.load(f)
                    if isinstance(query_ids_data, dict):
                        query_ids = set(query_ids_data.keys())
                    elif isinstance(query_ids_data, list):
                        query_ids = set(query_ids_data)
                    else:
                        query_ids = None
                    
                    if query_ids:
                        all_queries = [
                            q for q in all_queries
                            if str(q.get("query_id", "")) in query_ids
                        ]
        
        # Limit number of queries
        if max_queries:
            all_queries = all_queries[:max_queries]
        
        return all_queries

    @staticmethod
    def extract_gold_tool_calls(query: Dict[str, Any]) -> List[List[str]]:
        """
        Extract gold tool calls (relevant APIs) from query.
        
        Args:
            query: Query dictionary
            
        Returns:
            List of [tool_name, api_name] pairs
        """
        relevant_apis = query.get("relevant APIs", [])
        if not relevant_apis:
            relevant_apis = query.get("relevant_apis", [])
        if not relevant_apis:
            relevant_apis = query.get("api_list", [])
        
        return relevant_apis if relevant_apis else []
    
    @staticmethod
    def extract_available_tools(query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract available tools from query.
        If query has 'available_tools', use that.
        Otherwise, extract from 'api_list' if present.
        
        Args:
            query: Query dictionary
            
        Returns:
            List of tool dictionaries
        """
        # First check if available_tools is directly in query
        if "available_tools" in query and query["available_tools"]:
            return query["available_tools"]
        
        # Otherwise, extract from api_list
        api_list = query.get("api_list", [])
        if api_list:
            # Extract unique tools from api_list
            tools_dict = {}
            for api_info in api_list:
                if isinstance(api_info, dict):
                    tool_name = api_info.get("tool_name", "")
                    category = api_info.get("category_name", "")
                    if tool_name:
                        tool_key = f"{category}_{tool_name}"
                        if tool_key not in tools_dict:
                            tools_dict[tool_key] = {
                                "category": category,
                                "tool_name": tool_name,
                                "apis": []
                            }
                        # Add API info
                        api_name = api_info.get("api_name", "")
                        if api_name:
                            tools_dict[tool_key]["apis"].append({
                                "api_name": api_name,
                                "api_description": api_info.get("api_description", "")
                            })
            return list(tools_dict.values())
        
        return []

