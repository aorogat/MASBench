"""
Test QueryLoader component independently.
"""
import os
import sys

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

from utils import QueryLoader

def test_query_loader():
    """Test QueryLoader functionality."""
    print("Testing QueryLoader...")
    
    STB_ROOT = os.path.join(CURRENT_DIR, "StableToolBench")
    QUERY_INSTRUCTION_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_instruction")
    QUERY_IDS_DIR = os.path.join(STB_ROOT, "solvable_queries", "test_query_ids")
    
    loader = QueryLoader(
        query_instruction_dir=QUERY_INSTRUCTION_DIR,
        query_ids_dir=QUERY_IDS_DIR
    )
    
    queries = loader.load_queries(test_set="G1_instruction", max_queries=2)
    print(f"✓ Loaded {len(queries)} queries")
    
    if queries:
        query = queries[0]
        gold_apis = QueryLoader.extract_gold_tool_calls(query)
        print(f"✓ Extracted {len(gold_apis)} gold APIs from first query")
        print(f"✓ QueryLoader test passed!")
        return True
    else:
        print("✗ No queries loaded")
        return False

if __name__ == "__main__":
    test_query_loader()

