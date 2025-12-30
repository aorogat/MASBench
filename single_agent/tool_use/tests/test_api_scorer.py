"""
Test APIScorer component independently.
"""
import os
import sys
import json

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

from evaluation.api_scorer import APIScorer

def test_api_scorer():
    """Test APIScorer functionality."""
    print("Testing APIScorer...")
    
    scorer = APIScorer()
    
    # Test answer with correct API calls
    answer = {
        "answer": {
            "answer_details": [{
                "role": "system",
                "message": "",
                "next": [{
                    "role": "tool",
                    "message": json.dumps({"name": "Tool1_API1", "arguments": {}}),
                    "next": [{
                        "role": "tool",
                        "message": json.dumps({"name": "Tool2_API2", "arguments": {}}),
                        "next": []
                    }]
                }]
            }]
        }
    }
    
    gold_apis = [["Tool1", "API1"], ["Tool2", "API2"]]
    score = scorer.score(answer, gold_apis)
    print(f"✓ Perfect score (2/2): {score}")
    assert score == 1.0, f"Expected 1.0, got {score}"
    
    # Test answer with partial API calls
    answer_partial = {
        "answer": {
            "answer_details": [{
                "role": "system",
                "message": "",
                "next": [{
                    "role": "tool",
                    "message": json.dumps({"name": "Tool1_API1", "arguments": {}}),
                    "next": []
                }]
            }]
        }
    }
    
    score = scorer.score(answer_partial, gold_apis)
    print(f"✓ Partial score (1/2): {score}")
    assert score == 0.5, f"Expected 0.5, got {score}"
    
    # Test answer with no API calls
    answer_empty = {
        "answer": {
            "answer_details": [{
                "role": "system",
                "message": "",
                "next": []
            }]
        }
    }
    
    score = scorer.score(answer_empty, gold_apis)
    print(f"✓ Zero score (0/2): {score}")
    assert score == 0.0, f"Expected 0.0, got {score}"
    
    print(f"✓ APIScorer test passed!")
    return True

if __name__ == "__main__":
    test_api_scorer()

