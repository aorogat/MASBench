"""
Test AnswerValidator component independently.
"""
import os
import sys
import json

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

from utils import AnswerValidator

def test_answer_validator():
    """Test AnswerValidator functionality."""
    print("Testing AnswerValidator...")
    
    validator = AnswerValidator()
    
    # Test answer with Finish call
    answer_with_finish = {
        "answer_details": [{
            "role": "system",
            "message": "",
            "next": [{
                "role": "tool",
                "message": json.dumps({"name": "Finish", "arguments": {"final_answer": "test"}}),
                "next": []
            }]
        }]
    }
    
    has_finish = validator.check_has_finish(answer_with_finish)
    print(f"✓ Finish call detected: {has_finish}")
    assert has_finish, "Should detect Finish call"
    
    # Test answer without Finish call
    answer_without_finish = {
        "answer_details": [{
            "role": "system",
            "message": "",
            "next": [{
                "role": "tool",
                "message": json.dumps({"name": "SomeTool_API", "arguments": {}}),
                "next": []
            }]
        }]
    }
    
    has_finish = validator.check_has_finish(answer_without_finish)
    print(f"✓ No Finish call detected: {not has_finish}")
    assert not has_finish, "Should not detect Finish call"
    
    # Test structure validation
    valid_answer = {
        "answer": {
            "final_answer": "test",
            "answer_details": []
        }
    }
    is_valid = validator.validate_answer_structure(valid_answer)
    print(f"✓ Valid structure: {is_valid}")
    assert is_valid, "Should validate correct structure"
    
    print(f"✓ AnswerValidator test passed!")
    return True

if __name__ == "__main__":
    test_answer_validator()

