"""
Test FakeAnswerGenerator component independently.
"""
import os
import sys

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

from tests.fake_answer_generator import FakeAnswerGenerator
from utils import AnswerValidator

def test_fake_answer_generator():
    """Test FakeAnswerGenerator functionality."""
    print("Testing FakeAnswerGenerator...")
    
    generator = FakeAnswerGenerator()
    validator = AnswerValidator()
    
    relevant_apis = [["Tool1", "API1"], ["Tool2", "API2"]]
    
    # Test good answer
    good_answer = generator.create(
        query_id="123",
        query_text="Test query",
        relevant_apis=relevant_apis,
        include_finish=True,
        answer_quality="good"
    )
    
    assert "answer" in good_answer
    assert "final_answer" in good_answer["answer"]
    assert len(good_answer["answer"]["final_answer"]) > 50
    has_finish = validator.check_has_finish(good_answer)
    print(f"✓ Good answer generated with Finish: {has_finish}")
    assert has_finish, "Good answer should have Finish call"
    
    # Test bad answer
    bad_answer = generator.create(
        query_id="123",
        query_text="Test query",
        relevant_apis=relevant_apis,
        include_finish=True,
        answer_quality="bad"
    )
    
    assert "sorry" in bad_answer["answer"]["final_answer"].lower()
    print(f"✓ Bad answer generated: {bad_answer['answer']['final_answer'][:50]}...")
    
    # Test partial answer
    partial_answer = generator.create(
        query_id="123",
        query_text="Test query",
        relevant_apis=relevant_apis,
        include_finish=True,
        answer_quality="partial"
    )
    
    assert "partial" in partial_answer["answer"]["final_answer"].lower()
    print(f"✓ Partial answer generated: {partial_answer['answer']['final_answer'][:50]}...")
    
    print(f"✓ FakeAnswerGenerator test passed!")
    return True

if __name__ == "__main__":
    test_fake_answer_generator()

