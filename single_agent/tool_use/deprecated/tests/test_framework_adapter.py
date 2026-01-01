"""
Test FrameworkAdapter component independently.
"""
import os
import sys

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)

from core import FrameworkInterface, FrameworkAdapter

class MockFramework(FrameworkInterface):
    """Mock framework for testing."""
    
    def __init__(self):
        self.tools = {}
        self.reset_called = False
        self.answer_called = False
    
    def setup_tools(self, tools):
        self.tools = tools
    
    def reset(self):
        self.reset_called = True
    
    def answer(self, query: str) -> str:
        self.answer_called = True
        return f"Answer to: {query}"

def test_framework_adapter():
    """Test FrameworkAdapter functionality."""
    print("Testing FrameworkAdapter...")
    
    mock_framework = MockFramework()
    adapter = FrameworkAdapter(mock_framework)
    
    # Test reset
    adapter.reset()
    assert mock_framework.reset_called, "Reset should be called"
    print("✓ Reset works")
    
    # Test set_tools
    tools = {"tool1": {"name": "tool1"}}
    adapter.set_tools(tools)
    assert mock_framework.tools == tools, "Tools should be set"
    print("✓ Set tools works")
    
    # Test chat
    result = adapter.chat("test query")
    assert mock_framework.answer_called, "Answer should be called"
    assert result["response"] == "Answer to: test query"
    assert result["tool_calls"] == []
    print("✓ Chat works")
    
    print(f"✓ FrameworkAdapter test passed!")
    return True

if __name__ == "__main__":
    test_framework_adapter()

