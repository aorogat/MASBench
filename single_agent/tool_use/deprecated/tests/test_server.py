"""
Test script for StableToolBench server.

This script tests the /virtual endpoint to verify:
1. Server is running
2. Cache lookup works
3. GPT fallback works (if cache miss)
4. Response format is correct
"""
import os
import sys
import json
import requests
from pathlib import Path

# Add parent directory to path
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CURRENT_DIR)


def test_server_health():
    """Test if server is running."""
    print("=" * 80)
    print("Test 1: Server Health Check")
    print("=" * 80)
    
    server_url = "http://localhost:8080/virtual"
    
    try:
        # Simple health check - try to connect
        response = requests.post(
            server_url,
            json={
                "category": "Data",
                "tool_name": "TheClique",
                "api_name": "Transfermarkt details",
                "tool_input": {},
                "strip": "",
                "toolbench_key": "test"
            },
            timeout=5
        )
        print(f"✅ Server is running and responding")
        print(f"   Status code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"❌ Server is not running")
        print(f"   Please start the server with: cd StableToolBench/server && python main.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False


def test_cache_hit():
    """Test API call that should be in cache."""
    print("\n" + "=" * 80)
    print("Test 2: Cache Hit Test")
    print("=" * 80)
    
    server_url = "http://localhost:8080/virtual"
    
    # Example API call - adjust based on what's in your cache
    test_cases = [
        {
            "category": "Data",
            "tool_name": "TheClique",
            "api_name": "Transfermarkt details",
            "tool_input": {"player_id": "messi"},
            "description": "Transfermarkt player details"
        },
        {
            "category": "Data",
            "tool_name": "TheClique",
            "api_name": "Transfermarkt details",
            "tool_input": {},
            "description": "Transfermarkt details (empty params)"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {test_case['description']}")
        print(f"  Category: {test_case['category']}")
        print(f"  Tool: {test_case['tool_name']}")
        print(f"  API: {test_case['api_name']}")
        print(f"  Parameters: {test_case['tool_input']}")
        
        try:
            response = requests.post(
                server_url,
                json={
                    "category": test_case["category"],
                    "tool_name": test_case["tool_name"],
                    "api_name": test_case["api_name"],
                    "tool_input": test_case["tool_input"],
                    "strip": "",
                    "toolbench_key": "test"
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Response received")
                print(f"  Error field: {result.get('error', 'N/A')}")
                print(f"  Response preview: {str(result.get('response', ''))[:100]}...")
                
                # Check if it was cached (would be fast) or generated (slower)
                if result.get('error') == '':
                    print(f"  ✅ Valid response (no errors)")
                else:
                    print(f"  ⚠️  Response has error: {result.get('error')}")
            else:
                print(f"  ❌ Server returned status code: {response.status_code}")
                print(f"  Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_cache_miss_gpt_fallback():
    """Test API call that's not in cache (should trigger GPT fallback)."""
    print("\n" + "=" * 80)
    print("Test 3: Cache Miss + GPT Fallback Test")
    print("=" * 80)
    print("Note: This will make an OpenAI API call if cache miss occurs")
    
    server_url = "http://localhost:8080/virtual"
    
    # Use a unique parameter combination that's unlikely to be cached
    import time
    unique_param = f"test_{int(time.time())}"
    
    test_case = {
        "category": "Data",
        "tool_name": "TheClique",
        "api_name": "Transfermarkt details",
        "tool_input": {"player_id": unique_param},
        "description": "Unique test parameter (should trigger GPT)"
    }
    
    print(f"\nTest case: {test_case['description']}")
    print(f"  Category: {test_case['category']}")
    print(f"  Tool: {test_case['tool_name']}")
    print(f"  API: {test_case['api_name']}")
    print(f"  Parameters: {test_case['tool_input']}")
    print(f"  (This should trigger GPT fallback if not in cache)")
    
    try:
        import time as time_module
        start_time = time_module.time()
        
        response = requests.post(
            server_url,
            json={
                "category": test_case["category"],
                "tool_name": test_case["tool_name"],
                "api_name": test_case["api_name"],
                "tool_input": test_case["tool_input"],
                "strip": "",
                "toolbench_key": "test"
            },
            timeout=60  # GPT calls can take longer
        )
        
        elapsed = time_module.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Response received (took {elapsed:.2f} seconds)")
            print(f"  Error field: {result.get('error', 'N/A')}")
            print(f"  Response preview: {str(result.get('response', ''))[:200]}...")
            
            if elapsed > 2.0:
                print(f"  ℹ️  Slow response suggests GPT fallback was used")
            else:
                print(f"  ℹ️  Fast response suggests cache hit")
                
        else:
            print(f"  ❌ Server returned status code: {response.status_code}")
            print(f"  Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


def test_response_format():
    """Test that response format is correct."""
    print("\n" + "=" * 80)
    print("Test 4: Response Format Validation")
    print("=" * 80)
    
    server_url = "http://localhost:8080/virtual"
    
    try:
        response = requests.post(
            server_url,
            json={
                "category": "Data",
                "tool_name": "TheClique",
                "api_name": "Transfermarkt details",
                "tool_input": {},
                "strip": "",
                "toolbench_key": "test"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Check required fields
            required_fields = ["error", "response"]
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                print(f"  ❌ Missing required fields: {missing_fields}")
            else:
                print(f"  ✅ Response has all required fields: {required_fields}")
                print(f"  Error type: {type(result['error'])}")
                print(f"  Response type: {type(result['response'])}")
                
        else:
            print(f"  ❌ Server returned status code: {response.status_code}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("StableToolBench Server Test Suite")
    print("=" * 80)
    print("\nPrerequisites:")
    print("  1. Server should be running: cd StableToolBench/server && python main.py")
    print("  2. .env file should exist with OPENAI_API_KEY")
    print("  3. Cache folder should exist (optional, for cache hit tests)")
    print("=" * 80)
    
    # Run tests
    if not test_server_health():
        print("\n❌ Server is not running. Please start it first.")
        return
    
    test_cache_hit()
    test_cache_miss_gpt_fallback()
    test_response_format()
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Check server logs for detailed execution flow")
    print("  - Verify cache files are being created/used")
    print("  - Test with your actual framework integration")


if __name__ == "__main__":
    main()

