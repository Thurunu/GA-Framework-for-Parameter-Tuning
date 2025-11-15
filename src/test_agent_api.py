"""
Test Agent API Server

This script tests the agent API server to ensure it's working correctly.
Run this after starting agent_api_server.py to verify all endpoints.
"""

import requests
import json
import sys
import time

def test_endpoint(url, method="GET", name=""):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    print(f"Method: {method}")
    print(f"{'='*60}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ SUCCESS")
            
            # Try to pretty print JSON
            try:
                data = response.json()
                print("\nResponse (JSON):")
                print(json.dumps(data, indent=2)[:500])  # First 500 chars
                if len(json.dumps(data)) > 500:
                    print("... (truncated)")
            except:
                # Not JSON, show text
                print("\nResponse (Text):")
                print(response.text[:500])
                if len(response.text) > 500:
                    print("... (truncated)")
        else:
            print(f"❌ FAILED: Status {response.status_code}")
            print(response.text[:200])
        
        return response.status_code == 200
    
    except requests.exceptions.ConnectionError:
        print("❌ CONNECTION ERROR: Cannot connect to server")
        print("   Make sure agent_api_server.py is running!")
        return False
    except requests.exceptions.Timeout:
        print("❌ TIMEOUT: Server took too long to respond")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def main():
    """Run all tests"""
    
    # Configuration
    host = "localhost"
    port = 9300
    
    # Check command line args
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}")
            sys.exit(1)
    
    base_url = f"http://{host}:{port}"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              Agent API Server Test Suite                     ║
╠══════════════════════════════════════════════════════════════╣
║  Testing agent API at: {base_url:<35}║
║                                                              ║
║  This works with BOTH:                                       ║
║    • agent_api_server.py (FastAPI - 50MB RAM)                ║
║    • agent_api_server_lightweight.py (http.server - 5MB)    ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    results = []
    
    # Test 1: Root endpoint
    results.append(test_endpoint(
        f"{base_url}/",
        name="Root Endpoint"
    ))
    
    time.sleep(0.5)
    
    # Test 2: Health check
    results.append(test_endpoint(
        f"{base_url}/health",
        name="Health Check"
    ))
    
    time.sleep(0.5)
    
    # Test 3: System info
    results.append(test_endpoint(
        f"{base_url}/info",
        name="System Information"
    ))
    
    time.sleep(0.5)
    
    # Test 4: Agent metrics
    results.append(test_endpoint(
        f"{base_url}/metrics",
        name="Agent Metrics"
    ))
    
    time.sleep(0.5)
    
    # Test 5: JSON status
    results.append(test_endpoint(
        f"{base_url}/status",
        name="JSON Status"
    ))
    
    time.sleep(0.5)
    
    # Test 6: Current metrics
    results.append(test_endpoint(
        f"{base_url}/metrics/current",
        name="Current Metrics (JSON)"
    ))
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED!")
        print("\nYour agent API server is working correctly.")
        print(f"You can now configure your backend to pull from: {base_url}")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease check:")
        print("1. Is agent_api_server.py running?")
        print("2. Is it running on the correct port?")
        print("3. Are there any errors in the server logs?")
    
    print(f"\n{'='*60}\n")


def quick_curl_test():
    """Show curl commands for manual testing"""
    print("\n" + "="*60)
    print("MANUAL TESTING WITH CURL")
    print("="*60)
    print("\nYou can also test manually with these curl commands:")
    print("\n# Health check:")
    print("curl http://localhost:9300/health")
    print("\n# Get status (JSON):")
    print("curl http://localhost:9300/status")
    print("\n# Get Agent metrics:")
    print("curl http://localhost:9300/metrics")
    print("\n# Get system info:")
    print("curl http://localhost:9300/info")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
        quick_curl_test()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(0)