#!/usr/bin/env python3
"""
Simple interactive testing for the Intent Classification API
"""
import httpx
import json
import time

def test_api(text, endpoint="http://localhost:40012/classify"):
    """Test a single classification"""
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                endpoint,
                json={"text": text}
            )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Input: \"{text}\"")
            print(f"   Action: {result['type']}")
            print(f"   Confidence: {result['confidence']:.2f}")
            if result.get('params'):
                params = result['params']
                if params:
                    key_params = []
                    if 'type' in params:
                        key_params.append(f"Order Type: {params['type']}")
                    if 'takingAmount' in params:
                        key_params.append(f"Amount: {params['takingAmount']}")
                    if 'makingAmount' in params:
                        key_params.append(f"Amount: {params['makingAmount']}")
                    if 'limitPrice' in params:
                        key_params.append(f"Price: {params['limitPrice']}")
                    if 'chainId' in params:
                        key_params.append(f"Chain ID: {params['chainId']}")
                    if 'orderId' in params:
                        key_params.append(f"Order ID: {params['orderId']}")
                    if 'pair' in params:
                        key_params.append(f"Pair: {params['pair']}")
                    if key_params:
                        print(f"   Params: {', '.join(key_params)}")
            return result
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def main():
    print("🚀 Simple Intent Classification API Tester")
    print("=" * 50)
    print("Server should be running on http://localhost:40012")
    print("Type 'quit' to exit\n")
    
    # Test server health first
    try:
        with httpx.Client(timeout=5.0) as client:
            health_response = client.get("http://localhost:40012/health")
        if health_response.status_code == 200:
            print("✅ Server is healthy and ready!\n")
        else:
            print("❌ Server health check failed")
            return
    except:
        print("❌ Cannot connect to server. Make sure it's running:")
        print("   uv run python run_server.py")
        return

    while True:
        try:
            user_input = input("Enter text to classify (or 'quit'): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
                
            if not user_input:
                continue
                
            print()
            test_api(user_input)
            print()
            
            # Small delay to prevent overwhelming the server
            time.sleep(0.1)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()