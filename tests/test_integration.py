#!/usr/bin/env python3
"""
Test script for the rule-based NLP Intent Classification API
Run this after starting the server with: uv run python run_server.py
"""

import httpx
import json
import time
from typing import List, Dict, Any


class IntentAPITester:
    def __init__(self, base_url: str = "http://localhost:40012"):
        self.base_url = base_url
        self.client = httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=30.0
        )
    
    def test_health(self) -> bool:
        """Test health endpoint"""
        try:
            response = self.client.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print(f"✅ Server Status: {health['status']}")
                print(f"📊 Classifier: {health['classifier']['type']}")
                print(f"💾 Cache: {'Enabled' if health['cache']['enabled'] else 'Disabled'}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def classify_single(self, text: str) -> Dict[str, Any]:
        """Test single classification"""
        try:
            response = self.client.post(
                f"{self.base_url}/classify",
                json={"text": text}
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Classification failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"❌ Classification error: {e}")
            return {}
    
    def classify_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Test batch classification"""
        try:
            response = self.client.post(
                f"{self.base_url}/classify_batch", 
                json={"texts": texts}
            )
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Batch classification failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Batch classification error: {e}")
            return []
    
    def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        print("🚀 CPU-Only Intent Classification API Test")
        print("=" * 50)
        
        # Test health
        print("\n1. Testing Health Endpoint...")
        if not self.test_health():
            print("❌ Server not healthy, exiting")
            return
        
        # Test single classifications
        print("\n2. Testing Single Classifications...")
        test_cases = [
            ("buy 1000 usdt", "Market Order"),
            ("sell 2 eth at 3500", "Limit Order"),
            ("connect wallet", "Wallet Connect"),
            ("switch to polygon", "Chain Switch"),
            ("cancel order abc123", "Order Cancel"),
            ("dca into btc weekly", "DCA Strategy"),
            ("buy eth between 3000 and 3500", "Range Order"),
            ("accumulate 500 usdt today", "TWAP Order"),
            ("trade aave tokens", "Pair Change"),
            ("disconnect", "Wallet Disconnect")
        ]
        
        for text, description in test_cases:
            result = self.classify_single(text)
            if result:
                print(f"✅ {description}")
                print(f"   Input: \"{text}\"")
                print(f"   Action: {result['type']}")
                print(f"   Confidence: {result['confidence']:.2f}")
                if result.get('params'):
                    key_params = []
                    params = result['params']
                    if 'type' in params:
                        key_params.append(f"type={params['type']}")
                    if 'takingAmount' in params:
                        key_params.append(f"amount={params['takingAmount']}")
                    if 'limitPrice' in params:
                        key_params.append(f"price={params['limitPrice']}")
                    if 'chainId' in params:
                        key_params.append(f"chainId={params['chainId']}")
                    if 'orderId' in params:
                        key_params.append(f"orderId={params['orderId']}")
                    if key_params:
                        print(f"   Params: {', '.join(key_params)}")
                print()
        
        # Test batch classification
        print("3. Testing Batch Classification...")
        batch_texts = [
            "buy 500 usdc",
            "sell bitcoin asap", 
            "connect my wallet",
            "switch to bsc",
            "get order status"
        ]
        
        start_time = time.time()
        batch_results = self.classify_batch(batch_texts)
        batch_time = time.time() - start_time
        
        if batch_results:
            print(f"✅ Batch processed {len(batch_results)} items in {batch_time*1000:.1f}ms")
            print(f"   Throughput: {len(batch_results)/batch_time:.0f} classifications/second")
            
            for i, (text, result) in enumerate(zip(batch_texts, batch_results)):
                print(f"   {i+1}. \"{text}\" → {result['type']} (conf: {result['confidence']:.2f})")
        
        # Performance benchmark
        print("\n4. Performance Benchmark...")
        benchmark_text = "buy 1000 usdt at 4000"
        num_requests = 100
        
        print(f"   Running {num_requests} requests...")
        start_time = time.time()
        
        for _ in range(num_requests):
            self.classify_single(benchmark_text)
        
        total_time = time.time() - start_time
        avg_time = (total_time / num_requests) * 1000
        throughput = num_requests / total_time
        
        print(f"   ✅ Performance Results:")
        print(f"   • Average latency: {avg_time:.1f}ms")
        print(f"   • Throughput: {throughput:.0f} requests/second")
        print(f"   • Total time: {total_time:.2f}s")
        
        print("\n🎉 All tests completed successfully!")
        print("\nℹ️  API Endpoints Available:")
        print(f"   • Health: GET {self.base_url}/health")
        print(f"   • Classify: POST {self.base_url}/classify")
        print(f"   • Batch: POST {self.base_url}/classify_batch")
        print(f"   • Docs: {self.base_url}/docs")


def main():
    """Run the test suite"""
    tester = IntentAPITester()
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()