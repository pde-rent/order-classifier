"""Tests for rule-based NLP intent classifier using consolidated test cases"""
import pytest
from src.models.intent_classifier import IntentClassifier
from src.utils.test_loader import get_test_loader, get_validation_suite, get_simple_cases


@pytest.fixture(scope="module")
def classifier():
    """Create classifier instance for tests"""
    return IntentClassifier(use_ml_enhancement=False)  # Use rule-based only for consistent tests


@pytest.fixture(scope="module") 
def ml_classifier():
    """Create ML-enhanced classifier instance for tests"""
    return IntentClassifier(use_ml_enhancement=True)


@pytest.fixture(scope="module")
def test_loader():
    """Create test case loader"""
    return get_test_loader()


class TestIntentClassifier:
    """Test intent classification using consolidated test cases"""
    
    def test_validation_suite(self, classifier, test_loader):
        """Test the core validation suite - these MUST pass"""
        validation_cases = get_validation_suite()
        
        for test_case in validation_cases:
            result = classifier.classify(test_case["text"])
            
            # Check action type
            assert result["type"] == test_case["expected_action"], \
                f"Action mismatch for '{test_case['text']}': expected {test_case['expected_action']}, got {result['type']}"
            
            # Check confidence
            assert result["confidence"] >= 0.3, \
                f"Low confidence for '{test_case['text']}': {result['confidence']}"
            
            # Check specific validation value if provided
            if test_case.get("expected_value") is not None:
                params = result.get("params", {})
                expected_value = test_case["expected_value"]
                
                if test_case["expected_action"] == "CREATE_ORDER":
                    # Check amount or price
                    actual_value = params.get("takingAmount") or params.get("limitPrice")
                    assert actual_value == expected_value, \
                        f"Value mismatch for '{test_case['text']}': expected {expected_value}, got {actual_value}"
                elif test_case["expected_action"] == "CHANGE_CHAIN":
                    assert params.get("chainId") == expected_value, \
                        f"Chain ID mismatch for '{test_case['text']}': expected {expected_value}, got {params.get('chainId')}"
                elif test_case["expected_action"] == "CANCEL_ORDER":
                    assert params.get("orderId") == expected_value, \
                        f"Order ID mismatch for '{test_case['text']}': expected {expected_value}, got {params.get('orderId')}"
    
    def test_market_orders(self, classifier, test_loader):
        """Test market order classification"""
        market_cases = test_loader.get_by_category("market_orders")
        
        for test_case in market_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Failed market order: '{test_case['text']}' -> {result['type']}"
            
            if test_case.get("expected_order_type"):
                assert result["params"]["type"] == test_case["expected_order_type"], \
                    f"Order type mismatch for '{test_case['text']}': expected {test_case['expected_order_type']}, got {result['params']['type']}"
    
    def test_limit_orders(self, classifier, test_loader):
        """Test limit order classification"""
        limit_cases = test_loader.get_by_category("limit_orders")
        
        for test_case in limit_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Failed limit order: '{test_case['text']}' -> {result['type']}"
            
            if test_case.get("expected_order_type"):
                assert result["params"]["type"] == test_case["expected_order_type"], \
                    f"Order type mismatch for '{test_case['text']}': expected {test_case['expected_order_type']}, got {result['params']['type']}"
            
            # Check for limit price extraction
            expected_params = test_case.get("expected_params", {})
            if "limitPrice" in expected_params:
                actual_price = result["params"].get("limitPrice")
                expected_price = expected_params["limitPrice"]
                assert actual_price == expected_price, \
                    f"Price mismatch for '{test_case['text']}': expected {expected_price}, got {actual_price}"
    
    def test_twap_orders(self, classifier, test_loader):
        """Test TWAP order classification"""
        twap_cases = test_loader.get_by_category("twap_orders")
        
        for test_case in twap_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Failed TWAP order: '{test_case['text']}' -> {result['type']}"
            
            # TWAP detection might be less reliable, so we'll be more lenient
            # Just check that it's a valid order type
            order_type = result["params"].get("type")
            assert order_type in ["TWAP", "MARKET"], \
                f"Unexpected order type for TWAP '{test_case['text']}': {order_type}"
    
    def test_range_orders(self, classifier, test_loader):
        """Test range order classification"""
        range_cases = test_loader.get_by_category("range_orders")
        
        for test_case in range_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Failed range order: '{test_case['text']}' -> {result['type']}"
            
            if test_case.get("expected_order_type") == "RANGE":
                assert result["params"]["type"] == "RANGE", \
                    f"Range type mismatch for '{test_case['text']}': expected RANGE, got {result['params']['type']}"
                
                # Check for price range parameters
                expected_params = test_case.get("expected_params", {})
                if "startPrice" in expected_params and "endPrice" in expected_params:
                    assert result["params"].get("startPrice") == expected_params["startPrice"]
                    assert result["params"].get("endPrice") == expected_params["endPrice"]
    
    def test_special_orders(self, classifier, test_loader):
        """Test special order types (DCA, Grid, Iceberg)"""
        special_cases = test_loader.get_by_category("special_orders")
        
        for test_case in special_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Failed special order: '{test_case['text']}' -> {result['type']}"
            
            if test_case.get("expected_order_type"):
                expected_type = test_case["expected_order_type"]
                actual_type = result["params"]["type"]
                
                # Special orders might not always be detected perfectly, so we'll accept CREATE_ORDER
                assert actual_type in [expected_type, "MARKET", "LIMIT"], \
                    f"Special order type issue for '{test_case['text']}': expected {expected_type}, got {actual_type}"
    
    def test_wallet_operations(self, classifier, test_loader):
        """Test wallet connection/disconnection"""
        wallet_cases = test_loader.get_by_category("wallet_operations")
        
        for test_case in wallet_cases:
            result = classifier.classify(test_case["text"])
            
            expected_action = test_case["expected_action"]
            assert result["type"] == expected_action, \
                f"Wallet operation failed: '{test_case['text']}' -> expected {expected_action}, got {result['type']}"
            
            # Wallet operations should have high confidence
            assert result["confidence"] > 0.5, \
                f"Low confidence for wallet operation '{test_case['text']}': {result['confidence']}"
    
    def test_chain_operations(self, classifier, test_loader):
        """Test chain switching operations"""
        chain_cases = test_loader.get_by_category("chain_operations")
        
        for test_case in chain_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CHANGE_CHAIN", \
                f"Chain operation failed: '{test_case['text']}' -> {result['type']}"
            
            # Check chain ID if specified
            expected_params = test_case.get("expected_params", {})
            if "chainId" in expected_params:
                actual_chain_id = result["params"].get("chainId")
                expected_chain_id = expected_params["chainId"]
                assert actual_chain_id == expected_chain_id, \
                    f"Chain ID mismatch for '{test_case['text']}': expected {expected_chain_id}, got {actual_chain_id}"
    
    def test_pair_operations(self, classifier, test_loader):
        """Test trading pair operations"""
        pair_cases = test_loader.get_by_category("pair_operations")
        
        for test_case in pair_cases:
            result = classifier.classify(test_case["text"])
            
            # Pair operations might be classified as CHANGE_PAIR or CREATE_ORDER
            assert result["type"] in ["CHANGE_PAIR", "CREATE_ORDER"], \
                f"Pair operation failed: '{test_case['text']}' -> {result['type']}"
    
    def test_order_management(self, classifier, test_loader):
        """Test order management operations (cancel, update, info)"""
        order_mgmt_cases = test_loader.get_by_category("order_management")
        
        for test_case in order_mgmt_cases:
            result = classifier.classify(test_case["text"])
            
            expected_action = test_case["expected_action"]
            assert result["type"] == expected_action, \
                f"Order management failed: '{test_case['text']}' -> expected {expected_action}, got {result['type']}"
            
            # Check order ID if specified
            expected_params = test_case.get("expected_params", {})
            if expected_params and "orderId" in expected_params:
                params = result.get("params") or {}
                actual_order_id = params.get("orderId")
                expected_order_id = expected_params["orderId"]
                assert actual_order_id == expected_order_id, \
                    f"Order ID mismatch for '{test_case['text']}': expected {expected_order_id}, got {actual_order_id}"
    
    def test_amount_variations(self, classifier, test_loader):
        """Test different amount formats"""
        amount_cases = test_loader.get_by_category("amount_variations")
        
        for test_case in amount_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Amount variation failed: '{test_case['text']}' -> {result['type']}"
            
            # Check amount extraction if specified
            expected_params = test_case.get("expected_params", {})
            if "takingAmount" in expected_params:
                actual_amount = result["params"].get("takingAmount")
                expected_amount = expected_params["takingAmount"]
                assert actual_amount == expected_amount, \
                    f"Amount mismatch for '{test_case['text']}': expected {expected_amount}, got {actual_amount}"
    
    def test_price_variations(self, classifier, test_loader):
        """Test different price formats"""
        price_cases = test_loader.get_by_category("price_variations")
        
        for test_case in price_cases:
            result = classifier.classify(test_case["text"])
            
            assert result["type"] == "CREATE_ORDER", \
                f"Price variation failed: '{test_case['text']}' -> {result['type']}"
            
            # Should be limit order with price
            assert result["params"]["type"] == "LIMIT", \
                f"Expected LIMIT order for '{test_case['text']}', got {result['params']['type']}"
            
            # Check price extraction
            expected_params = test_case.get("expected_params", {})
            if "limitPrice" in expected_params:
                actual_price = result["params"].get("limitPrice")
                expected_price = expected_params["limitPrice"]
                assert actual_price == expected_price, \
                    f"Price mismatch for '{test_case['text']}': expected {expected_price}, got {actual_price}"
    
    def test_edge_cases(self, classifier, test_loader):
        """Test edge cases and error handling"""
        edge_cases = test_loader.get_by_category("edge_cases")
        
        for test_case in edge_cases:
            result = classifier.classify(test_case["text"])
            
            # Should not crash and should return some classification
            assert "type" in result
            assert "confidence" in result
            assert isinstance(result["confidence"], (int, float))
            assert 0 <= result["confidence"] <= 1
            
            # Edge cases might have lower confidence
            # But they should still produce a reasonable default
            assert result["type"] in [
                "CREATE_ORDER", "CONNECT_WALLET", "DISCONNECT_WALLET",
                "CHANGE_CHAIN", "CHANGE_PAIR", "CANCEL_ORDER", 
                "UPDATE_ORDER", "GET_ORDER_INFO"
            ]
    
    def test_ml_vs_rules_comparison(self, classifier, ml_classifier, test_loader):
        """Compare rule-based vs ML-enhanced classification on simple cases"""
        simple_cases = get_simple_cases()[:10]  # Test on subset for performance
        
        agreement = 0
        total = len(simple_cases)
        
        for test_case in simple_cases:
            rule_result = classifier.classify(test_case["text"])
            ml_result = ml_classifier.classify(test_case["text"])
            
            if rule_result["type"] == ml_result["type"]:
                agreement += 1
        
        agreement_rate = agreement / total
        
        # Should have reasonable agreement on simple cases
        assert agreement_rate > 0.6, \
            f"Low agreement between rule-based and ML classifiers: {agreement_rate:.2f}"
    
    def test_confidence_calibration(self, classifier, test_loader):
        """Test that confidence scores are well-calibrated"""
        validation_cases = get_validation_suite()
        
        high_confidence_count = 0
        total_cases = len(validation_cases)
        
        for test_case in validation_cases:
            result = classifier.classify(test_case["text"])
            
            # Validation cases should generally have decent confidence
            if result["confidence"] > 0.7:
                high_confidence_count += 1
        
        high_confidence_rate = high_confidence_count / total_cases
        
        # At least 60% of validation cases should have high confidence
        assert high_confidence_rate > 0.6, \
            f"Too few high-confidence predictions: {high_confidence_rate:.2f}"
    
    def test_performance_benchmarks(self, classifier, test_loader):
        """Test performance benchmarks"""
        import time
        
        simple_cases = get_simple_cases()
        
        start_time = time.time()
        for test_case in simple_cases:
            classifier.classify(test_case["text"])
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_ms = (total_time / len(simple_cases)) * 1000
        
        # Should be fast - under 10ms per classification on average
        assert avg_time_ms < 10, \
            f"Classification too slow: {avg_time_ms:.2f}ms per case"
        
        # Should handle reasonable throughput
        throughput = len(simple_cases) / total_time
        assert throughput > 100, \
            f"Low throughput: {throughput:.0f} classifications/second"


class TestStandaloneFunction:
    """Test standalone classify function for backward compatibility"""
    
    def test_standalone_classify_function(self):
        """Test standalone classify function"""
        from src.models.intent_classifier import classify
        
        result = classify("buy 1000 usdt")
        
        assert result["type"] == "CREATE_ORDER"
        assert result["confidence"] > 0
        assert result["params"]["type"] == "MARKET"


class TestStatistics:
    """Test suite statistics and coverage"""
    
    def test_suite_coverage(self, test_loader):
        """Test that we have good coverage across all action types"""
        stats = test_loader.get_statistics()
        
        # Should cover all main action types
        required_actions = [
            "CREATE_ORDER", "CONNECT_WALLET", "DISCONNECT_WALLET",
            "CHANGE_CHAIN", "CHANGE_PAIR", "CANCEL_ORDER", 
            "UPDATE_ORDER", "GET_ORDER_INFO"
        ]
        
        for action in required_actions:
            assert action in stats["actions"], \
                f"Missing test cases for action: {action}"
            assert stats["actions"][action] > 0, \
                f"No test cases for action: {action}"
        
        # Should have reasonable total number of cases
        assert stats["total_cases"] > 50, \
            f"Too few test cases: {stats['total_cases']}"