"""
Test case loader utility for consolidated test cases.
Provides a single source of truth for all test cases.
"""

import json
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

class TestCaseLoader:
    """Loads and provides access to consolidated test cases."""
    
    def __init__(self, test_file_path: Optional[str] = None):
        """Initialize the test case loader."""
        if test_file_path is None:
            # Default to test_cases.json in project root
            project_root = Path(__file__).parent.parent.parent
            test_file_path = project_root / "test_cases.json"
        
        self.test_file_path = test_file_path
        self._test_data = None
        self._load_test_cases()
    
    def _load_test_cases(self):
        """Load test cases from JSON file."""
        try:
            with open(self.test_file_path, 'r', encoding='utf-8') as f:
                self._test_data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load test cases from {self.test_file_path}: {e}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get test suite metadata."""
        return self._test_data.get("metadata", {})
    
    def get_all_test_cases(self) -> List[Dict[str, Any]]:
        """Get all test cases."""
        return self._test_data.get("test_cases", [])
    
    def get_validation_suite(self) -> List[Dict[str, Any]]:
        """Get validation test cases."""
        return self._test_data.get("validation_suite", [])
    
    def get_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get test cases by category."""
        return [case for case in self.get_all_test_cases() if case.get("category") == category]
    
    def get_by_action(self, action: str) -> List[Dict[str, Any]]:
        """Get test cases by expected action."""
        return [case for case in self.get_all_test_cases() if case.get("expected_action") == action]
    
    def get_by_order_type(self, order_type: str) -> List[Dict[str, Any]]:
        """Get test cases by expected order type."""
        return [case for case in self.get_all_test_cases() if case.get("expected_order_type") == order_type]
    
    def get_by_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific test case by ID."""
        for case in self.get_all_test_cases():
            if case.get("id") == test_id:
                return case
        return None
    
    def get_edge_cases(self) -> List[Dict[str, Any]]:
        """Get edge case test cases."""
        return self.get_by_category("edge_cases")
    
    def get_market_orders(self) -> List[Dict[str, Any]]:
        """Get market order test cases."""
        return self.get_by_category("market_orders")
    
    def get_limit_orders(self) -> List[Dict[str, Any]]:
        """Get limit order test cases."""
        return self.get_by_category("limit_orders")
    
    def get_wallet_operations(self) -> List[Dict[str, Any]]:
        """Get wallet operation test cases."""
        return self.get_by_category("wallet_operations")
    
    def get_chain_operations(self) -> List[Dict[str, Any]]:
        """Get chain operation test cases."""
        return self.get_by_category("chain_operations")
    
    def get_order_management(self) -> List[Dict[str, Any]]:
        """Get order management test cases."""
        return self.get_by_category("order_management")
    
    def get_simple_cases(self) -> List[Dict[str, Any]]:
        """Get simple, clear test cases for quick validation."""
        simple_categories = [
            "market_orders", "limit_orders", "wallet_operations", 
            "chain_operations", "order_management"
        ]
        cases = []
        for category in simple_categories:
            cases.extend(self.get_by_category(category)[:3])  # Take first 3 from each
        return cases
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the test suite."""
        all_cases = self.get_all_test_cases()
        validation_cases = self.get_validation_suite()
        
        # Count by category
        categories = {}
        for case in all_cases:
            category = case.get("category", "unknown")
            categories[category] = categories.get(category, 0) + 1
        
        # Count by action
        actions = {}
        for case in all_cases:
            action = case.get("expected_action", "unknown")
            actions[action] = actions.get(action, 0) + 1
        
        return {
            "total_cases": len(all_cases),
            "validation_cases": len(validation_cases),
            "categories": categories,
            "actions": actions,
            "metadata": self.get_metadata()
        }

# Global instance for easy access
_loader = None

def get_test_loader() -> TestCaseLoader:
    """Get the global test case loader instance."""
    global _loader
    if _loader is None:
        _loader = TestCaseLoader()
    return _loader

# Convenience functions
def get_all_test_cases() -> List[Dict[str, Any]]:
    """Get all test cases."""
    return get_test_loader().get_all_test_cases()

def get_validation_suite() -> List[Dict[str, Any]]:
    """Get validation test cases."""
    return get_test_loader().get_validation_suite()

def get_by_category(category: str) -> List[Dict[str, Any]]:
    """Get test cases by category."""
    return get_test_loader().get_by_category(category)

def get_simple_cases() -> List[Dict[str, Any]]:
    """Get simple test cases for quick validation."""
    return get_test_loader().get_simple_cases()

def get_statistics() -> Dict[str, Any]:
    """Get test suite statistics."""
    return get_test_loader().get_statistics()