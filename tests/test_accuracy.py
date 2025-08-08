#!/usr/bin/env python3
"""Test classifier accuracy on comprehensive test suite"""

import json
import sys
sys.path.append('.')
from src.models.intent_classifier import IntentClassifier

def main():
    # Load test cases
    with open('tests/test_cases.json', 'r') as f:
        test_data = json.load(f)

    classifier = IntentClassifier()

    total = 0
    correct = 0
    failures = []

    print('🧪 Running comprehensive accuracy test...')
    print('=' * 60)

    # Use correct key based on structure
    test_key = 'tests' if 'tests' in test_data else 'test_cases'

    for test in test_data[test_key]:
        total += 1
        result = classifier.classify(test['text'])
        
        # Check action type (handle None cases)
        expected_action = test['expected_action'] 
        actual_action = result['type']
        
        # Both None
        if expected_action is None and actual_action is None:
            action_match = True
        # One is None, other isn't  
        elif expected_action is None or actual_action is None:
            action_match = False
        # Both are strings
        else:
            action_match = actual_action == expected_action
        
        # Check order type if applicable
        order_match = True
        if test['expected_action'] == 'CREATE_ORDER' and 'expected_order_type' in test:
            if result.get('params'):
                result_order_type = result['params'].get('type', '').upper()
                expected_order_type = test['expected_order_type'].upper()
                order_match = result_order_type == expected_order_type
        
        if action_match and order_match:
            correct += 1
        else:
            failures.append({
                'text': test['text'],
                'expected': test['expected_action'],
                'expected_order': test.get('expected_order_type'),
                'got': result['type'],
                'got_order': result.get('params', {}).get('type') if result.get('params') else None
            })

    accuracy = (correct / total) * 100

    print(f'✅ Passed: {correct}/{total}')
    print(f'📊 Accuracy: {accuracy:.1f}%')

    if accuracy >= 85:
        print(f'🎯 Target accuracy of 85% achieved!')
    else:
        print(f'⚠️  Below target accuracy of 85%')

    if failures and len(failures) <= 20:
        print(f'\n❌ Failed {len(failures)} cases:')
        for f in failures[:10]:  # Show first 10 failures
            print(f'  "{f["text"]}"')
            exp_str = f["expected"]
            if f["expected_order"]:
                exp_str += f' ({f["expected_order"]})'
            got_str = f["got"]
            if f["got_order"]:
                got_str += f' ({f["got_order"]})'
            print(f'    Expected: {exp_str}')
            print(f'    Got: {got_str}')

if __name__ == "__main__":
    main()