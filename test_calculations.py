#!/usr/bin/env python3
"""
Comprehensive Test Script for Mathematical Calculations
Tests collateral calculations, liquidation math, and boundary conditions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collateral_calculator import CollateralCalculator
from liquidation_verifier import LiquidationVerifier

def test_collateral_calculations():
    """Test collateral calculation alignment between Python and Solidity"""
    print("🔍 Testing Collateral Calculations...")
    print("=" * 50)
    
    test_cases = [
        {"credit_score": 0.8, "loan_amount": 10000, "eth_price": 3000, "expected_ratio": 1.5},
        {"credit_score": 0.7, "loan_amount": 10000, "eth_price": 3000, "expected_ratio": 1.6},
        {"credit_score": 0.6, "loan_amount": 10000, "eth_price": 3000, "expected_ratio": 1.8},
        {"credit_score": 0.4, "loan_amount": 10000, "eth_price": 3000, "expected_ratio": 2.0},
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        credit_score = test_case["credit_score"]
        loan_amount = test_case["loan_amount"]
        eth_price = test_case["eth_price"]
        expected_ratio = test_case["expected_ratio"]
        
        # Calculate using our calculator
        calculated_ratio = CollateralCalculator.calculate_required_collateral_ratio(credit_score)
        required_usd = CollateralCalculator.calculate_required_collateral_usd(loan_amount, credit_score)
        required_eth = CollateralCalculator.calculate_required_collateral_eth(loan_amount, credit_score, eth_price)
        
        # Verify results
        ratio_correct = abs(calculated_ratio - expected_ratio) < 0.001
        usd_correct = abs(required_usd - (loan_amount * expected_ratio)) < 0.01
        eth_correct = abs(required_eth - (required_usd / eth_price)) < 0.0001
        
        status = "✅ PASS" if all([ratio_correct, usd_correct, eth_correct]) else "❌ FAIL"
        print(f"Test {i}: {status}")
        print(f"  Credit Score: {credit_score:.1%}")
        print(f"  Expected Ratio: {expected_ratio:.1%}")
        print(f"  Calculated Ratio: {calculated_ratio:.1%}")
        print(f"  Required USD: ${required_usd:,.2f}")
        print(f"  Required ETH: {required_eth:.6f}")
        print()
        
        if not all([ratio_correct, usd_correct, eth_correct]):
            all_passed = False
    
    return all_passed

def test_liquidation_math():
    """Test liquidation math verification"""
    print("⚡ Testing Liquidation Math...")
    print("=" * 50)
    
    test_cases = [
        {"collateral": 9.0, "loan": 30000, "eth_price": 3000, "description": "Below threshold (90% LTV)"},
        {"collateral": 5.0, "loan": 30000, "eth_price": 3000, "description": "Low collateral (50% LTV)"},
        {"collateral": 20.0, "loan": 30000, "eth_price": 3000, "description": "High collateral (200% LTV)"},
        {"collateral": 0.0009, "loan": 3, "eth_price": 3000, "description": "Very small below threshold"},
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        collateral = test_case["collateral"]
        loan = test_case["loan"]
        eth_price = test_case["eth_price"]
        description = test_case["description"]
        
        print(f"Test {i}: {description}")
        print(f"  Collateral: {collateral} ETH")
        print(f"  Loan: ${loan:,.2f}")
        print(f"  ETH Price: ${eth_price:,.2f}")
        
        # Test liquidation verification
        result = LiquidationVerifier.verify_liquidation_calculations(
            collateral, loan, eth_price
        )
        
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
            all_passed = False
        else:
            print(f"  Solidity: {result['solidity_collateral']:.6f} ETH, ${result['solidity_debt']:.2f}")
            print(f"  Python:   {result['python_collateral']:.6f} ETH, ${result['python_debt']:.2f}")
            print(f"  Match:    {'✅' if result['calculations_match'] else '❌'}")
            
            if not result['calculations_match']:
                all_passed = False
        
        print()
    
    return all_passed

def test_boundary_conditions():
    """Test boundary conditions and edge cases"""
    print("🚧 Testing Boundary Conditions...")
    print("=" * 50)
    
    result = LiquidationVerifier.test_boundary_conditions()
    
    print(f"Total Tests: {result['total_tests']}")
    print(f"Passed: {result['passed_tests']}")
    print(f"Failed: {result['failed_tests']}")
    print()
    
    for test_result in result['boundary_tests']:
        test_name = test_result['test_case']
        status = "✅ PASS" if test_result.get('calculations_match', False) else "❌ FAIL"
        
        print(f"{status}: {test_name}")
        if 'error' in test_result:
            print(f"  Error: {test_result['error']}")
        else:
            print(f"  LTV: {test_result['ltv_percentage']:.1f}%")
            print(f"  Solidity: {test_result['solidity_result']}")
            print(f"  Python: {test_result['python_result']}")
        print()
    
    return result['failed_tests'] == 0

def test_ltv_calculations():
    """Test LTV and health factor calculations"""
    print("📊 Testing LTV and Health Factor Calculations...")
    print("=" * 50)
    
    test_cases = [
        {"collateral": 10.0, "loan": 0, "eth_price": 3000, "description": "0% LTV - No loan"},
        {"collateral": 10.0, "loan": 30000, "eth_price": 3000, "description": "100% LTV - Exact collateral"},
        {"collateral": 20.0, "loan": 30000, "eth_price": 3000, "description": "50% LTV - Double collateral"},
        {"collateral": 0, "loan": 10000, "eth_price": 3000, "description": "Infinite LTV - No collateral"},
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        collateral = test_case["collateral"]
        loan = test_case["loan"]
        eth_price = test_case["eth_price"]
        description = test_case["description"]
        
        print(f"Test {i}: {description}")
        print(f"  Collateral: {collateral} ETH")
        print(f"  Loan: ${loan:,.2f}")
        print(f"  ETH Price: ${eth_price:,.2f}")
        
        try:
            ltv_bp, health_factor_bp = CollateralCalculator.calculate_ltv_and_health_factor(
                collateral, loan, eth_price
            )
            
            ltv_percentage = ltv_bp / 100
            health_factor_percentage = health_factor_bp / 100
            
            print(f"  LTV: {ltv_percentage:.1f}%")
            print(f"  Health Factor: {health_factor_percentage:.1f}%")
            
            # Verify calculations
            if loan == 0:
                expected_ltv = 0
                expected_health = float('inf')
            elif collateral == 0:
                expected_ltv = float('inf')
                expected_health = 0
            else:
                expected_ltv = (loan / (collateral * eth_price)) * 100
                expected_health = ((collateral * eth_price) / loan) * 100
            
            ltv_correct = abs(ltv_percentage - expected_ltv) < 0.1 if expected_ltv != float('inf') else True
            health_correct = abs(health_factor_percentage - expected_health) < 0.1 if expected_health != float('inf') else True
            
            if ltv_correct and health_correct:
                print("  ✅ LTV and Health Factor calculations correct")
            else:
                print("  ❌ LTV and Health Factor calculations incorrect")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_passed = False
        
        print()
    
    return all_passed

def main():
    """Run all tests"""
    print("🧮 COMPREHENSIVE MATHEMATICAL VERIFICATION")
    print("=" * 60)
    print()
    
    # Run all tests
    tests = [
        ("Collateral Calculations", test_collateral_calculations),
        ("LTV and Health Factor", test_ltv_calculations),
        ("Liquidation Math", test_liquidation_math),
        ("Boundary Conditions", test_boundary_conditions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print()
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All mathematical calculations are verified and consistent!")
    else:
        print("⚠️  Some calculations need attention. Check the failed tests above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 