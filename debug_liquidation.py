#!/usr/bin/env python3
"""
Debug script to understand liquidation math
"""

from liquidation_verifier import LiquidationVerifier

def debug_liquidation_case():
    """Debug a specific liquidation case"""
    
    # Test case: 9 ETH collateral, $30,000 loan, $3000 ETH price
    collateral = 9.0
    loan = 30000.0
    eth_price = 3000.0
    
    print("🔍 Debugging Liquidation Case")
    print("=" * 40)
    print(f"Collateral: {collateral} ETH")
    print(f"Loan: ${loan:,.2f}")
    print(f"ETH Price: ${eth_price:,.2f}")
    
    # Calculate values
    collateral_value = collateral * eth_price
    ltv = (loan / collateral_value) * 100
    health_factor = (collateral_value / loan) * 100
    
    print(f"Collateral Value: ${collateral_value:,.2f}")
    print(f"LTV: {ltv:.1f}%")
    print(f"Health Factor: {health_factor:.1f}%")
    
    # Check liquidation threshold
    liquidation_threshold = 110  # 110%
    needs_liquidation = health_factor < liquidation_threshold
    
    print(f"Liquidation Threshold: {liquidation_threshold}%")
    print(f"Needs Liquidation: {needs_liquidation}")
    
    if needs_liquidation:
        print("\n⚡ Calculating Liquidation Amount...")
        
        # Test Solidity calculation
        solidity_collateral, solidity_debt = LiquidationVerifier.calculate_partial_liquidation_amount_solidity(
            collateral, loan, eth_price, liquidation_discount_bp=500
        )
        
        print(f"Solidity Result: {solidity_collateral:.6f} ETH, ${solidity_debt:.2f}")
        
        # Test Python calculation
        python_collateral, python_debt = LiquidationVerifier.calculate_partial_liquidation_amount_python(
            collateral, loan, eth_price
        )
        
        print(f"Python Result: {python_collateral:.6f} ETH, ${python_debt:.2f}")
        
        # Calculate differences
        collateral_diff = abs(solidity_collateral - python_collateral)
        debt_diff = abs(solidity_debt - python_debt)
        
        print(f"Collateral Difference: {collateral_diff:.6f} ETH")
        print(f"Debt Difference: ${debt_diff:.2f}")
        
        # Check if calculations match
        tolerance = 0.001
        match = collateral_diff <= tolerance and debt_diff <= tolerance
        print(f"Calculations Match: {'✅' if match else '❌'}")
    
    print("\n" + "=" * 40)

if __name__ == "__main__":
    debug_liquidation_case() 