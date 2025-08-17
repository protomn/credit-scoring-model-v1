#!/usr/bin/env python3
"""
Debug Solidity math step by step
"""

def debug_solidity_math():
    """Debug the Solidity liquidation math step by step"""
    
    # Test case: 9 ETH collateral, $30,000 loan, $3000 ETH price
    collateral = 9.0
    loan = 30000.0
    eth_price = 3000.0
    
    print("🔍 Debugging Solidity Math Step by Step")
    print("=" * 50)
    print(f"Collateral: {collateral} ETH")
    print(f"Loan: ${loan:,.2f}")
    print(f"ETH Price: ${eth_price:,.2f}")
    
    # Constants from smart contract
    PRECISION = 1e18
    PERCENTAGE_PRECISION = 10000  # basis points
    LIQUIDATION_THRESHOLD = 11000  # 110% in basis points (110 * 10000 / 100)
    TARGET_HEALTH_FACTOR = 12500   # 125% in basis points (125 * 10000 / 100)
    MAX_LIQUIDATION_DISCOUNT = 500  # 5% in basis points
    
    print(f"\n📊 Constants:")
    print(f"PRECISION: {PRECISION}")
    print(f"PERCENTAGE_PRECISION: {PERCENTAGE_PRECISION}")
    print(f"LIQUIDATION_THRESHOLD: {LIQUIDATION_THRESHOLD}")
    print(f"TARGET_HEALTH_FACTOR: {TARGET_HEALTH_FACTOR}")
    print(f"MAX_LIQUIDATION_DISCOUNT: {MAX_LIQUIDATION_DISCOUNT}")
    
    # Step 1: Calculate current health factor
    collateral_value_usd = collateral * eth_price
    outstanding_debt_usd = loan
    
    print(f"\n📈 Step 1: Calculate Current Health Factor")
    print(f"Collateral Value USD: ${collateral_value_usd:,.2f}")
    print(f"Outstanding Debt USD: ${outstanding_debt_usd:,.2f}")
    
    # Health factor in basis points
    health_factor_bp = (collateral_value_usd / outstanding_debt_usd) * PERCENTAGE_PRECISION
    health_factor_percent = health_factor_bp / 100
    
    print(f"Health Factor (basis points): {health_factor_bp}")
    print(f"Health Factor (percent): {health_factor_percent:.1f}%")
    
    # Check if liquidation is needed
    needs_liquidation = health_factor_bp < LIQUIDATION_THRESHOLD
    print(f"Needs Liquidation: {needs_liquidation} (threshold: {LIQUIDATION_THRESHOLD})")
    
    if not needs_liquidation:
        print("❌ No liquidation needed - health factor above threshold")
        return
    
    # Step 2: Calculate liquidation discount
    liquidation_discount_bp = MAX_LIQUIDATION_DISCOUNT  # 500 basis points = 5%
    print(f"\n💰 Step 2: Liquidation Discount")
    print(f"Liquidation Discount: {liquidation_discount_bp} basis points ({liquidation_discount_bp/100:.1f}%)")
    
    # Step 3: Calculate target collateral value needed
    TF = TARGET_HEALTH_FACTOR  # 125
    PP = PERCENTAGE_PRECISION  # 10000
    
    print(f"\n🎯 Step 3: Calculate Target Collateral Value")
    print(f"TF (Target Health Factor): {TF}")
    print(f"PP (Percentage Precision): {PP}")
    
    # TFxD = (TF * D / PP) - target collateral value needed
    TFxD = (TF * outstanding_debt_usd) / PP
    print(f"TFxD = ({TF} * {outstanding_debt_usd:,.2f}) / {PP} = {TFxD:,.2f}")
    
    # Check if current collateral is sufficient
    if TFxD <= collateral_value_usd:
        print(f"❌ TFxD ({TFxD:,.2f}) <= Current Collateral ({collateral_value_usd:,.2f})")
        print("No liquidation needed - current collateral is sufficient")
        return
    
    print(f"✅ TFxD ({TFxD:,.2f}) > Current Collateral ({collateral_value_usd:,.2f})")
    print("Liquidation calculation proceeds...")
    
    # Step 4: Calculate numerator
    numerator_usd = TFxD - collateral_value_usd
    print(f"\n🔢 Step 4: Calculate Numerator")
    print(f"Numerator USD = TFxD - Current Collateral = {TFxD:,.2f} - {collateral_value_usd:,.2f} = {numerator_usd:,.2f}")
    
    # Step 5: Calculate denominator
    left = TF * (PP - liquidation_discount_bp)
    right = PP * PP
    denom = left - right
    
    print(f"\n🔢 Step 5: Calculate Denominator")
    print(f"Left = TF * (PP - discount) = {TF} * ({PP} - {liquidation_discount_bp}) = {TF} * {PP - liquidation_discount_bp} = {left}")
    print(f"Right = PP * PP = {PP} * {PP} = {right}")
    print(f"Denominator = Left - Right = {left} - {right} = {denom}")
    
    # Safety check
    if denom <= 0:
        print(f"❌ Denominator <= 0: {denom}")
        print("Invalid liquidation parameters")
        return
    
    print(f"✅ Denominator > 0: {denom}")
    
    # Step 6: Calculate collateral to liquidate (USD)
    numerator_mul = numerator_usd * PP * PP
    collateral_to_liquidate_usd = numerator_mul / denom
    
    print(f"\n💸 Step 6: Calculate Collateral to Liquidate (USD)")
    print(f"Numerator * PP * PP = {numerator_usd:,.2f} * {PP} * {PP} = {numerator_mul:,.2f}")
    print(f"Collateral to Liquidate USD = {numerator_mul:,.2f} / {denom} = {collateral_to_liquidate_usd:,.2f}")
    
    # Step 7: Calculate debt to repay (USD)
    debt_to_repay_usd = collateral_to_liquidate_usd * (PP - liquidation_discount_bp) / PP
    
    print(f"\n💳 Step 7: Calculate Debt to Repay (USD)")
    print(f"Debt to Repay USD = {collateral_to_liquidate_usd:,.2f} * ({PP} - {liquidation_discount_bp}) / {PP}")
    print(f"Debt to Repay USD = {collateral_to_liquidate_usd:,.2f} * {PP - liquidation_discount_bp} / {PP} = {debt_to_repay_usd:,.2f}")
    
    # Step 8: Convert to ETH
    collateral_to_liquidate_eth = collateral_to_liquidate_usd / eth_price
    
    print(f"\n🪙 Step 8: Convert to ETH")
    print(f"Collateral to Liquidate ETH = {collateral_to_liquidate_usd:,.2f} / {eth_price:,.2f} = {collateral_to_liquidate_eth:.6f}")
    
    print(f"\n📋 Final Results:")
    print(f"Collateral to Liquidate: {collateral_to_liquidate_eth:.6f} ETH (${collateral_to_liquidate_usd:,.2f})")
    print(f"Debt to Repay: ${debt_to_repay_usd:,.2f}")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    debug_solidity_math() 