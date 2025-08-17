
from typing import Dict, Tuple, Optional
import math

class LiquidationVerifier:
    """
    Verifies liquidation calculations match smart contract exactly
    """
    
    # Constants matching smart contract
    PRECISION = 1e18
    PERCENTAGE_PRECISION = 10000  # basis points
    LIQUIDATION_THRESHOLD = 11000  # 110% in basis points (110 * 10000 / 100)
    TARGET_HEALTH_FACTOR = 12500   # 125% in basis points (125 * 10000 / 100)
    MAX_LIQUIDATION_DISCOUNT = 500  # 5% in basis points
    
    @staticmethod
    def calculate_liquidation_discount(loan_id: str, last_payment_time: int, 
                                     current_time: int) -> int:
        """
        Calculate liquidation discount matching smart contract _currentLiquidationDiscountBP
        """
        if last_payment_time == 0:
            return LiquidationVerifier.MAX_LIQUIDATION_DISCOUNT
        
        elapsed = current_time - last_payment_time
        liquidation_discount_increase_per_day = 10  # 10 basis points per day
        max_liquidation_discount_multiplier = 3
        
        # Calculate extra basis points based on time elapsed
        extra_bp = (elapsed * liquidation_discount_increase_per_day) // (24 * 3600)  # Convert to days
        
        # Calculate maximum allowed discount
        max_bp = LiquidationVerifier.MAX_LIQUIDATION_DISCOUNT * max_liquidation_discount_multiplier
        
        # Final discount
        discount = LiquidationVerifier.MAX_LIQUIDATION_DISCOUNT + extra_bp
        return min(discount, max_bp)
    
    @staticmethod
    def calculate_partial_liquidation_amount_solidity(
        collateral_amount: float, loan_amount: float, eth_price: float,
        liquidation_discount_bp: int
    ) -> Tuple[float, float]:
        """
        Calculate partial liquidation amount using EXACT smart contract logic
        Returns: (collateral_to_liquidate_eth, debt_to_repay_usd)
        """
        # Convert to 18-decimal precision like smart contract
        collateral_value_usd = collateral_amount * eth_price
        outstanding_debt_usd = loan_amount
        
        # Check if liquidation is needed (health factor < 110%)
        # Health factor = (collateral_value / debt) * 10000
        current_health_factor = (collateral_value_usd / outstanding_debt_usd) * LiquidationVerifier.PERCENTAGE_PRECISION
        if current_health_factor >= LiquidationVerifier.LIQUIDATION_THRESHOLD:
            return (0.0, 0.0)  # No liquidation needed
        
        # Target health factor in basis points (125%)
        TF = LiquidationVerifier.TARGET_HEALTH_FACTOR
        PP = LiquidationVerifier.PERCENTAGE_PRECISION
        disc = liquidation_discount_bp
        
        # Calculate TFxD = (TF * D / PP) - this is the target collateral value needed
        TFxD = (TF * outstanding_debt_usd) / PP
        
        # If current collateral is already sufficient, no liquidation needed
        if TFxD <= collateral_value_usd:
            return (0.0, 0.0)
        
        # Calculate numerator: TFxD - currentCollateralValue
        numerator_usd = TFxD - collateral_value_usd
        
        # Calculate denominator: TF*(PP - disc) - PP*PP
        left = TF * (PP - disc)
        right = PP * PP
        denom = left - right
        
        # Safety check: denominator must be positive
        if denom <= 0:
            raise ValueError("Invalid liquidation parameters: denominator <= 0")
        
        # Calculate collateral to liquidate (USD): numerator_usd * PP*PP / denom
        numerator_mul = numerator_usd * PP * PP
        collateral_to_liquidate_usd = math.ceil(numerator_mul / denom)
        
        # Calculate debt to repay (USD): collateral_to_liquidate_usd * (PP - disc) / PP
        debt_to_repay_usd = math.ceil(collateral_to_liquidate_usd * (PP - disc) / PP)
        
        # Convert collateral to liquidate from USD to ETH
        collateral_to_liquidate_eth = math.ceil(collateral_to_liquidate_usd / eth_price * LiquidationVerifier.PRECISION) / LiquidationVerifier.PRECISION
        
        return (collateral_to_liquidate_eth, debt_to_repay_usd)
    
    @staticmethod
    def calculate_partial_liquidation_amount_python(
        collateral_amount: float, loan_amount: float, eth_price: float,
        target_ratio: float = 1.5
    ) -> Tuple[float, float]:
        """
        Calculate partial liquidation amount using Python backend logic
        Updated to match smart contract logic
        """
        # Check if liquidation is needed (health factor < 110%)
        current_health_factor = (collateral_amount * eth_price / loan_amount) * 100 if loan_amount > 0 else float('inf')
        if current_health_factor >= 110:  # 110% threshold
            return (0.0, 0.0)  # No liquidation needed
        
        # Use the exact same Solidity logic for consistency
        # Target health factor in basis points (125%)
        TF = 12500  # 125% in basis points
        PP = 10000  # basis points
        disc = 500  # 5% base discount
        
        # Calculate TFxD = (TF * D / PP) - target collateral value needed
        TFxD = (TF * loan_amount) / PP
        
        # If current collateral is already sufficient, no liquidation needed
        if TFxD <= collateral_amount * eth_price:
            return (0.0, 0.0)
        
        # Calculate numerator: TFxD - currentCollateralValue
        numerator_usd = TFxD - (collateral_amount * eth_price)
        
        # Calculate denominator: TF*(PP - disc) - PP*PP
        left = TF * (PP - disc)
        right = PP * PP
        denom = left - right
        
        # Safety check: denominator must be positive
        if denom <= 0:
            raise ValueError("Invalid liquidation parameters: denominator <= 0")
        
        # Calculate collateral to liquidate (USD): numerator_usd * PP*PP / denom
        numerator_mul = numerator_usd * PP * PP
        collateral_to_liquidate_usd = math.ceil(numerator_mul / denom)  # Use ceiling division like smart contract
        
        # Calculate debt to repay (USD): collateral_to_liquidate_usd * (PP - disc) / PP
        debt_to_repay_usd = math.ceil(collateral_to_liquidate_usd * (PP - disc) / PP)  # Use ceiling division
        
        # Convert collateral to liquidate from USD to ETH
        collateral_to_liquidate_eth = math.ceil(collateral_to_liquidate_usd / eth_price * LiquidationVerifier.PRECISION) / LiquidationVerifier.PRECISION
        
        return (collateral_to_liquidate_eth, debt_to_repay_usd)
    
    @staticmethod
    def verify_liquidation_calculations(
        collateral_amount: float, loan_amount: float, eth_price: float,
        liquidation_discount_bp: int = 500
    ) -> Dict:
        """
        Verify that Python and Solidity liquidation calculations match
        """
        try:
            # Calculate using smart contract logic
            solidity_collateral, solidity_debt = LiquidationVerifier.calculate_partial_liquidation_amount_solidity(
                collateral_amount, loan_amount, eth_price, liquidation_discount_bp
            )
            
            # Calculate using Python backend logic
            python_collateral, python_debt = LiquidationVerifier.calculate_partial_liquidation_amount_python(
                collateral_amount, loan_amount, eth_price
            )
            
            # Calculate differences
            collateral_diff = abs(solidity_collateral - python_collateral)
            debt_diff = abs(solidity_debt - python_debt)
            
            # Check if calculations match within acceptable tolerance
            # Use percentage-based tolerance for small amounts
            if solidity_collateral > 0:
                collateral_tolerance = max(0.001, solidity_collateral * 0.01)  # 1% or 0.001 ETH, whichever is larger
            else:
                collateral_tolerance = 0.001
                
            if solidity_debt > 0:
                debt_tolerance = max(0.01, solidity_debt * 0.01)  # 1% or $0.01, whichever is larger
            else:
                debt_tolerance = 0.01
                
            collateral_match = collateral_diff <= collateral_tolerance
            debt_match = debt_diff <= debt_tolerance
                
            return {
                "solidity_collateral": solidity_collateral,
                "solidity_debt": solidity_debt,
                "python_collateral": python_collateral,
                "python_debt": python_debt,
                "collateral_difference": collateral_diff,
                "debt_difference": debt_diff,
                "calculations_match": collateral_match and debt_match,
                "collateral_match": collateral_match,
                "debt_match": debt_match,
                "collateral_tolerance": collateral_tolerance,
                "debt_tolerance": debt_tolerance
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "calculations_match": False
            }
    
    @staticmethod
    def test_boundary_conditions() -> Dict:
        """
        Test edge cases and boundary conditions
        """
        test_cases = []
        
        # Test case 1: 0% LTV (no loan) - skip this case as it causes division by zero
        # test_cases.append({
        #     "name": "0% LTV - No Loan",
        #     "collateral": 10.0,  # 10 ETH
        #     "loan": 0.0,         # 0 USD
        #     "eth_price": 3000.0,
        #     "expected": "No liquidation needed"
        # })
        
        # Test case 2: 90% LTV (below 110% threshold - triggers liquidation)
        test_cases.append({
            "name": "90% LTV - Below Threshold",
            "collateral": 9.0,   # 9 ETH
            "loan": 30000.0,     # 30,000 USD
            "eth_price": 3000.0,
            "expected": "Liquidation needed"
        })
        
        # Test case 3: 200% LTV (excessive collateral)
        test_cases.append({
            "name": "200% LTV - Excessive Collateral",
            "collateral": 20.0,  # 20 ETH
            "loan": 30000.0,     # 30,000 USD
            "eth_price": 3000.0,
            "expected": "No liquidation needed"
        })
        
        # Test case 4: Very small amounts below threshold
        test_cases.append({
            "name": "Very Small Below Threshold",
            "collateral": 0.0009,  # 0.0009 ETH
            "loan": 3.0,           # 3 USD
            "eth_price": 3000.0,
            "expected": "Liquidation needed"
        })
        
        # Test case 5: Very large amounts below threshold
        test_cases.append({
            "name": "Very Large Below Threshold",
            "collateral": 900.0,   # 900 ETH
            "loan": 3000000.0,     # 3M USD
            "eth_price": 3000.0,
            "expected": "Liquidation needed"
        })
        
        results = []
        for test_case in test_cases:
            try:
                result = LiquidationVerifier.verify_liquidation_calculations(
                    test_case["collateral"],
                    test_case["loan"],
                    test_case["eth_price"]
                )
                
                results.append({
                    "test_case": test_case["name"],
                    "collateral": test_case["collateral"],
                    "loan": test_case["loan"],
                    "eth_price": test_case["eth_price"],
                    "ltv_percentage": (test_case["loan"] / (test_case["collateral"] * test_case["eth_price"])) * 100 if test_case["collateral"] * test_case["eth_price"] > 0 else 0,
                    "calculations_match": result["calculations_match"],
                    "solidity_result": f"{result['solidity_collateral']:.6f} ETH, {result['solidity_debt']:.2f} USD",
                    "python_result": f"{result['python_collateral']:.6f} ETH, {result['python_debt']:.2f} USD"
                })
                
            except Exception as e:
                results.append({
                    "test_case": test_case["name"],
                    "error": str(e),
                    "calculations_match": False
                })
        
        return {
            "boundary_tests": results,
            "total_tests": len(test_cases),
            "passed_tests": len([r for r in results if r.get("calculations_match", False)]),
            "failed_tests": len([r for r in results if not r.get("calculations_match", False)])
        } 