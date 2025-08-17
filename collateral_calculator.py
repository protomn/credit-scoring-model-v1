"""
Collateral Calculator - Matches Smart Contract Logic Exactly
Ensures Python backend and Solidity produce identical results
"""

from typing import Dict, Tuple
import math

class CollateralCalculator:
    """
    Calculates collateral requirements matching smart contract logic
    """
    
    # Constants matching smart contract
    PRECISION = 1e18
    PERCENTAGE_PRECISION = 10000  # basis points
    
    @staticmethod
    def calculate_required_collateral_ratio(credit_score: float) -> float:
        """
        Calculate required collateral ratio based on credit score
        Matches smart contract calculateRequiredCollateral function
        """
        if credit_score >= 0.75:  # Excellent credit (750+)
            return 1.5  # 150% collateral
        elif credit_score >= 0.65:  # Good credit (650+)
            return 1.6  # 160% collateral
        elif credit_score >= 0.55:  # Fair credit (550+)
            return 1.8  # 180% collateral
        else:  # Poor credit (<550)
            return 2.0  # 200% collateral
    
    @staticmethod
    def calculate_required_collateral_usd(loan_amount: float, credit_score: float) -> float:
        """
        Calculate required collateral in USD
        Matches smart contract: borrowAmount18 * ratio / PERCENTAGE_PRECISION
        """
        ratio = CollateralCalculator.calculate_required_collateral_ratio(credit_score)
        return loan_amount * ratio
    
    @staticmethod
    def calculate_required_collateral_eth(loan_amount: float, credit_score: float, eth_price: float) -> float:
        """
        Calculate required collateral in ETH
        Matches smart contract: ceil(requiredCollateralUSD18 * PRECISION / ethPrice)
        """
        required_usd = CollateralCalculator.calculate_required_collateral_usd(loan_amount, credit_score)
        # Use ceiling division to match smart contract _ceilDiv function
        return math.ceil(required_usd / eth_price * CollateralCalculator.PRECISION) / CollateralCalculator.PRECISION
    
    @staticmethod
    def calculate_ltv_and_health_factor(collateral_amount: float, loan_amount: float, eth_price: float) -> Tuple[float, float]:
        """
        Calculate LTV and health factor matching smart contract
        Returns: (ltv_basis_points, health_factor_basis_points)
        """
        if loan_amount == 0:
            return (0, float('inf'))
        if collateral_amount == 0:
            return (float('inf'), 0)
        
        current_collateral_value = collateral_amount * eth_price
        
        # LTV in basis points (loan_amount / collateral_value * 10000)
        ltv_basis_points = (loan_amount / current_collateral_value) * CollateralCalculator.PERCENTAGE_PRECISION
        
        # Health factor in basis points (collateral_value / loan_amount * 10000)
        health_factor_basis_points = (current_collateral_value / loan_amount) * CollateralCalculator.PERCENTAGE_PRECISION
        
        return (ltv_basis_points, health_factor_basis_points)
    
    @staticmethod
    def verify_collateral_sufficiency(collateral_amount: float, loan_amount: float, 
                                   eth_price: float, credit_score: float) -> Dict:
        """
        Verify if collateral is sufficient for loan approval
        """
        required_collateral = CollateralCalculator.calculate_required_collateral_eth(
            loan_amount, credit_score, eth_price
        )
        
        current_ltv, health_factor = CollateralCalculator.calculate_ltv_and_health_factor(
            collateral_amount, loan_amount, eth_price
        )
        
        # Convert to percentages for easier reading
        ltv_percentage = current_ltv / 100
        health_factor_percentage = health_factor / 100
        
        return {
            "required_collateral_eth": required_collateral,
            "current_collateral_eth": collateral_amount,
            "collateral_sufficient": collateral_amount >= required_collateral,
            "ltv_percentage": ltv_percentage,
            "health_factor_percentage": health_factor_percentage,
            "required_ratio": CollateralCalculator.calculate_required_collateral_ratio(credit_score),
            "current_ratio": collateral_amount * eth_price / loan_amount if loan_amount > 0 else 0
        } 