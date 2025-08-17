
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import time
import hashlib
import json
import random
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the web application
app = FastAPI(title="DeFi Lending Credit Scoring API", version="1.0.0")

@app.get("/health")
async def health_check():
    """Health check endpoint for cloud deployment"""
    return {"status": "healthy", "timestamp": time.time()}

# Allow CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (replace with database in production)
credit_scores_cache = {}
loan_requests_storage = {}

# Data models for API
class CreditScoreRequest(BaseModel):
    address: str

class LoanRequest(BaseModel):
    borrower_address: str
    collateral_amount: float
    requested_amount: float
    loan_duration_days: int
    interest_rate: float

# Mock blockchain data (replace with real Web3 calls later)
MOCK_BLOCKCHAIN_DATA = {
    "0xd8da6bf26964af9d7eed9e03e53415d37aa96045": {  # Vitalik's address
        "transaction_count": 12847,
        "total_volume": 156432.5,
        "defi_interactions": 342,
        "gas_efficiency": 0.89,
        "unique_tokens": 47,
        "liquidation_history": 0,
        "on_time_payments": 0.95
    },
    "0x742d35cc6634c0532925a3b8d654b63cf196c8e4": {  # Example good address
        "transaction_count": 2341,
        "total_volume": 23451.2,
        "defi_interactions": 67,
        "gas_efficiency": 0.67,
        "unique_tokens": 12,
        "liquidation_history": 1,
        "on_time_payments": 0.78
    }
}

def validate_ethereum_address(address: str) -> bool:
    """Validate Ethereum address format"""
    if not address.startswith("0x"):
        return False
    if len(address) != 42:
        return False
    try:
        int(address[2:], 16)  # Check if it's valid hex
        return True
    except ValueError:
        return False

def calculate_credit_score_aloe_method(address: str) -> dict:
    """
    Calculate credit score using the ALOE (Autonomous Lending Organization on Ethereum) methodology
    
    This implements the 5 core metrics from the research paper:
    1. Underpay Ratio (UPR)
    2. Current Debt Burden Ratio (CDBR) 
    3. Current Payment Burden Ratio (CPBR)
    4. Repayment Age Ratio (RAR)
    5. Average Number of Credit Lines (ANCL)
    """
    
    # Get mock data or generate deterministic scores
    if address.lower() in MOCK_BLOCKCHAIN_DATA:
        data = MOCK_BLOCKCHAIN_DATA[address.lower()]
    else:
        # Generate deterministic but varied data based on address
        address_hash = hashlib.sha256(address.encode()).hexdigest()
        hash_int = int(address_hash[:16], 16)
        
        data = {
            "transaction_count": (hash_int % 5000) + 100,
            "total_volume": ((hash_int % 100000) + 1000) / 10,
            "defi_interactions": (hash_int % 200) + 5,
            "gas_efficiency": ((hash_int % 80) + 20) / 100,
            "unique_tokens": (hash_int % 30) + 3,
            "liquidation_history": hash_int % 3,
            "on_time_payments": ((hash_int % 60) + 40) / 100
        }
    
    # ALOE Metric 1: Underpay Ratio (lower is better)
    # Based on payment punctuality
    underpay_ratio = max(0, 1 - data["on_time_payments"])
    
    # ALOE Metric 2: Current Debt Burden Ratio (lower is better)
    # Simulate based on transaction patterns
    avg_tx_size = data["total_volume"] / data["transaction_count"]
    debt_burden_ratio = min(1.0, avg_tx_size / 10000)  # Normalize to 0-1
    
    # ALOE Metric 3: Current Payment Burden Ratio (lower is better)
    # Based on recent transaction size vs historical average
    recent_payment_burden = min(1.0, data["liquidation_history"] * 0.3)
    
    # ALOE Metric 4: Repayment Age Ratio (higher is better for established users)
    # Based on account age proxy (transaction count)
    repayment_age_ratio = min(1.0, data["transaction_count"] / 10000)
    
    # ALOE Metric 5: Average Number of Credit Lines (moderate is best)
    # Based on DeFi interaction diversity
    credit_lines_score = min(1.0, data["defi_interactions"] / 500)
    
    # Calculate base ALOE score using weighted average (as per paper)
    aloe_weights = [0.35, 0.30, 0.15, 0.10, 0.10]  # UPR, CDBR, CPBR, RAR, ANCL
    
    # Invert negative metrics for scoring (lower underpay/debt burden = higher score)
    aloe_metrics = [
        1 - underpay_ratio,           # Higher is better
        1 - debt_burden_ratio,        # Higher is better  
        1 - recent_payment_burden,    # Higher is better
        repayment_age_ratio,          # Higher is better
        credit_lines_score            # Higher is better
    ]
    
    # Weighted sum
    basic_aloe_score = sum(metric * weight for metric, weight in zip(aloe_metrics, aloe_weights))
    
    # Enhanced scoring with blockchain-specific factors
    defi_bonus = min(0.1, data["defi_interactions"] / 1000)  # Up to 10% bonus for DeFi activity
    gas_efficiency_bonus = (data["gas_efficiency"] - 0.5) * 0.1  # Bonus/penalty for gas efficiency
    diversity_bonus = min(0.05, data["unique_tokens"] / 200)  # Up to 5% bonus for token diversity
    
    # Penalty for liquidation history
    liquidation_penalty = data["liquidation_history"] * 0.1
    
    # Final composite score
    composite_score = basic_aloe_score + defi_bonus + gas_efficiency_bonus + diversity_bonus - liquidation_penalty
    composite_score = max(0.0, min(1.0, composite_score))  # Clamp to [0,1]
    
    # Risk assessment
    volatility_risk = max(0.1, 1 - data["gas_efficiency"])
    portfolio_risk = max(0.1, 1 - (data["unique_tokens"] / 50))
    
    return {
        "address": address,
        "basic_credit_score": round(basic_aloe_score, 3),
        "composite_score": round(composite_score, 3),
        "aloe_metrics": {
            "underpay_ratio": round(underpay_ratio, 3),
            "debt_burden_ratio": round(debt_burden_ratio, 3),
            "payment_burden_ratio": round(recent_payment_burden, 3),
            "repayment_age_ratio": round(repayment_age_ratio, 3),
            "avg_credit_lines": round(credit_lines_score, 3)
        },
        "defi_metrics": {
            "defi_participation_score": round(min(1.0, data["defi_interactions"] / 200), 3),
            "liquidity_provision_ratio": round(random.uniform(0.1, 0.8), 3),
            "protocol_diversity": min(data["defi_interactions"] // 20, 8),
            "gas_efficiency": round(data["gas_efficiency"], 3)
        },
        "risk_assessment": {
            "volatility_risk": round(volatility_risk, 3),
            "portfolio_risk": round(portfolio_risk, 3),
            "liquidation_history": data["liquidation_history"],
            "overall_risk": round((volatility_risk + portfolio_risk) / 2, 3)
        },
        "transaction_metrics": {
            "transaction_count": data["transaction_count"],
            "total_volume_eth": round(data["total_volume"], 2),
            "average_transaction_size": round(data["total_volume"] / data["transaction_count"], 4),
            "unique_tokens": data["unique_tokens"]
        },
        "calculated_at": time.time(),
        "methodology": "ALOE (Autonomous Lending Organization on Ethereum)"
    }

def assess_loan_risk(credit_data: dict, loan_request: LoanRequest) -> dict:
    """
    Assess loan risk combining credit score with market factors
    """
    
    # Extract key metrics
    credit_score = credit_data["composite_score"]
    volatility_risk = credit_data["risk_assessment"]["volatility_risk"]
    
    # Market risk assessment (simplified)
    eth_price = 3000  # Mock current ETH price
    collateral_value = loan_request.collateral_amount * eth_price
    loan_to_collateral_ratio = loan_request.requested_amount / collateral_value
    
    # Risk factors
    credit_risk = 1 - credit_score
    market_risk = max(0, loan_to_collateral_ratio - 0.5)  # Risk if LTV > 50%
    duration_risk = min(loan_request.loan_duration_days / 365, 0.3)  # Max 30% from duration
    volatility_adjustment = volatility_risk * 0.2
    
    # Combined risk score (0 = no risk, 1 = maximum risk)
    combined_risk = (
        credit_risk * 0.4 +
        market_risk * 0.3 + 
        duration_risk * 0.2 +
        volatility_adjustment * 0.1
    )
    
    return {
        "combined_risk_score": round(combined_risk, 3),
        "credit_risk": round(credit_risk, 3),
        "market_risk": round(market_risk, 3),
        "duration_risk": round(duration_risk, 3),
        "volatility_risk": round(volatility_risk, 3),
        "loan_to_collateral_ratio": round(loan_to_collateral_ratio, 3),
        "collateral_value_usd": round(collateral_value, 2)
    }

def make_loan_decision(credit_data: dict, risk_assessment: dict, loan_request: LoanRequest) -> dict:
    """
    Make intelligent loan decision based on credit score and risk assessment
    """
    
    credit_score = credit_data["composite_score"]
    combined_risk = risk_assessment["combined_risk_score"]
    
    # Decision thresholds
    MIN_CREDIT_SCORE = 0.3
    MAX_RISK_SCORE = 0.7
    
    # Base approval logic
    credit_acceptable = credit_score >= MIN_CREDIT_SCORE
    risk_acceptable = combined_risk <= MAX_RISK_SCORE
    collateral_sufficient = risk_assessment["loan_to_collateral_ratio"] <= 0.8  # Max 80% LTV
    
    approved = credit_acceptable and risk_acceptable and collateral_sufficient
    
    # Calculate adjusted terms
    risk_premium = combined_risk * 0.05  # Up to 5% additional interest
    adjusted_interest_rate = loan_request.interest_rate + risk_premium
    
    # Adjust loan amount based on credit score
    credit_adjustment = 0.5 + (credit_score * 0.5)  # 50-100% of requested amount
    max_loan_amount = loan_request.requested_amount * credit_adjustment if approved else 0
    
    # Required collateral ratio (150% base + risk adjustment)
    base_collateral_ratio = 1.5
    risk_collateral_adjustment = combined_risk * 0.5
    required_collateral_ratio = base_collateral_ratio + risk_collateral_adjustment
    
    # Generate decision reasons
    reasons = []
    if not credit_acceptable:
        reasons.append(f"Credit score too low: {credit_score:.2%} (minimum: {MIN_CREDIT_SCORE:.2%})")
    if not risk_acceptable:
        reasons.append(f"Risk score too high: {combined_risk:.2%} (maximum: {MAX_RISK_SCORE:.2%})")
    if not collateral_sufficient:
        reasons.append(f"Insufficient collateral: {risk_assessment['loan_to_collateral_ratio']:.2%} LTV (maximum: 80%)")
    
    if approved:
        reasons.append("Good credit history and acceptable risk profile")
        if risk_premium > 0:
            reasons.append(f"Risk premium applied: +{risk_premium:.2%}")
    
    return {
        "approved": approved,
        "credit_score": credit_score,
        "risk_score": combined_risk,
        "adjusted_interest_rate": round(adjusted_interest_rate, 4),
        "max_loan_amount": round(max_loan_amount, 2),
        "required_collateral_ratio": round(required_collateral_ratio, 2),
        "price_prediction_confidence": 0.75,  # Mock confidence
        "reasons": reasons
    }

# API Endpoints

@app.get("/")
async def root():
    """
    Root endpoint - shows API info
    """
    
    return {
        "message": "DeFi Lending Credit Scoring API",
        "version": "1.0.0",
        "status": "running",
        "methodology": "ALOE (Autonomous Lending Organization on Ethereum)",
        "endpoints": {
            "credit_score": "POST /api/credit-score",
            "loan_request": "POST /api/loan/request",
            "health": "GET /api/health",
            "demo": "GET /api/demo",
            "docs": "GET /docs"
        },
        "demo_addresses": {
            "vitalik": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "example": "0x742d35Cc6634C0532925a3b8D654B63cF196c8e4"
        }
    }

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint
    """
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "systems": {
            "api": True,
            "credit_scoring": True,
            "loan_processing": True,
            "cache_size": len(credit_scores_cache),
            "loans_processed": len(loan_requests_storage)
        },
        "uptime_seconds": time.time() - start_time if 'start_time' in globals() else 0
    }

@app.post("/api/credit-score")
async def get_credit_score(request: CreditScoreRequest):
    """
    Get credit score for a wallet address
    """
    
    try:
        logger.info(f"Calculating credit score for {request.address}")
        
        # Validate address format
        if not validate_ethereum_address(request.address):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address format")
        
        # Check cache first
        cache_key = request.address.lower()
        if cache_key in credit_scores_cache:
            cached_result = credit_scores_cache[cache_key]
            if time.time() - cached_result["calculated_at"] < 300:  # 5 minute cache
                logger.info("Returning cached credit score")
                return {
                    "success": True,
                    "address": request.address,
                    "credit_data": cached_result,
                    "cached": True
                }
        
        # Calculate credit score using our built-in ALOE method
        credit_data = calculate_credit_score_aloe_method(request.address)
        
        # Cache the result
        credit_scores_cache[cache_key] = credit_data
        
        logger.info(f"Credit score calculated: {credit_data['composite_score']:.3f}")
        
        return {
            "success": True,
            "address": request.address,
            "credit_data": credit_data,
            "cached": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating credit score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credit scoring failed: {str(e)}")

@app.post("/api/loan/request")
async def process_loan_request(request: LoanRequest):
    """
    Process a loan request with intelligent risk assessment
    """
    
    try:
        logger.info(f"💰 Processing loan request for {request.borrower_address}")
        
        # Validate request
        if request.collateral_amount <= 0:
            raise HTTPException(status_code=400, detail="Collateral amount must be positive")
        
        if request.requested_amount <= 0:
            raise HTTPException(status_code=400, detail="Loan amount must be positive")
        
        if not validate_ethereum_address(request.borrower_address):
            raise HTTPException(status_code=400, detail="Invalid Ethereum address format")
        
        # Get credit score using our built-in method
        credit_data = calculate_credit_score_aloe_method(request.borrower_address)
        
        # Assess loan risk
        risk_assessment = assess_loan_risk(credit_data, request)
        
        # Make loan decision
        loan_decision = make_loan_decision(credit_data, risk_assessment, request)
        
        # Store the request
        request_id = f"req_{int(time.time())}_{hash(request.borrower_address) % 1000}"
        loan_requests_storage[request_id] = {
            "request": request.dict(),
            "decision": loan_decision,
            "risk_assessment": risk_assessment,
            "timestamp": time.time()
        }
        
        logger.info(f"Loan decision: {'APPROVED' if loan_decision['approved'] else 'REJECTED'}")
        
        return {
            "success": True,
            "request_id": request_id,
            "status": "approved" if loan_decision["approved"] else "rejected",
            "loan_decision": loan_decision,
            "risk_assessment": risk_assessment,
            "credit_data": credit_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing loan request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loan processing failed: {str(e)}")

@app.get("/api/demo")
async def demo_endpoint():
    """
    Demo endpoint with sample data
    """
    
    # Sample addresses for testing
    demo_addresses = [
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # Vitalik
        "0x742d35Cc6634C0532925a3b8D654B63cF196c8e4",  # Example
        "0x1234567890123456789012345678901234567890"   # Random
    ]
    
    results = {}
    
    for address in demo_addresses:
        
        try:
            credit_data = calculate_credit_score_aloe_method(address)
            results[address] = {
                "credit_score": credit_data["composite_score"],
                "basic_score": credit_data["basic_credit_score"],
                "risk_level": "Low" if credit_data["composite_score"] > 0.7 else "Medium" if credit_data["composite_score"] > 0.4 else "High",
                "defi_participation": credit_data["defi_metrics"]["defi_participation_score"],
                "gas_efficiency": credit_data["defi_metrics"]["gas_efficiency"]
            }
        except Exception as e:
            results[address] = {"error": str(e)}
    
    return {
        "demo_results": results,
        "instructions": "Use these addresses to test the credit scoring system",
        "endpoints": {
            "test_credit": "POST /api/credit-score with address",
            "test_loan": "POST /api/loan/request with full loan details"
        },
        "sample_loan_request": {
            "borrower_address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "collateral_amount": 5.0,
            "requested_amount": 12000,
            "loan_duration_days": 30,
            "interest_rate": 0.08
        }
    }

@app.get("/api/loan/requests")
async def get_loan_requests():
    """
    Get all loan requests (for debugging/monitoring)
    """
    
    return {
        "total_requests": len(loan_requests_storage),
        "requests": list(loan_requests_storage.values())
    }

@app.get("/api/statistics")
async def get_statistics():
    """
    Get system statistics
    """
    
    approved_loans = [req for req in loan_requests_storage.values() if req["decision"]["approved"]]
    rejected_loans = [req for req in loan_requests_storage.values() if not req["decision"]["approved"]]
    
    return {
        "total_credit_scores_calculated": len(credit_scores_cache),
        "total_loan_requests": len(loan_requests_storage),
        "approved_loans": len(approved_loans),
        "rejected_loans": len(rejected_loans),
        "approval_rate": len(approved_loans) / len(loan_requests_storage) if loan_requests_storage else 0,
        "average_credit_score": sum(data["composite_score"] for data in credit_scores_cache.values()) / len(credit_scores_cache) if credit_scores_cache else 0,
        "cache_hit_rate": "Available in production",
        "system_uptime": time.time() - start_time if 'start_time' in globals() else 0
    }

# Initialize startup time
start_time = time.time()

@app.get("/api/smart-contract/credit-score/{address}")
async def get_credit_score_for_contract(address: str):
    """
    Smart contract compatible endpoint
    """
    
    try:
        credit_data = calculate_credit_score_aloe_method(address)
        
        # Convert to smart contract format
        credit_score = int(credit_data["composite_score"] * 1000)  # Scale to 0-1000
        risk_score = int(credit_data["risk_assessment"]["overall_risk"] * 1000)
        approved = credit_data["composite_score"] >= 0.4
        
        return {
            "creditScore": credit_score,
            "riskScore": risk_score, 
            "approved": approved,
            "adjustedInterestRate": int((0.08 + (risk_score / 1000) * 0.05) * 10000)  # Basis points
        }
    except Exception as e:
        return {
            "creditScore": 0,
            "riskScore": 1000,
            "approved": False,
            "error": str(e)
        }

# Startup event
@app.on_event("startup")
async def startup_event():
    global start_time
    start_time = time.time()
    logger.info("DeFi Lending Credit Scoring API Started Successfully!")
    logger.info("Available at: http://localhost:8001")
    logger.info("API Docs at: http://localhost:8001/docs")
    logger.info("Demo data at: http://localhost:8001/api/demo")
    logger.info("System stats at: http://localhost:8001/api/statistics")
    logger.info("Methodology: ALOE (Autonomous Lending Organization on Ethereum)")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)