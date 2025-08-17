
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import logging
from typing import Dict, Any, Optional

# Import the systems I built for you
from credit_scoring import CreditScoringEngine
from security import BlockchainDataCollector, CreditScoringAPI
from config import config


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the web application
app = FastAPI(title="DeFi Lending Credit Scoring API", version="1.0.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store systems
credit_engine: Optional[CreditScoringEngine] = None
blockchain_collector: Optional[BlockchainDataCollector] = None
credit_api: Optional[CreditScoringAPI] = None


class LoanRequest(BaseModel):
    borrower_address: str
    collateral_amount: float
    requested_amount: float
    loan_duration_days: int
    interest_rate: float

class CreditScoreRequest(BaseModel):
    address: str


@app.on_event("startup")
async def startup_event():
    global credit_engine, blockchain_collector, credit_api
    
    logger.info("Starting DeFi Lending System...")
    
    try:
        # Initialize the credit scoring engine
        logger.info("Initializing credit scoring engine...")
        credit_engine = CreditScoringEngine(k_neighbors=5)
        
        # Initialize blockchain data collector
        logger.info("⛓️ Initializing blockchain collector...")
        
        api_keys = {
            'etherscan': config.ETHERSCAN_API_KEY
        }
        
        blockchain_collector = BlockchainDataCollector(config.WEB3_PROVIDER, api_keys)
        
        
        logger.info("Initializing credit scoring API...")
        credit_api = CreditScoringAPI(credit_engine, blockchain_collector, None)
        
        logger.info("All systems initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize systems: {str(e)}")
        raise

# API endpoint to get credit score
@app.post("/api/credit-score")
async def get_credit_score(request: CreditScoreRequest):
    """
    Get credit score for a wallet address
    """
    
    if not credit_api:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        logger.info(f"Calculating credit score for {request.address}")
        
        # Use our credit scoring system
        result = await credit_api.get_comprehensive_credit_score(request.address)
        
        logger.info(f"Credit score calculated: {result.get('composite_score', 0):.3f}")
        
        return {
            "success": True,
            "address": request.address,
            "credit_data": result
        }
        
    except Exception as e:
        logger.error(f"Error calculating credit score: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Credit scoring failed: {str(e)}")

# API endpoint to process loan request
@app.post("/api/loan/request")
async def process_loan_request(request: LoanRequest):
    """
    Process a loan request with intelligent risk assessment
    """
    
    if not credit_api:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        logger.info(f"💰 Processing loan request for {request.borrower_address}")
        
    
        credit_result = await credit_api.get_comprehensive_credit_score(request.borrower_address)
        credit_score = credit_result.get('composite_score', 0.5)
        
        
        risk_factors = {
            'credit_risk': 1 - credit_score,
            'loan_to_collateral_ratio': request.requested_amount / (request.collateral_amount * 3000),  
            'duration_risk': min(request.loan_duration_days / 365, 1.0)
        }
        
        combined_risk = (
            risk_factors['credit_risk'] * 0.5 +
            risk_factors['loan_to_collateral_ratio'] * 0.3 +
            risk_factors['duration_risk'] * 0.2
        )
        
    
        if credit_score >= 0.4 and combined_risk <= 0.6:
            approved = True
            
            risk_premium = combined_risk * 0.05  # Up to 5% extra
            adjusted_rate = request.interest_rate + risk_premium
        
        else:
            approved = False
            adjusted_rate = request.interest_rate
        
        decision = {
            "approved": approved,
            "credit_score": credit_score,
            "risk_score": combined_risk,
            "adjusted_interest_rate": adjusted_rate,
            "max_loan_amount": request.requested_amount if approved else 0,
            "reasons": "Good credit history and manageable risk" if approved else "Credit score or risk too high"
        }
        
        logger.info(f"Loan decision: {'APPROVED' if approved else 'REJECTED'}")
        
        return {
            "success": True,
            "loan_decision": decision,
            "credit_data": credit_result
        }
        
    except Exception as e:
        logger.error(f"Error processing loan request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Loan processing failed: {str(e)}")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Check if the system is running properly"""
    
    return {
        "status": "healthy",
        "systems": {
            "credit_engine": credit_engine is not None,
            "blockchain_collector": blockchain_collector is not None,
            "credit_api": credit_api is not None
        }
    }

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "DeFi Lending Credit Scoring API",
        "version": "1.0.0",
        "endpoints": {
            "credit_score": "/api/credit-score",
            "loan_request": "/api/loan/request",
            "health": "/api/health",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)