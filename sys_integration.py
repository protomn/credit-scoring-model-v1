"""
Complete System Integration Architecture
Connects Credit Scoring + ML Price Prediction + Smart Contracts + Frontend
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from web3 import Web3
from eth_account import Account
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import redis
from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

# Import our credit scoring system
from credit_scoring import CreditScoringEngine
from security import BlockchainDataCollector, CreditScoringAPI

logger = logging.getLogger(__name__)

@dataclass
class LoanRequest:
    """
    Loan request from frontend
    """
    
    borrower_address: str
    collateral_amount: float  # ETH amount
    requested_amount: float   # USDT/USDC amount
    loan_duration_days: int
    interest_rate: float
    collateral_ratio: float   # e.g., 150% = 1.5

@dataclass
class LoanApproval:
    """
    Loan approval decision
    """
    
    approved: bool
    credit_score: float
    adjusted_interest_rate: float
    max_loan_amount: float
    collateral_requirement: float
    risk_score: float
    price_prediction_confidence: float

@dataclass
class PriceAlert:
    """
    Price alert for liquidation prevention
    """
    
    loan_id: str
    current_eth_price: float
    predicted_price: float
    liquidation_threshold: float
    risk_level: str  # 'low', 'medium', 'high', 'critical'
    recommended_action: str

class InferenceEngine:
    """
    Central Inference Engine - The Brain of the System
    Coordinates between Credit Scoring, Price Prediction, and Smart Contracts
    """
    
    def __init__(self, web3_provider: str, contract_address: str, contract_abi: List):
    
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.contract = self.w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Initialize subsystems
        self.credit_engine = CreditScoringEngine()
        self.price_predictor = ETHPricePredictor()
        self.liquidation_monitor = LiquidationMonitor()
        self.smart_contract_interface = SmartContractInterface(self.w3, self.contract)
        
        # Redis for real-time coordination
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        # Active loans monitoring
        self.active_loans: Dict[str, Dict] = {}
        
        logger.info("Inference Engine initialized successfully")
    
    async def process_loan_request(self, loan_request: LoanRequest) -> LoanApproval:
        """
        Main orchestration function - processes a complete loan request
        This is where everything comes together!
        """
        try:
            logger.info(f"Processing loan request for {loan_request.borrower_address}")
            
            # STEP 1: Get Credit Score
            credit_score = await self._get_credit_assessment(loan_request)
            
            # STEP 2: Get ETH Price Prediction
            price_prediction = await self._get_price_prediction(loan_request.loan_duration_days)
            
            # STEP 3: Calculate Risk Assessment
            risk_assessment = await self._calculate_comprehensive_risk(
                loan_request, credit_score, price_prediction
            )
            
            # STEP 4: Make Loan Decision
            loan_decision = await self._make_loan_decision(
                loan_request, credit_score, risk_assessment, price_prediction
            )
            
            # STEP 5: If approved, prepare smart contract parameters
            if loan_decision.approved:
                await self._prepare_smart_contract_execution(loan_request, loan_decision)
            
            logger.info(f"Loan decision: {'APPROVED' if loan_decision.approved else 'REJECTED'}")
            return loan_decision
            
        except Exception as e:
            logger.error(f"Error processing loan request: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Loan processing failed: {str(e)}")
    
    async def _get_credit_assessment(self, loan_request: LoanRequest) -> Dict[str, float]:
        """
        Get comprehensive credit score from our credit scoring system
        """
        

        collector = BlockchainDataCollector('https://mainnet.infura.io/v3/c783c42e60d140aca3debcda20873e3c')
        credit_api = CreditScoringAPI(self.credit_engine, collector, None)
        

        credit_result = await credit_api.get_comprehensive_credit_score(
            loan_request.borrower_address,
            {
                'amount': loan_request.requested_amount,
                'interest_rate': loan_request.interest_rate,
                'num_payments': loan_request.loan_duration_days // 30,  # Monthly payments
                'payment_interval': 30
            }
        )
        
        return {
            'basic_score': credit_result.get('basic_credit_score', 0.5),
            'composite_score': credit_result.get('composite_score', 0.5),
            'defi_score': credit_result.get('defi_metrics', {}).get('defi_participation_score', 0.0),
            'risk_factors': credit_result.get('risk_assessment', {})
        }
    
    async def _get_price_prediction(self, duration_days: int) -> Dict[str, Any]:
        """
        Get ETH price prediction for loan duration
        """
        
        prediction_result = await self.price_predictor.predict_price_movement(
            days_ahead=duration_days,
            confidence_intervals=True
        )
        
        return prediction_result
    
    async def _calculate_comprehensive_risk(self, loan_request: LoanRequest, 
                                          credit_score: Dict, price_prediction: Dict) -> Dict[str, float]:
        """
        Calculate overall risk score combining credit risk + market risk
        """
        
        # Credit Risk Component (0-1)
        credit_risk = 1 - credit_score['composite_score']
        
        # Market Risk Component - probability of liquidation
        current_eth_price = await self.price_predictor.get_current_price()
        liquidation_price = (loan_request.requested_amount * loan_request.collateral_ratio) / loan_request.collateral_amount
        
        # Calculate probability of hitting liquidation price
        predicted_price = price_prediction['predicted_price']
        price_volatility = price_prediction['volatility']
        
        # Use normal distribution to estimate liquidation probability
        z_score = (liquidation_price - predicted_price) / (price_volatility * predicted_price)
        liquidation_probability = max(0, min(1, 0.5 + z_score / 6))  # Normalize to 0-1
        
        # Combined Risk Score
        combined_risk = (credit_risk * 0.6) + (liquidation_probability * 0.4)
        
        return {
            'credit_risk': credit_risk,
            'market_risk': liquidation_probability,
            'combined_risk': combined_risk,
            'liquidation_price': liquidation_price,
            'current_price': current_eth_price,
            'predicted_price': predicted_price
        }
    
    async def _make_loan_decision(self, loan_request: LoanRequest, credit_score: Dict, 
                                risk_assessment: Dict, price_prediction: Dict) -> LoanApproval:
        """
        Make final loan approval decision based on all factors
        """
        
        # Decision thresholds
        MIN_CREDIT_SCORE = 0.3
        MAX_RISK_SCORE = 0.7
        MIN_CONFIDENCE = 0.6
        
        # Basic approval criteria
        credit_ok = credit_score['composite_score'] >= MIN_CREDIT_SCORE
        risk_ok = risk_assessment['combined_risk'] <= MAX_RISK_SCORE
        confidence_ok = price_prediction['confidence'] >= MIN_CONFIDENCE
        
        approved = credit_ok and risk_ok and confidence_ok
        
        # Adjust terms based on risk
        base_interest = loan_request.interest_rate
        risk_premium = risk_assessment['combined_risk'] * 0.05  # Up to 5% extra
        adjusted_interest = base_interest + risk_premium
        
        # Adjust loan amount based on credit score
        credit_factor = credit_score['composite_score']
        max_loan_amount = loan_request.requested_amount * credit_factor
        
        # Adjust collateral requirement based on market risk
        market_risk_factor = 1 + risk_assessment['market_risk'] * 0.5  # Up to 50% more collateral
        collateral_requirement = loan_request.collateral_ratio * market_risk_factor
        
        return LoanApproval(
            approved=approved,
            credit_score=credit_score['composite_score'],
            adjusted_interest_rate=adjusted_interest,
            max_loan_amount=max_loan_amount,
            collateral_requirement=collateral_requirement,
            risk_score=risk_assessment['combined_risk'],
            price_prediction_confidence=price_prediction['confidence']
        )
    
    async def _prepare_smart_contract_execution(self, loan_request: LoanRequest, 
                                              loan_decision: LoanApproval):
        """Prepare parameters for smart contract execution"""
        
        contract_params = {
            'borrower': loan_request.borrower_address,
            'collateral_amount': int(loan_request.collateral_amount * 1e18),  # Convert to Wei
            'loan_amount': int(loan_decision.max_loan_amount * 1e6),  # USDT has 6 decimals
            'interest_rate': int(loan_decision.adjusted_interest_rate * 1e4),  # Basis points
            'duration': loan_request.loan_duration_days * 86400,  # Convert to seconds
            'liquidation_threshold': int(loan_decision.collateral_requirement * 1e4)
        }
        
        # Store in Redis for smart contract to pickup
        await self.redis_client.setex(
            f"loan_params:{loan_request.borrower_address}",
            3600,  # 1 hour expiry
            json.dumps(contract_params)
        )
        
        logger.info(f"Smart contract parameters prepared for {loan_request.borrower_address}")
    
    async def monitor_active_loans(self):
        """
        Continuous monitoring of active loans for liquidation prevention
        """
        
        while True:
            try:
                # Get all active loans from smart contract
                active_loans = await self.smart_contract_interface.get_active_loans()
                
                for loan in active_loans:
                    # Check if loan needs attention
                    alert = await self._check_loan_health(loan)
                    
                    if alert.risk_level in ['high', 'critical']:
                        # Send alert to frontend via WebSocket
                        await self._send_liquidation_alert(alert)
                        
                        # If critical, trigger partial liquidation
                        if alert.risk_level == 'critical':
                            await self._trigger_partial_liquidation(loan['loan_id'])
                
                # Sleep for 30 seconds before next check
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in loan monitoring: {str(e)}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _check_loan_health(self, loan: Dict) -> PriceAlert:
        """
        Check individual loan health and liquidation risk
        """
        
        # Get current ETH price
        current_price = await self.price_predictor.get_current_price()
        
        # Get price prediction for next 24 hours
        prediction = await self.price_predictor.predict_price_movement(days_ahead=1)
        
        # Calculate current and predicted collateral values
        current_collateral_value = loan['collateral_amount'] * current_price
        predicted_collateral_value = loan['collateral_amount'] * prediction['predicted_price']
        
        # Calculate liquidation threshold
        liquidation_threshold = loan['loan_amount'] * loan['liquidation_ratio']
        
        # Determine risk level
        current_ratio = current_collateral_value / loan['loan_amount']
        predicted_ratio = predicted_collateral_value / loan['loan_amount']
        
        if predicted_ratio < 1.1:  # Less than 110%
            risk_level = 'critical'
            recommended_action = 'Immediate partial liquidation required'
        elif predicted_ratio < 1.2:  # Less than 120%
            risk_level = 'high'
            recommended_action = 'Consider adding collateral or partial repayment'
        elif predicted_ratio < 1.3:  # Less than 130%
            risk_level = 'medium'
            recommended_action = 'Monitor closely, consider adding collateral'
        else:
            risk_level = 'low'
            recommended_action = 'No action needed'
        
        return PriceAlert(
            loan_id=loan['loan_id'],
            current_eth_price=current_price,
            predicted_price=prediction['predicted_price'],
            liquidation_threshold=liquidation_threshold,
            risk_level=risk_level,
            recommended_action=recommended_action
        )

class ETHPricePredictor:
    """
    ML-based ETH price prediction system
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = [
            'price', 'volume', 'market_cap', 'gas_price', 'active_addresses',
            'transaction_count', 'defi_tvl', 'volatility_7d', 'rsi', 'moving_avg_ratio'
        ]
    
    async def train_model(self, historical_data: pd.DataFrame):
        """
        Train the price prediction model
        """
        
        # Feature engineering
        features = self._engineer_features(historical_data)
        
        # Prepare training data
        X = features[self.feature_columns].values
        y = features['price_change_24h'].values  # Predict 24h price change
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train ensemble model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("Price prediction model trained successfully")
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for price prediction
        """
        
        # Technical indicators
        data['rsi'] = self._calculate_rsi(data['price'])
        data['moving_avg_ratio'] = data['price'] / data['price'].rolling(20).mean()
        data['volatility_7d'] = data['price'].rolling(7).std()
        
        # Price change target
        data['price_change_24h'] = data['price'].shift(-1) / data['price'] - 1
        
        return data.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index
        """
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    async def predict_price_movement(self, days_ahead: int = 1, 
                                   confidence_intervals: bool = True) -> Dict[str, Any]:
        """
        Predict ETH price movement
        """
        
        if not self.is_trained:
        
            return await self._baseline_prediction(days_ahead)
        
        # Get current market data
        current_features = await self._get_current_features()
        
        # Make prediction
        X_scaled = self.scaler.transform([current_features])
        price_change_prediction = self.model.predict(X_scaled)[0]
        
        # Get current price
        current_price = await self.get_current_price()
        predicted_price = current_price * (1 + price_change_prediction)
        
        # Calculate confidence intervals using ensemble predictions
        if confidence_intervals:
            predictions = []
            
            for estimator in self.model.estimators_:
                
                pred = estimator.predict(X_scaled)[0]
                predictions.append(current_price * (1 + pred))
            
            confidence_90 = np.percentile(predictions, [5, 95])
            volatility = np.std(predictions) / current_price
        
        else:
            confidence_90 = [predicted_price * 0.9, predicted_price * 1.1]
            volatility = 0.1
        
        return {
            'predicted_price': predicted_price,
            'current_price': current_price,
            'price_change_percent': price_change_prediction * 100,
            'confidence_interval_90': confidence_90,
            'volatility': volatility,
            'confidence': max(0.1, 1 - volatility),  # Higher volatility = lower confidence
            'prediction_horizon_days': days_ahead
        }
    
    async def _baseline_prediction(self, days_ahead: int) -> Dict[str, Any]:
        """
        Simple baseline prediction when model isn't trained
        """
        
        current_price = await self.get_current_price()
        
        # Simple mean reversion model
        predicted_change = 0.0  # Assume no change as baseline
        volatility = 0.15  # 15% daily volatility assumption
        
        return {
            'predicted_price': current_price,
            'current_price': current_price,
            'price_change_percent': 0.0,
            'confidence_interval_90': [current_price * 0.85, current_price * 1.15],
            'volatility': volatility,
            'confidence': 0.5,  # Medium confidence for baseline
            'prediction_horizon_days': days_ahead
        }
    
    async def _get_current_features(self) -> List[float]:
        """
        Get current market features for prediction
        """
        

        # For now, return dummy features
        return [
            4500.0,  # price
            1000000000,  # volume
            360000000000,  # market_cap
            30,  # gas_price
            500000,  # active_addresses
            1000000,  # transaction_count
            50000000000,  # defi_tvl
            0.15,  # volatility_7d
            50,  # rsi
            1.05  # moving_avg_ratio
        ]
    
    async def get_current_price(self) -> float:
        """
        Get current ETH price from external API
        """
        
        return 4500.0

class LiquidationMonitor:
    """
    Monitors loans and triggers partial liquidations when necessary
    """
    
    def __init__(self, smart_contract_interface):
        self.smart_contract_interface = smart_contract_interface
        self.liquidation_threshold = 1.1  # 110% collateralization ratio
    
    async def check_liquidation_needed(self, loan_id: str, current_eth_price: float) -> bool:
        """
        Check if a loan needs liquidation
        """
        
        loan_details = await self.smart_contract_interface.get_loan_details(loan_id)
        
        collateral_value = loan_details['collateral_amount'] * current_eth_price
        collateral_ratio = collateral_value / loan_details['loan_amount']
        
        return collateral_ratio <= self.liquidation_threshold
    
    async def calculate_partial_liquidation_amount(self, loan_id: str, 
                                                 current_eth_price: float) -> float:
        """
        Calculate how much collateral to liquidate
        """
        
        loan_details = await self.smart_contract_interface.get_loan_details(loan_id)
        
        # Calculate amount needed to bring back to safe ratio (150%)
        target_ratio = 1.5
        required_collateral_value = loan_details['loan_amount'] * target_ratio
        current_collateral_value = loan_details['collateral_amount'] * current_eth_price
        
        if current_collateral_value > required_collateral_value:
            return 0.0  # No liquidation needed
        
        # Calculate excess collateral to liquidate
        excess_debt = loan_details['loan_amount'] - (current_collateral_value / target_ratio)
        liquidation_amount = excess_debt / current_eth_price
        
        return min(liquidation_amount, loan_details['collateral_amount'] * 0.5)  # Max 50% liquidation

class SmartContractInterface:
    """
    Interface to interact with Solidity smart contracts
    """
    
    def __init__(self, w3: Web3, contract):
        self.w3 = w3
        self.contract = contract
        self.account = None  # Set this up with your account
    
    async def create_loan(self, loan_params: Dict) -> str:
        """
        Create a new loan in the smart contract
        """
        
        try:
            # Build transaction
            transaction = self.contract.functions.createLoan(
                loan_params['borrower'],
                loan_params['collateral_amount'],
                loan_params['loan_amount'],
                loan_params['interest_rate'],
                loan_params['duration'],
                loan_params['liquidation_threshold']
            ).buildTransaction({
                'from': self.account.address,
                'gas': 500000,
                'gasPrice': self.w3.toWei('20', 'gwei'),
                'nonce': self.w3.eth.getTransactionCount(self.account.address)
            })
            
            # Sign and send transaction
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.privateKey)
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            # Wait for confirmation
            receipt = self.w3.eth.waitForTransactionReceipt(tx_hash)
            
            # Extract loan ID from logs
            loan_created_event = self.contract.events.LoanCreated().processReceipt(receipt)
            loan_id = loan_created_event[0]['args']['loanId']
            
            return loan_id
            
        except Exception as e:
            logger.error(f"Error creating loan: {str(e)}")
            raise
    
    async def get_active_loans(self) -> List[Dict]:
        """
        Get all active loans from the contract
        """
        
        try:
            # Call contract function
            active_loans = self.contract.functions.getActiveLoans().call()
            
            # Convert to Python dict format
            loans = []
            
            for loan in active_loans:
                loans.append({
                    'loan_id': loan[0],
                    'borrower': loan[1],
                    'collateral_amount': loan[2] / 1e18,  # Convert from Wei
                    'loan_amount': loan[3] / 1e6,  # Convert from USDT decimals
                    'interest_rate': loan[4] / 1e4,  # Convert from basis points
                    'liquidation_ratio': loan[5] / 1e4,
                    'created_at': loan[6]
                })
            
            return loans
            
        except Exception as e:
            logger.error(f"Error getting active loans: {str(e)}")
            return []
    
    async def trigger_partial_liquidation(self, loan_id: str, liquidation_amount: float):
        """
        Trigger partial liquidation for a loan
        """
        
        try:
            # Build transaction
            transaction = self.contract.functions.partialLiquidate(
                loan_id,
                int(liquidation_amount * 1e18)  # Convert to Wei
            ).buildTransaction({
                'from': self.account.address,
                'gas': 300000,
                'gasPrice': self.w3.toWei('25', 'gwei'),  # Higher gas for urgent liquidation
                'nonce': self.w3.eth.getTransactionCount(self.account.address)
            })
            
            # Sign and send
            signed_txn = self.w3.eth.account.sign_transaction(transaction, self.account.privateKey)
            tx_hash = self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
            
            logger.info(f"Partial liquidation triggered for loan {loan_id}, tx: {tx_hash.hex()}")
            
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Error triggering partial liquidation: {str(e)}")
            raise

# FastAPI Application - This ties everything together for your frontend
app = FastAPI(title="P2P DeFi Lending Platform", version="1.0.0")

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global inference engine instance
inference_engine: Optional[InferenceEngine] = None

@app.on_event("startup")
async def startup_event():
    """
    Initialize the inference engine on startup
    """
    
    global inference_engine
    
    # Initialize with your contract details
    WEB3_PROVIDER = "https://mainnet.infura.io/v3/c783c42e60d140aca3debcda20873e3c"
    CONTRACT_ADDRESS = "0xYourContractAddress"
    CONTRACT_ABI = []  # Your contract ABI here
    
    inference_engine = InferenceEngine(WEB3_PROVIDER, CONTRACT_ADDRESS, CONTRACT_ABI)
    
    # Start background monitoring
    asyncio.create_task(inference_engine.monitor_active_loans())

@app.post("/api/loan/request")
async def request_loan(loan_request: LoanRequest):
    """
    Main endpoint for loan requests from frontend
    """
    
    if not inference_engine:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Process the loan request through our inference engine
    loan_decision = await inference_engine.process_loan_request(loan_request)
    
    return {
        "status": "approved" if loan_decision.approved else "rejected",
        "loan_decision": asdict(loan_decision),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/credit-score/{address}")
async def get_credit_score(address: str):
    """
    Get credit score for an address
    """
    
    if not inference_engine:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Get credit assessment
    credit_result = await inference_engine._get_credit_assessment(
        LoanRequest(
            borrower_address=address,
            collateral_amount=0,
            requested_amount=0,
            loan_duration_days=30,
            interest_rate=0.05,
            collateral_ratio=1.5
        )
    )
    
    return credit_result

@app.get("/api/price/prediction")
async def get_price_prediction(days_ahead: int = 1):
    """
    Get ETH price prediction
    """
    
    if not inference_engine:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    prediction = await inference_engine.price_predictor.predict_price_movement(days_ahead)
    return prediction

@app.get("/api/loans/active")
async def get_active_loans():
    """
    Get all active loans
    """
    
    if not inference_engine:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    loans = await inference_engine.smart_contract_interface.get_active_loans()
    return {"active_loans": loans}

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time liquidation alerts
    """
    
    await websocket.accept()
    
    try:
        while True:
            # Check for new alerts (this would be triggered by the monitoring system)
            # For now, just keep connection alive
            await asyncio.sleep(10)
            await websocket.send_json({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
            
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)