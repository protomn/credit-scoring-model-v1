
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass, asdict
import hashlib
import hmac
import secrets
from web3 import Web3
from eth_account import Account
import redis
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
from functools import wraps
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class SecureLoanRecord(Base):
    """
    Database model for secure loan storage
    """
    
    __tablename__ = 'secure_loans'
    
    id = Column(String, primary_key=True)
    borrower_hash = Column(String, nullable=False, index=True)
    amount_encrypted = Column(Text, nullable=False)
    interest_rate = Column(Float, nullable=False)
    num_payments = Column(Integer, nullable=False)
    payment_interval = Column(Integer, nullable=False)
    start_timestamp = Column(Integer, nullable=False)
    status = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

@dataclass
class BlockchainTransaction:
    """
    Represents a blockchain transaction for credit analysis
    """
    
    tx_hash: str
    from_address: str
    to_address: str
    value: float
    gas_used: int
    gas_price: int
    timestamp: int
    block_number: int
    status: bool

@dataclass
class DeFiProtocolInteraction:
    """
    Represents interaction with DeFi protocols
    """
    
    protocol_name: str
    interaction_type: str  # 'lend', 'borrow', 'stake', 'swap'
    amount: float
    timestamp: int
    tx_hash: str
    success: bool

class SecurityAuditLogger:
    """
    Enhanced security logging and monitoring
    """
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.alert_thresholds = {
            'failed_auth_attempts': 5,
            'rate_limit_violations': 10,
            'suspicious_score_requests': 20
        }
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], severity: str = 'INFO'):
        """Log security events with structured data"""
        event = {
            'timestamp': datetime.utcnow().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'event_id': secrets.token_hex(16)
        }
        
        # Log to standard logger
        logger.log(getattr(logging, severity), f"Security Event: {event}")
        
        # Store in Redis for real-time monitoring
        if self.redis_client:
            self.redis_client.lpush('security_events', json.dumps(event))
            self.redis_client.expire('security_events', 86400 * 7)  # Keep for 7 days
    
    def check_threat_patterns(self, address: str) -> Dict[str, bool]:
        """
        Check for suspicious patterns in user behavior
        """
        
        threats = {
            'rapid_score_requests': False,
            'multiple_identity_attempts': False,
            'unusual_transaction_patterns': False
        }
        
        # Check recent activity
        recent_events = self._get_recent_events(address, hours=1)
        
        # Detect rapid score requests
        score_requests = [e for e in recent_events if e.get('event_type') == 'score_request']
        if len(score_requests) > 10:
            threats['rapid_score_requests'] = True
        
        return threats
    
    def _get_recent_events(self, address: str, hours: int = 24) -> List[Dict]:
        """
        Retrieve recent security events for an address
        """
        
        if not self.redis_client:
            return []
        
        events = []
        event_data = self.redis_client.lrange('security_events', 0, -1)
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        for event_json in event_data:
            try:
                event = json.loads(event_json)
                event_time = datetime.fromisoformat(event['timestamp'])
                
                if (event_time > cutoff_time and 
                    event['details'].get('address') == address):
                    events.append(event)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        
        return events

class BlockchainDataCollector:
    """
    Collects and validates blockchain data for credit scoring
    """
    
    def __init__(self, web3_provider: str, api_keys: Dict[str, str] = None):
        
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.api_keys = api_keys or {}
        self.session = None
        
    async def __aenter__(self):
        
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        
        if self.session:
            await self.session.close()
    
    async def get_address_transactions(self, address: str, limit: int = 100) -> List[BlockchainTransaction]:
        """
        Fetch transaction history for an address
        """
        
        try:
            
            if 'etherscan' not in self.api_keys:
                logger.warning("Etherscan API key not provided, using limited data")
                return await self._get_transactions_from_node(address, limit)
            
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'account',
                'action': 'txlist',
                'address': address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': limit,
                'sort': 'desc',
                'apikey': self.api_keys['etherscan']
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data['status'] != '1':
                    raise Exception(f"API Error: {data.get('message', 'Unknown error')}")
                
                transactions = []
                for tx in data['result']:
                    transactions.append(BlockchainTransaction(
                        tx_hash=tx['hash'],
                        from_address=tx['from'],
                        to_address=tx['to'],
                        value=float(Web3.fromWei(int(tx['value']), 'ether')),
                        gas_used=int(tx['gasUsed']),
                        gas_price=int(tx['gasPrice']),
                        timestamp=int(tx['timeStamp']),
                        block_number=int(tx['blockNumber']),
                        status=bool(int(tx.get('txreceipt_status', '1')))
                    ))
                
                return transactions
                
        except Exception as e:
            logger.error(f"Error fetching transactions for {address}: {str(e)}")
            return []
    
    async def _get_transactions_from_node(self, address: str, limit: int) -> List[BlockchainTransaction]:
        """
        Fallback method to get transactions directly from node
        """
        
        transactions = []
        
        try:
            latest_block = self.w3.eth.block_number
            
            # Check recent blocks for transactions
            for block_num in range(max(0, latest_block - 1000), latest_block + 1):
                
                block = self.w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block.transactions:
                
                    if (tx['from'].lower() == address.lower() or 
                
                        tx['to'] and tx['to'].lower() == address.lower()):
                        
                        receipt = self.w3.eth.get_transaction_receipt(tx['hash'])
                        
                        transactions.append(BlockchainTransaction(
                            tx_hash=tx['hash'].hex(),
                            from_address=tx['from'],
                            to_address=tx['to'] or '',
                            value=float(Web3.fromWei(tx['value'], 'ether')),
                            gas_used=receipt['gasUsed'],
                            gas_price=tx['gasPrice'],
                            timestamp=block['timestamp'],
                            block_number=block['number'],
                            status=receipt['status'] == 1
                        ))
                        
                        if len(transactions) >= limit:
                            return transactions
        
        except Exception as e:
            logger.error(f"Error getting transactions from node: {str(e)}")
        
        return transactions
    
    async def analyze_defi_interactions(self, address: str) -> List[DeFiProtocolInteraction]:
        """
        Analyze DeFi protocol interactions for enhanced credit scoring
        """
        
        interactions = []
        
        # Known DeFi protocol addresses (example)
        defi_protocols = {
            '0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9': 'Aave',
            '0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b': 'Compound',
            '0x1f98431c8ad98523631ae4a59f267346ea31f984': 'Uniswap V3',
            '0x5c69bee701ef814a2b6a3edd4b1652cb9cc5aa6f': 'Uniswap V2'
        }
        
        transactions = await self.get_address_transactions(address)
        
        for tx in transactions:
            to_address = tx.to_address.lower()
            
            if to_address in defi_protocols:
                protocol_name = defi_protocols[to_address]
                
                # Determine interaction type based on transaction data
                interaction_type = self._classify_defi_interaction(tx, protocol_name)
                
                interactions.append(DeFiProtocolInteraction(
                    protocol_name=protocol_name,
                    interaction_type=interaction_type,
                    amount=tx.value,
                    timestamp=tx.timestamp,
                    tx_hash=tx.tx_hash,
                    success=tx.status
                ))
        
        return interactions
    
    def _classify_defi_interaction(self, tx: BlockchainTransaction, protocol: str) -> str:
        """
        Classify the type of DeFi interaction
        """
        # This is a simplified classification

        
        if tx.value > 0:
            
            if protocol in ['Aave', 'Compound']:
                return 'deposit'
            
            elif protocol.startswith('Uniswap'):
                return 'swap'
        
        else:
        
            if protocol in ['Aave', 'Compound']:
                return 'withdraw'
        
            elif protocol.startswith('Uniswap'):
                return 'swap'
        
        return 'unknown'
    
    async def get_erc20_transfers(self, address: str, contract_address: str = None) -> List[Dict]:
        """
        Get ERC-20 token transfers for an address
        """
        
        try:
        
            if 'etherscan' not in self.api_keys:
                return []
            
            url = "https://api.etherscan.io/api"
            params = {
                'module': 'account',
                'action': 'tokentx',
                'address': address,
                'startblock': 0,
                'endblock': 99999999,
                'page': 1,
                'offset': 100,
                'sort': 'desc',
                'apikey': self.api_keys['etherscan']
            }
            
            if contract_address:
                params['contractaddress'] = contract_address
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data['status'] != '1':
                    return []
                
                return data['result']
                
        except Exception as e:
            logger.error(f"Error fetching ERC-20 transfers: {str(e)}")
            return []

class AdvancedCreditAnalyzer:
    """
    Advanced credit analysis using multiple data sources
    """
    
    def __init__(self, blockchain_collector: BlockchainDataCollector):
        self.blockchain_collector = blockchain_collector
        
    async def calculate_transaction_velocity(self, address: str, days: int = 30) -> Dict[str, float]:
        """
        Calculate transaction velocity metrics
        """
        
        transactions = await self.blockchain_collector.get_address_transactions(address, limit=1000)
        
        cutoff_time = int(time.time()) - (days * 86400)
        recent_txs = [tx for tx in transactions if tx.timestamp >= cutoff_time]
        
        if not recent_txs:
            return {'velocity': 0.0, 'avg_value': 0.0, 'frequency': 0.0}
        
        total_value = sum(tx.value for tx in recent_txs)
        avg_value = total_value / len(recent_txs)
        frequency = len(recent_txs) / days
        
        return {
            'velocity': total_value / days,
            'avg_value': avg_value,
            'frequency': frequency,
            'total_transactions': len(recent_txs)
        }
    
    async def analyze_liquidity_behavior(self, address: str) -> Dict[str, float]:
        """
        Analyze liquidity providing and DeFi behavior
        """
        
        defi_interactions = await self.blockchain_collector.analyze_defi_interactions(address)
        
        metrics = {
            'defi_participation_score': 0.0,
            'liquidity_provision_ratio': 0.0,
            'protocol_diversity': 0.0,
            'avg_interaction_value': 0.0
        }
        
        if not defi_interactions:
            return metrics
        
        # Calculate DeFi participation score
        unique_protocols = set(interaction.protocol_name for interaction in defi_interactions)
        successful_interactions = sum(1 for i in defi_interactions if i.success)
        
        metrics['defi_participation_score'] = min(len(unique_protocols) / 5.0, 1.0)
        metrics['protocol_diversity'] = len(unique_protocols)
        
        # Calculate liquidity provision ratio
        deposits = [i for i in defi_interactions if i.interaction_type in ['deposit', 'lend']]
        total_deposit_value = sum(i.amount for i in deposits)
        total_interaction_value = sum(i.amount for i in defi_interactions)
        
        if total_interaction_value > 0:
            metrics['liquidity_provision_ratio'] = total_deposit_value / total_interaction_value
            metrics['avg_interaction_value'] = total_interaction_value / len(defi_interactions)
        
        return metrics
    
    async def calculate_portfolio_diversity(self, address: str) -> Dict[str, float]:
        """
        Calculate portfolio diversity based on token holdings
        """
        
        erc20_transfers = await self.blockchain_collector.get_erc20_transfers(address)
        
        # Calculate token diversity
        tokens_held = set()
        for transfer in erc20_transfers:
            if transfer['to'].lower() == address.lower():
                tokens_held.add(transfer['contractAddress'])
        
        diversity_score = min(len(tokens_held) / 10.0, 1.0)  # Max score at 10+ tokens
        
        return {
            'token_diversity_score': diversity_score,
            'unique_tokens': len(tokens_held),
            'total_token_transfers': len(erc20_transfers)
        }

class RiskAssessment:
    """
    Advanced risk assessment algorithms
    """
    
    @staticmethod
    def calculate_volatility_risk(transactions: List[BlockchainTransaction], window_days: int = 30) -> float:
        """Calculate transaction value volatility"""
        if len(transactions) < 2:
            return 0.5  # Neutral risk for insufficient data
        
        values = [tx.value for tx in transactions[-window_days:] if tx.value > 0]
        
        if len(values) < 2:
            return 0.5
        
        mean_value = np.mean(values)
        std_value = np.std(values)
        
        if mean_value == 0:
            return 1.0  # High risk
        
        coefficient_of_variation = std_value / mean_value
        
        # Normalize to 0-1 scale (higher CV = higher risk)
        risk_score = min(coefficient_of_variation / 2.0, 1.0)
        
        return risk_score
    
    @staticmethod
    def assess_gas_efficiency(transactions: List[BlockchainTransaction]) -> float:
        """
        Assess gas usage efficiency as a proxy for financial sophistication
        """
        
        if not transactions:
            return 0.5
        
        # Calculate average gas price relative to network average
        gas_prices = [tx.gas_price for tx in transactions if tx.gas_price > 0]
        
        if not gas_prices:
            return 0.5
        
        avg_gas_price = np.mean(gas_prices)
        
        # Estimate network average (this would be fetched from external source in production)
        network_avg_gas = 20e9  # 20 Gwei as example
        
        efficiency_ratio = network_avg_gas / avg_gas_price
        
        # Score between 0 and 1 (higher efficiency = better score)
        return min(max(efficiency_ratio - 0.5, 0.0) * 2.0, 1.0)

def security_monitor(func):
    """
    Decorator for monitoring security-sensitive functions
    """
    
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        function_name = func.__name__
        
        try:
            result = await func(*args, **kwargs)
            
            # Log successful execution
            execution_time = time.time() - start_time
            logger.info(f"Security-monitored function {function_name} executed successfully in {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            # Log security-relevant errors
            logger.error(f"Security-monitored function {function_name} failed: {str(e)}")
            raise
    
    return wrapper

class CreditScoringAPI:
    """
    Production-ready API for credit scoring system
    """
    
    def __init__(self, scoring_engine, blockchain_collector, security_logger):
        self.scoring_engine = scoring_engine
        self.blockchain_collector = blockchain_collector
        self.security_logger = security_logger
        self.analyzer = AdvancedCreditAnalyzer(blockchain_collector)
    
    @security_monitor
    async def get_comprehensive_credit_score(self, address: str, loan_params: Dict = None) -> Dict[str, Any]:
        """
        Get comprehensive credit score with multiple data source
        s"""
        
        # Log the request
        self.security_logger.log_security_event(
            'score_request',
            {'address': address, 'timestamp': datetime.utcnow().isoformat()},
            'INFO'
        )
        
        # Check for threats
        threats = self.security_logger.check_threat_patterns(address)
        
        if any(threats.values()):
            
            self.security_logger.log_security_event(
                'threat_detected',
                {'address': address, 'threats': threats},
                'WARNING'
            )
        
        try:
            # Get basic credit score
            basic_score = self.scoring_engine.get_credit_score(
                address,
                loan_params.get('amount') if loan_params else None,
                loan_params.get('interest_rate') if loan_params else None,
                loan_params.get('num_payments') if loan_params else None,
                loan_params.get('payment_interval') if loan_params else None
            )
            
            # Get enhanced metrics
            velocity_metrics = await self.analyzer.calculate_transaction_velocity(address)
            liquidity_metrics = await self.analyzer.analyze_liquidity_behavior(address)
            diversity_metrics = await self.analyzer.calculate_portfolio_diversity(address)
            
            # Get transactions for risk assessment
            transactions = await self.blockchain_collector.get_address_transactions(address)
            volatility_risk = RiskAssessment.calculate_volatility_risk(transactions)
            gas_efficiency = RiskAssessment.assess_gas_efficiency(transactions)
            
            # Calculate composite score
            composite_score = self._calculate_composite_score(
                basic_score or 0.5,
                velocity_metrics,
                liquidity_metrics,
                diversity_metrics,
                volatility_risk,
                gas_efficiency
            )
            
            return {
                'address': address,
                'basic_credit_score': basic_score,
                'composite_score': composite_score,
                'risk_assessment': {
                    'volatility_risk': volatility_risk,
                    'gas_efficiency': gas_efficiency
                },
                'defi_metrics': liquidity_metrics,
                'transaction_metrics': velocity_metrics,
                'portfolio_metrics': diversity_metrics,
                'timestamp': datetime.utcnow().isoformat(),
                'threats_detected': threats
            }
            
        except Exception as e:
            
            self.security_logger.log_security_event(
                'score_calculation_error',
                {'address': address, 'error': str(e)},
                'ERROR'
            )
            raise
    
    def _calculate_composite_score(self, basic_score: float, velocity: Dict, 
                                 liquidity: Dict, diversity: Dict, 
                                 volatility_risk: float, gas_efficiency: float) -> float:
        """
        Calculate composite credit score from multiple factors
        """
        
        # Weights for different components
        weights = {
            'basic_score': 0.40,
            'defi_participation': 0.15,
            'transaction_velocity': 0.10,
            'portfolio_diversity': 0.10,
            'volatility_risk': 0.15,  # Negative weight
            'gas_efficiency': 0.10
        }
        
        # Normalize velocity score
        velocity_score = min(velocity.get('frequency', 0) / 10.0, 1.0)
        
        # Calculate weighted composite score
        composite = (
            basic_score * weights['basic_score'] +
            liquidity.get('defi_participation_score', 0) * weights['defi_participation'] +
            velocity_score * weights['transaction_velocity'] +
            diversity.get('token_diversity_score', 0) * weights['portfolio_diversity'] +
            (1 - volatility_risk) * weights['volatility_risk'] +  # Invert risk
            gas_efficiency * weights['gas_efficiency']
        )
        
        return max(0.0, min(1.0, composite))

# Example usage and testing
async def main():
    """
    Example usage of the security and integration module
    """
    
    # Initialize components
    security_logger = SecurityAuditLogger()
    
    # Initialize blockchain collector
    api_keys = {
        'etherscan': 'PDW6NTZXAAYNBFYIKSI7ZG6P8YD96BPJVS'
    }
    
    async with BlockchainDataCollector('https://mainnet.infura.io/v3/c783c42e60d140aca3debcda20873e3c', api_keys) as collector:
        
        # Initialize credit scoring engine
        from credit_scoring import CreditScoringEngine
        scoring_engine = CreditScoringEngine()
        
        # Initialize API
        api = CreditScoringAPI(scoring_engine, collector, security_logger)
        
        # Example address (Vitalik's address)
        test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
        
        # Get comprehensive credit score
        try:
            result = await api.get_comprehensive_credit_score(
                test_address,
                {
                    'amount': 5.0,
                    'interest_rate': 0.05,
                    'num_payments': 12,
                    'payment_interval': 30
                }
            )
            
            print("Comprehensive Credit Analysis:")
            print(f"Address: {result['address']}")
            print(f"Basic Credit Score: {result['basic_credit_score']:.3f}")
            print(f"Composite Score: {result['composite_score']:.3f}")
            print(f"DeFi Participation: {result['defi_metrics']['defi_participation_score']:.3f}")
            print(f"Volatility Risk: {result['risk_assessment']['volatility_risk']:.3f}")
            print(f"Gas Efficiency: {result['risk_assessment']['gas_efficiency']:.3f}")
            
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())