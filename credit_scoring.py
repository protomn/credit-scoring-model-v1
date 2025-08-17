
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import json
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging for security monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LoanRecord:
    """
    Represents a single loan record with all relevant data
    """
    
    loan_id: str
    borrower_address: str
    amount: float
    interest_rate: float
    num_payments: int
    payment_interval_days: int
    start_timestamp: int
    payments_made: List[Tuple[int, float]]  # (timestamp, amount)
    status: str  # 'active', 'completed', 'defaulted'
    collateral_ratio: float = 0.0

@dataclass
class CreditMetrics:
    """
    Credit scoring metrics based on ALOE paper
    """

    underpay_ratio: float = 0.0
    current_debt_burden_ratio: float = 0.0
    current_payment_burden_ratio: float = 0.0
    repayment_age_ratio: float = 0.0
    avg_num_credit_lines: float = 0.0
    odds_stay_current: Dict[int, float] = None
    
    def __post_init__(self):
        if self.odds_stay_current is None:
            self.odds_stay_current = {}

class SecurityManager:
    """
    Handles encryption, hashing, and security operations
    """
    
    def __init__(self):
        self.salt = secrets.token_bytes(32)
        self._setup_encryption()
    
    def _setup_encryption(self):
        """
        Initialize encryption for sensitive data
        """
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(b"credit_scoring_key"))
        self.cipher_suite = Fernet(key)
    
    def hash_address(self, address: str) -> str:
        """
        Securely hash wallet address for privacy
        """
        
        return hashlib.sha256(address.encode()).hexdigest()
    
    def encrypt_sensitive_data(self, data: str) -> bytes:
        """
        Encrypt sensitive information
        """
        
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> str:
        """
        Decrypt sensitive information
        """
       
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    def verify_signature(self, message: str, signature: str, public_key: str) -> bool:
        """
        Verify digital signature (placeholder - implement with actual crypto library)
        """
   
        expected_sig = hmac.new(public_key.encode(), message.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(signature, expected_sig)

class BlockchainDataValidator:
    """
    Validates blockchain data integrity and authenticity
    """
    
    @staticmethod
    def validate_transaction_data(tx_data: Dict) -> bool:
        """
        Validate transaction data structure and integrity
        """
        
        required_fields = ['from_address', 'to_address', 'amount', 'timestamp', 'tx_hash']
        return all(field in tx_data for field in required_fields)
    
    @staticmethod
    def validate_address_format(address: str) -> bool:
        """
        Validate Ethereum address format
        """
        
        if not address.startswith('0x'):
            return False
        
        if len(address) != 42:
            return False
        
        try:
            int(address[2:], 16)
            return True
        
        except ValueError:
            return False
    
    @staticmethod
    def validate_loan_parameters(loan: LoanRecord) -> bool:
        """
        Validate loan parameters for security
        """
        
        if loan.amount <= 0 or loan.interest_rate < 0:
            return False
        
        if loan.num_payments <= 0 or loan.payment_interval_days <= 0:
            return False
        
        return True

class CreditScoringEngine:
    """
    Main credit scoring engine implementing ALOE algorithm
    """
    
    def __init__(self, k_neighbors: int = 5, max_model_size: int = 10000):
        
        self.k_neighbors = k_neighbors
        self.max_model_size = max_model_size
        self.security_manager = SecurityManager()
        self.validator = BlockchainDataValidator()
        self.scaler = StandardScaler()
        
        # Storage for credit data
        self.loan_records: Dict[str, List[LoanRecord]] = {}
        self.credit_metrics: Dict[str, CreditMetrics] = {}
        self.model_data: List[Tuple[np.ndarray, float]] = []
        
        # Security parameters
        self.min_loan_threshold = 0.1  # ETH
        self.max_loan_threshold = 1000  # ETH
        self.rate_limit_window = 3600  # 1 hour
        self.max_requests_per_window = 100
        self.request_history: Dict[str, List[int]] = {}
        
        logger.info("Credit Scoring Engine initialized with security protocols")
    
    def rate_limit_check(self, address: str) -> bool:
        """
        Implement rate limiting for API security
        """
        
        current_time = int(datetime.now().timestamp())
        
        if address not in self.request_history:
            self.request_history[address] = []
        
        # Clean old requests
        self.request_history[address] = [
            req_time for req_time in self.request_history[address]
            if current_time - req_time < self.rate_limit_window
        ]
        
        if len(self.request_history[address]) >= self.max_requests_per_window:
            logger.warning(f"Rate limit exceeded for address: {address}")
            return False
        
        self.request_history[address].append(current_time)
        return True
    
    def register_borrower(self, address: str, initial_fico_score: int = None) -> bool:
        """
        Register a new borrower with optional FICO score
        """
        try:
            if not self.validator.validate_address_format(address):
                raise ValueError("Invalid address format")
            
            if not self.rate_limit_check(address):
                return False
            
            hashed_address = self.security_manager.hash_address(address)
            
            if hashed_address in self.credit_metrics:
                logger.warning(f"Borrower already registered: {address}")
                return False
            
            # Initialize credit metrics
            self.credit_metrics[hashed_address] = CreditMetrics()
            self.loan_records[hashed_address] = []
            
            logger.info(f"New borrower registered: {address}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering borrower {address}: {str(e)}")
            return False
    
    def add_loan_record(self, loan: LoanRecord) -> bool:
        """
        Add a new loan record to the system
        """
        
        try:
            if not self.validator.validate_loan_parameters(loan):
                raise ValueError("Invalid loan parameters")
            
            if not self.rate_limit_check(loan.borrower_address):
                return False
            
            hashed_address = self.security_manager.hash_address(loan.borrower_address)
            
            if hashed_address not in self.loan_records:
                self.register_borrower(loan.borrower_address)
            
            self.loan_records[hashed_address].append(loan)
            self._update_credit_metrics(hashed_address)
            
            logger.info(f"Loan record added for {loan.borrower_address}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding loan record: {str(e)}")
            return False
    
    def _calculate_underpay_ratio(self, loans: List[LoanRecord]) -> float:
        """
        Calculate Underpay Ratio (UPR) based on ALOE paper
        """
        
        if not loans:
            return 0.0
        
        total_time_behind = 0
        total_time_span = 0
        
        for loan in loans:
            
            if not loan.payments_made:
                continue
            
            expected_payment_amount = loan.amount / loan.num_payments
            payment_schedule_days = loan.payment_interval_days
            
            for i, (payment_timestamp, payment_amount) in enumerate(loan.payments_made):
                
                expected_timestamp = loan.start_timestamp + (i * payment_schedule_days * 86400)
                time_behind = max(0, payment_timestamp - expected_timestamp)
                total_time_behind += time_behind
                total_time_span += payment_schedule_days * 86400
        
        return total_time_behind / max(total_time_span, 1)
    
    def _calculate_debt_burden_ratio(self, loans: List[LoanRecord]) -> float:
        """
        Calculate Current Debt Burden Ratio (CDBR)
        """
        
        if not loans:
            return 0.0
        
        current_outstanding = 0
        total_ever_loaned = 0
        
        for loan in loans:
            
            total_ever_loaned += loan.amount
            
            if loan.status == 'active':
                payments_made = sum(payment[1] for payment in loan.payments_made)
                current_outstanding += max(0, loan.amount - payments_made)
        
        average_outstanding = total_ever_loaned / len(loans)
        
        if average_outstanding == 0:
            return 0.0
        
        # Added pseudo standard deviation as per ALOE paper
        pseudo_std = average_outstanding * 0.3  # Approximation
        denominator = average_outstanding + 3 * pseudo_std
        
        ratio = current_outstanding / max(denominator, 1)
        return min(ratio, 1.0)
    
    def _calculate_payment_burden_ratio(self, loans: List[LoanRecord]) -> float:
        """
        Calculate Current Payment Burden Ratio (CPBR)
        """
        
        if not loans:
            return 0.0
        
        current_daily_payment = 0
        total_payments = 0
        total_days = 0
        
        for loan in loans:
            
            expected_payment = loan.amount / loan.num_payments
            payment_frequency = loan.payment_interval_days
            
            if loan.status == 'active':
                current_daily_payment += expected_payment / payment_frequency
            
            total_payments += expected_payment * loan.num_payments
            total_days += loan.num_payments * payment_frequency
        
        avg_daily_payment = total_payments / max(total_days, 1)
        pseudo_std = avg_daily_payment * 0.2
        
        ratio = current_daily_payment / max(avg_daily_payment + 3 * pseudo_std, 1)
        return min(ratio, 1.0)
    
    def _calculate_repayment_age_ratio(self, loans: List[LoanRecord]) -> float:
        """
        Calculate Repayment Age Ratio (RAR)
        """
        
        if not loans:
            return 0.0
        
        current_time = int(datetime.now().timestamp())
        first_loan_time = min(loan.start_timestamp for loan in loans)
        
        weighted_age_sum = 0
        total_amount = 0
        
        for loan in loans:
            loan_age = current_time - loan.start_timestamp
            weighted_age_sum += loan.amount * loan_age
            total_amount += loan.amount
        
        time_since_first = current_time - first_loan_time
        
        if time_since_first == 0 or total_amount == 0:
            return 0.0
        
        return weighted_age_sum / (total_amount * time_since_first)
    
    def _calculate_avg_credit_lines(self, loans: List[LoanRecord]) -> float:
        """
        Calculate Average Number of Credit Lines (ANCL)
        """
        
        if not loans:
            return 0.0
        
        # Simplified calculation - count overlapping active loans over time
        loan_timeline = []
        for loan in loans:
            
            loan_timeline.append((loan.start_timestamp, 1))  # Start
            # Estimate end time
            estimated_end = loan.start_timestamp + (loan.num_payments * loan.payment_interval_days * 86400)
            loan_timeline.append((estimated_end, -1))  # End
        
        loan_timeline.sort()
        
        active_loans = 0
        total_weighted_time = 0
        total_time = 0
        last_time = None
        
        for timestamp, change in loan_timeline:
            
            if last_time is not None:
            
                time_diff = timestamp - last_time
                total_weighted_time += active_loans * time_diff
                total_time += time_diff
            
            active_loans += change
            last_time = timestamp
        
        return total_weighted_time / max(total_time, 1)
    
    def _calculate_odds_stay_current(self, loans: List[LoanRecord]) -> Dict[int, float]:
        """
        Calculate Odds Stay Current for various time windows
        """
        
        windows = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        odds = {}
        
        for window_days in windows:
            
            window_seconds = window_days * 86400
            successful_periods = 0
            total_periods = 0
            
            for loan in loans:
                
                if loan.status in ['completed', 'active']:

                    # Check if borrower stayed current during this window
                    loan_duration = len(loan.payments_made) * loan.payment_interval_days * 86400
                    periods_in_loan = max(1, int(loan_duration / window_seconds))
                    
                    for period in range(periods_in_loan):

                        period_start = loan.start_timestamp + (period * window_seconds)
                        period_end = period_start + window_seconds
                        
                        # Check if payments were made on time in this period
                        on_time = True
                        for payment_time, _ in loan.payments_made:

                            if period_start <= payment_time <= period_end:
                                expected_time = period_start + (loan.payment_interval_days * 86400)
                                
                                if payment_time > expected_time + 86400:  # 1 day grace period
                                    on_time = False
                                    break
                        
                        total_periods += 1
                        
                        if on_time:
                            successful_periods += 1
            
            odds[window_days] = successful_periods / max(total_periods, 1)
        
        return odds
    
    def _update_credit_metrics(self, hashed_address: str):
        """
        Update credit metrics for a borrower
        """
        loans = self.loan_records[hashed_address]
        
        metrics = CreditMetrics(
            underpay_ratio=self._calculate_underpay_ratio(loans),
            current_debt_burden_ratio=self._calculate_debt_burden_ratio(loans),
            current_payment_burden_ratio=self._calculate_payment_burden_ratio(loans),
            repayment_age_ratio=self._calculate_repayment_age_ratio(loans),
            avg_num_credit_lines=self._calculate_avg_credit_lines(loans),
            odds_stay_current=self._calculate_odds_stay_current(loans)
        )
        
        self.credit_metrics[hashed_address] = metrics
    
    def get_credit_score(self, address: str, requested_amount: float = None, 
                        interest_rate: float = None, num_payments: int = None,
                        payment_interval: int = None) -> Optional[float]:
        """
        Calculate credit score using k-nearest neighbors algorithm
        Returns score between 0 and 1 (higher is better)
        """
        
        try:
            if not self.validator.validate_address_format(address):
                raise ValueError("Invalid address format")
            
            if not self.rate_limit_check(address):
                return None
            
            hashed_address = self.security_manager.hash_address(address)
            
            # Handle new borrowers or those with minimal history
            if hashed_address not in self.credit_metrics:
                logger.info(f"New borrower, returning base score for {address}")
                return 0.5  # Neutral score for new borrowers
            
            metrics = self.credit_metrics[hashed_address]
            
            # Create feature vector
            feature_vector = np.array([
                metrics.underpay_ratio,
                metrics.current_debt_burden_ratio,
                metrics.current_payment_burden_ratio,
                metrics.repayment_age_ratio,
                metrics.avg_num_credit_lines
            ])
            
            # If we have loan parameters, simulate the effect
            if all(param is not None for param in [requested_amount, interest_rate, num_payments, payment_interval]):
                feature_vector = self._simulate_loan_impact(
                    feature_vector, hashed_address, requested_amount, 
                    interest_rate, num_payments, payment_interval
                )
            
            # Use k-nearest neighbors if we have enough data
            if len(self.model_data) >= self.k_neighbors:
                score = self._knn_score(feature_vector)
            
            else:
                # Fallback scoring for limited data
                score = self._fallback_score(metrics)
            
            # Apply security bounds
            score = max(0.0, min(1.0, score))
            
            logger.info(f"Credit score calculated for {address}: {score:.3f}")
            return score
            
        except Exception as e:
            logger.error(f"Error calculating credit score for {address}: {str(e)}")
            return None
    
    def _simulate_loan_impact(self, feature_vector: np.ndarray, hashed_address: str,
                             amount: float, rate: float, payments: int, interval: int) -> np.ndarray:
        """
        Simulate the impact of a proposed loan on credit metrics
        """
        
        loans = self.loan_records[hashed_address]
        
        # Create a simulated loan
        simulated_loan = LoanRecord(
            loan_id=f"sim_{secrets.token_hex(8)}",
            borrower_address="simulated",
            amount=amount,
            interest_rate=rate,
            num_payments=payments,
            payment_interval_days=interval,
            start_timestamp=int(datetime.now().timestamp()),
            payments_made=[],
            status='active'
        )
        
        # Temporarily add to loans for calculation
        temp_loans = loans + [simulated_loan]
        
        # Recalculate metrics with simulated loan
        simulated_vector = np.array([
            self._calculate_underpay_ratio(temp_loans),
            self._calculate_debt_burden_ratio(temp_loans),
            self._calculate_payment_burden_ratio(temp_loans),
            self._calculate_repayment_age_ratio(temp_loans),
            self._calculate_avg_credit_lines(temp_loans)
        ])
        
        return simulated_vector
    
    def _knn_score(self, feature_vector: np.ndarray) -> float:
        """
        Calculate score using k-nearest neighbors
        """
        
        if len(self.model_data) == 0:
            return 0.5
        
        # Prepare data for KNN
        X = np.array([data[0] for data in self.model_data])
        y = np.array([data[1] for data in self.model_data])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        feature_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Fit KNN model
        knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(X)))
        knn.fit(X_scaled)
        
        # Find neighbors
        distances, indices = knn.kneighbors(feature_scaled)
        
        # Calculate inverse distance weighted average
        weights = 1 / (distances[0] + 1e-8)  # Add small epsilon to avoid division by zero
        weighted_scores = weights * y[indices[0]]
        
        return np.sum(weighted_scores) / np.sum(weights)
    
    def _fallback_score(self, metrics: CreditMetrics) -> float:
        """
        Fallback scoring method when insufficient data for KNN
        """
        
        # Simple weighted average of metrics
        score = (
            (1 - metrics.underpay_ratio) * 0.35 +
            (1 - metrics.current_debt_burden_ratio) * 0.30 +
            (1 - metrics.current_payment_burden_ratio) * 0.15 +
            metrics.repayment_age_ratio * 0.10 +
            min(metrics.avg_num_credit_lines / 5, 1.0) * 0.10
        )
        
        return score
    
    def update_model_data(self, feature_vector: np.ndarray, outcome_score: float):
        """
        Update the model with new training data
        """
        
        self.model_data.append((feature_vector, outcome_score))
        
        # Maintain model size limit
        if len(self.model_data) > self.max_model_size:
            # Remove oldest data points
            self.model_data = self.model_data[-self.max_model_size:]
    
    def export_model_state(self) -> Dict[str, Any]:
        """
        Export model state for backup/persistence
        """
        
        return {
            'model_data': [(vec.tolist(), score) for vec, score in self.model_data],
            'k_neighbors': self.k_neighbors,
            'max_model_size': self.max_model_size,
            'timestamp': int(datetime.now().timestamp())
        }
    
    def import_model_state(self, state: Dict[str, Any]):
        """
        Import model state from backup
        """
        
        self.model_data = [(np.array(vec), score) for vec, score in state['model_data']]
        self.k_neighbors = state.get('k_neighbors', self.k_neighbors)
        self.max_model_size = state.get('max_model_size', self.max_model_size)
        
        logger.info("Model state imported successfully")

# Example usage and testing
if __name__ == "__main__":
    
    # Initialize the credit scoring engine
    engine = CreditScoringEngine(k_neighbors=5)
    
    # Register a borrower
    test_address = "0x1234567890123456789012345678901234567890"
    engine.register_borrower(test_address)
    
    # Add some loan records
    loan1 = LoanRecord(
        loan_id="loan_001",
        borrower_address=test_address,
        amount=10.0,  # 10 ETH
        interest_rate=0.05,  # 5%
        num_payments=12,
        payment_interval_days=30,
        start_timestamp=int(datetime.now().timestamp()) - 86400 * 180,  # 6 months ago
        payments_made=[
            (int(datetime.now().timestamp()) - 86400 * 150, 1.0),
            (int(datetime.now().timestamp()) - 86400 * 120, 1.0),
            (int(datetime.now().timestamp()) - 86400 * 90, 1.0),
        ],
        status='active'
    )
    
    engine.add_loan_record(loan1)
    
    # Calculate credit score
    score = engine.get_credit_score(test_address, requested_amount=5.0, 
                                  interest_rate=0.04, num_payments=6, 
                                  payment_interval=30)
    
    print(f"Credit score for {test_address}: {score:.3f}")
    
    # Export model state
    model_state = engine.export_model_state()
    print(f"Model exported with {len(model_state['model_data'])} data points")