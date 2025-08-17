from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from database import Base
import datetime

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    ethereum_address = Column(String, unique=True, index=True)
    credit_score = Column(Float, default=0.0)
    reputation_score = Column(Float, default=0.0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    loans_as_borrower = relationship("Loan", back_populates="borrower", foreign_keys="Loan.borrower_id")
    loans_as_lender = relationship("Loan", back_populates="lender", foreign_keys="Loan.lender_id")

class Loan(Base):
    __tablename__ = "loans"
    
    id = Column(Integer, primary_key=True, index=True)
    loan_id = Column(String, unique=True, index=True)
    borrower_id = Column(Integer, ForeignKey("users.id"))
    lender_id = Column(Integer, ForeignKey("users.id"))
    
    # Loan terms
    collateral_amount = Column(Float)  # ETH amount
    principal_amount = Column(Float)   # USDC/USDT amount
    interest_rate = Column(Float)      # Annual percentage
    duration_days = Column(Integer)
    
    # Current state
    outstanding_debt = Column(Float)
    status = Column(String, default="proposed")  # proposed, active, repaid, defaulted
    collateral_ratio = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    activated_at = Column(DateTime)
    due_date = Column(DateTime)
    
    # Relationships
    borrower = relationship("User", back_populates="loans_as_borrower", foreign_keys=[borrower_id])
    lender = relationship("User", back_populates="loans_as_lender", foreign_keys=[lender_id])
    transactions = relationship("Transaction", back_populates="loan")

class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    tx_hash = Column(String, unique=True, index=True)
    loan_id = Column(Integer, ForeignKey("loans.id"))
    from_address = Column(String)
    to_address = Column(String)
    amount = Column(Float)
    gas_used = Column(Integer)
    gas_price = Column(Integer)
    block_number = Column(Integer)
    status = Column(String)  # pending, confirmed, failed
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    loan = relationship("Loan", back_populates="transactions")

class CreditScore(Base):
    __tablename__ = "credit_scores"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    score = Column(Float)
    calculation_method = Column(String)  # ALOE, traditional, etc.
    factors = Column(Text)  # JSON string of scoring factors
    calculated_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User")

class BlockchainData(Base):
    __tablename__ = "blockchain_data"
    
    id = Column(Integer, primary_key=True, index=True)
    address = Column(String, index=True)
    transaction_count = Column(Integer)
    total_volume = Column(Float)
    defi_interactions = Column(Integer)
    gas_efficiency = Column(Float)
    unique_tokens = Column(Integer)
    liquidation_history = Column(Integer)
    on_time_payments = Column(Float)
    last_updated = Column(DateTime, default=func.now()) 