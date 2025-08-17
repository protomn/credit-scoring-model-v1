from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import get_db
from models import User
from auth import get_password_hash, verify_password, create_access_token, get_current_active_user
from pydantic import BaseModel
from typing import Optional
from datetime import timedelta

router = APIRouter(prefix="/users", tags=["users"])

# Pydantic models
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    ethereum_address: str

class UserLogin(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    ethereum_address: str
    credit_score: float
    reputation_score: float
    is_active: bool

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/register", response_model=UserResponse)
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    
    # Check if username already exists
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )
    
    # Check if email already exists
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    # Check if ethereum address already exists
    if db.query(User).filter(User.ethereum_address == user.ethereum_address).first():
        raise HTTPException(
            status_code=400,
            detail="Ethereum address already registered"
        )
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        ethereum_address=user.ethereum_address
    )
    
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    return UserResponse(
        id=db_user.id,
        username=db_user.username,
        email=db_user.email,
        ethereum_address=db_user.ethereum_address,
        credit_score=db_user.credit_score,
        reputation_score=db_user.reputation_score,
        is_active=db_user.is_active
    )

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and get access token"""
    
    # Find user by username
    user = db.query(User).filter(User.username == form_data.username).first()
    
    # Verify password
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=30)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        ethereum_address=current_user.ethereum_address,
        credit_score=current_user.credit_score,
        reputation_score=current_user.reputation_score,
        is_active=current_user.is_active
    )

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        ethereum_address=user.ethereum_address,
        credit_score=user.credit_score,
        reputation_score=user.reputation_score,
        is_active=user.is_active
    ) 