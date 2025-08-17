import os

class Config:

    ETHERSCAN_API_KEY = "PDW6NTZXAAYNBFYIKSI7ZG6P8YD96BPJVS" 
    INFURA_PROJECT_ID = "c783c42e60d140aca3debcda20873e3c" 
    

    WEB3_PROVIDER = f"https://mainnet.infura.io/v3/{INFURA_PROJECT_ID}"
    CONTRACT_ADDRESS = "0x0000000000000000000000000000000000000000"  
    
    # SQLite database
    DATABASE_URL = "sqlite:///./loans.db"
    
    # Redis caching
    REDIS_URL = "redis://localhost:6379/0"
    
    # Security
    SECRET_KEY = "your-secret-key-change-this-in-production"
    
    # Credit scoring settings
    CREDIT_SCORING = {
        'k_neighbors': 5,
        'max_model_size': 1000,  
        'cache_duration': 300,   
    }


config = Config()