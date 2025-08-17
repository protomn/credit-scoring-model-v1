#!/usr/bin/env python3
"""
Start script for production deployment
"""
import uvicorn
from main_new import app

if __name__ == "__main__":
    uvicorn.run(
        "main_new:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    ) 