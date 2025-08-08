#!/usr/bin/env python3
"""
Simple script to run the FastAPI server for testing
"""
import uvicorn
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == "__main__":
    # Get configuration from environment
    port = int(os.getenv("PORT", "40012"))
    host = os.getenv("HOST", "127.0.0.1")
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("Starting Order Classifier NLP Server...")
    print(f"Server: http://localhost:{port}")
    print(f"API Documentation: http://localhost:{port}/docs")
    print(f"Health Check: http://localhost:{port}/health")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=False,
        log_level=log_level
    )