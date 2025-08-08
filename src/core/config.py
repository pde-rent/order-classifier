"""Configuration settings for the service"""
import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Cache settings (in-memory TTL+LRU)
    cache_ttl: int = 300  # seconds (5 minutes)
    cache_max_size: int = 2000  # max entries in LRU cache
    cache_max_memory_mb: int = 500  # max memory usage in MB
    
    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 40012
    api_workers: int = 1
    
    # Performance settings
    max_batch_size: int = 32
    batch_timeout_ms: int = 50
    
    # Rate limiting
    rate_limit_per_second: int = 100
    rate_limit_burst: int = 200
    
    class Config:
        env_prefix = "EDGE_LLM_"
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()