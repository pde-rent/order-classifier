"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional


class ClassifyRequest(BaseModel):
    """Request model for single text classification"""
    text: str = Field(..., min_length=1, max_length=500, description="Text to classify")
    
    @validator('text')
    def validate_text(cls, v):
        # Clean up whitespace
        v = v.strip()
        if not v:
            raise ValueError("Text cannot be empty")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "buy 1000 usdt at 4000"
            }
        }


class ClassifyResponse(BaseModel):
    """Response model for classification results"""
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    type: Optional[str] = Field(None, description="Action type")
    params: Optional[Dict[str, Any]] = Field(None, description="Extracted parameters")
    
    class Config:
        json_schema_extra = {
            "example": {
                "confidence": 0.95,
                "type": "CREATE_ORDER",
                "params": {
                    "orderParams": {
                        "type": "limit",
                        "takingAmount": 1000,
                        "price": 4000
                    }
                }
            }
        }


class BatchClassifyRequest(BaseModel):
    """Request model for batch classification"""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=100,
        description="List of texts to classify"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        # Clean up each text
        cleaned = []
        for text in v:
            text = text.strip()
            if not text:
                raise ValueError("Text cannot be empty")
            if len(text) > 500:
                raise ValueError(f"Text too long: {len(text)} > 500")
            cleaned.append(text)
        return cleaned
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "buy 1000 usdt",
                    "sell 0.5 eth at 3500",
                    "connect my wallet"
                ]
            }
        }