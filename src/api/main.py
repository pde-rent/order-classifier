"""FastAPI service for intent classification"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field

from ..models.intent_classifier import IntentClassifier
from ..core.config import settings
from ..core.cache import CacheManager
from .models import ClassifyRequest, ClassifyResponse, BatchClassifyRequest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Lightweight metrics using logging and in-memory counters
@dataclass
class Metrics:
    requests_total: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    latencies: List[float] = field(default_factory=list)
    model_load_time: float = 0.0
    
    def log_request(self, latency: float, cache_hit: bool = False):
        self.requests_total += 1
        self.latencies.append(latency)
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        # Log every 100 requests for efficiency
        if self.requests_total % 100 == 0:
            avg_latency = sum(self.latencies[-100:]) / min(100, len(self.latencies))
            hit_rate = self.cache_hits / self.requests_total * 100
            logger.info(f"METRICS: requests={self.requests_total}, avg_latency={avg_latency:.3f}s, hit_rate={hit_rate:.1f}%")

metrics = Metrics()

# Global instances
classifier: IntentClassifier = None
cache: CacheManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global classifier, cache
    
    start_time = time.time()
    
    # Initialize classifier
    logger.info("Loading CPU-only intent classifier...")
    classifier = IntentClassifier(
        use_ml_enhancement=True  # Use scikit-learn if available
    )
    
    load_time = time.time() - start_time
    metrics.model_load_time = load_time
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    # Initialize cache
    cache = CacheManager(
        ttl=settings.cache_ttl,
        max_size=settings.cache_max_size,
        max_memory_mb=settings.cache_max_memory_mb
    )
    await cache.connect()
    
    yield
    
    # Cleanup
    await cache.disconnect()


app = FastAPI(
    title="Order Classifier - NLP Intent Classification",
    description="Rule-based NLP system for trading intent classification using proven pattern matching and semantic analysis techniques",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - Allow localhost (all ports) and 1edge.trade domains
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?|https?://(\w+\.)?1edge\.trade",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassifyResponse)
async def classify_intent(request: ClassifyRequest) -> Dict[str, Any]:
    """
    Classify user intent from text input.
    
    Returns action type, confidence, and extracted parameters.
    """
    start_time = time.time()
    
    try:
        # Check cache first
        cached_result = await cache.get(request.text)
        if cached_result:
            latency = time.time() - start_time
            metrics.log_request(latency, cache_hit=True)
            return cached_result
        
        # Classify with model
        result = classifier.classify(request.text)
        
        # Cache result
        await cache.set(request.text, result)
        
        # Record metrics
        latency = time.time() - start_time
        metrics.log_request(latency, cache_hit=False)
        
        return result
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify_batch")
async def classify_batch(request: BatchClassifyRequest) -> List[Dict[str, Any]]:
    """
    Classify multiple texts in a single request.
    
    More efficient for high-volume scenarios.
    """
    start_time = time.time()
    
    try:
        # Check cache for all texts
        cached_results = await cache.get_batch(request.texts)
        
        # Separate cached and uncached
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(request.texts)
        cache_hits = 0
        
        for i, text in enumerate(request.texts):
            if cached_results.get(text):
                results[i] = cached_results[text]
                cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts
        if uncached_texts:
            new_results = []
            for text in uncached_texts:
                new_results.append(classifier.classify(text))
            
            # Cache new results
            cache_items = {}
            for text, result in zip(uncached_texts, new_results):
                cache_items[text] = result
            await cache.set_batch(cache_items)
            
            # Merge results
            for idx, result in zip(uncached_indices, new_results):
                results[idx] = result
        
        # Record batch metrics
        total_latency = time.time() - start_time
        for i in range(len(request.texts)):
            metrics.log_request(total_latency / len(request.texts), cache_hit=(i < cache_hits))
        
        return results
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check model
        if classifier is None:
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "reason": "Model not loaded"}
            )
        
        # Check cache
        cache_stats = await cache.get_stats()
        
        return {
            "status": "healthy",
            "classifier": {
                "loaded": True,
                "type": "CPU-only rule-based with optional ML enhancement"
            },
            "cache": cache_stats,
            "metrics": {
                "requests_total": metrics.requests_total,
                "cache_hits": metrics.cache_hits,
                "cache_misses": metrics.cache_misses,
                "cache_hit_rate": round(metrics.cache_hits / max(metrics.requests_total, 1) * 100, 2),
                "avg_latency_ms": round(sum(metrics.latencies[-1000:]) / max(len(metrics.latencies[-1000:]), 1) * 1000, 2) if metrics.latencies else 0,
                "model_load_time_s": round(metrics.model_load_time, 3)
            },
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/metrics")
async def get_metrics():
    """Lightweight metrics endpoint in JSON format"""
    return {
        "requests_total": metrics.requests_total,
        "cache_hits": metrics.cache_hits,
        "cache_misses": metrics.cache_misses,
        "cache_hit_rate": round(metrics.cache_hits / max(metrics.requests_total, 1) * 100, 2),
        "avg_latency_ms": round(sum(metrics.latencies[-1000:]) / max(len(metrics.latencies[-1000:]), 1) * 1000, 2) if metrics.latencies else 0,
        "model_load_time_s": round(metrics.model_load_time, 3),
        "recent_requests": len([l for l in metrics.latencies[-100:] if l]),
        "p95_latency_ms": round(sorted(metrics.latencies[-1000:])[-int(len(metrics.latencies[-1000:]) * 0.05):][0] * 1000, 2) if len(metrics.latencies) >= 20 else 0
    }


@app.post("/cache/clear")
async def clear_cache():
    """Clear all cached results"""
    try:
        success = await cache.clear()
        return {"success": success, "message": "Cache cleared"}
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Order Classifier - NLP Intent Classification",
        "version": "1.0.0",
        "type": "Rule-based NLP with pattern matching and semantic analysis",
        "endpoints": {
            "POST /classify": "Classify single text",
            "POST /classify_batch": "Classify multiple texts",
            "GET /health": "Health check",
            "GET /metrics": "Prometheus metrics",
            "POST /cache/clear": "Clear cache"
        }
    }


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )