"""Tests for FastAPI endpoints"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, MagicMock
import json


@pytest.fixture
def client():
    """Create test client"""
    with patch('src.api.main.IntentClassifier') as mock_classifier:
        with patch('src.api.main.CacheManager') as mock_cache:
            # Setup mocks
            mock_instance = MagicMock()
            mock_instance.classify.return_value = {
                "confidence": 0.95,
                "type": "CREATE_ORDER",
                "params": {"orderParams": {"type": "market"}}
            }
            mock_classifier.return_value = mock_instance
            
            mock_cache_instance = AsyncMock()
            mock_cache_instance.get.return_value = None
            mock_cache_instance.set.return_value = True
            mock_cache_instance.connect.return_value = None
            mock_cache_instance.disconnect.return_value = None
            mock_cache_instance.get_stats.return_value = {"enabled": True}
            mock_cache.return_value = mock_cache_instance
            
            from src.api.main import app
            return TestClient(app)


class TestAPI:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "Order Classifier - NLP Intent Classification"
    
    def test_health_endpoint(self, client):
        """Test health check"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_classify_endpoint(self, client):
        """Test single classification"""
        response = client.post(
            "/classify",
            json={"text": "buy 1000 usdt"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "confidence" in data
        assert "type" in data
        assert "params" in data
    
    def test_classify_empty_text(self, client):
        """Test classification with empty text"""
        response = client.post(
            "/classify",
            json={"text": ""}
        )
        assert response.status_code == 422  # Validation error
    
    def test_classify_long_text(self, client):
        """Test classification with text exceeding max length"""
        long_text = "a" * 501  # Max is 500
        response = client.post(
            "/classify",
            json={"text": long_text}
        )
        assert response.status_code == 422
    
    def test_batch_classify(self, client):
        """Test batch classification"""
        with patch('src.api.main.classifier') as mock_classifier:
            mock_classifier.classify_batch.return_value = [
                {"confidence": 0.9, "type": "CREATE_ORDER", "params": {}},
                {"confidence": 0.8, "type": "CONNECT_WALLET", "params": {}}
            ]
            
            response = client.post(
                "/classify_batch",
                json={"texts": ["buy 1000 usdt", "connect wallet"]}
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
    
    def test_batch_classify_empty_list(self, client):
        """Test batch classification with empty list"""
        response = client.post(
            "/classify_batch",
            json={"texts": []}
        )
        assert response.status_code == 422
    
    def test_batch_classify_too_many(self, client):
        """Test batch classification with too many texts"""
        texts = ["test"] * 101  # Max is 100
        response = client.post(
            "/classify_batch",
            json={"texts": texts}
        )
        assert response.status_code == 422
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        # Should return Prometheus metrics
    
    def test_cache_clear(self, client):
        """Test cache clearing"""
        with patch('src.api.main.cache') as mock_cache:
            mock_cache.clear = AsyncMock(return_value=True)
            
            response = client.post("/cache/clear")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
    
    def test_classify_with_cache_hit(self, client):
        """Test classification with cache hit"""
        with patch('src.api.main.cache') as mock_cache:
            cached_result = {
                "confidence": 0.95,
                "type": "CREATE_ORDER",
                "params": {"orderParams": {"type": "market"}}
            }
            mock_cache.get = AsyncMock(return_value=cached_result)
            
            response = client.post(
                "/classify",
                json={"text": "buy 1000 usdt"}
            )
            assert response.status_code == 200
            data = response.json()
            assert data == cached_result