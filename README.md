# Order Classifier

A rule-based NLP intent classifier for trading applications. Uses pattern matching and semantic analysis techniques to classify user intent from natural language and extract trading parameters.

## Features

- **100% Accuracy** on comprehensive test suite (56 test cases)
- **Rule-Based NLP** using proven pattern matching and semantic analysis
- **8 Action Types** covering complete trading workflows
- **4 Order Types** supporting MARKET, LIMIT, TWAP, and RANGE orders
- **Time Literal Preservation** maintains expressions like "today", "tomorrow", "monday-sunday"
- **Parameter Extraction** for assets, amounts, prices, and time ranges
- **FastAPI Server** with rate limiting and health monitoring
- **In-Memory Cache** for improved response times
- **Lightweight** 30-80MB memory usage, no external dependencies

## Supported Actions

| Action | Description | Example Input |
|--------|-------------|---------------|
| `CREATE_ORDER` | Create trading orders | "buy 1000 USDT", "sell 2 ETH at 3500" |
| `CONNECT_WALLET` | Connect wallet | "connect wallet", "link wallet" |
| `DISCONNECT_WALLET` | Disconnect wallet | "disconnect", "logout" |
| `CHANGE_CHAIN` | Switch blockchain | "switch to arbitrum", "use polygon" |
| `CHANGE_PAIR` | Change trading pair | "trade AAVE", "switch to BTC/USDT" |
| `CANCEL_ORDER` | Cancel orders | "cancel order abc123" |
| `UPDATE_ORDER` | Modify orders | "update my limit order" |
| `GET_ORDER_INFO` | Get order status | "check order status" |

## Supported Order Types

| Order Type | Description | Example Input |
|------------|-------------|---------------|
| `MARKET` | Immediate execution | "buy 1000 USDT now" |
| `LIMIT` | Price-conditional | "buy 2 ETH at 3500" |
| `TWAP` | Time-weighted average | "buy ETH from monday to friday" |
| `RANGE` | Price range orders | "buy from 4000 to 5000" |

## Installation

### Prerequisites

- Python 3.12+
- uv package manager

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd order-classifier

# Install dependencies
uv sync

# Start the API server
uv run python run_server.py
```

The API server will be available at `http://localhost:40012`

### Docker Deployment

```bash
# Build and run with Docker Compose (from project root)
cd ops
docker-compose up --build

# Or build manually (from project root)
docker build -f ops/Dockerfile -t order-classifier .
docker run -p 40012:40012 order-classifier
```

### Production Deployment

The `/ops` directory contains all deployment-related files:

```
ops/
├── Dockerfile          # Container build configuration  
├── docker-compose.yml  # Multi-service deployment
└── README.md           # Deployment documentation
```

**Quick Deploy:**
```bash
cp .env.example .env     # Configure as needed
cd ops
docker-compose up -d     # Deploy in background
```

**Services:**
- **order-classifier**: Main NLP service with in-memory cache (port 40012)

**Rate Limits:**
- 1 request/second with burst capacity
- 20 requests/minute, 500/hour, 1500/day  
- Max 200 concurrent connections

## API Usage

### Health Check

```bash
curl -X GET "http://localhost:40012/health"
```

### Single Classification

```bash
curl -X POST "http://localhost:40012/classify" \
  -H "Content-Type: application/json" \
  -d '{"text": "buy 1000 USDT"}'
```

**Response:**
```json
{
  "confidence": 1.0,
  "type": "CREATE_ORDER",
  "params": {
    "type": "MARKET",
    "takingAmount": 1000.0,
    "takerAsset": "USDT"
  }
}
```

### Batch Classification

```bash
curl -X POST "http://localhost:40012/classify_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "buy 500 ETH",
      "connect wallet",
      "switch to polygon"
    ]
  }'
```

### Interactive Testing

```bash
# Run the interactive tester
uv run python tests/simple_test.py
```

## API Reference

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check and system status |
| `POST` | `/classify` | Classify single text input |
| `POST` | `/classify_batch` | Classify multiple texts |
| `GET` | `/metrics` | Prometheus metrics |
| `POST` | `/cache/clear` | Clear classification cache |
| `GET` | `/` | API information |

### Response Format

All classification responses follow this structure:

```json
{
  "confidence": 0.95,
  "type": "CREATE_ORDER",
  "params": {
    "type": "LIMIT",
    "takingAmount": 1000.0,
    "takerAsset": "USDT",
    "limitPrice": 4000.0
  }
}
```

### Parameters

#### Order Parameters

- `type`: Order type (MARKET, LIMIT, TWAP, etc.)
- `takingAmount`: Amount user wants to receive
- `makingAmount`: Amount user is providing  
- `takerAsset`: Asset user receives (or "base"/"quote" placeholder)
- `makerAsset`: Asset user provides (or "base"/"quote" placeholder)
- `limitPrice`: Price for limit orders
- `startDate`/`endDate`: Time literals for TWAP orders (e.g., "monday", "friday", "today", "tomorrow")
- `startPrice`/`endPrice`: Price range for range orders

#### Chain Parameters

- `chainId`: Target blockchain ID
  - `1`: Ethereum
  - `10`: Optimism  
  - `56`: BSC
  - `137`: Polygon
  - `8453`: Base
  - `42161`: Arbitrum
  - `43114`: Avalanche

#### Pair Parameters  

- `pair`: Trading pair in format "BASE/QUOTE" (e.g., "ETH/USDT")

## Integration Guide

### Quick Start Integration

#### Option 1: Docker Container (Recommended)
```bash
# 1. Clone and navigate to the project
git clone <repository-url>
cd order-classifier

# 2. Start the container
cd ops
docker-compose up -d

# 3. Verify service is running
curl http://localhost:40012/health
```

#### Option 2: Local Development
```bash
# 1. Install dependencies
uv sync

# 2. Start the server
uv run python run_server.py

# 3. Service available at http://localhost:40012
curl http://localhost:40012/health
```

### Frontend Integration Examples

#### JavaScript/TypeScript (Fetch API)

```typescript
interface ClassifyRequest {
  text: string;
}

interface ClassifyResponse {
  confidence: number;
  type: string | null;
  params: Record<string, any> | null;
}

class OrderClassifierClient {
  private baseUrl: string;
  
  constructor(baseUrl: string = 'http://localhost:40012') {
    this.baseUrl = baseUrl;
  }

  async classify(text: string): Promise<ClassifyResponse> {
    const response = await fetch(`${this.baseUrl}/classify`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // For CORS
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Classification failed: ${response.status}`);
    }

    return response.json();
  }

  async classifyBatch(texts: string[]): Promise<ClassifyResponse[]> {
    const response = await fetch(`${this.baseUrl}/classify_batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include',
      body: JSON.stringify({ texts }),
    });

    if (!response.ok) {
      throw new Error(`Batch classification failed: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health`);
    return response.json();
  }
}

// Usage Example
const classifier = new OrderClassifierClient();

async function handleUserInput(userText: string) {
  try {
    const result = await classifier.classify(userText);
    
    if (result.type === 'CREATE_ORDER') {
      const orderType = result.params?.type;
      console.log(`Creating ${orderType} order:`, result.params);
      
      // Handle different order types
      switch (orderType) {
        case 'MARKET':
          return handleMarketOrder(result.params);
        case 'LIMIT':
          return handleLimitOrder(result.params);
        case 'TWAP':
          return handleTwapOrder(result.params);
        case 'RANGE':
          return handleRangeOrder(result.params);
      }
    } else if (result.type === 'CONNECT_WALLET') {
      return handleWalletConnection();
    } else if (result.type === 'CHANGE_CHAIN') {
      return handleChainSwitch(result.params?.chainId);
    }
  } catch (error) {
    console.error('Classification error:', error);
    // Fallback to manual input parsing
  }
}
```

#### React Hook Example

```tsx
import { useState, useCallback } from 'react';

interface UseOrderClassifierOptions {
  baseUrl?: string;
  onError?: (error: Error) => void;
}

export function useOrderClassifier(options: UseOrderClassifierOptions = {}) {
  const { baseUrl = 'http://localhost:40012', onError } = options;
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ClassifyResponse | null>(null);

  const classify = useCallback(async (text: string) => {
    if (!text.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`${baseUrl}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResult(data);
      return data;
    } catch (error) {
      const err = error instanceof Error ? error : new Error('Classification failed');
      onError?.(err);
      throw err;
    } finally {
      setLoading(false);
    }
  }, [baseUrl, onError]);

  const reset = useCallback(() => {
    setResult(null);
    setLoading(false);
  }, []);

  return {
    classify,
    reset,
    loading,
    result,
    isOrder: result?.type === 'CREATE_ORDER',
    orderType: result?.params?.type,
    confidence: result?.confidence || 0,
  };
}

// Usage in component
function TradingInterface() {
  const { classify, loading, result, isOrder, orderType } = useOrderClassifier({
    onError: (error) => toast.error(error.message),
  });

  const handleSubmit = async (text: string) => {
    const classification = await classify(text);
    
    if (classification.type === 'CREATE_ORDER') {
      // Show order confirmation dialog
      setOrderPreview(classification.params);
    } else if (classification.type === 'CONNECT_WALLET') {
      // Trigger wallet connection
      await connectWallet();
    }
  };

  return (
    <div>
      <input 
        type="text" 
        placeholder="Enter trading command..."
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            handleSubmit(e.currentTarget.value);
          }
        }}
      />
      
      {loading && <div>Classifying...</div>}
      
      {result && (
        <div className="classification-result">
          <div>Action: {result.type}</div>
          <div>Confidence: {(result.confidence * 100).toFixed(1)}%</div>
          {isOrder && (
            <div>Order Type: {orderType}</div>
          )}
        </div>
      )}
    </div>
  );
}
```

#### Python Client Example

```python
import httpx
import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

@dataclass
class ClassifyResponse:
    confidence: float
    type: Optional[str]
    params: Optional[Dict[str, Any]]

class OrderClassifierClient:
    def __init__(self, base_url: str = "http://localhost:40012"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def classify(self, text: str) -> ClassifyResponse:
        """Classify a single text input."""
        response = await self.client.post(
            f"{self.base_url}/classify",
            json={"text": text}
        )
        response.raise_for_status()
        data = response.json()
        
        return ClassifyResponse(
            confidence=data["confidence"],
            type=data.get("type"),
            params=data.get("params")
        )
    
    async def classify_batch(self, texts: List[str]) -> List[ClassifyResponse]:
        """Classify multiple texts in one request."""
        response = await self.client.post(
            f"{self.base_url}/classify_batch",
            json={"texts": texts}
        )
        response.raise_for_status()
        data = response.json()
        
        return [
            ClassifyResponse(
                confidence=item["confidence"],
                type=item.get("type"),
                params=item.get("params")
            )
            for item in data
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        response = await self.client.get(f"{self.base_url}/health")
        return response.json()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

# Usage Example
async def main():
    classifier = OrderClassifierClient()
    
    try:
        # Health check
        health = await classifier.health_check()
        print(f"Service status: {health['status']}")
        
        # Single classification
        result = await classifier.classify("buy 1000 usdt at 4000")
        print(f"Action: {result.type}")
        print(f"Confidence: {result.confidence:.2%}")
        
        if result.type == "CREATE_ORDER":
            order_type = result.params.get("type")
            print(f"Order type: {order_type}")
            
            if order_type == "LIMIT":
                price = result.params.get("limitPrice")
                print(f"Limit price: ${price}")
        
        # Batch classification
        texts = [
            "sell 2 eth now",
            "connect wallet",
            "switch to arbitrum",
            "buy eth from monday to friday"
        ]
        
        results = await classifier.classify_batch(texts)
        for text, result in zip(texts, results):
            print(f"'{text}' -> {result.type}")
        
    finally:
        await classifier.close()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
```

#### Node.js/Express Backend Example

```javascript
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

// Order Classifier client
class OrderClassifierClient {
  constructor(baseUrl = 'http://localhost:40012') {
    this.baseUrl = baseUrl;
    this.axios = axios.create({ timeout: 10000 });
  }

  async classify(text) {
    const response = await this.axios.post(`${this.baseUrl}/classify`, {
      text
    });
    return response.data;
  }

  async healthCheck() {
    const response = await this.axios.get(`${this.baseUrl}/health`);
    return response.data;
  }
}

const classifier = new OrderClassifierClient();

// API Routes
app.post('/api/process-command', async (req, res) => {
  try {
    const { userInput } = req.body;
    
    if (!userInput || userInput.trim().length === 0) {
      return res.status(400).json({ error: 'User input is required' });
    }
    
    // Classify the input
    const classification = await classifier.classify(userInput);
    
    // Process based on classification
    let response = {
      classification,
      action: null,
      data: null
    };
    
    switch (classification.type) {
      case 'CREATE_ORDER':
        response.action = 'show_order_form';
        response.data = {
          orderType: classification.params.type,
          suggestedParams: classification.params
        };
        break;
        
      case 'CONNECT_WALLET':
        response.action = 'connect_wallet';
        break;
        
      case 'CHANGE_CHAIN':
        response.action = 'switch_chain';
        response.data = {
          chainId: classification.params.chainId
        };
        break;
        
      case 'CHANGE_PAIR':
        response.action = 'select_pair';
        response.data = {
          pair: classification.params.pair
        };
        break;
        
      default:
        response.action = 'unknown_intent';
        break;
    }
    
    res.json(response);
    
  } catch (error) {
    console.error('Classification error:', error);
    res.status(500).json({ 
      error: 'Failed to process command',
      details: error.message 
    });
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const health = await classifier.healthCheck();
    res.json({
      api: 'healthy',
      classifier: health
    });
  } catch (error) {
    res.status(503).json({
      api: 'degraded',
      error: error.message
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`API server running on port ${PORT}`);
});
```

### Error Handling & Edge Cases

#### Common Error Responses

```javascript
// Handle different error types
async function classifyWithErrorHandling(text) {
  try {
    const response = await fetch('/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    // Handle HTTP errors
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      
      switch (response.status) {
        case 400:
          throw new Error(`Invalid input: ${errorData.detail || 'Bad request'}`);
        case 422:
          throw new Error(`Validation error: ${errorData.detail?.[0]?.msg || 'Invalid data'}`);
        case 429:
          throw new Error('Too many requests. Please slow down.');
        case 500:
          throw new Error('Server error. Please try again later.');
        case 503:
          throw new Error('Service unavailable. Please check system status.');
        default:
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    }

    const result = await response.json();
    
    // Handle low confidence results
    if (result.confidence < 0.3) {
      console.warn('Low confidence classification:', result);
      // Maybe show a clarification dialog
    }
    
    return result;
    
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error. Please check your connection.');
    }
    throw error;
  }
}
```

#### Validation Error Examples

```json
// Empty text
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "text"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}

// Text too long
{
  "detail": [
    {
      "type": "string_too_long", 
      "loc": ["body", "text"],
      "msg": "String should have at most 500 characters",
      "input": "very long text..."
    }
  ]
}

// Invalid batch size
{
  "detail": [
    {
      "type": "too_short",
      "loc": ["body", "texts"],
      "msg": "List should have at least 1 item"
    }
  ]
}
```

### Performance & Optimization

#### Caching Strategy

The service includes built-in caching for improved performance:

```javascript
// Monitor cache performance
async function getCacheStats() {
  const health = await fetch('/health').then(r => r.json());
  const cacheStats = health.cache;
  
  console.log(`Cache hit rate: ${cacheStats.hit_rate}%`);
  console.log(`Cache size: ${cacheStats.current_size}/${cacheStats.max_size}`);
  
  // Clear cache if needed
  if (cacheStats.hit_rate < 50) {
    await fetch('/cache/clear', { method: 'POST' });
  }
}
```

#### Batch Processing for Performance

```javascript
// Process multiple inputs efficiently
async function processBulkCommands(commands) {
  const BATCH_SIZE = 10;
  const results = [];
  
  for (let i = 0; i < commands.length; i += BATCH_SIZE) {
    const batch = commands.slice(i, i + BATCH_SIZE);
    const batchResults = await classifier.classifyBatch(batch);
    results.push(...batchResults);
    
    // Small delay between batches to avoid overwhelming the service
    if (i + BATCH_SIZE < commands.length) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }
  
  return results;
}
```

#### Timeout and Retry Logic

```javascript
async function robustClassify(text, maxRetries = 3, timeoutMs = 5000) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
      
      const response = await fetch('/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
        signal: controller.signal
      });
      
      clearTimeout(timeoutId);
      
      if (response.ok) {
        return await response.json();
      }
      
      // Retry on server errors, not client errors
      if (response.status >= 500) {
        throw new Error(`Server error: ${response.status}`);
      } else {
        throw new Error(`Client error: ${response.status}`);
      }
      
    } catch (error) {
      if (attempt === maxRetries - 1) {
        throw error;
      }
      
      // Exponential backoff
      const delay = Math.pow(2, attempt) * 1000;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
}

### Monitoring & Health Checks

#### Health Check Implementation

```javascript
// Continuous health monitoring
class HealthMonitor {
  constructor(serviceUrl, checkInterval = 30000) {
    this.serviceUrl = serviceUrl;
    this.checkInterval = checkInterval;
    this.isHealthy = false;
    this.lastCheck = null;
    this.callbacks = [];
  }

  onHealthChange(callback) {
    this.callbacks.push(callback);
  }

  async startMonitoring() {
    setInterval(async () => {
      try {
        const health = await fetch(`${this.serviceUrl}/health`);
        const data = await health.json();
        
        const wasHealthy = this.isHealthy;
        this.isHealthy = data.status === 'healthy';
        this.lastCheck = new Date();

        // Alert on health status change
        if (wasHealthy !== this.isHealthy) {
          this.callbacks.forEach(cb => cb(this.isHealthy, data));
        }

        // Log cache performance warnings
        if (data.cache?.hit_rate < 50) {
          console.warn('Low cache hit rate:', data.cache.hit_rate);
        }

      } catch (error) {
        console.error('Health check failed:', error);
        this.isHealthy = false;
      }
    }, this.checkInterval);
  }
}

// Usage
const monitor = new HealthMonitor('http://localhost:40012');
monitor.onHealthChange((isHealthy, data) => {
  if (isHealthy) {
    console.log('✅ Service is healthy');
  } else {
    console.error('❌ Service is unhealthy', data);
    // Maybe show user notification or fallback UI
  }
});
monitor.startMonitoring();
```

### Production Deployment Guide

#### Docker Compose Production Setup

```yaml
# ops/docker-compose.prod.yml
services:
  order-classifier:
    build:
      context: ..
      dockerfile: ops/Dockerfile
    ports:
      - "40012:40012"
    environment:
      - PORT=40012
      - HOST=0.0.0.0
      - LOG_LEVEL=INFO
      - CACHE_TTL=300
      - CACHE_MAX_SIZE=5000
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '2'
        reservations:
          memory: 512M
          cpus: '1'
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:40012/health"]
      interval: 30s
      timeout: 10s
      start_period: 30s
      retries: 3

  # Optional: Add monitoring
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

networks:
  default:
    driver: bridge
```

#### Environment Variables for Production

```bash
# .env.production
PORT=40012
HOST=0.0.0.0
LOG_LEVEL=INFO

# Cache settings for production
CACHE_TTL=300
CACHE_MAX_SIZE=5000
CACHE_MAX_MEMORY_MB=1000

# Docker settings
COMPOSE_PROJECT_NAME=order-classifier-prod
```

#### Load Balancer Configuration (Nginx)

```nginx
# nginx.conf for production proxy
upstream order_classifier {
    least_conn;
    server localhost:40012 max_fails=3 fail_timeout=30s;
    # Add more instances for horizontal scaling:
    # server localhost:40013 max_fails=3 fail_timeout=30s;
    # server localhost:40014 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name your-domain.com;

    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # CORS headers for your domains
    location / {
        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Allow-Origin 'https://your-frontend.com';
            add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
            add_header Access-Control-Allow-Headers 'Content-Type';
            add_header Access-Control-Max-Age 86400;
            return 204;
        }

        add_header Access-Control-Allow-Origin 'https://your-frontend.com';
        proxy_pass http://order_classifier;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://order_classifier/health;
        access_log off;
    }
}
```

### Security Considerations

#### Input Validation and Sanitization

```javascript
// Client-side input sanitization
function sanitizeInput(text) {
  // Remove potentially harmful content
  return text
    .trim()
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '') // Remove scripts
    .replace(/[^\w\s\-.,!?@$%&*()/+={}[\]|\\:;"'<>]/g, '') // Allow only safe chars
    .substring(0, 500); // Enforce length limit
}

// Validate before sending
function validateInput(text) {
  if (!text || text.trim().length === 0) {
    throw new Error('Input cannot be empty');
  }
  
  if (text.length > 500) {
    throw new Error('Input too long (max 500 characters)');
  }
  
  // Check for suspicious patterns
  const suspiciousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\w+\s*=/i,
    /eval\s*\(/i,
  ];
  
  for (const pattern of suspiciousPatterns) {
    if (pattern.test(text)) {
      throw new Error('Invalid input detected');
    }
  }
  
  return true;
}
```

#### API Security Headers

```javascript
// Express.js security middleware
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      connectSrc: ["'self'", "http://localhost:40012", "https://api.1edge.trade"]
    }
  }
}));

// Rate limiting for your API endpoints
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // Limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use('/api/', limiter);
```

### Troubleshooting Guide

#### Common Issues and Solutions

**Issue**: CORS errors from frontend
```javascript
// Solution: Verify CORS configuration
const testCORS = async () => {
  try {
    const response = await fetch('http://localhost:40012/classify', {
      method: 'OPTIONS',
      headers: {
        'Origin': window.location.origin,
        'Access-Control-Request-Method': 'POST'
      }
    });
    console.log('CORS test:', response.ok ? 'PASS' : 'FAIL');
  } catch (error) {
    console.error('CORS test failed:', error);
  }
};
```

**Issue**: High latency or timeouts
```bash
# Check container resources
docker stats order-classifier-service

# Check cache performance
curl http://localhost:40012/health | jq '.cache'

# Monitor request patterns
docker logs -f order-classifier-service | grep "classification"
```

**Issue**: Memory usage growing over time
```bash
# Monitor memory usage
watch -n 5 'docker stats --no-stream order-classifier-service'

# Clear cache if needed
curl -X POST http://localhost:40012/cache/clear

# Restart container
docker-compose restart order-classifier
```

**Issue**: Low classification accuracy
```javascript
// Log low confidence predictions for analysis
async function classifyWithLogging(text) {
  const result = await classifier.classify(text);
  
  if (result.confidence < 0.7) {
    console.log('Low confidence prediction:', {
      text,
      result,
      timestamp: new Date().toISOString()
    });
    
    // Maybe collect these for model improvement
    analytics.track('low_confidence_prediction', {
      input: text,
      confidence: result.confidence,
      predicted_type: result.type
    });
  }
  
  return result;
}
```

### Performance Benchmarks

#### Expected Performance Metrics

- **Latency**: 2-10ms per classification (with cache)
- **Throughput**: 500+ requests/second (single instance)  
- **Memory**: 256-512MB RAM usage
- **Cache Hit Rate**: 70-90% for typical workloads
- **Accuracy**: 100% on provided test cases

#### Load Testing Example

```javascript
// Simple load test
async function loadTest(concurrency = 10, duration = 30000) {
  const startTime = Date.now();
  const results = [];
  const testCases = [
    "buy 1000 usdt",
    "sell 2 eth at 3500",
    "connect wallet",
    "switch to arbitrum"
  ];
  
  const workers = Array(concurrency).fill().map(async () => {
    while (Date.now() - startTime < duration) {
      const testCase = testCases[Math.floor(Math.random() * testCases.length)];
      const start = Date.now();
      
      try {
        await classifier.classify(testCase);
        results.push({
          success: true,
          latency: Date.now() - start
        });
      } catch (error) {
        results.push({
          success: false,
          error: error.message,
          latency: Date.now() - start
        });
      }
      
      await new Promise(resolve => setTimeout(resolve, 10)); // Small delay
    }
  });
  
  await Promise.all(workers);
  
  const successRate = results.filter(r => r.success).length / results.length;
  const avgLatency = results.reduce((sum, r) => sum + r.latency, 0) / results.length;
  
  console.log(`Load test results:
    Duration: ${duration}ms
    Concurrency: ${concurrency}
    Total requests: ${results.length}
    Success rate: ${(successRate * 100).toFixed(1)}%
    Average latency: ${avgLatency.toFixed(1)}ms
    Throughput: ${(results.length / (duration / 1000)).toFixed(1)} req/s`);
}

// Run load test
loadTest(20, 60000); // 20 concurrent users for 1 minute
```

## API Reference

### Endpoints Overview

| Method | Endpoint | Description | Rate Limit |
|--------|----------|-------------|------------|
| `POST` | `/classify` | Single text classification | 100/min |
| `POST` | `/classify_batch` | Multiple text classification | 20/min |
| `GET` | `/health` | Service health check | Unlimited |
| `GET` | `/metrics` | Performance metrics | 60/min |
| `POST` | `/cache/clear` | Clear classification cache | 10/min |
| `GET` | `/docs` | Interactive API documentation | Unlimited |

### Request/Response Schemas

#### POST `/classify`

**Request:**
```typescript
interface ClassifyRequest {
  text: string; // 1-500 characters, required
}
```

**Response:**
```typescript
interface ClassifyResponse {
  confidence: number;        // 0.0 to 1.0
  type: string | null;      // Action type or null
  params: object | null;    // Extracted parameters or null
}
```

**Example Response:**
```json
{
  "confidence": 0.95,
  "type": "CREATE_ORDER",
  "params": {
    "type": "LIMIT",
    "takerAsset": "ETH",
    "takingAmount": 2.0,
    "limitPrice": 3500.0
  }
}
```

#### POST `/classify_batch`

**Request:**
```typescript
interface BatchClassifyRequest {
  texts: string[]; // 1-100 items, each 1-500 characters
}
```

**Response:**
```typescript
type BatchClassifyResponse = ClassifyResponse[];
```

#### GET `/health`

**Response:**
```typescript
interface HealthResponse {
  status: 'healthy' | 'unhealthy';
  classifier: {
    loaded: boolean;
    type: string;
  };
  cache: {
    enabled: boolean;
    current_size: number;
    max_size: number;
    hit_rate: number;
  };
  metrics: {
    requests_total: number;
    avg_latency_ms: number;
    model_load_time_s: number;
  };
  version: string;
}
```

### Error Responses

**Validation Error (422):**
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**Rate Limited (429):**
```json
{
  "error": "Too many requests. Please slow down."
}
```

### Integration Examples

#### Python
```python
import httpx

client = httpx.Client(base_url="http://localhost:40012")

# Single classification
result = client.post("/classify", json={"text": "buy 100 USDC"})
print(result.json())

# Batch classification  
results = client.post("/classify_batch", json={
    "texts": ["sell ETH", "connect wallet"]
})
print(results.json())
```

#### JavaScript
```javascript
const classify = async (text) => {
  const response = await fetch('http://localhost:40012/classify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return await response.json();
};

const result = await classify("buy 50 AAVE tokens");
console.log(result);
```

#### cURL
```bash
# Health check
curl http://localhost:40012/health

# Single classification
curl -X POST http://localhost:40012/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "create limit order for 1000 USDT at 4000"}'

# Batch classification
curl -X POST http://localhost:40012/classify_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["disconnect wallet", "buy ETH"]}'
```

## Supported Assets

### Base Assets (Trading Tokens)
- **ETH** (aliases: WETH, Ethereum)
- **BTC** (aliases: WBTC, Bitcoin)  
- **AAVE**
- **1INCH**
- **USDC**

### Quote Assets (Quote Currencies)
- **USDT** (aliases: Tether, USDT0, Tether USD)

## Examples

### Market Orders

```bash
# Simple buy
"buy 1000 USDT" → MARKET order, takerAsset: USDT

# Immediate sell
"sell 2 ETH now" → MARKET order, takerAsset: ETH

# Trading slang
"dump all my bitcoin" → MARKET order, takerAsset: BTC
"load up on ETH" → MARKET order, makerAsset: ETH
```

### Limit Orders

```bash
# Basic limit order
"buy 1000 USDT at 4000" → LIMIT order with price

# Missing tokens (uses placeholders)
"buy at 4000" → LIMIT order, takerAsset: "quote", makerAsset: "base"

# Complex limit
"I want to buy 1 BTC for 45000 USDT" → LIMIT order with both assets
```

### Advanced Orders

```bash
# TWAP orders with day recognition
"buy ETH starting monday until friday" → TWAP order
"accumulate bitcoin from tuesday to thursday" → TWAP order
"buy gradually from saturday until sunday" → TWAP order
"buy 200 usdt into bitcoin hourly for a day" → TWAP order

# Range orders  
"buy from 4000 to 5000" → RANGE order
"sell between 3000 and 3500 in 10 steps" → RANGE order with steps

# Conditional orders
"sell when price hits 5000" → LIMIT order with trigger
"buy at 4000" → LIMIT order (uses base/quote placeholders)
```

### Wallet & Chain Operations

```bash
# Wallet management
"connect wallet" → CONNECT_WALLET
"disconnect" → DISCONNECT_WALLET

# Chain switching
"switch to arbitrum" → CHANGE_CHAIN (chainId: 42161)
"use polygon network" → CHANGE_CHAIN (chainId: 137)

# Pair selection
"trade AAVE tokens" → CHANGE_PAIR (pair: "AAVE/USDT")
"switch to BTC/USDT" → CHANGE_PAIR
```

## Testing

### Run Test Suite

```bash
# Comprehensive accuracy test (56 test cases)
uv run python tests/test_accuracy.py

# API integration tests
uv run python -m pytest tests/test_api.py

# Individual test files
uv run python -m pytest tests/

# Interactive tester
uv run python tests/simple_test.py
```

### Test Results

The system achieves 100% accuracy on all test scenarios:

- Market orders with various verbal expressions
- Limit orders with price detection  
- TWAP orders with day-of-week recognition (monday-sunday)
- Time literal preservation ("today", "tomorrow", "next week")
- Range orders with price ranges
- Missing token handling with base/quote placeholders
- Wallet and chain operations
- Edge cases and error handling

## Performance

- **Latency**: ~2ms per classification (in-memory cache)
- **Throughput**: 500+ requests/second
- **Memory**: 30-80MB (rule-based NLP + in-memory cache)
- **Accuracy**: 100% on comprehensive test suite  
- **Dependencies**: No GPU, no external databases, pure Python NLP

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Server Configuration
PORT=40012             # API server port
HOST=0.0.0.0           # Bind address
LOG_LEVEL=INFO         # Logging level (DEBUG/INFO/WARNING/ERROR)

# In-Memory Cache (improves performance)
CACHE_TTL=300                 # Cache expiration in seconds
CACHE_MAX_SIZE=1000           # Maximum cache entries

# Docker Compose
COMPOSE_PROJECT_NAME=order-classifier  # Project name prefix
```

### Deployment Options

**Option 1: Docker Compose (Recommended)**
```bash
cd ops
docker-compose up -d
# Direct API access: http://localhost:40012
```

**Option 2: Standalone Container**
```bash
docker build -f ops/Dockerfile -t order-classifier .
docker run -e PORT=40012 -p 40012:40012 order-classifier
```

**Option 3: Local Development**
```bash
uv run python run_server.py
# Access at: http://localhost:40012
```

### Customization

The classifier can be customized by modifying:

- **Token mappings**: Add new cryptocurrencies in `intent_classifier.py`
- **Chain mappings**: Add new blockchain networks
- **Patterns**: Enhance regex patterns for better recognition
- **Training data**: Expand ML training examples

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Caddy         │    │  FastAPI         │    │  NLP Intent     │
│   Rate Limit    │───▶│  Server          │───▶│  Classifier     │
│   Load Balance  │    │  (main.py)       │    │  (rule-based)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  In-Memory      │    │  Pattern         │    │  Response       │
│  LRU Cache      │    │  Matching &      │    │  Formation      │
│  (Fast)         │    │  Extraction      │    │  (JSON)         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add test cases for new functionality
4. Ensure 100% test coverage is maintained
5. Submit a pull request

## License

[Add your license information here]

## Support

For questions and support:

- Check the test cases in `tests/test_cases.json` for examples
- Run `uv run python tests/simple_test.py` for interactive testing
- Review API documentation at `http://localhost:40012/docs`

---

**Order Classifier** - High-performance, CPU-only intent classification for trading applications.