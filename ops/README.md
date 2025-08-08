# Deployment Configuration

This directory contains all deployment-related files for the Order Classifier.

## Files

- **`Dockerfile`**: Container build configuration for the Order Classifier
- **`docker-compose.yml`**: Single-service deployment
## Quick Start

```bash
# 1. Configure environment from project root (optional)
cp ../.env.example ../.env
vim ../.env

# 3. Deploy stack
docker-compose up -d

# 4. Check status
docker-compose ps
docker-compose logs -f
```

## Services

### order-classifier (Main NLP API)
- **Port**: 40012
- **Resources**: 512MB RAM, 1 CPU
- **Cache**: In-memory LRU cache (fast, lightweight)
- **Health Check**: `curl http://localhost:40012/health`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 40012 | API server port |
| `HOST` | 0.0.0.0 | Server bind address |
| `LOG_LEVEL` | INFO | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `CACHE_TTL` | 300 | Cache expiration in seconds |
| `CACHE_MAX_SIZE` | 1000 | Maximum in-memory cache entries |
| `COMPOSE_PROJECT_NAME` | order-classifier | Docker Compose project name |

## Commands

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f [service]

# Scale the API service
docker-compose up -d --scale order-classifier=3

# Stop services
docker-compose down

# Remove volumes
docker-compose down -v

# Rebuild and restart
docker-compose up -d --build --force-recreate
```

## Troubleshooting

**Service won't start:**
```bash
docker-compose logs order-classifier
```

**Cache performance issues:**
```bash
curl http://localhost:40012/metrics  # Check cache hit rates
```

**Rate limiting too strict:**
```bash
# Edit nginx.conf rate limits and restart
docker-compose restart nginx
```

**Health check failing:**
```bash
curl http://localhost:40012/health
curl http://localhost/health  # via nginx
```