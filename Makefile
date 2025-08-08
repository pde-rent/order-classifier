.PHONY: help install dev test run docker-build docker-up docker-down clean

help:
	@echo "Available commands:"
	@echo "  make install      Install dependencies"
	@echo "  make dev         Install with dev dependencies"
	@echo "  make test        Run tests"
	@echo "  make run         Run the service locally"
	@echo "  make docker-build Build Docker image"
	@echo "  make docker-up   Start services with docker-compose"
	@echo "  make docker-down Stop docker-compose services"
	@echo "  make clean       Clean up cache and temp files"

install:
	pip install uv
	uv pip install --system .

dev:
	pip install uv
	uv pip install --system -e ".[dev]"

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services started. Check health at http://localhost/health"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f llm-service

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf dist build *.egg-info

format:
	black src tests
	ruff check src tests --fix

lint:
	black --check src tests
	ruff check src tests

benchmark:
	@echo "Running performance benchmark..."
	python -m tests.benchmark