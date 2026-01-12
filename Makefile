.PHONY: install test lint format clean notebook data help

# Default target
help:
	@echo "Dynamic Pricing Engine - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies with Poetry"
	@echo "  make test       - Run pytest test suite"
	@echo "  make lint       - Run ruff linter"
	@echo "  make format     - Format code with black"
	@echo "  make clean      - Remove Python cache files"
	@echo "  make notebook   - Start Jupyter notebook server"
	@echo "  make data       - Download Kaggle dataset (requires kaggle CLI)"
	@echo ""

# Install dependencies
install:
	poetry install

# Run tests
test:
	poetry run pytest tests/ -v

# Lint code
lint:
	poetry run ruff check src/ tests/

# Format code
format:
	poetry run black src/ tests/ notebooks/
	poetry run ruff check --fix src/ tests/

# Clean cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true

# Start Jupyter notebook
notebook:
	poetry run jupyter notebook notebooks/

# Download Kaggle dataset
data:
	@echo "Downloading Uber & Lyft Cab Prices dataset from Kaggle..."
	@mkdir -p data/raw
	kaggle datasets download -d ravi72munde/uber-lyft-cab-prices -p data/raw --unzip
	@echo "Dataset downloaded to data/raw/"
