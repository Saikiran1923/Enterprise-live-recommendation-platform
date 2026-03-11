.PHONY: install test run docker-up docker-down lint clean generate-data

install:
pip install -r requirements.txt

test:
pytest tests/ -v --asyncio-mode=auto

run:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

docker-up:
docker-compose up --build -d

docker-down:
docker-compose down

lint:
flake8 . --max-line-length=120 --exclude=.git,__pycache__

clean:
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete

generate-data:
python scripts/generate_synthetic_data.py --users 1000 --videos 5000 --interactions 50000
