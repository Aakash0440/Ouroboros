"""Generates a production-ready Dockerfile for OUROBOROS deployment."""

DOCKERFILE = '''FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc g++ make \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \\
    fastapi uvicorn httpx pydantic

# Copy source
COPY . .
RUN pip install -e . --no-cache-dir

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
  CMD python -c "import urllib.request; urllib.request.urlopen(\'http://localhost:8000/health\')"

# Run the API
EXPOSE 8000
CMD ["uvicorn", "ouroboros.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
'''

DOCKER_COMPOSE = '''version: "3.8"
services:
  ouroboros:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./results:/app/results
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: "2.0"
'''

def generate(output_dir: str = ".") -> None:
    from pathlib import Path
    Path(f"{output_dir}/Dockerfile").write_text(DOCKERFILE)
    Path(f"{output_dir}/docker-compose.yml").write_text(DOCKER_COMPOSE)
    print(f"Generated Dockerfile and docker-compose.yml in {output_dir}")
    print("To deploy:")
    print("  docker build -t ouroboros .")
    print("  docker run -p 8000:8000 ouroboros")
    print("  # or:")
    print("  docker-compose up -d")
    