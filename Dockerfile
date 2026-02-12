FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python path for imports
ENV PYTHONPATH=/app/src

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY outputs/ ./outputs/
COPY run_batch.py ./run_batch.py

# Create directories (models may not exist yet, created as empty)
RUN mkdir -p outputs/predictions models

# Expose ports
EXPOSE 8000 8050

# Run API (dashboard overrides CMD in docker-compose)
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
