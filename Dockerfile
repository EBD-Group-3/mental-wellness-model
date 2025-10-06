# Use Python 3.10 slim image as base
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Install the package in development mode
RUN pip install -e .

# Create directories for data, models, and credentials
RUN mkdir -p /app/data /app/models /app/output /app/credentials

# Copy GCS credentials
COPY credentials/ /app/credentials/

# Ensure basic trained model is available in the container
RUN if [ -f "./models/basic_trained_model.joblib" ]; then \
        cp ./models/basic_trained_model.joblib /app/models/; \
        echo "Copied basic_trained_model.joblib to /app/models/"; \
    else \
        echo "Warning: basic_trained_model.joblib not found"; \
    fi

# Set environment variable for GCS credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/credentials/mentalwellness-473814-key.json"

# Copy and set up the startup script
COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use the startup script as entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command runs FastAPI with Gunicorn
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]