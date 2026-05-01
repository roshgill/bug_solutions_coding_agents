FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for chromadb and building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Fly.io will use PORT env var)
EXPOSE 8000

# Run the app
CMD ["python", "mcp_server.py"]
