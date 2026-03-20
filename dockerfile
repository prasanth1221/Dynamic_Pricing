# Base image
FROM python:3.10.11-slim

# Prevents Python from writing .pyc files & enables clean logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# System dependencies for scientific libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Working directory inside container
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt .

# Install uv
RUN pip install --no-cache-dir uv

# Use uv to install dependencies (blazing fast)
RUN uv pip install --system --no-cache -r requirements.txt

# Copy full project
COPY . .

# Run setup: create directories + sample data (non-interactive)
RUN echo "y" | python setup.py

# Train the RL agent if no model exists
RUN python -c "\
import os; \
model_dir = 'models/trained_models'; \
has_model = os.path.isdir(model_dir) and bool(os.listdir(model_dir)); \
import subprocess, sys; \
subprocess.run([sys.executable, 'training/train.py']) if not has_model else print('Model already exists, skipping training.')"

# Expose port 8080
EXPOSE 8080

# Start server with Uvicorn on port 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]


