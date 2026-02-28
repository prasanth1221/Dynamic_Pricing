# Base image that matches your local Python exactly
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

# Upgrade pip and install dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Expose Flask port
EXPOSE 5000

# Start server with Gunicorn (production-ready)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120"]

