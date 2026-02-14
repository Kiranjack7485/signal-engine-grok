# Use official Python slim image (lightweight, compatible with Railway)
FROM python:3.12-slim-bookworm

# Install system dependencies needed for TA-Lib compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download, compile, and install TA-Lib C library from source
RUN wget https://downloads.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create virtual env and install Python deps
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install -r requirements.txt

# Copy your code
COPY . .

# Use the venv for runtime
ENV PATH="/app/.venv/bin:$PATH"

# Command to run your script
CMD ["python", "main.py"]