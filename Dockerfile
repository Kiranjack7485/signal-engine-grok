# Downgrade to Python 3.11 for TA-Lib compatibility
FROM python:3.11-slim-bookworm

# Install build tools and wget
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download, compile, install TA-Lib to /usr/local
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Set env for TA-Lib wrapper to find C lib/headers
ENV TA_INCLUDE_PATH=/usr/local/include
ENV TA_LIBRARY_PATH=/usr/local/lib

# Install Python deps
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Activate venv
ENV PATH="/app/.venv/bin:$PATH"

# Run script
CMD ["python", "main.py"]