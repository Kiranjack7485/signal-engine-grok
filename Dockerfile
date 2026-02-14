# Use a slim Python base image (compatible with Railway, lighter than full)
FROM python:3.12-slim-bookworm

# Install build tools and wget for downloading TA-Lib source
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Download and compile TA-Lib C library from official GitHub release (safer than SourceForge)
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr \
    && make \
    && make install \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set working directory for your app
WORKDIR /app

# Copy requirements first (caching optimization)
COPY requirements.txt .

# Create virtual env and install Python packages
RUN python -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . .

# Activate venv in PATH
ENV PATH="/app/.venv/bin:$PATH"

# Run your script (no web server needed)
CMD ["python", "main.py"]