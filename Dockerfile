FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    wget \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C core 0.4.0 (IMPORTANT VERSION)
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/0.4.0/ta-lib-0.4.0-src.tar.gz \
    && tar -xzf ta-lib-0.4.0-src.tar.gz \
    && cd ta-lib \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

WORKDIR /app

RUN pip install --upgrade pip
RUN pip install numpy==1.26.4
RUN pip install --no-cache-dir --no-build-isolation TA-Lib==0.4.28
RUN pip install --no-cache-dir python-binance==1.0.19 requests==2.32.3 pytz==2024.1

COPY . .

CMD ["python", "main.py"]
