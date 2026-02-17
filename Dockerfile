FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib C lib (still needed even for wheels in some envs)
RUN wget https://github.com/TA-Lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    && tar -xzf ta-lib-0.6.4-src.tar.gz \
    && cd ta-lib-0.6.4 \
    && ./configure --prefix=/usr/local \
    && make \
    && make install \
    && ldconfig \
    && cd .. \
    && rm -rf ta-lib-0.6.4*

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "main.py"]