FROM python:3.13-slim

WORKDIR /app

# Install minimal build dependencies (just in case)
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Force pip to use pre-built wheels only
RUN pip install --no-cache-dir --upgrade pip

# Install numpy and pandas first with --only-binary flag
RUN pip install --no-cache-dir --only-binary :all: numpy==1.26.4 pandas==2.2.3

# Install the rest of requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY . .

CMD ["python", "ml_trading_bot.py"]
