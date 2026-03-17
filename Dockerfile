FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# First install numpy and pandas from pre-built wheels
RUN pip install --no-cache-dir --only-binary :all: numpy==1.23.5 pandas==1.5.3

# Then install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ml_trading_bot.py"]
