FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install numpy and pandas with correct versions for Python 3.9
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --only-binary :all: numpy==1.23.5 pandas==1.5.3

# Install the rest
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary :all: -r requirements.txt

COPY . .

CMD ["python", "ml_trading_bot.py"]
