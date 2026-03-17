FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

# Install numpy and pandas first
RUN pip install --no-cache-dir numpy==1.23.5 pandas==1.5.3

# Install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "ml_trading_bot.py"]
