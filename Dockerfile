FROM python:3.9-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir pandas==1.5.3 numpy==1.23.5
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "ml_trading_bot.py"]
