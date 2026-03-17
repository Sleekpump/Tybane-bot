FROM python:3.9-slim

WORKDIR /app

# Install everything with --only-binary flag
RUN pip install --no-cache-dir --only-binary :all: \
    numpy==1.23.5 \
    pandas==1.5.3 \
    python-telegram-bot==20.7 \
    ccxt==4.1.22 \
    scikit-learn==1.2.2 \
    joblib==1.2.0 \
    python-dotenv==1.0.0 \
    feedparser==6.0.10 \
    groq==0.4.2

COPY . .

CMD ["python", "ml_trading_bot.py"]
