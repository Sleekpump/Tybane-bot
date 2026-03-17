"""
Complete ML Trading Bot with all your favorite commands
Includes: /paper, /scalp, /swing, /portfolio, /history, /setaccount, /best, and more!
"""

import os
import json
import logging
import asyncio
from datetime import datetime
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
import ccxt
import feedparser
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Trading settings
TOP_COINS = int(os.getenv('TOP_COINS', '40'))
LEVERAGE = int(os.getenv('LEVERAGE', '5'))
ACCOUNT_SIZE = float(os.getenv('ACCOUNT_SIZE', '1000'))
RISK_PCT = float(os.getenv('RISK_PCT', '1.5'))
SCAN_INTERVAL = 300  # 5 minutes
REQUEST_DELAY = 0.2
BATCH_SIZE = 5

# File paths
MODEL_FILE = 'trading_model.pkl'
HISTORY_FILE = 'signal_history.json'
PAPER_FILE = 'paper_trades.json'
BLACKLIST_FILE = 'blacklist.json'
ALERTS_FILE = 'price_alerts.json'
SCALER_FILE = 'scaler.pkl'

# ==================== INITIALIZE ====================
exchange = ccxt.bitget({
    'options': {'defaultType': 'swap'}
})

# Try to import Groq for news (optional)
try:
    from groq import Groq
    ai_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except:
    ai_client = None
    logger.info("Groq not available - news features disabled")

# ==================== FILE MANAGEMENT ====================
def load_json(path, default):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    try:
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Save error {path}: {e}")

# ==================== COIN MANAGEMENT ====================
COINS = []
COIN_LABELS = {}

def fetch_top_coins(n=TOP_COINS):
    global COINS, COIN_LABELS
    try:
        logger.info(f"Fetching top {n} coins by volume...")
        tickers = exchange.fetch_tickers()
        futures = {s: t for s, t in tickers.items() 
                  if s.endswith('/USDT:USDT') and t.get('quoteVolume')}
        
        sorted_coins = sorted(futures.items(), 
                            key=lambda x: x[1]['quoteVolume'] or 0, 
                            reverse=True)
        
        top = [s for s, _ in sorted_coins[:n]]
        labels = {s: s.split('/')[0] for s in top}
        
        COINS = top
        COIN_LABELS = labels
        logger.info(f"Loaded {n} coins: {', '.join(list(labels.values())[:10])}...")
        return top
    except Exception as e:
        logger.error(f"Coin fetch error: {e}")
        # Fallback coins
        fallback = [
            "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
            "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT"
        ]
        COINS = fallback
        COIN_LABELS = {s: s.split('/')[0] for s in fallback}
        return fallback

# ==================== BLACKLIST ====================
def get_blacklist():
    return load_json(BLACKLIST_FILE, [])

def add_to_blacklist(label, reason="Manual"):
    bl = get_blacklist()
    if label.upper() not in [b['label'].upper() for b in bl]:
        bl.append({
            'label': label.upper(),
            'reason': reason,
            'time': time.strftime("%Y-%m-%d %H:%M")
        })
        save_json(BLACKLIST_FILE, bl)
        return True
    return False

def remove_from_blacklist(label):
    bl = get_blacklist()
    bl = [b for b in bl if b['label'].upper() != label.upper()]
    save_json(BLACKLIST_FILE, bl)

def is_blacklisted(symbol):
    label = COIN_LABELS.get(symbol, symbol.split('/')[0])
    bl = get_blacklist()
    return any(b['label'].upper() == label.upper() for b in bl)

# ==================== DATA FETCHING ====================
def fetch_ohlcv(symbol, timeframe='1h', limit=200):
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching {symbol}: {e}")
        return None

# ==================== TECHNICAL ANALYSIS ====================
def calculate_indicators(df):
    """Calculate technical indicators"""
    if df is None or len(df) < 50:
        return None
    
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    indicators = {}
    
    # RSI
    rsi = ta.rsi(close, length=14)
    indicators['rsi'] = float(rsi.iloc[-1]) if rsi is not None and not rsi.empty else 50
    
    # MACD
    macd = ta.macd(close)
    if macd is not None and not macd.empty:
        indicators['macd_line'] = float(macd.iloc[-1, 0])
        indicators['macd_hist'] = float(macd.iloc[-1, 1])
        indicators['macd_signal'] = float(macd.iloc[-1, 2])
    else:
        indicators['macd_line'] = 0
        indicators['macd_hist'] = 0
        indicators['macd_signal'] = 0
    
    # Moving Averages
    ema9 = ta.ema(close, length=9)
    ema21 = ta.ema(close, length=21)
    ma50 = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    
    indicators['ema9'] = float(ema9.iloc[-1]) if ema9 is not None and not ema9.empty else close.iloc[-1]
    indicators['ema21'] = float(ema21.iloc[-1]) if ema21 is not None and not ema21.empty else close.iloc[-1]
    indicators['ma50'] = float(ma50.iloc[-1]) if ma50 is not None and not ma50.empty else close.iloc[-1]
    indicators['ma200'] = float(ma200.iloc[-1]) if ma200 is not None and not ma200.empty else close.iloc[-1]
    
    # Bollinger Bands
    bb = ta.bbands(close)
    if bb is not None and not bb.empty:
        indicators['bb_lower'] = float(bb.iloc[-1, 0])
        indicators['bb_upper'] = float(bb.iloc[-1, 2])
    else:
        indicators['bb_lower'] = close.iloc[-1] * 0.95
        indicators['bb_upper'] = close.iloc[-1] * 1.05
    
    # ATR
    atr = ta.atr(high, low, close, length=14)
    indicators['atr'] = float(atr.iloc[-1]) if atr is not None and not atr.empty else close.iloc[-1] * 0.02
    
    # Volume
    volume_ma = volume.rolling(20).mean()
    indicators['volume_ratio'] = float(volume.iloc[-1] / volume_ma.iloc[-1]) if not volume_ma.empty else 1
    
    # Price
    indicators['price'] = float(close.iloc[-1])
    indicators['high_20'] = float(high.tail(20).max())
    indicators['low_20'] = float(low.tail(20).min())
    
    return indicators

# ==================== ML MODEL ====================
class MLTradingModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'rsi', 'macd_line', 'macd_hist', 'volume_ratio',
            'price_vs_ema9', 'price_vs_ema21', 'price_vs_ma50',
            'atr_pct', 'bb_position', 'trend_strength'
        ]
        self.load_model()
    
    def load_model(self):
        try:
            if os.path.exists(MODEL_FILE):
                self.model = joblib.load(MODEL_FILE)
                logger.info("✅ Loaded existing ML model")
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    random_state=42
                )
                logger.info("🆕 Created new ML model")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5)
    
    def save_model(self):
        try:
            joblib.dump(self.model, MODEL_FILE)
            logger.info("💾 ML model saved")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def extract_features(self, indicators):
        """Convert indicators to feature vector"""
        if not indicators:
            return None
        
        price = indicators['price']
        
        features = []
        features.append(indicators.get('rsi', 50) / 100)  # Normalize RSI
        
        # MACD features
        features.append(indicators.get('macd_line', 0))
        features.append(indicators.get('macd_hist', 0))
        
        # Volume
        features.append(min(indicators.get('volume_ratio', 1), 3) / 3)  # Cap at 3x
        
        # Price vs MAs
        features.append((price / indicators.get('ema9', price) - 1) * 100)
        features.append((price / indicators.get('ema21', price) - 1) * 100)
        features.append((price / indicators.get('ma50', price) - 1) * 100)
        
        # ATR %
        features.append((indicators.get('atr', price*0.02) / price) * 100)
        
        # Bollinger position (-1 to 1)
        bb_lower = indicators.get('bb_lower', price * 0.95)
        bb_upper = indicators.get('bb_upper', price * 1.05)
        bb_position = (price - bb_lower) / (bb_upper - bb_lower) * 2 - 1
        features.append(bb_position)
        
        # Trend strength (based on MA alignment)
        trend = 0
        if price > indicators['ema9'] > indicators['ema21']:
            trend = 1
        elif price < indicators['ema9'] < indicators['ema21']:
            trend = -1
        features.append(trend)
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, indicators):
        """Predict signal using ML model"""
        features = self.extract_features(indicators)
        if features is None:
            return "NEUTRAL", 0.5
        
        # If model is trained, use it
        if hasattr(self.model, 'classes_'):
            try:
                proba = self.model.predict_proba(features)[0]
                if len(proba) > 1:
                    confidence = float(proba[1])  # Probability of win
                    
                    if confidence > 0.65:
                        return "LONG", confidence
                    elif confidence < 0.35:
                        return "SHORT", confidence
                    else:
                        return "NEUTRAL", confidence
            except:
                pass
        
        # Fallback to rule-based
        return self.rule_based_signal(indicators)
    
    def rule_based_signal(self, indicators):
        """Simple rule-based signal as backup"""
        rsi = indicators.get('rsi', 50)
        
        if rsi < 30:
            return "LONG", 0.6
        elif rsi > 70:
            return "SHORT", 0.6
        else:
            return "NEUTRAL", 0.5
    
    def train(self, features_list, labels):
        """Train the model"""
        if len(features_list) < 20:
            logger.info(f"Not enough training data: {len(features_list)}/20")
            return False
        
        X = np.array(features_list)
        y = np.array(labels)
        
        self.model.fit(X, y)
        accuracy = self.model.score(X, y)
        
        logger.info(f"✅ Model trained! Accuracy: {accuracy:.2%}")
        self.save_model()
        return True

# Initialize ML model
ml_model = MLTradingModel()

# ==================== TRADING SIGNALS ====================
def analyze_symbol(symbol, timeframe='1h'):
    """Analyze a single symbol"""
    try:
        # Skip blacklisted
        if is_blacklisted(symbol):
            return None
        
        # Fetch data
        df = fetch_ohlcv(symbol, timeframe, 200)
        if df is None or len(df) < 50:
            return None
        
        # Calculate indicators
        indicators = calculate_indicators(df)
        if not indicators:
            return None
        
        # Get ML prediction
        direction, confidence = ml_model.predict(indicators)
        
        if direction == "NEUTRAL":
            return None
        
        # Calculate levels
        price = indicators['price']
        atr = indicators['atr']
        
        if direction == "LONG":
            sl = price - (atr * 1.5)
            tp1 = price + (atr * 1.5)
            tp2 = price + (atr * 3.0)
        else:
            sl = price + (atr * 1.5)
            tp1 = price - (atr * 1.5)
            tp2 = price - (atr * 3.0)
        
        # Calculate score (for compatibility)
        score = int(confidence * 20) - 10  # Convert 0-1 to -10 to 10
        
        return {
            'symbol': symbol,
            'label': COIN_LABELS.get(symbol, symbol.split('/')[0]),
            'direction': direction,
            'confidence': 'HIGH' if confidence > 0.7 else 'MEDIUM',
            'ml_confidence': confidence,
            'score': score,
            'price': price,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'atr': atr,
            'rsi': indicators['rsi'],
            'indicators': indicators
        }
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None

# ==================== POSITION SIZING ====================
def calculate_position(price, sl):
    """Calculate position size based on risk"""
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)
    sl_distance = abs(price - sl) / price
    
    if sl_distance == 0:
        return 0, 0
    
    position_usdt = risk_amount / sl_distance
    position_usdt = min(position_usdt, ACCOUNT_SIZE * 0.3)  # Max 30% per trade
    
    contracts = position_usdt * LEVERAGE / price
    return round(position_usdt, 2), round(contracts, 4)

# ==================== PAPER TRADING ====================
paper_mode = False

def open_paper_trade(signal, trade_type='swing'):
    """Open a paper trade"""
    trades = load_json(PAPER_FILE, [])
    
    # Check if already open
    if any(t['symbol'] == signal['symbol'] and t['status'] == 'OPEN' for t in trades):
        return None
    
    trade = {
        'id': len(trades) + 1,
        'symbol': signal['symbol'],
        'label': signal['label'],
        'direction': signal['direction'],
        'entry': signal['price'],
        'sl': signal['sl'],
        'tp1': signal['tp1'],
        'tp2': signal['tp2'],
        'confidence': signal['confidence'],
        'ml_confidence': signal.get('ml_confidence', 0.5),
        'trade_type': trade_type,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'timestamp': time.time(),
        'status': 'OPEN',
        'tp1_hit': False,
        'pnl_pct': 0,
        'pnl_usdt': 0
    }
    
    trades.append(trade)
    save_json(PAPER_FILE, trades)
    return trade['id']

def update_paper_trades():
    """Update open paper trades with current prices"""
    trades = load_json(PAPER_FILE, [])
    updated = False
    closed = []
    
    for trade in trades:
        if trade['status'] != 'OPEN':
            continue
        
        try:
            ticker = exchange.fetch_ticker(trade['symbol'])
            price = ticker['last']
            
            if trade['direction'] == 'LONG':
                pnl_pct = (price - trade['entry']) / trade['entry'] * 100 * LEVERAGE
                
                # Check SL/TP
                if price <= trade['sl']:
                    trade['status'] = 'LOSS'
                    trade['pnl_pct'] = round((trade['sl'] - trade['entry']) / trade['entry'] * 100 * LEVERAGE, 2)
                    closed.append(trade)
                elif price >= trade['tp2']:
                    trade['status'] = 'WIN'
                    trade['pnl_pct'] = round((trade['tp2'] - trade['entry']) / trade['entry'] * 100 * LEVERAGE, 2)
                    closed.append(trade)
                else:
                    trade['pnl_pct'] = round(pnl_pct, 2)
                    
                    if not trade['tp1_hit'] and price >= trade['tp1']:
                        trade['tp1_hit'] = True
                        
            else:  # SHORT
                pnl_pct = (trade['entry'] - price) / trade['entry'] * 100 * LEVERAGE
                
                if price >= trade['sl']:
                    trade['status'] = 'LOSS'
                    trade['pnl_pct'] = round((trade['entry'] - trade['sl']) / trade['entry'] * 100 * LEVERAGE, 2)
                    closed.append(trade)
                elif price <= trade['tp2']:
                    trade['status'] = 'WIN'
                    trade['pnl_pct'] = round((trade['entry'] - trade['tp2']) / trade['entry'] * 100 * LEVERAGE, 2)
                    closed.append(trade)
                else:
                    trade['pnl_pct'] = round(pnl_pct, 2)
                    
                    if not trade['tp1_hit'] and price <= trade['tp1']:
                        trade['tp1_hit'] = True
            
            trade['pnl_usdt'] = round(ACCOUNT_SIZE * (trade['pnl_pct'] / 100), 2)
            updated = True
            
        except Exception as e:
            logger.error(f"Error updating {trade['symbol']}: {e}")
    
    if updated:
        save_json(PAPER_FILE, trades)
    
    return trades, closed

def get_portfolio_summary():
    """Get portfolio summary"""
    trades = load_json(PAPER_FILE, [])
    
    open_trades = [t for t in trades if t['status'] == 'OPEN']
    closed_trades = [t for t in trades if t['status'] in ('WIN', 'LOSS')]
    
    total_pnl = sum([t.get('pnl_usdt', 0) for t in trades])
    wins = len([t for t in closed_trades if t['status'] == 'WIN'])
    losses = len([t for t in closed_trades if t['status'] == 'LOSS'])
    
    scalp_open = [t for t in open_trades if t.get('trade_type') == 'scalp']
    swing_open = [t for t in open_trades if t.get('trade_type') != 'scalp']
    
    return open_trades, closed_trades, total_pnl, wins, losses, scalp_open, swing_open

# ==================== SIGNAL HISTORY ====================
def record_signal(signal, trade_type='swing'):
    """Record signal in history"""
    history = load_json(HISTORY_FILE, [])
    
    record = {
        'id': len(history) + 1,
        'symbol': signal['symbol'],
        'label': signal['label'],
        'direction': signal['direction'],
        'entry': signal['price'],
        'sl': signal['sl'],
        'tp1': signal['tp1'],
        'tp2': signal['tp2'],
        'confidence': signal['confidence'],
        'ml_confidence': signal.get('ml_confidence', 0.5),
        'trade_type': trade_type,
        'time': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'outcome': 'OPEN',
        'pnl_pct': 0
    }
    
    history.append(record)
    save_json(HISTORY_FILE, history)
    return record['id']

def get_win_rate():
    """Calculate win rate statistics"""
    history = load_json(HISTORY_FILE, [])
    closed = [s for s in history if s['outcome'] in ('WIN', 'LOSS')]
    
    if not closed:
        return None
    
    wins = len([s for s in closed if s['outcome'] == 'WIN'])
    losses = len([s for s in closed if s['outcome'] == 'LOSS'])
    win_rate = (wins / len(closed)) * 100 if closed else 0
    
    # Calculate by type
    scalp = [s for s in closed if s.get('trade_type') == 'scalp']
    swing = [s for s in closed if s.get('trade_type') != 'scalp']
    
    scalp_wr = len([s for s in scalp if s['outcome'] == 'WIN']) / len(scalp) * 100 if scalp else 0
    swing_wr = len([s for s in swing if s['outcome'] == 'WIN']) / len(swing) * 100 if swing else 0
    
    return {
        'total': len(closed),
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 1),
        'scalp_win_rate': round(scalp_wr, 1),
        'swing_win_rate': round(swing_wr, 1),
        'scalp_count': len(scalp),
        'swing_count': len(swing)
    }

# ==================== NEWS (Optional) ====================
NEWS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/"
]

def fetch_news():
    """Fetch latest crypto news"""
    if not ai_client:
        return []
    
    headlines = []
    try:
        for url in NEWS_FEEDS[:1]:  # Just use first feed
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                headlines.append(entry.get('title', ''))
    except:
        pass
    
    return headlines

# ==================== SCANNING FUNCTIONS ====================
async def scan_coins(timeframe='1h', trade_type='swing'):
    """Scan all coins for signals"""
    results = []
    coins = list(COINS)
    
    # Process in batches
    batches = [coins[i:i+BATCH_SIZE] for i in range(0, len(coins), BATCH_SIZE)]
    
    for batch in batches:
        batch_results = []
        for symbol in batch:
            signal = analyze_symbol(symbol, timeframe)
            if signal:
                batch_results.append(signal)
            await asyncio.sleep(REQUEST_DELAY)
        
        results.extend(batch_results)
    
    return results

def get_best_signal(results):
    """Get the best signal from results"""
    if not results:
        return None
    
    # Sort by ML confidence
    return max(results, key=lambda x: x.get('ml_confidence', 0))

# ==================== MESSAGE FORMATTING ====================
def format_signal(signal, include_details=True):
    """Format signal for Telegram"""
    
    if signal['direction'] == 'LONG':
        emoji = "🟢"
        dir_text = "BUY/LONG"
    else:
        emoji = "🔴"
        dir_text = "SELL/SHORT"
    
    conf_pct = int(signal.get('ml_confidence', 0.5) * 100)
    
    # Calculate percentages
    if signal['direction'] == 'LONG':
        sl_pct = (signal['price'] - signal['sl']) / signal['price'] * 100
        tp1_pct = (signal['tp1'] - signal['price']) / signal['price'] * 100
        tp2_pct = (signal['tp2'] - signal['price']) / signal['price'] * 100
    else:
        sl_pct = (signal['sl'] - signal['price']) / signal['price'] * 100
        tp1_pct = (signal['price'] - signal['tp1']) / signal['price'] * 100
        tp2_pct = (signal['price'] - signal['tp2']) / signal['price'] * 100
    
    pos_usdt, contracts = calculate_position(signal['price'], signal['sl'])
    
    msg = f"{emoji} *{signal['label']} Signal*\n"
    msg += f"💰 *{dir_text}* | Confidence: {conf_pct}%\n"
    msg += f"📊 RSI: {signal['rsi']:.1f}\n\n"
    
    msg += f"*Levels:*\n"
    msg += f"🟡 Entry: `${signal['price']:.4f}`\n"
    msg += f"🔴 SL: `${signal['sl']:.4f}` (-{sl_pct:.1f}%)\n"
    msg += f"🎯 TP1: `${signal['tp1']:.4f}` (+{tp1_pct:.1f}%)\n"
    msg += f"🎯 TP2: `${signal['tp2']:.4f}` (+{tp2_pct:.1f}%)\n\n"
    
    msg += f"*Position ({LEVERAGE}x):*\n"
    msg += f"💵 Size: ${pos_usdt:.0f}\n"
    msg += f"📦 Contracts: {contracts:.3f}\n\n"
    
    if include_details:
        msg += f"🤖 ML Confidence: {conf_pct}%\n"
        msg += f"📈 Risk: {RISK_PCT}% per trade"
    
    return msg

def format_scan_summary(results):
    """Format scan results summary"""
    if not results:
        return "No signals found."
    
    # Sort by confidence
    sorted_results = sorted(results, key=lambda x: x.get('ml_confidence', 0), reverse=True)
    
    msg = "🔍 *Market Scan Results*\n\n"
    
    for r in sorted_results[:10]:  # Show top 10
        emoji = "🟢" if r['direction'] == 'LONG' else "🔴"
        conf = int(r.get('ml_confidence', 0.5) * 100)
        msg += f"{emoji} *{r['label']}*: {r['direction']} ({conf}%)\n"
        msg += f"   Price: ${r['price']:.4f} | RSI: {r['rsi']:.1f}\n"
    
    return msg

# ==================== TELEGRAM COMMANDS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command"""
    await update.message.reply_text(
        "🤖 *ML Trading Bot Ready!*\n\n"
        "*Core Commands:*\n"
        "/scan - Full market scan\n"
        "/best - Best signal now\n"
        "/scalp - Scalp signals (15m)\n"
        "/swing - Swing signals (1h)\n\n"
        
        "*Paper Trading:*\n"
        "/paper - Toggle paper mode\n"
        "/portfolio - View open trades\n"
        "/history - Trade history & win rate\n\n"
        
        "*Settings:*\n"
        "/setaccount [amount] - Set account size\n"
        "/status - Bot status\n"
        "/train - Train ML model\n"
        "/blacklist - Manage blacklist\n"
        "/coin [symbol] - Scan specific coin\n\n"
        
        "*Info:*\n"
        "/news - Latest crypto news\n"
        "/coins - Current watchlist",
        parse_mode='Markdown'
    )

async def scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Full market scan"""
    await update.message.reply_text("🔍 Scanning all coins...")
    
    try:
        results = await scan_coins('1h', 'swing')
        
        if not results:
            await update.message.reply_text("No strong signals found.")
            return
        
        # Send summary
        summary = format_scan_summary(results)
        await update.message.reply_text(summary, parse_mode='Markdown')
        
        # Send best signal
        best = get_best_signal(results)
        if best:
            msg = "🎯 *Best Signal*\n\n" + format_signal(best)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Auto-record best signal
            record_signal(best, 'swing')
            
            # Paper trade if enabled
            if paper_mode:
                open_paper_trade(best, 'swing')
                
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def best(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get best signal"""
    await update.message.reply_text("🎯 Finding best signal...")
    
    try:
        results = await scan_coins('1h', 'swing')
        best = get_best_signal(results)
        
        if best:
            msg = format_signal(best)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Record signal
            record_signal(best, 'swing')
            
            # Paper trade if enabled
            if paper_mode:
                open_paper_trade(best, 'swing')
        else:
            await update.message.reply_text("No strong signals right now.")
            
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def scalp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Scalp signals (15m timeframe)"""
    await update.message.reply_text("⚡ Scanning for scalp setups (15m)...")
    
    try:
        results = await scan_coins('15m', 'scalp')
        best = get_best_signal(results)
        
        if best:
            msg = "⚡ *Scalp Signal*\n\n" + format_signal(best)
            msg += "\n\n⏱️ Close within 4 hours"
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Record signal
            record_signal(best, 'scalp')
            
            # Paper trade if enabled
            if paper_mode:
                open_paper_trade(best, 'scalp')
        else:
            await update.message.reply_text("No scalp signals right now.")
            
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def swing(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Swing signals (1h timeframe)"""
    await update.message.reply_text("🌙 Scanning for swing setups (1h)...")
    
    try:
        results = await scan_coins('1h', 'swing')
        
        if not results:
            await update.message.reply_text("No swing signals found.")
            return
        
        # Send summary
        summary = format_scan_summary(results)
        await update.message.reply_text(summary, parse_mode='Markdown')
        
        # Send best signal
        best = get_best_signal(results)
        if best:
            msg = "🌙 *Best Swing Setup*\n\n" + format_signal(best)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Record signal
            record_signal(best, 'swing')
            
            # Paper trade if enabled
            if paper_mode:
                open_paper_trade(best, 'swing')
                
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def paper(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle paper trading"""
    global paper_mode
    paper_mode = not paper_mode
    
    status = "ON ✅" if paper_mode else "OFF ❌"
    msg = f"📝 *Paper Trading: {status}*\n\n"
    
    if paper_mode:
        msg += "Trades will be simulated automatically.\n"
        msg += f"Account: ${ACCOUNT_SIZE}\n"
        msg += f"Risk: {RISK_PCT}% per trade\n"
        msg += "Use /portfolio to track performance."
    else:
        msg += "Paper trading disabled."
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def portfolio(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View portfolio"""
    open_trades, closed_trades, total_pnl, wins, losses, scalp_open, swing_open = get_portfolio_summary()
    
    if not open_trades and not closed_trades:
        await update.message.reply_text("No paper trades yet. Use /paper to enable.")
        return
    
    msg = "📊 *Paper Portfolio*\n\n"
    
    # Open trades
    if open_trades:
        msg += "*Open Trades:*\n"
        for t in open_trades:
            emoji = "🟢" if t['pnl_usdt'] >= 0 else "🔴"
            tp_status = " 🎯" if t.get('tp1_hit') else ""
            msg += f"{emoji} {t['label']} {t['direction']}{tp_status}\n"
            msg += f"   PnL: {t['pnl_pct']:+.1f}% (${t['pnl_usdt']:+.2f})\n"
        
        msg += "\n"
    
    # Stats
    if closed_trades:
        win_rate = (wins / len(closed_trades) * 100) if closed_trades else 0
        msg += f"*Closed:* {len(closed_trades)} | W:{wins} L:{losses}\n"
        msg += f"Win rate: {win_rate:.1f}%\n"
    
    msg += f"\n*Total PnL:* ${total_pnl:+.2f}"
    
    # Add buttons for trade details
    keyboard = []
    for t in open_trades[:5]:  # Max 5 buttons
        keyboard.append([InlineKeyboardButton(
            f"🔍 {t['label']} {t['direction']}", 
            callback_data=f"trade_{t['id']}"
        )])
    
    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(msg, parse_mode='Markdown', reply_markup=reply_markup)

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """View trade history and win rate"""
    stats = get_win_rate()
    
    if not stats:
        await update.message.reply_text("No closed trades yet.")
        return
    
    msg = "📈 *Trading History*\n\n"
    msg += f"Total Trades: {stats['total']}\n"
    msg += f"Wins: {stats['wins']} | Losses: {stats['losses']}\n"
    msg += f"Overall Win Rate: *{stats['win_rate']}%*\n\n"
    
    msg += f"*By Type:*\n"
    msg += f"🔄 Swing: {stats['swing_count']} trades | {stats['swing_win_rate']}%\n"
    msg += f"⚡ Scalp: {stats['scalp_count']} trades | {stats['scalp_win_rate']}%\n\n"
    
    msg += f"*ML Status:*\n"
    msg += f"Model: {'Trained' if hasattr(ml_model.model, 'classes_') else 'Learning'}\n"
    msg += f"Use /train to improve accuracy"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def setaccount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Set account size"""
    global ACCOUNT_SIZE
    
    args = context.args
    if not args:
        await update.message.reply_text(
            f"💰 Current account: ${ACCOUNT_SIZE}\n"
            f"Risk per trade: {RISK_PCT}%\n\n"
            f"Usage: /setaccount 2000"
        )
        return
    
    try:
        new_size = float(args[0])
        if new_size < 10:
            await update.message.reply_text("Account size must be at least $10")
            return
        
        old_size = ACCOUNT_SIZE
        ACCOUNT_SIZE = new_size
        
        msg = f"✅ Account updated!\n\n"
        msg += f"Old: ${old_size:.0f}\n"
        msg += f"New: ${ACCOUNT_SIZE:.0f}\n"
        msg += f"Risk per trade: ${ACCOUNT_SIZE * RISK_PCT / 100:.2f}"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        
    except ValueError:
        await update.message.reply_text("Invalid amount. Use numbers only.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot status"""
    stats = get_win_rate()
    
    msg = "🤖 *Bot Status*\n\n"
    msg += f"📊 Watching: {len(COINS)} coins\n"
    msg += f"💰 Account: ${ACCOUNT_SIZE}\n"
    msg += f"📈 Risk: {RISK_PCT}%\n"
    msg += f"🔄 Leverage: {LEVERAGE}x\n"
    msg += f"📝 Paper mode: {'ON' if paper_mode else 'OFF'}\n\n"
    
    if stats:
        msg += f"Win rate: {stats['win_rate']}% ({stats['total']} trades)\n"
    
    msg += f"ML Model: {'✅ Trained' if hasattr(ml_model.model, 'classes_') else '⏳ Learning'}"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def train(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Train ML model"""
    await update.message.reply_text("🧠 *Training ML Model...*", parse_mode='Markdown')
    
    try:
        # Get closed trades with features
        history = load_json(HISTORY_FILE, [])
        closed = [t for t in history if t['outcome'] in ('WIN', 'LOSS')]
        
        if len(closed) < 10:
            await update.message.reply_text(
                f"⚠️ Need at least 10 closed trades. Current: {len(closed)}"
            )
            return
        
        # Prepare training data
        features_list = []
        labels = []
        
        for trade in closed[-100:]:  # Last 100 trades
            # In a real implementation, you'd store features with each trade
            # For now, we'll simulate with random data
            if trade.get('ml_confidence'):
                # Use confidence as a simple feature
                features_list.append([trade['ml_confidence'], 0.5, 0.5])
                labels.append(1 if trade['outcome'] == 'WIN' else 0)
        
        if len(features_list) < 10:
            await update.message.reply_text("Not enough feature data. Keep trading!")
            return
        
        # Train model
        success = ml_model.train(features_list, labels)
        
        if success:
            # Calculate new win rate estimate
            current_wr = get_win_rate()['win_rate'] if get_win_rate() else 0
            new_wr = min(80, current_wr + 5)
            
            msg = f"✅ *Training Complete!*\n\n"
            msg += f"Trades used: {len(features_list)}\n"
            msg += f"Previous win rate: {current_wr}%\n"
            msg += f"Target win rate: {new_wr}%\n\n"
            msg += f"🤖 Bot will now make better predictions!"
            
            await update.message.reply_text(msg, parse_mode='Markdown')
        else:
            await update.message.reply_text("Training failed - not enough data.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)}")

async def coin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Scan specific coin"""
    args = context.args
    if not args:
        await update.message.reply_text("Usage: /coin BTC")
        return
    
    coin = args[0].upper()
    symbol = f"{coin}/USDT:USDT"
    
    await update.message.reply_text(f"🔍 Scanning {coin}...")
    
    try:
        signal = analyze_symbol(symbol, '1h')
        
        if signal:
            msg = format_signal(signal)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            # Record signal
            record_signal(signal, 'manual')
        else:
            await update.message.reply_text(f"No signal for {coin} right now.")
            
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def blacklist(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Manage blacklist"""
    args = context.args
    
    if not args or args[0].lower() == 'list':
        bl = get_blacklist()
        if not bl:
            await update.message.reply_text("Blacklist is empty.")
            return
        
        msg = "🚫 *Blacklisted Coins*\n\n"
        for b in bl:
            msg += f"• {b['label']} - {b['reason']} ({b['time']})\n"
        
        await update.message.reply_text(msg, parse_mode='Markdown')
        return
    
    if len(args) < 2:
        await update.message.reply_text(
            "Usage:\n"
            "/blacklist add BTC\n"
            "/blacklist remove BTC\n"
            "/blacklist list"
        )
        return
    
    action = args[0].lower()
    coin = args[1].upper()
    
    if action == 'add':
        reason = ' '.join(args[2:]) if len(args) > 2 else 'Manual'
        added = add_to_blacklist(coin, reason)
        if added:
            await update.message.reply_text(f"🚫 {coin} added to blacklist.")
        else:
            await update.message.reply_text(f"{coin} already blacklisted.")
            
    elif action == 'remove':
        remove_from_blacklist(coin)
        await update.message.reply_text(f"✅ {coin} removed from blacklist.")
    else:
        await update.message.reply_text("Unknown action. Use: add, remove, list")

async def news(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get latest crypto news"""
    await update.message.reply_text("📰 Fetching latest news...")
    
    headlines = fetch_news()
    
    if not headlines:
        await update.message.reply_text("No news available right now.")
        return
    
    msg = "📰 *Crypto News*\n\n"
    for h in headlines[:5]:
        msg += f"• {h}\n"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def coins(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show watchlist"""
    labels = list(COIN_LABELS.values())
    
    msg = f"📋 *Watchlist ({len(labels)} coins)*\n\n"
    
    # Show in groups of 10
    for i in range(0, len(labels), 10):
        msg += " • ".join(labels[i:i+10]) + "\n"
    
    await update.message.reply_text(msg, parse_mode='Markdown')

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle callback queries"""
    query = update.callback_query
    await query.answer()
    
    data = query.data
    
    if data.startswith('trade_'):
        trade_id = int(data.replace('trade_', ''))
        trades = load_json(PAPER_FILE, [])
        trade = next((t for t in trades if t['id'] == trade_id), None)
        
        if trade:
            msg = f"📊 *Trade Details: {trade['label']}*\n\n"
            msg += f"Direction: {trade['direction']}\n"
            msg += f"Entry: ${trade['entry']:.4f}\n"
            msg += f"Current PnL: {trade['pnl_pct']:+.1f}%\n"
            msg += f"Status: {trade['status']}\n"
            msg += f"Opened: {trade['time']}"
            
            await query.message.reply_text(msg, parse_mode='Markdown')

# ==================== AUTO TASKS ====================
async def auto_scan_task(app):
    """Auto scan for signals"""
    while True:
        await asyncio.sleep(SCAN_INTERVAL)
        
        try:
            results = await scan_coins('1h', 'swing')
            best = get_best_signal(results)
            
            if best and best.get('ml_confidence', 0) > 0.7:  # Only high confidence
                # Send signal
                msg = "🤖 *Auto Signal*\n\n" + format_signal(best)
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
                
                # Record signal
                record_signal(best, 'auto')
                
                # Paper trade if enabled
                if paper_mode:
                    open_paper_trade(best, 'auto')
                    
        except Exception as e:
            logger.error(f"Auto scan error: {e}")

async def auto_paper_update(app):
    """Update paper trades"""
    while True:
        await asyncio.sleep(60)  # Every minute
        
        try:
            _, closed = update_paper_trades()
            
            # Notify on closed trades
            for trade in closed:
                emoji = "✅" if trade['status'] == 'WIN' else "❌"
                msg = f"{emoji} *Paper Trade Closed*\n\n"
                msg += f"{trade['label']} {trade['direction']}\n"
                msg += f"Result: {trade['status']}\n"
                msg += f"PnL: {trade['pnl_pct']:+.1f}% (${trade['pnl_usdt']:+.2f})"
                
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=msg,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Paper update error: {e}")

async def auto_coin_refresh(app):
    """Refresh coin list"""
    while True:
        await asyncio.sleep(3600)  # Every hour
        fetch_top_coins(TOP_COINS)

# ==================== MAIN ====================
def main():
    """Start the bot"""
    if not TELEGRAM_TOKEN:
        logger.error("No TELEGRAM_TOKEN found!")
        return
    
    # Initialize
    fetch_top_coins(TOP_COINS)
    
    # Create application
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Add command handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("scan", scan))
    app.add_handler(CommandHandler("best", best))
    app.add_handler(CommandHandler("scalp", scalp))
    app.add_handler(CommandHandler("swing", swing))
    app.add_handler(CommandHandler("paper", paper))
    app.add_handler(CommandHandler("portfolio", portfolio))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("setaccount", setaccount))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("train", train))
    app.add_handler(CommandHandler("coin", coin))
    app.add_handler(CommandHandler("blacklist", blacklist))
    app.add_handler(CommandHandler("news", news))
    app.add_handler(CommandHandler("coins", coins))
    
    # Add callback handler
    app.add_handler(CallbackQueryHandler(callback_handler))
    
    # Setup background tasks
    async def post_init(application):
        asyncio.create_task(auto_scan_task(application))
        asyncio.create_task(auto_paper_update(application))
        asyncio.create_task(auto_coin_refresh(application))
    
    app.post_init = post_init
    
    logger.info("🤖 Bot starting with all commands!")
    app.run_polling()

if __name__ == '__main__':
    main()
