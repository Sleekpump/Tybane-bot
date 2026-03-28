"""
Phyrobot — Multi-Coin Signal Bot
Timeframes: 4H + 1D | 40 coins dynamic | Score-based signals
"""

import os, json, logging, asyncio, feedparser
import time as _time
from datetime import datetime, timedelta
import ccxt
import pandas as pd
import pandas_ta as ta
from groq import Groq
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from dotenv import load_dotenv
from signal_engine import analyze_v2
from ai_validator  import run_full_pipeline, format_ai_block
from risk_manager  import risk_gate, format_risk_block, cmd_risk, get_portfolio_heat
from backtester    import cmd_backtest

load_dotenv()
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TOP_COINS      = int(os.getenv("TOP_COINS", "40"))
LEVERAGE       = int(os.getenv("LEVERAGE", "10"))
SCAN_INTERVAL  = int(os.getenv("SCAN_INTERVAL", "300"))
NEWS_INTERVAL  = int(os.getenv("NEWS_INTERVAL", "600"))
COIN_REFRESH   = int(os.getenv("COIN_REFRESH", "3600"))
REQUEST_DELAY  = float(os.getenv("REQUEST_DELAY", "0.2"))
BATCH_SIZE     = int(os.getenv("BATCH_SIZE", "5"))
ACCOUNT_SIZE   = float(os.getenv("ACCOUNT_SIZE", "200"))
RISK_PCT       = float(os.getenv("RISK_PCT", "1.5"))
MAX_VOLATILITY = float(os.getenv("MAX_VOLATILITY", "8.0"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT  = os.getenv("TELEGRAM_CHAT_ID")
GROQ_KEY       = os.getenv("GROQ_API_KEY")

# ─── STATE ─────────────────────────────────────────────────────────────────────
COINS               = []
COIN_LABELS         = {}
last_signal         = {}
active_signals      = {}
reentry_cooldown    = {}  # {symbol: timestamp} — prevents re-entry spam

# ─── FILES ─────────────────────────────────────────────────────────────────────
HISTORY_FILE   = "signal_history.json"
PAPER_FILE     = "paper_trades.json"
BLACKLIST_FILE = "blacklist.json"
ALERTS_FILE    = "price_alerts.json"

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        log.error("Save error " + path + ": " + str(e))

# ─── EXCHANGE ──────────────────────────────────────────────────────────────────
exchange  = ccxt.bitget({"options": {"defaultType": "swap"}})
ai_client = Groq(api_key=GROQ_KEY) if GROQ_KEY else None

# ─── DYNAMIC COIN SELECTION ────────────────────────────────────────────────────
def fetch_top_coins(n=TOP_COINS):
    global COINS, COIN_LABELS
    try:
        log.info("Fetching top " + str(n) + " coins by volume...")
        tickers = exchange.fetch_tickers()
        futures = {s: t for s, t in tickers.items()
                   if s.endswith("/USDT:USDT") and t.get("quoteVolume")}
        sorted_coins = sorted(futures.items(), key=lambda x: x[1]["quoteVolume"] or 0, reverse=True)
        top    = [s for s, _ in sorted_coins[:n]]
        labels = {s: s.split("/")[0] for s in top}
        COINS       = top
        COIN_LABELS = labels
        log.info("Top " + str(n) + " coins loaded: " + ", ".join(labels.values()))
        return top
    except Exception as e:
        log.error("Coin fetch error: " + str(e))
        fallback = [
            "BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","BNB/USDT:USDT",
            "XRP/USDT:USDT","DOGE/USDT:USDT","ADA/USDT:USDT","AVAX/USDT:USDT",
            "LINK/USDT:USDT","BGB/USDT:USDT"
        ]
        COINS       = fallback
        COIN_LABELS = {s: s.split("/")[0] for s in fallback}
        return fallback

# ─── DATA FETCHING ─────────────────────────────────────────────────────────────
def fetch_ohlcv(symbol, timeframe="4h", limit=200):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df  = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

# ─── TECHNICAL ANALYSIS ────────────────────────────────────────────────────────
def score_timeframe(df):
    score   = 0
    signals = []
    close   = df["close"]
    high    = df["high"]
    low     = df["low"]
    volume  = df["volume"]
    price   = float(close.iloc[-1])

    # RSI
    rsi     = ta.rsi(close, length=14)
    rsi_val = float(rsi.iloc[-1])
    if rsi_val < 35:
        score += 2; signals.append("RSI oversold (" + str(round(rsi_val,1)) + ")")
    elif rsi_val > 65:
        score -= 2; signals.append("RSI overbought (" + str(round(rsi_val,1)) + ")")

    # RSI Divergence (last 20 candles)
    prices_w = close.tail(20).values
    rsis_w   = rsi.tail(20).values
    if len(prices_w) >= 20 and len(rsis_w) >= 20:
        if prices_w[-1] < min(prices_w[:-1]) and rsis_w[-1] > min(rsis_w[:-1]):
            score += 3; signals.append("Bullish RSI divergence")
        elif prices_w[-1] > max(prices_w[:-1]) and rsis_w[-1] < max(rsis_w[:-1]):
            score -= 3; signals.append("Bearish RSI divergence")

    # MACD
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty:
        macd_line = float(macd_df.iloc[-1, 0])
        sig_line  = float(macd_df.iloc[-1, 2])
        hist_now  = float(macd_df.iloc[-1, 1])
        hist_prev = float(macd_df.iloc[-2, 1])
        macd_prev = float(macd_df.iloc[-2, 0])
        sig_prev  = float(macd_df.iloc[-2, 2])
        if macd_line > sig_line and macd_prev <= sig_prev:
            score += 2; signals.append("MACD bullish crossover")
        elif macd_line < sig_line and macd_prev >= sig_prev:
            score -= 2; signals.append("MACD bearish crossover")
        elif hist_now > 0 and hist_now > hist_prev:
            score += 1; signals.append("MACD histogram rising")
        elif hist_now < 0 and hist_now < hist_prev:
            score -= 1; signals.append("MACD histogram falling")

    # EMA 9/21
    ema9  = ta.ema(close, length=9)
    ema21 = ta.ema(close, length=21)
    if ema9 is not None and ema21 is not None:
        e9_now  = float(ema9.iloc[-1]);  e9_prev  = float(ema9.iloc[-2])
        e21_now = float(ema21.iloc[-1]); e21_prev = float(ema21.iloc[-2])
        if e9_now > e21_now and e9_prev <= e21_prev:
            score += 2; signals.append("EMA 9/21 bullish cross")
        elif e9_now < e21_now and e9_prev >= e21_prev:
            score -= 2; signals.append("EMA 9/21 bearish cross")
        elif e9_now > e21_now:
            score += 1
        else:
            score -= 1

    # MA50 / MA200
    ma50  = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    if ma50 is not None and ma200 is not None:
        ma50_val  = float(ma50.dropna().iloc[-1]) if len(ma50.dropna()) > 0 else None
        ma200_val = float(ma200.dropna().iloc[-1]) if len(ma200.dropna()) > 0 else None
        if ma50_val and ma200_val:
            if ma50_val > ma200_val:
                score += 1; signals.append("Golden cross (MA50 > MA200)")
            else:
                score -= 1; signals.append("Death cross (MA50 < MA200)")
            if price > ma50_val:
                score += 1
            else:
                score -= 1

    # Bollinger Bands
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None and not bb.empty:
        bb_lower = float(bb.iloc[-1, 0])
        bb_upper = float(bb.iloc[-1, 2])
        if price < bb_lower:
            score += 2; signals.append("Below Bollinger lower band")
        elif price > bb_upper:
            score -= 2; signals.append("Above Bollinger upper band")

    # Volume spike
    vol_ma = volume.rolling(20).mean()
    if float(volume.iloc[-1]) > float(vol_ma.iloc[-1]) * 1.5:
        if price > float(close.iloc[-2]):
            score += 1; signals.append("High volume bullish candle")
        else:
            score -= 1; signals.append("High volume bearish candle")

    # Support / Resistance
    support    = float(low.tail(20).min())
    resistance = float(high.tail(20).max())
    if abs(price - support) / price < 0.015:
        score += 1; signals.append("Near support $" + str(round(support, 4)))
    if abs(price - resistance) / price < 0.015:
        score -= 1; signals.append("Near resistance $" + str(round(resistance, 4)))

    # Stochastic RSI
    stoch = ta.stochrsi(close, length=14)
    if stoch is not None and not stoch.empty:
        stoch_k = float(stoch.iloc[-1, 0])
        stoch_d = float(stoch.iloc[-1, 1])
        if stoch_k < 20 and stoch_d < 20:
            score += 1; signals.append("StochRSI oversold")
        elif stoch_k > 80 and stoch_d > 80:
            score -= 1; signals.append("StochRSI overbought")

    # CCI
    cci = ta.cci(high, low, close, length=20)
    if cci is not None:
        cci_val = float(cci.iloc[-1])
        if cci_val < -100:
            score += 1; signals.append("CCI oversold")
        elif cci_val > 100:
            score -= 1; signals.append("CCI overbought")

    # Williams %R
    willr = ta.willr(high, low, close, length=14)
    if willr is not None:
        willr_val = float(willr.iloc[-1])
        if willr_val < -80:
            score += 1; signals.append("Williams %R oversold")
        elif willr_val > -20:
            score -= 1; signals.append("Williams %R overbought")

    # ATR
    atr     = ta.atr(high, low, close, length=14)
    atr_val = float(atr.iloc[-1]) if atr is not None else 0

    # ── Volume Exhaustion Detection ────────────────────────────────────────────
    # Pump on declining volume = exhaustion = likely reversal
    try:
        vol_series  = volume.tail(10).values
        price_series = close.tail(10).values
        recent_pump  = price_series[-1] > price_series[-5] * 1.05  # 5%+ rise in last 5 candles
        recent_dump  = price_series[-1] < price_series[-5] * 0.95  # 5%+ fall in last 5 candles
        vol_declining = vol_series[-1] < vol_series[-3:].mean() * 0.7  # volume dropping off
        if recent_pump and vol_declining:
            score -= 2; signals.append("Volume exhaustion on pump — potential reversal")
        elif recent_dump and vol_declining:
            score += 2; signals.append("Volume exhaustion on dump — potential reversal")
    except Exception:
        pass

    # ── Candle Pattern Detection ───────────────────────────────────────────────
    try:
        o = float(df["open"].iloc[-1])
        h = float(df["high"].iloc[-1])
        l = float(df["low"].iloc[-1])
        c = float(df["close"].iloc[-1])
        body     = abs(c - o)
        candle_range = h - l

        if candle_range > 0:
            # Shooting star — small body at bottom, long upper wick = bearish reversal
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l
            if upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
                score -= 2; signals.append("Shooting star candle — bearish reversal")

            # Hammer — small body at top, long lower wick = bullish reversal
            if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
                score += 2; signals.append("Hammer candle — bullish reversal")

            # Bearish engulfing — current red candle body engulfs previous green candle
            prev_o = float(df["open"].iloc[-2])
            prev_c = float(df["close"].iloc[-2])
            if prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                score -= 2; signals.append("Bearish engulfing candle")

            # Bullish engulfing — current green candle body engulfs previous red candle
            if prev_c < prev_o and c > o and o < prev_c and c > prev_o:
                score += 2; signals.append("Bullish engulfing candle")
    except Exception:
        pass

    return score, signals, support, resistance, rsi_val, atr_val, price

# ─── MOMENTUM & REVERSAL CLASSIFICATION ───────────────────────────────────────
def classify_signal(df, direction, score, btc_change_24h=None):
    """
    Classify signal as MOMENTUM or REVERSAL.
    MOMENTUM: breakout, higher highs/lows, outperforming BTC, or strong price action
    REVERSAL: volume exhaustion + extreme RSI
    WEAK: oversold stacking without confirmation
    """
    try:
        close  = df["close"].values
        volume = df["volume"].values
        high   = df["high"].values
        low    = df["low"].values

        if len(close) < 20:
            return "REVERSAL", "LOW", "Insufficient data"

        reasons     = []
        momentum_pts = 0

        # ── 1. Breakout / Breakdown Detection ──────────────────────────────────
        resistance_20 = max(high[-21:-1])  # high of last 20 candles excluding current
        support_20    = min(low[-21:-1])   # low of last 20 candles excluding current
        breakout_vol  = volume[-1] > volume[-5:].mean() * 1.2  # volume confirmation

        if direction == "LONG":
            breakout = close[-1] > resistance_20 and breakout_vol
            if breakout:
                momentum_pts += 2
                reasons.append("breakout above $" + "{:.4f}".format(resistance_20))
        else:
            breakdown = close[-1] < support_20 and breakout_vol
            if breakdown:
                momentum_pts += 2
                reasons.append("breakdown below $" + "{:.4f}".format(support_20))

        # ── 2. Higher Highs / Higher Lows (LONG) or Lower Highs / Lower Lows (SHORT)
        # Find last 3 swing highs and lows
        if len(high) >= 15:
            swing_highs = []
            swing_lows  = []
            for i in range(2, min(15, len(high)-2)):
                if high[-i] > high[-i-1] and high[-i] > high[-i+1]:
                    swing_highs.append(high[-i])
                if low[-i] < low[-i-1] and low[-i] < low[-i+1]:
                    swing_lows.append(low[-i])

            if direction == "LONG" and len(swing_highs) >= 2 and len(swing_lows) >= 2:
                hh = swing_highs[0] > swing_highs[1]  # most recent high > previous high
                hl = swing_lows[0] > swing_lows[1]    # most recent low > previous low
                if hh and hl:
                    momentum_pts += 2
                    reasons.append("higher highs + higher lows")
            elif direction == "SHORT" and len(swing_highs) >= 2 and len(swing_lows) >= 2:
                lh = swing_highs[0] < swing_highs[1]  # most recent high < previous high
                ll = swing_lows[0] < swing_lows[1]    # most recent low < previous low
                if lh and ll:
                    momentum_pts += 2
                    reasons.append("lower highs + lower lows")

        # ── 3. Relative Strength vs BTC ────────────────────────────────────────
        if btc_change_24h is not None:
            coin_change_24h = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
            relative_strength = coin_change_24h - btc_change_24h
            if direction == "LONG" and relative_strength > 3:
                momentum_pts += 1
                reasons.append("outperforming BTC by +" + str(round(relative_strength, 1)) + "%")
            elif direction == "SHORT" and relative_strength < -3:
                momentum_pts += 1
                reasons.append("underperforming BTC by " + str(round(relative_strength, 1)) + "%")

        # ── 4. Rate of Change + Volume ─────────────────────────────────────────
        roc = (close[-1] - close[-5]) / close[-5] * 100
        vol_increasing = volume[-1] > volume[-5:].mean() * 1.1
        if direction == "LONG":
            roc_ok = roc > 1.0
        else:
            roc_ok = roc < -1.0
        if roc_ok and vol_increasing:
            momentum_pts += 1
            reasons.append("accelerating with volume")

        # ── Reversal checks ────────────────────────────────────────────────────
        vol_ma = volume[-20:].mean() if len(volume) >= 20 else volume.mean()
        if direction == "LONG":
            price_falling  = close[-1] < close[-5]
            vol_exhaustion = volume[-1] < vol_ma * 0.7
            reversal_setup = price_falling and vol_exhaustion
        else:
            price_rising   = close[-1] > close[-5]
            vol_exhaustion = volume[-1] < vol_ma * 0.7
            reversal_setup = price_rising and vol_exhaustion

        rsi_series = []
        for i in range(14, min(len(close), 50)):
            gains  = [max(close[j]-close[j-1], 0) for j in range(i-13, i+1)]
            losses = [max(close[j-1]-close[j], 0) for j in range(i-13, i+1)]
            avg_gain = sum(gains) / 14
            avg_loss = sum(losses) / 14
            if avg_loss == 0:
                rsi_series.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi_series.append(100 - (100 / (1 + rs)))
        current_rsi = rsi_series[-1] if rsi_series else 50
        extreme_rsi = current_rsi < 25 if direction == "LONG" else current_rsi > 75

        reversal_score = sum([reversal_setup, extreme_rsi])

        # ── Classification ─────────────────────────────────────────────────────
        if momentum_pts >= 2:
            type_conf = "HIGH" if momentum_pts >= 4 else "MEDIUM"
            return "MOMENTUM", type_conf, " + ".join(reasons)
        elif reversal_score >= 2 or (reversal_score >= 1 and score >= 9):
            rev_reasons = []
            if reversal_setup:
                rev_reasons.append("volume exhaustion")
            if extreme_rsi:
                rev_reasons.append("extreme RSI (" + str(round(current_rsi, 1)) + ")")
            type_conf = "HIGH" if reversal_score == 2 else "MEDIUM"
            return "REVERSAL", type_conf, " + ".join(rev_reasons)
        else:
            return "WEAK", "LOW", "no breakout, structure, or reversal confirmation"

    except Exception as e:
        return "REVERSAL", "LOW", "classification error"


def analyze(symbol):
    df_4h = fetch_ohlcv(symbol, "1h", 200)
    df_1d = fetch_ohlcv(symbol, "4h", 100)

    score_4h, signals_4h, support, resistance, rsi_4h, atr_4h, price = score_timeframe(df_4h)
    score_1d, signals_1d, _, _, rsi_1d, atr_1d, _ = score_timeframe(df_1d)

    # 4H has higher weight — acts as trend filter
    combined_score = score_4h + (score_1d * 2)
    tf_agree = (score_4h > 0 and score_1d > 0) or (score_4h < 0 and score_1d < 0)

    # Bonus if both agree, penalty if they disagree
    if tf_agree:
        combined_score = int(combined_score * 1.2)
    else:
        combined_score = int(combined_score * 0.7)

    abs_score = abs(combined_score)

    if abs_score >= 5:
        direction  = "LONG" if combined_score > 0 else "SHORT"
        confidence = "HIGH" if abs_score >= 9 else "MEDIUM"
    else:
        direction  = "NEUTRAL"
        confidence = "LOW"

    # Merge signals, 1D first as it's the trend
    all_signals = ["[4H] " + s for s in signals_1d] + ["[1H] " + s for s in signals_4h]

    # Funding rate
    try:
        funding = exchange.fetch_funding_rate(symbol)
        funding_rate = float(funding["fundingRate"]) if funding else 0
    except Exception:
        funding_rate = 0

    label = COIN_LABELS.get(symbol, symbol.split("/")[0])

    # Fetch BTC 24hr change for relative strength
    btc_change_24h = None
    try:
        if symbol != "BTC/USDT:USDT":
            btc_df = fetch_ohlcv("BTC/USDT:USDT", "1h", 25)
            btc_change_24h = (float(btc_df["close"].iloc[-1]) - float(btc_df["close"].iloc[-24])) / float(btc_df["close"].iloc[-24]) * 100
    except Exception:
        btc_change_24h = None

    # Classify signal type
    if direction != "NEUTRAL":
        signal_type, type_conf, type_reason = classify_signal(df_4h, direction, abs_score, btc_change_24h)
    else:
        signal_type, type_conf, type_reason = "NEUTRAL", "LOW", ""

    return {
        "symbol": symbol, "label": label,
        "direction": direction, "confidence": confidence,
        "score": combined_score, "abs_score": abs_score,
        "score_4h": score_4h, "score_1d": score_1d,
        "tf_agree": tf_agree,
        "signals": all_signals, "price": price,
        "support": support, "resistance": resistance,
        "rsi_4h": rsi_4h, "rsi_1d": rsi_1d,
        "atr": atr_4h, "funding": funding_rate,
        "df_4h": df_4h,
        "tf_labels": ("1H", "4H"),
        "signal_type": signal_type,
        "type_conf": type_conf,
        "type_reason": type_reason,
    }

# ─── PARALLEL SCANNING ─────────────────────────────────────────────────────────
async def analyze_async(symbol):
    try:
        result = await run_full_pipeline(
            symbol, fetch_ohlcv, COIN_LABELS,
            ai_client, exchange, news_context
        )
        log.info(result["label"] + ": " + result["direction"] + " | Q:" + str(result.get("abs_score", 0)))
        return result
    except Exception as e:
        log.error("Error " + symbol + ": " + str(e))
        return None

async def scan_all_async():
    results  = []
    coins    = list(COINS)
    batches  = [coins[i:i+BATCH_SIZE] for i in range(0, len(coins), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        batch_results = await asyncio.gather(*[analyze_async(s) for s in batch])
        results.extend([r for r in batch_results if r is not None])
        if i < len(batches) - 1:
            await asyncio.sleep(REQUEST_DELAY * BATCH_SIZE)
    return results

def get_actionable(results):
    # Filter out WEAK signals — only MOMENTUM and REVERSAL pass
    return [r for r in results if r["direction"] != "NEUTRAL"
            and r["confidence"] in ("HIGH","MEDIUM")
            and r.get("signal_type", "REVERSAL") != "WEAK"]

# ─── ORDER BOOK ANALYSIS ───────────────────────────────────────────────────────
def get_order_book_bias(symbol, depth=20):
    try:
        ob      = exchange.fetch_order_book(symbol, limit=depth)
        bid_vol = sum([b[1] for b in ob["bids"][:depth]])
        ask_vol = sum([a[1] for a in ob["asks"][:depth]])
        total   = bid_vol + ask_vol
        if total == 0:
            return "NEUTRAL", 0.5
        bid_ratio = bid_vol / total
        if bid_ratio > 0.6:
            return "BULLISH", round(bid_ratio, 2)
        elif bid_ratio < 0.4:
            return "BEARISH", round(bid_ratio, 2)
        return "NEUTRAL", round(bid_ratio, 2)
    except Exception:
        return "NEUTRAL", 0.5

# ─── NEWS SCANNER (CONTEXT ONLY) ───────────────────────────────────────────────
NEWS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://coindesk.com/arc/outboundfeeds/rss/",
    "https://decrypt.co/feed",
    "https://feeds.reuters.com/reuters/businessNews",
]
seen_headlines = set()
news_context   = {"sentiment": "NEUTRAL", "headlines": [], "last_update": 0}

def fetch_headlines(max_per_feed=5):
    headlines = []
    for url in NEWS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = entry.get("title", "").strip()
                if title and title not in seen_headlines:
                    headlines.append(title)
                    seen_headlines.add(title)
        except Exception as e:
            log.warning("Feed error " + url + ": " + str(e))
    return headlines

def analyze_news_context(headlines):
    if not headlines or not ai_client:
        return "NEUTRAL", []
    try:
        prompt = (
            "You are a crypto market analyst. Analyze these headlines and return JSON only:\n"
            + "\n".join(["- " + h for h in headlines[:20]])
            + '\n\nReturn: {"sentiment": "BULLISH|BEARISH|NEUTRAL", "key_points": ["point1", "point2"]}'
            + "\nJSON only, no other text."
        )
        response = ai_client.chat.completions.create(
            model="llama3-70b-8192",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.choices[0].message.content.strip()
        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        return data.get("sentiment", "NEUTRAL"), data.get("key_points", [])
    except Exception as e:
        log.error("News analysis error: " + str(e))
        return "NEUTRAL", []

# ─── POSITION SIZING ───────────────────────────────────────────────────────────
def calc_position_size(entry, sl):
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)
    sl_distance = abs(entry - sl) / entry
    if sl_distance == 0:
        return 0, 0
    position_usdt = round(risk_amount / sl_distance, 2)
    contracts     = round(position_usdt * LEVERAGE / entry, 4)
    return position_usdt, contracts

TRAILING_BUFFER = 0.12  # 12% trailing stop buffer after TP1 hit

def calc_levels(direction, price, atr):
    if direction == "LONG":
        sl  = round(price - (atr * 1.5), 6)
        tp1 = round(price + (atr * 1.5), 6)
        tp2 = round(price + (atr * 4.0), 6)
    else:
        sl  = round(price + (atr * 1.5), 6)
        tp1 = round(price - (atr * 1.5), 6)
        tp2 = round(price - (atr * 4.0), 6)
    return sl, tp1, tp2

# ─── SIGNAL HISTORY ────────────────────────────────────────────────────────────
def record_signal(symbol, direction, entry, sl, tp1, tp2, confidence, trade_type="swing", signal_type="REVERSAL"):
    history = load_json(HISTORY_FILE, [])
    sig_id  = len(history) + 1
    history.append({
        "id": sig_id, "symbol": symbol,
        "label": COIN_LABELS.get(symbol, symbol.split("/")[0]),
        "direction": direction, "entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "confidence": confidence,
        "trade_type": trade_type,
        "signal_type": signal_type,
        "time": _time.strftime("%Y-%m-%d %H:%M"),
        "outcome": "OPEN", "pnl_pct": 0,
    })
    save_json(HISTORY_FILE, history)
    return sig_id

def get_win_rate():
    history = load_json(HISTORY_FILE, [])
    closed  = [s for s in history if s["outcome"] in ("WIN","LOSS","BREAKEVEN","EXPIRED")]
    if not closed:
        return None

    def calc_stats(trades):
        if not trades:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "avg_pnl": 0}
        wins   = len([s for s in trades if s["outcome"] == "WIN"])
        losses = len([s for s in trades if s["outcome"] in ("LOSS","EXPIRED")])
        total  = len(trades)
        avg_pnl = sum([s["pnl_pct"] for s in trades]) / total if total else 0
        return {
            "total": total, "wins": wins, "losses": losses,
            "win_rate": round(wins / total * 100, 1) if total else 0,
            "avg_pnl": round(avg_pnl, 2),
        }

    scalp_closed    = [s for s in closed if s.get("trade_type") == "scalp"]
    swing_closed    = [s for s in closed if s.get("trade_type") != "scalp"]
    momentum_closed = [s for s in closed if s.get("signal_type") == "MOMENTUM"]
    reversal_closed = [s for s in closed if s.get("signal_type") == "REVERSAL"]

    return {
        "scalp":    calc_stats(scalp_closed),
        "swing":    calc_stats(swing_closed),
        "momentum": calc_stats(momentum_closed),
        "reversal": calc_stats(reversal_closed),
        "overall":  calc_stats(closed),
        "open": len([s for s in history if s["outcome"] == "OPEN"]),
    }

# ─── PAPER TRADING ─────────────────────────────────────────────────────────────
paper_mode = False

def open_paper_trade(symbol, direction, entry, sl, tp1, tp2, confidence, trade_type="swing", signal_type="REVERSAL"):
    trades = load_json(PAPER_FILE, [])
    trade  = {
        "id": len(trades) + 1,
        "symbol": symbol,
        "label": COIN_LABELS.get(symbol, symbol.split("/")[0]),
        "direction": direction, "entry": entry,
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "confidence": confidence,
        "time": _time.strftime("%Y-%m-%d %H:%M"),
        "open_timestamp": _time.time(),
        "trade_type": trade_type,
        "signal_type": signal_type,
        "status": "OPEN", "tp1_hit": False,
        "pnl_pct": 0, "pnl_usdt": 0,
        "original_signal": {
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "confidence": confidence, "direction": direction,
            "trade_type": trade_type, "signal_type": signal_type,
            "time": _time.strftime("%Y-%m-%d %H:%M"),
        }
    }
    trades.append(trade)
    save_json(PAPER_FILE, trades)
    return trade["id"]

def update_paper_trades():
    trades  = load_json(PAPER_FILE, [])
    closed_now = []
    for t in trades:
        if t["status"] != "OPEN":
            continue
        try:
            price     = exchange.fetch_ticker(t["symbol"])["last"]
            direction = t["direction"]
            entry     = t["entry"]

            # Auto-close scalp trades after 24 hours
            if t.get("trade_type") == "scalp":
                open_time = t.get("open_timestamp", _time.time())
                if _time.time() - open_time > 86400:
                    pnl_pct  = (price - entry) / entry * 100 * LEVERAGE if direction == "LONG" else (entry - price) / entry * 100 * LEVERAGE
                    t["status"]   = "EXPIRED"
                    t["pnl_pct"]  = round(pnl_pct, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * pnl_pct / 100, 2)
                    closed_now.append(t)
                    continue

            if direction == "LONG":
                pnl_pct = (price - entry) / entry * 100 * LEVERAGE
                if price <= t["sl"]:
                    # LONG LOSS — SL < entry, so result is negative
                    t["status"]   = "LOSS"
                    t["pnl_pct"]  = round((t["sl"] - entry) / entry * 100 * LEVERAGE, 2)  # negative
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price >= t["tp2"]:
                    # LONG WIN — TP2 > entry, so result is positive
                    t["status"]   = "WIN"
                    t["pnl_pct"]  = round((t["tp2"] - entry) / entry * 100 * LEVERAGE, 2)  # positive
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                else:
                    t["pnl_pct"]  = round(pnl_pct, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * pnl_pct / 100, 2)
                    if not t["tp1_hit"] and price >= t["tp1"]:
                        t["tp1_hit"] = True
            else:
                pnl_pct = (entry - price) / entry * 100 * LEVERAGE
                if price >= t["sl"]:
                    # SHORT LOSS — SL > entry, so (entry - sl) is negative
                    t["status"]   = "LOSS"
                    t["pnl_pct"]  = round((entry - t["sl"]) / entry * 100 * LEVERAGE, 2)  # negative
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price <= t["tp2"]:
                    # SHORT WIN — TP2 < entry, so (entry - tp2) is positive
                    t["status"]   = "WIN"
                    t["pnl_pct"]  = round((entry - t["tp2"]) / entry * 100 * LEVERAGE, 2)  # positive
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                else:
                    t["pnl_pct"]  = round(pnl_pct, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * pnl_pct / 100, 2)
                    if not t["tp1_hit"] and price <= t["tp1"]:
                        t["tp1_hit"] = True
        except Exception as e:
            log.error("Paper update error: " + str(e))
    save_json(PAPER_FILE, trades)
    return trades, closed_now

def get_paper_summary():
    trades      = load_json(PAPER_FILE, [])
    open_t      = [t for t in trades if t["status"] == "OPEN"]
    closed_t    = [t for t in trades if t["status"] in ("WIN","LOSS","EXPIRED")]
    scalp_open  = [t for t in open_t if t.get("trade_type") == "scalp"]
    swing_open  = [t for t in open_t if t.get("trade_type") != "scalp"]
    total_pnl   = sum([t["pnl_usdt"] for t in trades])
    wins        = len([t for t in closed_t if t["status"] == "WIN"])
    losses      = len([t for t in closed_t if t["status"] == "LOSS"])
    return open_t, closed_t, total_pnl, wins, losses, scalp_open, swing_open

# ─── SIGNAL FORMATTING ─────────────────────────────────────────────────────────
def format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts):
    emoji      = "\U0001f7e2" if r["direction"] == "LONG" else "\U0001f534"
    conf_emoji = "\U0001f525" if r["confidence"] == "HIGH" else "\u26a1"
    tf_emoji   = "\u2705" if r["tf_agree"] else "\u26a0"
    dir_text   = "Buy/Long" if r["direction"] == "LONG" else "Sell/Short" if r["direction"] == "SHORT" else "No Trade"
    sl_pct     = abs(r["price"] - sl) / r["price"] * 100
    tp1_pct    = abs(tp1 - r["price"]) / r["price"] * 100
    tp2_pct    = abs(tp2 - r["price"]) / r["price"] * 100

    # Order book emoji
    ob_emoji = "\U0001f7e2" if ob_bias == "BULLISH" else "\U0001f534" if ob_bias == "BEARISH" else "\U0001f7e1"
    ob_conflict = ob_bias == "BEARISH" and r["direction"] == "LONG"
    ob_conflict = ob_conflict or (ob_bias == "BULLISH" and r["direction"] == "SHORT")

    # Signal type emoji
    signal_type = r.get("signal_type", "REVERSAL")
    type_reason = r.get("type_reason", "")
    if signal_type == "MOMENTUM":
        type_emoji = "\u26a1"
        type_label = "MOMENTUM"
    elif signal_type == "REVERSAL":
        type_emoji = "\U0001f504"
        type_label = "REVERSAL"
    else:
        type_emoji = "\u26a0"
        type_label = "WEAK"

    regime_data = r.get("regime") or {}
    regime_name = regime_data.get("regime", "UNKNOWN") if isinstance(regime_data, dict) else str(regime_data)
    adx_val = regime_data.get("adx", 0) if isinstance(regime_data, dict) else 0
    
    msg  = emoji + " *" + r["label"] + " Signal | " + datetime.now().strftime("%H:%M UTC") + "*\n"
    msg += conf_emoji + " *" + r["direction"] + "* | " + r["confidence"] + " | Score: `" + str(r["score"]) + "`\n"
    msg += "📊 Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n"
    msg += type_emoji + " Type: *" + type_label + "* | " + type_reason + "\n"
    tf_label = r.get("tf_labels", ("1H", "4H"))
    msg += tf_emoji + " " + tf_label[0] + ": `" + str(r["score_4h"]) + "` | " + tf_label[1] + ": `" + str(r["score_1d"]) + "` | Funding: `" + "{:.3f}".format(r["funding"]*100) + "%`\n\n"

    msg += ob_emoji + " *Order Book: " + ob_bias + "* (bid ratio: " + str(ob_ratio) + ")"
    if ob_conflict:
        msg += " \u26a0 conflicts with signal"
    msg += "\n\n"

    # News context
    if news_context["sentiment"] != "NEUTRAL" or news_context["headlines"]:
        news_emoji = "\U0001f7e2" if news_context["sentiment"] == "BULLISH" else "\U0001f534" if news_context["sentiment"] == "BEARISH" else "\U0001f7e1"
        msg += "\U0001f4f0 *News Context: " + news_context["sentiment"] + "* " + news_emoji + "\n"
        for pt in news_context.get("key_points", [])[:2]:
            msg += "  \u2022 " + pt + "\n"
        msg += "\n"

    msg += "*How to trade (Bitget " + str(LEVERAGE) + "x Futures):*\n"
    msg += "  1\ufe0f\u20e3 Futures \u2192 " + r["label"] + "USDT \u2192 " + str(LEVERAGE) + "x\n"
    msg += "  2\ufe0f\u20e3 " + dir_text + " at market\n"
    msg += "  3\ufe0f\u20e3 Set levels below\n\n"

    msg += "*Levels:*\n"
    msg += "  \U0001f7e1 Entry: `$" + "{:.4f}".format(r["price"]) + "`\n"
    msg += "  \U0001f534 SL:  `$" + "{:.4f}".format(sl) + "` (-" + "{:.1f}".format(sl_pct) + "% / -" + "{:.0f}".format(sl_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n"
    msg += "  \U0001f3af TP1: `$" + "{:.4f}".format(tp1) + "` (+" + "{:.1f}".format(tp1_pct) + "% / +" + "{:.0f}".format(tp1_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n"
    msg += "  \U0001f3af TP2: `$" + "{:.4f}".format(tp2) + "` (+" + "{:.1f}".format(tp2_pct) + "% / +" + "{:.0f}".format(tp2_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n\n"

    msg += "*Position Sizing (" + str(RISK_PCT) + "% risk / $" + str(ACCOUNT_SIZE) + " account):*\n"
    msg += "  Size: $" + str(pos_usdt) + " | Contracts: " + str(contracts) + "\n\n"

    msg += "*Key Signals:*\n"
    for s in r["signals"][:6]:
        msg += "  \u2022 " + s + "\n"

    msg += "\n\u26a0 _Not financial advice. Trade at your own risk._"
    return msg

def format_scan_summary(results):
    sorted_r = sorted(results, key=lambda x: x["abs_score"], reverse=True)
    half     = len(sorted_r) // 2
    chunks   = [sorted_r[:half], sorted_r[half:]]
    messages = []
    for i, chunk in enumerate(chunks):
        msg = "\U0001f50d *Market Scan | " + datetime.now().strftime("%H:%M UTC") + "* (" + str(i+1) + "/2)\n\n"
        for r in chunk:
            if r["direction"] == "LONG":
                e = "\U0001f7e2"
            elif r["direction"] == "SHORT":
                e = "\U0001f534"
            else:
                e = "\u26aa"
            conf = " | " + r["confidence"] if r["direction"] != "NEUTRAL" else ""
            stype = r.get("signal_type", "")
            if stype == "MOMENTUM":
                type_tag = " \u26a1"
            elif stype == "REVERSAL":
                type_tag = " \U0001f504"
            elif stype == "WEAK":
                type_tag = " \u26a0"
            else:
                type_tag = ""
            msg += e + " *" + r["label"] + "* \u2014 " + r["direction"] + conf + type_tag + " | `" + ("{:+.0f}".format(r["score"])) + "` | `$" + ("{:.4f}".format(r["price"])) + "`\n"
        messages.append(msg)
    return messages

# ─── TELEGRAM COMMANDS ─────────────────────────────────────────────────────────
async def send_msg(app, text):
    await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=text, parse_mode="Markdown")

async def cmd_coin(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Scan any coin on demand. Usage: /coin LINK"""
    args = ctx.args
    if not args:
        await update.message.reply_text("Usage: /coin LINK\nScans any coin on Bitget futures.")
        return
    coin   = args[0].upper()
    symbol = coin + "/USDT:USDT"
    await update.message.reply_text("Scanning " + coin + " (1H+4H)...")
    try:
        # Check blacklist
        if is_blacklisted(symbol):
            await update.message.reply_text("\U0001f6ab " + coin + " is blacklisted. Remove with: /blacklist remove " + coin)
            return
        # Temporarily add to labels if not in watchlist
        original_label = COIN_LABELS.get(symbol)
        if not original_label:
            COIN_LABELS[symbol] = coin

        r = analyze(symbol)

        # Clean up if we added it temporarily
        if not original_label:
            del COIN_LABELS[symbol]

        sl, tp1, tp2 = calc_levels(r["direction"], r["price"], r["atr"])
        ob_bias, ob_ratio = get_order_book_bias(symbol)
        pos_usdt, contracts = calc_position_size(r["price"], sl)

        # Full signal format
        msg = format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)

        # Add extra detail header
        header  = "\U0001f50e *On-Demand Scan: " + coin + "*\n"
        header += "1H Score: `" + str(r["score_4h"]) + "` | 4H Score: `" + str(r["score_1d"]) + "`\n"
        header += "RSI 1H: `" + str(round(r["rsi_4h"],1)) + "` | RSI 4H: `" + str(round(r["rsi_1d"],1)) + "`\n"
        header += "TF Agreement: " + ("\u2705 Yes" if r["tf_agree"] else "\u26a0 No") + "\n\n"
        regime_data = r.get("regime") if isinstance(r.get("regime"), dict) else {}
        regime_name = regime_data.get("regime", "UNKNOWN")
        adx_val = round(float(regime_data.get("adx", 0)), 1)
        header += " Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n\n"
        
        await update.message.reply_text(header + msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text("Could not scan " + coin + ". Make sure it has a USDT futures pair on Bitget.\nError: " + str(e))


async def scalp_scan_coins():
    """Scan all coins using 15m+1H for scalp trades."""
    scalp_results = []
    coins   = list(COINS)
    batches = [coins[i:i+BATCH_SIZE] for i in range(0, len(coins), BATCH_SIZE)]
    for batch in batches:
        for symbol in batch:
            # Skip blacklisted coins
            if is_blacklisted(symbol):
                continue
            try:
                df_15m = fetch_ohlcv(symbol, "15m", 200)
                df_1h  = fetch_ohlcv(symbol, "1h", 100)
                score_15m, signals_15m, support, resistance, rsi_15m, atr_15m, price = score_timeframe(df_15m)
                score_1h,  signals_1h,  _,       _,          rsi_1h,  atr_1h,  _     = score_timeframe(df_1h)
                combined = score_15m + (score_1h * 2)
                tf_agree = (score_15m > 0 and score_1h > 0) or (score_15m < 0 and score_1h < 0)
                combined = int(combined * 1.2) if tf_agree else int(combined * 0.7)
                abs_score = abs(combined)
                if abs_score >= 5:
                    direction  = "LONG" if combined > 0 else "SHORT"
                    confidence = "HIGH" if abs_score >= 9 else "MEDIUM"
                else:
                    direction  = "NEUTRAL"
                    confidence = "LOW"
                try:
                    funding = exchange.fetch_funding_rate(symbol)
                    funding_rate = float(funding["fundingRate"]) if funding else 0
                except Exception:
                    funding_rate = 0
                scalp_results.append({
                    "symbol": symbol,
                    "label": COIN_LABELS.get(symbol, symbol.split("/")[0]),
                    "direction": direction, "confidence": confidence,
                    "score": combined, "abs_score": abs_score,
                    "score_4h": score_15m, "score_1d": score_1h,
                    "tf_agree": tf_agree,
                    "signals": ["[1H] " + s for s in signals_1h] + ["[15m] " + s for s in signals_15m],
                    "price": price, "support": support, "resistance": resistance,
                    "rsi_4h": rsi_15m, "rsi_1d": rsi_1h,
                    "atr": atr_15m, "funding": funding_rate,
                    "df_4h": df_15m,
                    "tf_labels": ("15m", "1H"),
                })
            except Exception as e:
                log.error("Scalp scan error " + symbol + ": " + str(e))
        await asyncio.sleep(REQUEST_DELAY * BATCH_SIZE)
    return scalp_results


async def cmd_scalp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Scalp scan using 15m+1H — single best signal, paper trade auto-closes in 24hrs."""
    await update.message.reply_text("\u26a1 Scalp scan (15m+1H)... finding best setup...")
    try:
        scalp_results = await scalp_scan_coins()
        actionable = [r for r in scalp_results if r["direction"] != "NEUTRAL" and r["confidence"] in ("HIGH","MEDIUM")]
        if not actionable:
            await update.message.reply_text("\u26a1 No strong scalp signals right now. Try again in 15 minutes or use /swing.")
            return
        best = max(actionable, key=lambda x: x["abs_score"])
        sl, tp1, tp2 = calc_levels(best["direction"], best["price"], best["atr"])
        ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
        pos_usdt, contracts = calc_position_size(best["price"], sl)

        # Open paper trade as scalp type and register in active_signals for monitoring
        record_signal(best["symbol"], best["direction"], best["price"], sl, tp1, tp2, best["confidence"], trade_type="scalp")
        if paper_mode:
            existing = load_json(PAPER_FILE, [])
            already_open = any(t["symbol"] == best["symbol"] and t["status"] == "OPEN" and t.get("trade_type") == "scalp" for t in existing)
            if not already_open:
                open_paper_trade(best["symbol"], best["direction"], best["price"], sl, tp1, tp2, best["confidence"], trade_type="scalp", signal_type=best.get("signal_type", "REVERSAL"))
                
        # Always register in active_signals so monitor tracks TP1, trailing stop, re-entry
        active_signals[best["symbol"]] = {
            "direction": best["direction"], "entry": best["price"],
            "sl": sl, "tp1": tp1, "tp2": tp2,
            "tp1_hit": False, "atr": best["atr"],
            "time": _time.time(),
            "trailing_extreme": best["price"],
            "trailing_stop": None,
            "trade_type": "scalp",
        }

        header  = "\u26a1 *Scalp Signal (15m+1H) | " + datetime.now().strftime("%H:%M UTC") + "*\n"
        header += "Act within 15 minutes | Paper trade closes in 24hrs\n\n"
        await update.message.reply_text(header + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts), parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))


async def cmd_swing(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Swing scan using 1H+4H — full list + best signal for overnight trades."""
    await update.message.reply_text("\U0001f319 Swing scan (1H+4H)... overnight setups...")
    try:
        results  = await scan_all_async()
        messages = format_scan_summary(results)
        for msg in messages:
            await update.message.reply_text(msg, parse_mode="Markdown")
        actionable = get_actionable(results)
        if actionable:
            best = max(actionable, key=lambda x: x["abs_score"])
            sl, tp1, tp2 = calc_levels(best["direction"], best["price"], best["atr"])
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            header  = "\U0001f319 *Swing Signal (1H+4H) | " + datetime.now().strftime("%H:%M UTC") + "*\n"
            header += "Overnight setup — hold until TP or SL\n\n"
            await update.message.reply_text(header + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts), parse_mode="Markdown")
        else:
            await update.message.reply_text("No strong swing signals right now. Full list above.")
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))


async def cmd_blacklist(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if not args or args[0].lower() == "list":
        bl = get_blacklist()
        if not bl:
            await update.message.reply_text("Blacklist is empty.\nAdd coins with: /blacklist add LYN")
            return
        msg = "\U0001f6ab *Blacklisted Coins*\n\n"
        for b in bl:
            msg += "\u2022 *" + b["label"] + "* — " + b["reason"] + " (" + b["time"] + ")\n"
        msg += "\nRemove with: /blacklist remove LYN"
        await update.message.reply_text(msg, parse_mode="Markdown")
        return
    if len(args) < 2:
        await update.message.reply_text(
            "Usage:\n"
            "/blacklist add LYN\n"
            "/blacklist add LYN bad signals\n"
            "/blacklist remove LYN\n"
            "/blacklist list"
        )
        return
    action = args[0].lower()
    coin   = args[1].upper()
    reason = " ".join(args[2:]) if len(args) > 2 else "Manual"
    if action == "add":
        added = add_to_blacklist(coin, reason)
        if added:
            await update.message.reply_text("\U0001f6ab " + coin + " added to blacklist.\nBot will skip all signals for this coin.")
        else:
            await update.message.reply_text(coin + " is already blacklisted.")
    elif action == "remove":
        remove_from_blacklist(coin)
        await update.message.reply_text("\u2705 " + coin + " removed from blacklist.")
    else:
        await update.message.reply_text("Unknown action. Use: add or remove")


async def cmd_whale(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args   = ctx.args
    coin   = args[0].upper() if args else "BTC"
    symbol = coin + "/USDT:USDT"
    await update.message.reply_text("\U0001f433 Checking whale activity for " + coin + "...")
    try:
        # Volume spike check
        df       = fetch_ohlcv(symbol, "1h", 50)
        vol_ma   = df["volume"].rolling(20).mean()
        last_vol = float(df["volume"].iloc[-1])
        avg_vol  = float(vol_ma.iloc[-1])
        ratio    = round(last_vol / avg_vol, 1) if avg_vol > 0 else 1
        price    = float(df["close"].iloc[-1])
        prev     = float(df["close"].iloc[-2])
        direction = "BUY" if price > prev else "SELL"
        is_whale  = ratio >= 3.0

        # Order book
        ob_bias, ob_ratio = get_order_book_bias(symbol)

        # Open interest
        try:
            oi = exchange.fetch_open_interest(symbol)
            oi_val = float(oi["openInterestAmount"]) if oi else None
        except Exception:
            oi_val = None

        msg  = "\U0001f433 *Whale Check: " + coin + "*\n\n"
        msg += "Volume spike: *" + str(ratio) + "x* average"
        if is_whale:
            msg += " \u26a0 WHALE DETECTED (" + direction + ")\n"
        else:
            msg += " (normal)\n"
        ob_emoji = "\U0001f7e2" if ob_bias == "BULLISH" else "\U0001f534" if ob_bias == "BEARISH" else "\U0001f7e1"
        msg += ob_emoji + " Order book: *" + ob_bias + "* (bid ratio: " + str(ob_ratio) + ")\n"
        if oi_val:
            msg += "\U0001f4ca Open Interest: *" + "{:,.0f}".format(oi_val) + "* contracts\n"
        msg += "\nCurrent price: $" + "{:.4f}".format(price)
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text("Could not fetch data for " + coin + ".\nError: " + str(e))


async def cmd_alert(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = ctx.args
    if len(args) < 3:
        await update.message.reply_text(
            "\U0001f514 *Price Alert Usage:*\n\n"
            "/alert BTC 90000 above\n"
            "/alert ETH 2000 below\n"
            "/alert LYN 0.05 above\n\n"
            "Bot pings you when price crosses the level."
        )
        return
    coin      = args[0].upper()
    try:
        target = float(args[1])
    except ValueError:
        await update.message.reply_text("Invalid price. Usage: /alert BTC 90000 above")
        return
    direction = args[2].upper()
    if direction not in ("ABOVE", "BELOW"):
        await update.message.reply_text("Direction must be ABOVE or BELOW.")
        return
    symbol = coin + "/USDT:USDT"
    add_price_alert(symbol, coin, target, direction)
    await update.message.reply_text(
        "\U0001f514 Alert set!\n\n"
        "*" + coin + " " + direction + " $" + str(target) + "*\n"
        "You will be pinged when price crosses this level."
    )


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f916 *Phyrobot Signal Bot*\n\n"
        "Watching: top 40 coins by volume\n"
        "Timeframes: 1H + 4H\n"
        "Signals fire when quality \u226545 (MEDIUM/HIGH)\n\n"
        "*Scanning:*\n"
        "/scan \u2014 Full market scan (1H+4H)\n"
        "/best \u2014 Best signal right now\n"
        "/scalp \u2014 Best scalp signal (15m+1H)\n"
        "/swing \u2014 Overnight signals + full list\n"
        "/coin LINK \u2014 Scan any coin on demand\n"
        "/news \u2014 Latest news context\n"
        "/coins \u2014 Current watchlist\n\n"
        "*Trading:*\n"
        "/paper \u2014 Toggle paper trading\n"
        "/portfolio \u2014 Paper trades & PnL\n"
        "/history \u2014 Signal history & win rate\n"
        "/weekly \u2014 Weekly PnL report\n"
        "/setaccount 200 \u2014 Set account size\n\n"
        "*Alerts & Tools:*\n"
        "/whale BTC \u2014 Check whale activity\n"
        "/alert BTC 90000 above \u2014 Price alert\n"
        "/blacklist add LYN \u2014 Block a coin\n"
        "/blacklist list \u2014 View blocked coins\n\n"
        "*Settings:*\n"
        "/status \u2014 Bot settings",
        parse_mode="Markdown"
    )


async def cmd_scan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Scanning all 40 coins (1H+4H)...")
    try:
        results = await scan_all_async()
        messages = format_scan_summary(results)
        for msg in messages:
            await update.message.reply_text(msg, parse_mode="Markdown")
        actionable = get_actionable(results)
        if actionable:
            best = max(actionable, key=lambda x: x["abs_score"])
            sl, tp1, tp2 = calc_levels(best["direction"], best["price"], best["atr"])
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            await update.message.reply_text(
                "\U0001f3c6 *Best Signal: " + best["label"] + "*\n\n" + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts),
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("No MEDIUM/HIGH signals right now. Full list above.")
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))

async def cmd_best(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Finding best signal...")
    try:
        results    = await scan_all_async()
        actionable = get_actionable(results)
        if actionable:
            best = max(actionable, key=lambda x: x["abs_score"])
            sl, tp1, tp2 = calc_levels(best["direction"], best["price"], best["atr"])
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            await update.message.reply_text(format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts), parse_mode="Markdown")
        else:
            await update.message.reply_text("No MEDIUM/HIGH signals right now. Market is ranging.")
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))

async def cmd_news(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Fetching latest news...")
    try:
        headlines = fetch_headlines()
        if not headlines:
            await update.message.reply_text("No fresh headlines found.")
            return
        sentiment, key_points = analyze_news_context(headlines)
        emoji = "\U0001f7e2" if sentiment == "BULLISH" else "\U0001f534" if sentiment == "BEARISH" else "\U0001f7e1"
        msg  = emoji + " *News Context: " + sentiment + "*\n\n"
        if key_points:
            msg += "*Key Points:*\n"
            for pt in key_points[:5]:
                msg += "\u2022 " + pt + "\n"
        msg += "\n*Recent Headlines:*\n"
        for h in headlines[:5]:
            msg += "\u2022 " + h + "\n"
        await update.message.reply_text(msg, parse_mode=None)
    except Exception as e:
        await update.message.reply_text("Error: " + str(e))

async def cmd_coins(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    labels = list(COIN_LABELS.values())
    msg    = "\U0001f4cb *Watchlist (" + str(len(labels)) + " coins)*\n\n"
    msg   += " \u00b7 ".join(labels)
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_paper(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global paper_mode
    paper_mode = not paper_mode
    status = "ON" if paper_mode else "OFF"
    msg  = "\U0001f4dd *Paper Trading: " + status + "*\n\n"
    if paper_mode:
        msg += "Signals will be simulated automatically.\n"
        msg += "Trades auto-close at TP2 or SL.\n"
        msg += "Check /portfolio to track performance."
    else:
        msg += "Paper trading disabled."
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_portfolio(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    open_t, closed_t, total_pnl, wins, losses, scalp_open, swing_open = get_paper_summary()
    if not open_t and not closed_t:
        await update.message.reply_text("No paper trades yet. Enable with /paper and wait for signals or use /scalp.")
        return

    msg = "\U0001f4bc *Paper Portfolio*\n\n"
    keyboard = []

    if swing_open:
        msg += "*Swing Trades (Open):*\n"
        for t in swing_open:
            emoji = "\U0001f7e2" if t["pnl_usdt"] >= 0 else "\U0001f534"
            tp1_status = " \U0001f3af" if t.get("tp1_hit") else ""
            try:
                current_price = exchange.fetch_ticker(t["symbol"])["last"]
                price_str = " | `$" + "{:.4f}".format(current_price) + "`"
            except Exception:
                price_str = ""
            msg += emoji + " " + t["label"] + " " + t["direction"] + tp1_status
            msg += " | " + "{:+.1f}".format(t["pnl_pct"]) + "% ($" + "{:+.2f}".format(t["pnl_usdt"]) + ")" + price_str + "\n"
            keyboard.append([InlineKeyboardButton("\U0001f50e " + t["label"] + " signal", callback_data="sig_" + str(t["id"]))])

    if scalp_open:
        msg += "\n*Scalp Trades (Open):*\n"
        for t in scalp_open:
            emoji = "\U0001f7e2" if t["pnl_usdt"] >= 0 else "\U0001f534"
            tp1_status = " \U0001f3af" if t.get("tp1_hit") else ""
            try:
                current_price = exchange.fetch_ticker(t["symbol"])["last"]
                price_str = " | `$" + "{:.4f}".format(current_price) + "`"
            except Exception:
                price_str = ""
            open_time  = t.get("open_timestamp", _time.time())
            hours_left = max(0, round((86400 - (_time.time() - open_time)) / 3600, 1))
            msg += emoji + " " + t["label"] + " " + t["direction"] + tp1_status
            msg += " | " + "{:+.1f}".format(t["pnl_pct"]) + "% ($" + "{:+.2f}".format(t["pnl_usdt"]) + ")" + price_str
            msg += " | " + str(hours_left) + "h left\n"
            keyboard.append([InlineKeyboardButton("\U0001f50e " + t["label"] + " signal", callback_data="sig_" + str(t["id"]))])

    if closed_t:
        msg += "\n*Closed:* " + str(len(closed_t)) + " | W:" + str(wins) + " L:" + str(losses) + "\n"
        msg += "Total PnL: $" + "{:+.2f}".format(total_pnl)

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_markup)


async def callback_signal_detail(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Show original signal when user taps View Signal button."""
    query = update.callback_query
    await query.answer()
    try:
        trade_id = int(query.data.replace("sig_", ""))
        trades   = load_json(PAPER_FILE, [])
        trade    = next((t for t in trades if t["id"] == trade_id), None)
        if not trade:
            await query.message.reply_text("Trade not found.")
            return
        sig  = trade.get("original_signal", {})
        entry = sig.get("entry", trade["entry"])
        sl    = sig.get("sl", trade["sl"])
        tp1   = sig.get("tp1", trade["tp1"])
        tp2   = sig.get("tp2", trade["tp2"])
        sl_pct  = abs(entry - sl) / entry * 100
        tp1_pct = abs(tp1 - entry) / entry * 100
        tp2_pct = abs(tp2 - entry) / entry * 100
        emoji   = "\U0001f7e2" if trade["direction"] == "LONG" else "\U0001f534"
        conf_emoji = "\U0001f525" if trade["confidence"] == "HIGH" else "\u26a1"
        msg  = emoji + " *Original Signal: " + trade["label"] + "*\n"
        msg += "Opened: " + sig.get("time", trade["time"]) + "\n"
        msg += conf_emoji + " *" + trade["direction"] + "* | " + trade["confidence"] + "\n"
        msg += "Type: " + trade.get("trade_type", "swing").upper() + "\n\n"
        msg += "*Original Levels:*\n"
        msg += "  \U0001f7e1 Entry: `$" + "{:.4f}".format(entry) + "`\n"
        msg += "  \U0001f534 SL:  `$" + "{:.4f}".format(sl) + "` (-" + "{:.1f}".format(sl_pct) + "% / -" + "{:.0f}".format(sl_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n"
        msg += "  \U0001f3af TP1: `$" + "{:.4f}".format(tp1) + "` (+" + "{:.1f}".format(tp1_pct) + "% / +" + "{:.0f}".format(tp1_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n"
        msg += "  \U0001f3af TP2: `$" + "{:.4f}".format(tp2) + "` (+" + "{:.1f}".format(tp2_pct) + "% / +" + "{:.0f}".format(tp2_pct*LEVERAGE) + "% at " + str(LEVERAGE) + "x)\n\n"
        msg += "*Current Status:*\n"
        current_emoji = "\U0001f7e2" if trade["pnl_usdt"] >= 0 else "\U0001f534"
        msg += current_emoji + " " + trade["status"] + " | PnL: " + "{:+.1f}".format(trade["pnl_pct"]) + "% ($" + "{:+.2f}".format(trade["pnl_usdt"]) + ")"
        await query.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await query.message.reply_text("Error: " + str(e))


async def cmd_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    stats = get_win_rate()
    if not stats:
        await update.message.reply_text("No closed signals yet. History builds as signals are tracked.")
        return
    msg  = "\U0001f4ca *Signal History*\n\n"

    swing = stats["swing"]
    if swing["total"] > 0:
        msg += "\U0001f319 *Swing Trades (1H+4H):*\n"
        msg += "  Total: " + str(swing["total"]) + " | Win rate: *" + str(swing["win_rate"]) + "%*\n"
        msg += "  Wins: " + str(swing["wins"]) + " | Losses: " + str(swing["losses"]) + "\n"
        msg += "  Avg PnL: " + str(swing["avg_pnl"]) + "%\n\n"

    scalp = stats["scalp"]
    if scalp["total"] > 0:
        msg += "\u26a1 *Scalp Trades (15m+1H):*\n"
        msg += "  Total: " + str(scalp["total"]) + " | Win rate: *" + str(scalp["win_rate"]) + "%*\n"
        msg += "  Wins: " + str(scalp["wins"]) + " | Losses: " + str(scalp["losses"]) + "\n"
        msg += "  Avg PnL: " + str(scalp["avg_pnl"]) + "%\n\n"

    momentum = stats["momentum"]
    if momentum["total"] > 0:
        msg += "\u26a1 *Momentum Signals:*\n"
        msg += "  Total: " + str(momentum["total"]) + " | Win rate: *" + str(momentum["win_rate"]) + "%*\n"
        msg += "  Wins: " + str(momentum["wins"]) + " | Losses: " + str(momentum["losses"]) + "\n"
        msg += "  Avg PnL: " + str(momentum["avg_pnl"]) + "%\n\n"

    reversal = stats["reversal"]
    if reversal["total"] > 0:
        msg += "\U0001f504 *Reversal Signals:*\n"
        msg += "  Total: " + str(reversal["total"]) + " | Win rate: *" + str(reversal["win_rate"]) + "%*\n"
        msg += "  Wins: " + str(reversal["wins"]) + " | Losses: " + str(reversal["losses"]) + "\n"
        msg += "  Avg PnL: " + str(reversal["avg_pnl"]) + "%\n\n"

    overall = stats["overall"]
    msg += "\U0001f4ca *Overall:*\n"
    msg += "  Total: " + str(overall["total"]) + " | Win rate: *" + str(overall["win_rate"]) + "%*\n"
    msg += "  Avg PnL: " + str(overall["avg_pnl"]) + "%\n"
    msg += "  Open signals: " + str(stats["open"])
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_weekly(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    history = load_json(HISTORY_FILE, [])
    paper   = load_json(PAPER_FILE, [])
    closed  = [s for s in history if s["outcome"] in ("WIN","LOSS","BREAKEVEN")]
    wins    = len([s for s in closed if s["outcome"] == "WIN"])
    losses  = len([s for s in closed if s["outcome"] == "LOSS"])
    total   = len(closed)
    wr      = round(wins/total*100, 1) if total else 0
    paper_pnl = sum([t["pnl_usdt"] for t in paper])
    msg  = "\U0001f4ca *Performance Report*\n\n"
    msg += "*Signal History:*\n"
    msg += "  Total: " + str(total) + " | Wins: " + str(wins) + " | Losses: " + str(losses) + "\n"
    msg += "  Win rate: *" + str(wr) + "%*\n\n"
    msg += "*Paper Trading PnL:*\n"
    msg += "  Total: $" + "{:+.2f}".format(paper_pnl) + "\n"
    msg += "  Open trades: " + str(len([t for t in paper if t["status"] == "OPEN"]))
    await update.message.reply_text(msg, parse_mode="Markdown")

async def cmd_setaccount(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    global ACCOUNT_SIZE
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "Account size: $" + str(ACCOUNT_SIZE) + "\n"
            "Risk per trade: " + str(RISK_PCT) + "%\n\n"
            "Usage: /setaccount 500"
        )
        return
    try:
        ACCOUNT_SIZE = float(args[0])
        await update.message.reply_text(
            "\u2705 Account updated to $" + str(ACCOUNT_SIZE) + "\n"
            "Risk per trade: $" + str(round(ACCOUNT_SIZE * RISK_PCT / 100, 2))
        )
    except ValueError:
        await update.message.reply_text("Invalid amount. Usage: /setaccount 500")

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "\U0001f916 *Bot Status*\n\n"
        "Coins: top " + str(TOP_COINS) + " by volume\n"
        "Timeframes: 1H + 4H\n"
        "Scan interval: every " + str(SCAN_INTERVAL//60) + " min\n"
        "Signal threshold: quality \u226545 (MEDIUM/HIGH)\n"
        "Account size: $" + str(ACCOUNT_SIZE) + "\n"
        "Risk per trade: " + str(RISK_PCT) + "%\n"
        "Paper mode: " + ("ON" if paper_mode else "OFF"),
        parse_mode="Markdown"
    )


# ─── BLACKLIST ─────────────────────────────────────────────────────────────────
def get_blacklist():
    return load_json(BLACKLIST_FILE, [])

def add_to_blacklist(label, reason="Manual"):
    bl = get_blacklist()
    if label.upper() not in [b["label"].upper() for b in bl]:
        bl.append({
            "label":  label.upper(),
            "reason": reason,
            "time":   _time.strftime("%Y-%m-%d %H:%M"),
        })
        save_json(BLACKLIST_FILE, bl)
        return True
    return False

def remove_from_blacklist(label):
    bl = get_blacklist()
    bl = [b for b in bl if b["label"].upper() != label.upper()]
    save_json(BLACKLIST_FILE, bl)

def is_blacklisted(symbol):
    label = COIN_LABELS.get(symbol, symbol.split("/")[0])
    bl    = get_blacklist()
    return any(b["label"].upper() == label.upper() for b in bl)

# ─── PRICE ALERTS ──────────────────────────────────────────────────────────────
def add_price_alert(symbol, label, target, direction):
    alerts = load_json(ALERTS_FILE, [])
    alerts.append({
        "symbol":    symbol,
        "label":     label,
        "target":    target,
        "direction": direction,
        "triggered": False,
        "time":      _time.strftime("%Y-%m-%d %H:%M"),
    })
    save_json(ALERTS_FILE, alerts)

async def check_price_alerts(app):
    alerts   = load_json(ALERTS_FILE, [])
    modified = False
    for alert in alerts:
        if alert["triggered"]:
            continue
        try:
            price = exchange.fetch_ticker(alert["symbol"])["last"]
            hit   = (alert["direction"] == "ABOVE" and price >= alert["target"]) or                     (alert["direction"] == "BELOW" and price <= alert["target"])
            if hit:
                alert["triggered"] = True
                modified = True
                msg  = "\U0001f514 *Price Alert: " + alert["label"] + "*\n\n"
                msg += "Target $" + str(alert["target"]) + " reached!\n"
                msg += "Current price: $" + "{:.4f}".format(price) + "\n"
                msg += "Alert: " + alert["label"] + " " + alert["direction"] + " $" + str(alert["target"])
                await send_msg(app, msg)
                log.info("Price alert triggered: " + alert["label"])
        except Exception as e:
            log.error("Alert check error: " + str(e))
    if modified:
        save_json(ALERTS_FILE, alerts)

# ─── AUTO LOOPS ────────────────────────────────────────────────────────────────
async def auto_scan(app):
    log.info("Auto scan started")
    while True:
        await asyncio.sleep(SCAN_INTERVAL)
        try:
            results    = await scan_all_async()
            actionable = get_actionable(results)
            now = _time.time()
            # Limit to top 3 signals per scan cycle
            actionable = sorted(actionable, key=lambda x: x["abs_score"], reverse=True)[:3]
            for r in actionable:
                key = r["symbol"] + "_" + r["direction"]
                last = last_signal.get(key)
                # Skip if same score fired within last 4 hours
                if last and last["score"] == r["score"] and now - last["time"] < 14400:
                    continue
                # Skip if coin is blacklisted
                if is_blacklisted(r["symbol"]):
                    log.info("Blacklisted — skipping: " + r["label"])
                    continue
                last_signal[key] = {"score": r["score"], "time": now}
                history = load_json(HISTORY_FILE, [])
                risk = risk_gate(
                    symbol=r["symbol"],
                    direction=r["direction"],
                    price=r["price"],
                    atr=r["atr"],
                    quality_score=r.get("abs_score", 0),
                    active_signals=active_signals,
                    trade_history=history,
                )
                if not risk["approved"]:
                    log.info("Risk gate blocked " + r["label"] + ": " + risk["reject_reason"])
                    continue
                sl        = risk["sl"]
                tp1       = risk["tp1"]
                tp2       = risk["tp2"]
                pos_usdt  = risk["position_usdt"]
                contracts = risk["contracts"]
                ob_bias, ob_ratio = get_order_book_bias(r["symbol"])
               
                # Record signal
                record_signal(r["symbol"], r["direction"], r["price"], sl, tp1, tp2, r["confidence"])

                # Open paper trade — only if no open trade already exists for this symbol
                if paper_mode:
                    existing = load_json(PAPER_FILE, [])
                    already_open = any(t["symbol"] == r["symbol"] and t["status"] == "OPEN" for t in existing)
                    if not already_open:
                        open_paper_trade(r["symbol"], r["direction"], r["price"], sl, tp1, tp2, r["confidence"], signal_type=r.get("signal_type", "REVERSAL"))
                    else:
                        log.info("Paper trade skipped — already open for " + r["label"])

                # Track for TP/re-entry alerts
                active_signals[r["symbol"]] = {
                    "direction": r["direction"], "entry": r["price"],
                    "sl": sl, "tp1": tp1, "tp2": tp2,
                    "tp1_hit": False, "atr": r["atr"],
                    "time": _time.time(),
                    "trailing_extreme": r["price"],  # tracks highest/lowest price seen
                    "trailing_stop": None,            # activated after TP1 hit
                }

                msg = format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                msg += format_ai_block(r.get("ai_result", {}))
                await send_msg(app, msg)
                log.info("Signal sent: " + r["label"] + " " + r["direction"] + " | Score: " + str(r["score"]))
        except Exception as e:
            log.error("Auto scan error: " + str(e))

async def auto_price_alerts(app):
    """Check custom price alerts every 2 minutes."""
    log.info("Price alert monitor started")
    while True:
        await asyncio.sleep(120)
        try:
            await check_price_alerts(app)
        except Exception as e:
            log.error("Price alert error: " + str(e))


async def auto_news(app):
    log.info("News loop started")
    while True:
        await asyncio.sleep(NEWS_INTERVAL)
        try:
            headlines = fetch_headlines()
            if headlines:
                sentiment, key_points = analyze_news_context(headlines)
                news_context["sentiment"]  = sentiment
                news_context["key_points"] = key_points
                news_context["headlines"]  = headlines
                news_context["last_update"] = _time.time()
                log.info("News context updated: " + sentiment)
        except Exception as e:
            log.error("News loop error: " + str(e))

async def auto_monitor(app):
    log.info("Signal monitor started")
    while True:
        await asyncio.sleep(60)
        try:
            price_cache = {}

            # ── Partial TP + Trailing Stop + Auto TP Adjustment ────────────
            for symbol, sig in list(active_signals.items()):
                try:
                    price = exchange.fetch_ticker(symbol)["last"]
                    price_cache[symbol] = price
                    label     = COIN_LABELS.get(symbol, symbol.split("/")[0])
                    direction = sig["direction"]
                    entry     = sig["entry"]
                    tp1       = sig["tp1"]
                    tp2       = sig["tp2"]

                    # TP1 hit
                    if not sig["tp1_hit"]:
                        if (direction == "LONG" and price >= tp1) or (direction == "SHORT" and price <= tp1):
                            active_signals[symbol]["tp1_hit"] = True
                            # FIX: trailing stop must be BELOW price for LONG, ABOVE for SHORT
                            if direction == "LONG":
                                trailing_stop = round(price * (1 - TRAILING_BUFFER), 6)  # below price
                            else:
                                trailing_stop = round(price * (1 + TRAILING_BUFFER), 6)  # above price
                            active_signals[symbol]["trailing_stop"]    = trailing_stop
                            active_signals[symbol]["trailing_extreme"] = price
                            msg  = "\U0001f3af *TP1 Hit: " + label + "*\n\n"
                            msg += "Price: $" + "{:.4f}".format(price) + "\n"
                            msg += "Action: Close 50% of your position\n"
                            msg += "Move SL to breakeven: $" + "{:.4f}".format(entry) + "\n"
                            msg += "Trailing stop set at: $" + "{:.4f}".format(trailing_stop) + " (12% buffer)\n"
                            msg += "Remaining 50% protected — stop follows price"
                            await send_msg(app, msg)

                    # Trailing stop management after TP1 hit
                    elif sig["tp1_hit"] and sig.get("trailing_stop"):
                        trailing_stop    = sig["trailing_stop"]
                        trailing_extreme = sig["trailing_extreme"]

                        # Update trailing extreme and move stop — only in favorable direction
                        if direction == "LONG" and price > trailing_extreme:
                            new_extreme = price
                            new_stop    = round(price * (1 - TRAILING_BUFFER), 6)
                            # FIX: only move stop UP for LONG, never down
                            if new_stop > trailing_stop:
                                active_signals[symbol]["trailing_extreme"] = new_extreme
                                active_signals[symbol]["trailing_stop"]    = new_stop
                                log.info("Trailing stop moved up for " + label + ": $" + str(new_stop))

                        elif direction == "SHORT" and price < trailing_extreme:
                            new_extreme = price
                            new_stop    = round(price * (1 + TRAILING_BUFFER), 6)
                            # FIX: only move stop DOWN for SHORT, never up
                            if new_stop < trailing_stop:
                                active_signals[symbol]["trailing_extreme"] = new_extreme
                                active_signals[symbol]["trailing_stop"]    = new_stop
                                log.info("Trailing stop moved down for " + label + ": $" + str(new_stop))

                        # Check if trailing stop triggered
                        trailing_hit = (direction == "LONG" and price <= trailing_stop) or                                        (direction == "SHORT" and price >= trailing_stop)
                        if trailing_hit:
                            pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
                            msg  = "\U0001f6d1 *Trailing Stop Hit: " + label + "*\n\n"
                            msg += "Price: $" + "{:.4f}".format(price) + "\n"
                            msg += "Action: Close remaining 50% now\n"
                            msg += "Entry was: $" + "{:.4f}".format(entry) + "\n"
                            msg += "Approx PnL on remainder: " + "{:+.1f}".format(pnl_pct * LEVERAGE) + "% at " + str(LEVERAGE) + "x"
                            await send_msg(app, msg)
                            del active_signals[symbol]

                        # Auto TP2 adjustment — FIX: only move TP2 in favorable direction
                        else:
                            try:
                                df_fresh  = fetch_ohlcv(symbol, "1h", 50)
                                atr_fresh = ta.atr(df_fresh["high"], df_fresh["low"], df_fresh["close"], length=14)
                                new_atr   = float(atr_fresh.iloc[-1])
                                _, _, new_tp2 = calc_levels(direction, price, new_atr)
                                old_tp2   = sig["tp2"]
                                improvement = abs(new_tp2 - old_tp2) / old_tp2
                                if improvement > 0.02:
                                    # LONG: new TP2 must be HIGHER than old TP2
                                    # SHORT: new TP2 must be LOWER than old TP2
                                    favorable = (direction == "LONG" and new_tp2 > old_tp2) or                                                 (direction == "SHORT" and new_tp2 < old_tp2)
                                    if favorable:
                                        active_signals[symbol]["tp2"] = new_tp2
                                        msg  = "\U0001f504 *TP2 Adjusted: " + label + "*\n\n"
                                        msg += "Old TP2: $" + "{:.4f}".format(old_tp2) + "\n"
                                        msg += "New TP2: $" + "{:.4f}".format(new_tp2) + " (improved)\n"
                                        msg += "Trailing stop: $" + "{:.4f}".format(trailing_stop)
                                        await send_msg(app, msg)
                            except Exception:
                                pass

                    # Expire after 72 hours or if paper trade already closed
                    trades = load_json(PAPER_FILE, [])
                    trade_closed = any(t["symbol"] == symbol and t["status"] != "OPEN" for t in trades)
                    if trade_closed or _time.time() - sig["time"] > 259200:
                        if symbol in active_signals:
                            del active_signals[symbol]
                            log.info("Removed from active_signals: " + label)

                except Exception as e:
                    log.error("Monitor error " + symbol + ": " + str(e))

            # ── Paper trade updates + closed alerts ────────────────────────
            if paper_mode:
                _, closed_now = update_paper_trades()
                for t in closed_now:
                    emoji = "\U0001f7e2" if t["status"] == "WIN" else "\U0001f534"
                    msg  = emoji + " *Paper Trade Closed: " + t["label"] + "*\n\n"
                    msg += "Result: *" + t["status"] + "*\n"
                    msg += "PnL: " + "{:+.1f}".format(t["pnl_pct"]) + "% ($" + "{:+.2f}".format(t["pnl_usdt"]) + ")\n"
                    msg += "Direction: " + t["direction"] + " | Entry: $" + "{:.4f}".format(t["entry"])
                    await send_msg(app, msg)

            # ── Re-entry alerts (4hr cooldown per coin) ────────────────
            history = load_json(HISTORY_FILE, [])
            recent  = [s for s in history if s["outcome"] == "OPEN"]
            seen_symbols = set()  # only one alert per coin per cycle
            now_ts = _time.time()
            for sig in recent[:10]:
                symbol = sig["symbol"]
                # Skip if in active signals
                if symbol in active_signals:
                    continue
                # Skip if blacklisted
                if is_blacklisted(symbol):
                    continue
                # Skip if already alerted this cycle
                if symbol in seen_symbols:
                    continue
                # Skip if within 4hr cooldown
                last_reentry = reentry_cooldown.get(symbol, 0)
                if now_ts - last_reentry < 14400:
                    continue
                try:
                    price = price_cache.get(symbol) or exchange.fetch_ticker(symbol)["last"]
                    entry = sig["entry"]
                    label = COIN_LABELS.get(symbol, symbol.split("/")[0])
                    if abs(price - entry) / entry < 0.005:
                        seen_symbols.add(symbol)
                        reentry_cooldown[symbol] = now_ts  # set cooldown
                        msg  = "\U0001f504 *Re-entry Alert: " + label + "*\n\n"
                        msg += "Price back near original entry\n"
                        msg += "Entry: $" + "{:.4f}".format(entry) + " | Current: $" + "{:.4f}".format(price) + "\n"
                        msg += "Direction: *" + sig["direction"] + "*\n"
                        msg += "SL: $" + "{:.4f}".format(sig["sl"]) + " | TP1: $" + "{:.4f}".format(sig["tp1"]) + "\n"
                        msg += "Next alert for this coin in 4 hours."
                        await send_msg(app, msg)
                        log.info("Re-entry alert sent: " + label)
                except Exception as e:
                    log.error("Re-entry error " + symbol + ": " + str(e))

        except Exception as e:
            log.error("Monitor loop error: " + str(e))

async def auto_coin_refresh(app):
    log.info("Coin refresh loop started")
    while True:
        await asyncio.sleep(COIN_REFRESH)
        try:
            fetch_top_coins(TOP_COINS)
            log.info("Coin list refreshed")
        except Exception as e:
            log.error("Coin refresh error: " + str(e))

async def auto_weekly_report(app):
    log.info("Weekly report loop started")
    while True:
        now = datetime.utcnow()
        days_until_sunday = (6 - now.weekday()) % 7 or 7
        target = now.replace(hour=19, minute=0, second=0, microsecond=0)
        target = target + timedelta(days=days_until_sunday)
        wait_seconds = (target - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        try:
            history = load_json(HISTORY_FILE, [])
            paper   = load_json(PAPER_FILE, [])
            closed  = [s for s in history if s["outcome"] in ("WIN","LOSS","BREAKEVEN")]
            wins    = len([s for s in closed if s["outcome"] == "WIN"])
            losses  = len([s for s in closed if s["outcome"] == "LOSS"])
            total   = len(closed)
            wr      = round(wins/total*100, 1) if total else 0
            paper_pnl = sum([t["pnl_usdt"] for t in paper])
            msg  = "\U0001f4ca *Weekly Performance Report*\n\n"
            msg += "Signals: " + str(total) + " | Win rate: *" + str(wr) + "%*\n"
            msg += "Wins: " + str(wins) + " | Losses: " + str(losses) + "\n\n"
            msg += "Paper PnL: $" + "{:+.2f}".format(paper_pnl) + "\n"
            msg += "New week, new opportunities. Stay disciplined."
            await send_msg(app, msg)
        except Exception as e:
            log.error("Weekly report error: " + str(e))

# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    fetch_top_coins(TOP_COINS)

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start",      cmd_start))
    app.add_handler(CommandHandler("help",       cmd_start))
    app.add_handler(CommandHandler("scan",       cmd_scan))
    app.add_handler(CommandHandler("best",       cmd_best))
    app.add_handler(CommandHandler("news",       cmd_news))
    app.add_handler(CommandHandler("coins",      cmd_coins))
    app.add_handler(CommandHandler("paper",      cmd_paper))
    app.add_handler(CommandHandler("portfolio",  cmd_portfolio))
    app.add_handler(CommandHandler("history",    cmd_history))
    app.add_handler(CommandHandler("weekly",     cmd_weekly))
    app.add_handler(CommandHandler("setaccount", cmd_setaccount))
    app.add_handler(CommandHandler("status",     cmd_status))
    app.add_handler(CommandHandler("coin",       cmd_coin))
    app.add_handler(CallbackQueryHandler(callback_signal_detail, pattern="^sig_"))
    app.add_handler(CommandHandler("scalp",      cmd_scalp))
    app.add_handler(CommandHandler("swing",      cmd_swing))
    app.add_handler(CommandHandler("blacklist",  cmd_blacklist))
    app.add_handler(CommandHandler("whale",      cmd_whale))
    app.add_handler(CommandHandler("alert",      cmd_alert))
    app.add_handler(CommandHandler("backtest", lambda u, c: cmd_backtest(u, c, exchange, COIN_LABELS)))
    app.add_handler(CommandHandler("risk",     lambda u, c: cmd_risk(u, c, active_signals, load_json(HISTORY_FILE, []))))
    
    async def post_init(application):
        asyncio.create_task(auto_scan(application))
        asyncio.create_task(auto_news(application))
        asyncio.create_task(auto_monitor(application))
        asyncio.create_task(auto_coin_refresh(application))
        asyncio.create_task(auto_weekly_report(application))
        asyncio.create_task(auto_price_alerts(application))

    app.post_init = post_init
    log.info("Phyrobot starting — top " + str(TOP_COINS) + " coins | 1H+4H")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
BOTEOF
