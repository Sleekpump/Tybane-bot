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
watched_trades      = {}  # {symbol: {direction, entry, start_time, last_alert}} — /watch command
btc_circuit_breaker = {   # BTC dump protection — blocks new LONGs when BTC is dumping
    "active":      False,
    "triggered_at": 0,
    "reason":      "",
    "last_check":  0,
}
coupon_monitor = {        # Auto-coupon scanner state
    "active":      False, # toggle via /coupon
    "last_signals": {},   # {symbol: timestamp} cooldown tracker
}
flip_trades = {}          # Active flip scalps: {symbol: {direction, ref_price, target_pct, entry, ...}}
FLIP_CHECK_INTERVAL = 120  # check every 2 minutes
FLIP_COOLDOWN       = 1800 # 30 min between flips on same coin
FLIP_MAX_ATR_MULT   = 3.0  # pause flipping if ATR > 3x normal (too erratic)

gainers_state = {         # Top gainers scanner state
    "last_scan":    0,
    "last_signals": {},   # {symbol: timestamp} 4H cooldown
    "known_pumps":  {},   # {symbol: first_seen_pct} track pumps over time
}
GAINERS_SCAN_INTERVAL = 1800  # check top gainers every 30 minutes
GAINERS_MIN_CHANGE    = 15.0  # minimum 24H % gain to analyze
GAINERS_MAX_COINS     = 20    # top N gainers to check
GAINERS_COOLDOWN      = 14400 # 4H cooldown per coin

alpha_state = {           # Binance Alpha scanner state
    "coins":        [],   # symbols available on Bitget futures with $500k+ volume
    "labels":       {},   # {symbol: label}
    "volumes":      {},   # {symbol: 24h_volume_usdt} for context
    "last_refresh": 0,
    "last_signals": {},   # {symbol: timestamp} cooldown tracker
}
ALPHA_SCAN_INTERVAL = 600    # scan every 10 minutes
ALPHA_REFRESH_HOURS = 6      # refresh list every 6 hours
ALPHA_MAX_TOKENS    = 40     # max tokens to cross-reference
ALPHA_MIN_VOLUME    = 500000 # $500k minimum 24H Bitget volume
ALPHA_COOLDOWN      = 14400  # 4H cooldown per token
WATCH_MAX           = 6   # max coins watched simultaneously (6 saves credits vs 8)
BTC_DUMP_THRESHOLD  = 2.0  # % drop on 1H candle to trigger circuit breaker
BTC_RESET_RSI       = 48   # BTC 1H RSI must be above this to reset breaker
BTC_RESET_DROP      = 1.0  # BTC 1H drop must be below this % to reset
BTC_CHECK_INTERVAL  = 1800 # check BTC every 30 minutes for reset
WATCH_INTERVAL      = 120 # check every 2 minutes
WATCH_QUALITY_MIN_SCALP = 55  # scalp: reversal quality threshold
WATCH_QUALITY_MIN_SWING = 65  # swing: reversal quality threshold (default)
WATCH_COOLDOWN          = 1800  # 30 min cooldown between confirmed alerts per coin
WATCH_EARLY_COOLDOWN    = 3600  # 60 min cooldown between early warning alerts (less urgent)

# Confidence level thresholds — /watch COIN DIR PRICE swing high|medium|low
# HIGH:   4H needs 4/7 signals + 4H trend must be broken; 1H warnings suppressed entirely
# MEDIUM: default — 2/7 on 1H early warning, 2/7 on 4H confirmed
# LOW:    maximum sensitivity — any 2/7 on 1H fires immediately
WATCH_CONF_HIGH_L2_MIN = 4   # HIGH: 4H needs 4+ signals to fire
WATCH_CONF_MED_L2_MIN  = 2   # MEDIUM/LOW: 4H needs 2+ signals (default)

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
# ─── PARALLEL SCANNING ─────────────────────────────────────────────────────────
# BTC 24h change cache — fetched once per scan cycle, shared across all coins
_btc_change_cache = {"value": None, "ts": 0}

async def analyze_async(symbol):
    try:
        result = await run_full_pipeline(
            symbol, fetch_ohlcv, COIN_LABELS,
            ai_client, exchange, news_context
        )
        if not result or "direction" not in result:
            return None
        # ── Classify MOMENTUM vs REVERSAL vs WEAK ────────────────────────
        if result["direction"] not in (None, "NEUTRAL"):
            try:
                # Use cached BTC change — avoids 40 extra API calls per scan cycle
                btc_change_24h = _btc_change_cache["value"] if _time.time() - _btc_change_cache["ts"] < 300 else None
                df_classify = result.get("df_4h")
                if df_classify is not None:
                    signal_type, type_conf, type_reason = classify_signal(
                        df_classify, result["direction"], result.get("abs_score", 0), btc_change_24h
                    )
                    result["signal_type"]  = signal_type
                    result["type_conf"]    = type_conf
                    result["type_reason"]  = type_reason
            except Exception as e:
                log.warning("classify_signal failed for " + symbol + ": " + str(e))
                result["signal_type"] = "MOMENTUM"
        # ── Grade signal A/B/C ────────────────────────────────────────────
        if result["direction"] not in (None, "NEUTRAL"):
            try:
                from signal_engine import grade_signal
                df_grade = result.get("df_4h")
                df_4h_grade = fetch_ohlcv(symbol, "4h", 100)
                if df_grade is not None:
                    grade_result = grade_signal(
                        df_grade, df_4h_grade,
                        result["direction"],
                        result.get("abs_score", 0),
                        result.get("rsi_4h", 50),
                        result.get("funding", 0),
                    )
                    result["grade"]         = grade_result["grade"]
                    result["grade_score"]   = grade_result["grade_score"]
                    result["grade_reasons"] = grade_result["reasons"]
                    result["grade_warnings"]= grade_result["warnings"]
                    result["dow_phase"]     = grade_result.get("dow_phase", "UNCLEAR")
                    result["dow_confidence"]= grade_result.get("dow_confidence", "LOW")
                    result["dow_signals"]   = grade_result.get("dow_signals", [])
                    result["vwap"]          = grade_result.get("vwap", 0)
                    result["vwap_dist_pct"] = grade_result.get("vwap_dist_pct", 0)
                    result["vwap_bias"]     = grade_result.get("vwap_bias", "AT")
                    result["vwap_strength"] = grade_result.get("vwap_strength", "WEAK")
                    result["supertrend"]    = grade_result.get("supertrend", 0)
                    result["supertrend_dir"]= grade_result.get("supertrend_dir", 1)
                    result["supertrend_flip"]= grade_result.get("supertrend_flip", False)
                    # Dynamic leverage suggestion
                    try:
                        from signal_engine import suggest_leverage
                        lev = suggest_leverage(
                            result.get("df_4h") or df_ltf,
                            signal_quality=result.get("abs_score", 50),
                            grade=grade_result["grade"],
                        )
                        result["suggested_leverage"] = lev["suggested"]
                        result["max_safe_leverage"]  = lev["max_safe"]
                        result["leverage_reason"]    = lev["reason"]
                        result["atr_pct"]            = lev["atr_pct"]
                    except Exception:
                        result["suggested_leverage"] = LEVERAGE
                        result["max_safe_leverage"]  = LEVERAGE
            except Exception as e:
                log.warning("Grade signal failed for " + symbol + ": " + str(e))
                result["grade"] = "B"

        # ── Volume Profile — compute and store for level adjustment ──────
        try:
            from signal_engine import compute_volume_profile
            df_for_vp = result.get("df_4h")
            if df_for_vp is not None and len(df_for_vp) >= 20:
                vp = compute_volume_profile(df_for_vp)
                result["vp_poc"]          = vp.get("poc")
                result["vp_hvn_above"]    = vp.get("hvn_above")
                result["vp_hvn_below"]    = vp.get("hvn_below")
                result["vp_in_lvn"]       = vp.get("in_lvn", False)
                result["vp_adjustments"]  = []  # populated later by calc_levels_v2
                # Grade A boost if price is in LVN (fast move expected)
                if vp.get("in_lvn") and result.get("grade") == "B":
                    result["grade"] = "A"
                    result.setdefault("grade_reasons", []).append("In LVN — fast move expected")
        except Exception as e:
            log.warning("VP compute error for " + symbol + ": " + str(e))

        # ── Relative strength vs BTC ───────────────────────────────────────
        try:
            btc_val = _btc_change_cache.get("value")
            if btc_val is not None and symbol != "BTC/USDT:USDT":
                df_rs = result.get("df_4h")
                if df_rs is not None and len(df_rs) >= 24:
                    coin_chg = (float(df_rs["close"].iloc[-1]) - float(df_rs["close"].iloc[-24])) / float(df_rs["close"].iloc[-24]) * 100
                    result["relative_strength"] = round(coin_chg - btc_val, 2)
                else:
                    result["relative_strength"] = 0
            else:
                result["relative_strength"] = 0
        except Exception:
            result["relative_strength"] = 0

        # ── Grade / signal_type contradiction guard ──────────────────────
        # Grade A means early/accumulation setup. WEAK means no confirmation.
        # If signal_type is WEAK, Grade A is misleading — downgrade to B.
        # WEAK signals are excluded by get_actionable anyway, but the grade
        # should still be consistent for display purposes.
        if result.get("signal_type") == "WEAK" and result.get("grade") == "A":
            result["grade"] = "B"
            result.setdefault("grade_warnings", []).append("Grade capped at B — signal type is WEAK")

        log.info(result["label"] + ": " + result["direction"] + " | Q:" + str(result.get("abs_score", 0)) + " | " + result.get("signal_type", "?") + " | Grade:" + result.get("grade", "B"))
        return result
    except Exception as e:
        log.error("Error " + symbol + ": " + str(e))
        return None

async def scan_all_async():
    # Fetch BTC 24h change once for the whole scan cycle
    try:
        btc_df = fetch_ohlcv("BTC/USDT:USDT", "1h", 25)
        _btc_change_cache["value"] = (float(btc_df["close"].iloc[-1]) - float(btc_df["close"].iloc[-24])) / float(btc_df["close"].iloc[-24]) * 100
        _btc_change_cache["ts"] = _time.time()
    except Exception:
        pass
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
    """
    Filter and rank signals:
    1. MOMENTUM only (REVERSAL disabled)
    2. Grade C excluded — late entry / exit liquidity risk
    3. Ranked by: Grade A first, then relative strength vs BTC
    """
    filtered = [r for r in results
                if r["direction"] != "NEUTRAL"
                and r["confidence"] in ("HIGH", "MEDIUM")
                and r.get("signal_type", "REVERSAL") == "MOMENTUM"
                and r.get("grade", "B") != "C"]  # exclude Grade C

    # Sort: Grade A first, then by relative strength (strongest outperformers first)
    grade_order = {"A": 0, "B": 1, "C": 2}
    filtered.sort(key=lambda r: (
        grade_order.get(r.get("grade", "B"), 1),
        -abs(r.get("relative_strength", 0))
    ))
    return filtered

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

# ─── FEAR & GREED INDEX ────────────────────────────────────────────────────────
fear_greed_cache = {"value": None, "label": "Neutral", "ts": 0}

def fetch_fear_greed():
    """Fetch Fear & Greed Index from alternative.me API. Free, no key needed."""
    global fear_greed_cache
    try:
        import urllib.request, json as _json
        if _time.time() - fear_greed_cache["ts"] < 3600:  # cache 1 hour
            return fear_greed_cache
        with urllib.request.urlopen("https://api.alternative.me/fng/?limit=1", timeout=5) as r:
            data = _json.loads(r.read())
            val  = int(data["data"][0]["value"])
            lbl  = data["data"][0]["value_classification"]
            fear_greed_cache = {"value": val, "label": lbl, "ts": _time.time()}
            log.info(f"Fear & Greed: {val} ({lbl})")
    except Exception as e:
        log.warning("Fear & Greed fetch error: " + str(e))
    return fear_greed_cache


def get_fear_greed_signal_modifier(direction: str) -> dict:
    """
    Returns a modifier dict based on Fear & Greed value.
    Extreme Fear (< 20): boost LONG confidence, reduce SHORT
    Extreme Greed (> 80): boost SHORT confidence, reduce LONG (distribution risk)
    """
    fg = fetch_fear_greed()
    val = fg.get("value")
    if val is None:
        return {"modifier": "NEUTRAL", "note": ""}

    if val <= 20:
        note = f"Extreme Fear ({val}) — historically strong BUY zone"
        return {"modifier": "LONG_BOOST" if direction == "LONG" else "SHORT_CAUTION", "note": note}
    elif val >= 80:
        note = f"Extreme Greed ({val}) — distribution risk, caution on LONGs"
        return {"modifier": "SHORT_BOOST" if direction == "SHORT" else "LONG_CAUTION", "note": note}
    elif val <= 35:
        return {"modifier": "LONG_SLIGHT_BOOST" if direction == "LONG" else "NEUTRAL",
                "note": f"Fear ({val}) — cautiously bullish context"}
    elif val >= 65:
        return {"modifier": "NEUTRAL", "note": f"Greed ({val}) — watch for exhaustion"}
    return {"modifier": "NEUTRAL", "note": ""}


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
            model="llama-3.1-8b-instant",
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

TRAILING_BUFFER = 0.06  # 6% trailing stop buffer after TP1 hit (tighter = keep more profit)

def record_signal(symbol, direction, entry, sl, tp1, tp2, confidence,
                  trade_type="swing", signal_type="REVERSAL",
                  grade="B", dow_phase="UNCLEAR", vwap_bias="AT"):
    history = load_json(HISTORY_FILE, [])
    sig_id  = len(history) + 1
    history.append({
        "id":          sig_id,
        "symbol":      symbol,
        "label":       COIN_LABELS.get(symbol, symbol.split("/")[0]),
        "direction":   direction,
        "entry":       entry,
        "sl":          sl,
        "tp1":         tp1,
        "tp2":         tp2,
        "confidence":  confidence,
        "trade_type":  trade_type,
        "signal_type": signal_type,
        "grade":       grade,       # A / B / C
        "dow_phase":   dow_phase,   # ACCUMULATION / PARTICIPATION / DISTRIBUTION / UNCLEAR
        "vwap_bias":   vwap_bias,   # ABOVE / BELOW / AT
        "time":        _time.strftime("%Y-%m-%d %H:%M"),
        "outcome":     "OPEN",
        "pnl_pct":     0,
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
        losses = len([s for s in trades if s["outcome"] == "LOSS"])
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

            # Auto-close scalp trades after 4 hours
            if t.get("trade_type") == "scalp":
                open_time = t.get("open_timestamp", _time.time())
                if _time.time() - open_time > 14400:  # 4 hours
                    pnl_pct = (price - entry) / entry * 100 * LEVERAGE if direction == "LONG" else (entry - price) / entry * 100 * LEVERAGE
                    t["pnl_pct"]  = round(pnl_pct, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * pnl_pct / 100, 2)
                    # Reclassify by actual PnL — mirrors backtester logic
                    if t["pnl_usdt"] > 0:
                        t["status"] = "WIN"
                    elif t["pnl_usdt"] < 0:
                        t["status"] = "LOSS"
                    else:
                        t["status"] = "BREAKEVEN"
                    closed_now.append(t)
                    continue

            if direction == "LONG":
                pnl_pct = (price - entry) / entry * 100 * LEVERAGE

                # Check trailing stop first (only after TP1 hit)
                trailing_stop = None
                if t.get("tp1_hit"):
                    sig = active_signals.get(t["symbol"])
                    if sig and sig.get("trailing_stop"):
                        trailing_stop = sig["trailing_stop"]

                if trailing_stop and price <= trailing_stop:
                    t["status"]   = "WIN"   # trailing stop after TP1 = profit
                    t["pnl_pct"]  = round((trailing_stop - entry) / entry * 100 * LEVERAGE, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price <= t["sl"]:
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

                # Check trailing stop first (only after TP1 hit)
                trailing_stop = None
                if t.get("tp1_hit"):
                    sig = active_signals.get(t["symbol"])
                    if sig and sig.get("trailing_stop"):
                        trailing_stop = sig["trailing_stop"]

                if trailing_stop and price >= trailing_stop:
                    t["status"]   = "WIN"   # trailing stop after TP1 = profit
                    t["pnl_pct"]  = round((entry - trailing_stop) / entry * 100 * LEVERAGE, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price >= t["sl"]:
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
    closed_t    = [t for t in trades if t["status"] in ("WIN","LOSS","EXPIRED","BREAKEVEN")]
    scalp_open  = [t for t in open_t if t.get("trade_type") == "scalp"]
    swing_open  = [t for t in open_t if t.get("trade_type") != "scalp"]
    total_pnl   = sum([t["pnl_usdt"] for t in trades])
    wins        = len([t for t in closed_t if t["status"] == "WIN"])
    losses      = len([t for t in closed_t if t["status"] == "LOSS"])
    return open_t, closed_t, total_pnl, wins, losses, scalp_open, swing_open


def classify_signal(df, direction, score, btc_change_24h=None):
    """
    Classify signal as MOMENTUM or REVERSAL.
    MOMENTUM: breakout, higher highs/lows, outperforming BTC, or strong price action
    REVERSAL: volume exhaustion + extreme RSI
    """
    try:
        close  = df["close"].values
        volume = df["volume"].values
        high   = df["high"].values
        low    = df["low"].values

        if len(close) < 20:
            return "REVERSAL", "LOW", "Insufficient data"

        reasons      = []
        momentum_pts = 0

        # 1. Breakout / Breakdown
        resistance_20 = max(high[-21:-1])
        support_20    = min(low[-21:-1])
        breakout_vol  = volume[-1] > volume[-5:].mean() * 1.2

        if direction == "LONG":
            if close[-1] > resistance_20 and breakout_vol:
                momentum_pts += 2
                reasons.append("breakout above $" + "{:.4f}".format(resistance_20))
        else:
            if close[-1] < support_20 and breakout_vol:
                momentum_pts += 2
                reasons.append("breakdown below $" + "{:.4f}".format(support_20))

        # 2. Higher Highs / Lower Lows structure
        if len(high) >= 15:
            swing_highs, swing_lows = [], []
            for i in range(2, min(15, len(high)-2)):
                if high[-i] > high[-i-1] and high[-i] > high[-i+1]:
                    swing_highs.append(high[-i])
                if low[-i] < low[-i-1] and low[-i] < low[-i+1]:
                    swing_lows.append(low[-i])
            if direction == "LONG" and len(swing_highs) >= 2 and len(swing_lows) >= 2:
                if swing_highs[0] > swing_highs[1] and swing_lows[0] > swing_lows[1]:
                    momentum_pts += 2
                    reasons.append("higher highs + higher lows")
            elif direction == "SHORT" and len(swing_highs) >= 2 and len(swing_lows) >= 2:
                if swing_highs[0] < swing_highs[1] and swing_lows[0] < swing_lows[1]:
                    momentum_pts += 2
                    reasons.append("lower highs + lower lows")

        # 3. Relative strength vs BTC
        if btc_change_24h is not None:
            coin_chg = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
            rs = coin_chg - btc_change_24h
            if direction == "LONG" and rs > 3:
                momentum_pts += 1
                reasons.append("outperforming BTC by +" + str(round(rs, 1)) + "%")
            elif direction == "SHORT" and rs < -3:
                momentum_pts += 1
                reasons.append("underperforming BTC by " + str(round(rs, 1)) + "%")

        # 4. ROC + volume
        roc = (close[-1] - close[-5]) / close[-5] * 100
        vol_up = volume[-1] > volume[-5:].mean() * 1.1
        if (direction == "LONG" and roc > 1.0 and vol_up) or            (direction == "SHORT" and roc < -1.0 and vol_up):
            momentum_pts += 1
            reasons.append("accelerating with volume")

        # Reversal checks
        vol_ma = volume[-20:].mean() if len(volume) >= 20 else volume.mean()
        if direction == "LONG":
            reversal_setup = close[-1] < close[-5] and volume[-1] < vol_ma * 0.7
        else:
            reversal_setup = close[-1] > close[-5] and volume[-1] < vol_ma * 0.7

        gains  = [max(close[j]-close[j-1], 0) for j in range(max(1,len(close)-14), len(close))]
        losses = [max(close[j-1]-close[j], 0) for j in range(max(1,len(close)-14), len(close))]
        avg_g  = sum(gains)/max(len(gains),1)
        avg_l  = sum(losses)/max(len(losses),1)
        cur_rsi = 100 - (100/(1+(avg_g/avg_l))) if avg_l > 0 else 100
        extreme_rsi = cur_rsi < 25 if direction == "LONG" else cur_rsi > 75
        reversal_score = sum([reversal_setup, extreme_rsi])

        if momentum_pts >= 2:
            return "MOMENTUM", ("HIGH" if momentum_pts >= 4 else "MEDIUM"), " + ".join(reasons)
        elif reversal_score >= 2 or (reversal_score >= 1 and score >= 9):
            return "REVERSAL", ("HIGH" if reversal_score == 2 else "MEDIUM"),                    ("volume exhaustion" if reversal_setup else "") + (" extreme RSI" if extreme_rsi else "")
        else:
            return "WEAK", "LOW", "no breakout or reversal confirmation"
    except Exception:
        return "REVERSAL", "LOW", "classification error"

def _dow_confirms_short(df: "pd.DataFrame") -> tuple:
    """
    Runs detect_dow_phase on a 4H dataframe and returns whether
    Dow Theory confirms a SHORT entry (distribution or participation short).

    Returns (confirmed: bool, phase: str, confidence: str, reason: str)

    Confirmed SHORT when:
      - Phase is DISTRIBUTION (smart money exiting) → strong confirmation
      - Phase is PARTICIPATION with SHORT direction (downtrend participation) → moderate
      - Phase is UNCLEAR → allow SHORT (don't block, just neutral)

    Block SHORT when:
      - Phase is ACCUMULATION (smart money still buying) → price may continue up
      - Phase is PARTICIPATION in uptrend → move likely has more upside
    """
    try:
        from signal_engine import detect_dow_phase
        dow = detect_dow_phase(df, direction="SHORT")
        phase = dow["phase"]
        conf  = dow["confidence"]

        if phase == "DISTRIBUTION":
            return True, phase, conf, "DISTRIBUTION confirmed — smart money exiting"
        elif phase == "PARTICIPATION":
            # SHORT direction participation = downtrend in progress = good SHORT
            return True, phase, conf, "PARTICIPATION (downtrend) — momentum SHORT"
        elif phase == "UNCLEAR":
            return True, phase, conf, "UNCLEAR — allowing SHORT (neutral)"
        elif phase == "ACCUMULATION":
            # Smart money still buying — SHORT likely premature
            return False, phase, conf, "ACCUMULATION — smart money still buying, SHORT risky"
        else:
            return True, phase, conf, ""
    except Exception as e:
        log.warning("Dow SHORT confirm error: " + str(e))
        return True, "UNCLEAR", "LOW", ""  # fail open


def _two_candle_confirm(df: "pd.DataFrame", direction: str) -> bool:
    """
    Returns True if the last 2 CLOSED candles confirm the signal direction.
    Prevents firing on coins that reversed immediately after the signal candle.
    LONG: last 2 closes higher than their opens (bullish bodies)
    SHORT: last 2 closes lower than their opens (bearish bodies)
    Uses candles [-3] and [-2] — not the current forming candle [-1].
    """
    try:
        import pandas_ta as _ta
        o = df["open"].values
        c = df["close"].values
        if direction == "LONG":
            return c[-3] > o[-3] and c[-2] > o[-2]
        else:
            return c[-3] < o[-3] and c[-2] < o[-2]
    except Exception:
        return True  # fail open — don't block on error

# ─── SIGNAL FORMATTING ─────────────────────────────────────────────────────────
def get_recent_perf(n: int = 10) -> str:
    """Returns a one-liner like 'Recent 10: 7W 3L (70%)' for signal messages."""
    try:
        history = load_json(HISTORY_FILE, [])
        closed  = [s for s in history if s.get("outcome") in ("WIN","LOSS")][-n:]
        if len(closed) < 3:
            return ""
        wins    = len([s for s in closed if s.get("outcome") == "WIN"])
        losses  = len(closed) - wins
        wr      = round(wins / len(closed) * 100, 1)
        return f"Recent {len(closed)}: {wins}W {losses}L ({wr}%)"
    except Exception:
        return ""


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
    # Sanitise type_reason — strip chars that break Telegram Markdown parser
    type_reason = r.get("type_reason", "") or ""
    type_reason = type_reason.replace("`", "").replace("*", "").replace("_", "").replace("[", "").replace("]", "")
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
    
    # Grade, Relative Strength, Fear & Greed
    grade       = r.get("grade", "B")
    grade_emoji = {"A": "🟣", "B": "🟢", "C": "🟡"}.get(grade, "🟢")
    grade_label = {"A": "GRADE A — Early Setup", "B": "GRADE B — Confirmed", "C": "GRADE C — Late Entry"}.get(grade, "GRADE B")
    grade_warn  = " ⚠ Consider skipping" if grade == "C" else ""
    rs      = r.get("relative_strength", 0)
    rs_str  = (("+" if rs >= 0 else "") + str(rs) + "% vs BTC") if rs != 0 else ""
    fg_mod  = get_fear_greed_signal_modifier(r["direction"])
    fg_note = fg_mod.get("note", "")
    fg_val  = fear_greed_cache.get("value")
    fg_lbl  = fear_greed_cache.get("label", "")
    msg  = emoji + " *" + r["label"] + " Signal | " + datetime.now().strftime("%H:%M UTC") + "*\n"
    msg += grade_emoji + " *" + grade_label + "*" + grade_warn + "\n"
    # Supertrend display
    st_val  = r.get("supertrend", 0)
    st_dir  = r.get("supertrend_dir", 1)
    st_flip = r.get("supertrend_flip", False)
    if st_val and st_val > 0:
        st_emoji = "\U0001f7e2" if st_dir == 1 else "\U0001f534"
        st_label = "BULLISH" if st_dir == 1 else "BEARISH"
        st_flip_tag = " \u26a1 just flipped" if st_flip else ""
        msg += st_emoji + " Supertrend: *" + st_label + "* `$" + "{:.4f}".format(st_val) + "`" + st_flip_tag + "\n"

    dow_phase = r.get("dow_phase", "")
    dow_conf  = r.get("dow_confidence", "")
    if dow_phase and dow_phase not in ("UNCLEAR", ""):
        dow_emoji = {
            "ACCUMULATION":  "\U0001f7e3",
            "PARTICIPATION": "\U0001f7e2",
            "DISTRIBUTION":  "\U0001f7e1",
        }.get(dow_phase, "\u26aa")
        msg += dow_emoji + " Dow: *" + dow_phase + "* (" + dow_conf + ")\n"
    # VWAP display
    vwap_val  = r.get("vwap", 0)
    vwap_dist = r.get("vwap_dist_pct", 0)
    vwap_bias = r.get("vwap_bias", "")
    vwap_str  = r.get("vwap_strength", "")
    if vwap_val and vwap_val > 0 and vwap_bias:
        vwap_emoji = "\U0001f7e2" if vwap_bias == "BELOW" else "\U0001f534" if vwap_bias == "ABOVE" else "\u26aa"
        dist_str   = ("{:+.1f}".format(vwap_dist)) + "% vs VWAP"
        msg += vwap_emoji + " VWAP: `$" + "{:.4f}".format(vwap_val) + "` | " + dist_str + " (" + vwap_str + ")\n"
    msg += conf_emoji + " *" + r["direction"] + "* | " + r["confidence"] + " | Score: `" + str(r["score"]) + "`\n"
    msg += "📊 Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n"
    msg += type_emoji + " Type: *" + type_label + "* | " + type_reason + "\n"
    if rs_str:
        msg += "📈 Relative Strength: `" + rs_str + "`\n"
    if fg_val is not None:
        fg_emoji = "😱" if fg_val <= 25 else "😤" if fg_val >= 75 else "😐"
        msg += fg_emoji + " Fear & Greed: `" + str(fg_val) + " — " + fg_lbl + "`"
        if fg_note:
            msg += " | " + fg_note
        msg += "\n"

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
    msg += "  Size: $" + str(pos_usdt) + " | Contracts: " + str(contracts) + "\n"
    # Dynamic leverage suggestion
    sug_lev  = r.get("suggested_leverage")
    max_lev  = r.get("max_safe_leverage")
    lev_rsn  = r.get("leverage_reason", "")
    if sug_lev and sug_lev != LEVERAGE:
        lev_emoji = "\U0001f7e2" if sug_lev >= 8 else "\U0001f7e1" if sug_lev >= 5 else "\U0001f534"
        msg += "  " + lev_emoji + " Suggested leverage: *" + str(sug_lev) + "x* (max safe: " + str(max_lev) + "x)\n"
        if lev_rsn:
            msg += "  _" + lev_rsn + "_\n"
    msg += "\n"

    msg += "*Key Signals:*\n"
    for s in r["signals"][:6]:
        msg += "  \u2022 " + s + "\n"

    # Volume Profile block
    vp_adjs = r.get("vp_adjustments", [])
    vp_poc   = r.get("vp_poc")
    vp_in_lvn = r.get("vp_in_lvn", False)
    if vp_poc or vp_adjs or vp_in_lvn:
        msg += "\n\U0001f4ca *Volume Profile:*\n"
        if vp_poc:
            msg += "  POC: `$" + "{:.4f}".format(vp_poc) + "` (highest volume level)\n"
        if vp_in_lvn:
            msg += "  \u26a1 Price in LOW VOLUME ZONE — fast move expected\n"
        for adj in vp_adjs[:3]:
            msg += "  \u2022 " + adj + "\n"

    # Exhaustion warning display
    exh = r.get("exhaustion", {})
    if exh.get("short_opp"):
        msg += "\n\U0001f6a8 *EXHAUSTION ALERT: SHORT OPPORTUNITY*\n"
        msg += "_" + exh.get("reason","") + "_\n"
    elif exh.get("block_long") and r.get("direction") == "SHORT":
        msg += "\n\U0001f534 *Note: LONG blocked — reversed to SHORT due to blow-off top*\n"
        msg += "_" + exh.get("reason","") + "_\n"

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
    try:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=text, parse_mode="Markdown")
    except Exception:
        await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=text, parse_mode=None)

async def cmd_coin(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /coin SYMBOL — full on-demand scan with complete analysis.
    Runs Phase 1+2 + Grade + Dow phase + VWAP + Volume Profile + AI verdict.
    Identical depth to /scalp or /swing output.
    """
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "Usage: /coin LINK\n"
            "Scans any coin with full analysis — Grade, Dow phase, VWAP, VP, AI verdict."
        )
        return

    coin   = args[0].upper()
    symbol = coin + "/USDT:USDT"
    await update.message.reply_text("\U0001f50e Scanning " + coin + " (full analysis)...")

    try:
        if is_blacklisted(symbol):
            await update.message.reply_text("\U0001f6ab " + coin + " is blacklisted. Remove with: /blacklist remove " + coin)
            return

        # Temporarily add to labels if not in watchlist
        original_label = COIN_LABELS.get(symbol)
        if not original_label:
            COIN_LABELS[symbol] = coin

        r = await run_full_pipeline(symbol, fetch_ohlcv, COIN_LABELS, ai_client, exchange, news_context)

        if not original_label and symbol in COIN_LABELS:
            del COIN_LABELS[symbol]

        if not r or "direction" not in r:
            regime     = r.get("regime", {}) if r else {}
            regime_name = regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else str(regime)
            await update.message.reply_text("\u26a0 No signal for " + coin + " right now.\nRegime: " + regime_name)
            return

        # ── Signal type classification ────────────────────────────────────────
        if r["direction"] not in (None, "NEUTRAL"):
            try:
                df_c = r.get("df_4h")
                if df_c is not None:
                    st, tc, tr = classify_signal(df_c, r["direction"], r.get("abs_score", 0), None)
                    r["signal_type"] = st
                    r["type_conf"]   = tc
                    r["type_reason"] = tr
            except Exception:
                r["signal_type"] = "MOMENTUM"

        # ── Grade + Dow phase + VWAP ──────────────────────────────────────────
        if r["direction"] not in (None, "NEUTRAL"):
            try:
                from signal_engine import grade_signal
                df_grade  = r.get("df_4h")
                df_4h_htf = fetch_ohlcv(symbol, "4h", 100)
                if df_grade is not None:
                    gr = grade_signal(
                        df_grade, df_4h_htf,
                        r["direction"],
                        r.get("abs_score", 0),
                        r.get("rsi_4h", 50),
                        r.get("funding", 0),
                    )
                    r["grade"]          = gr["grade"]
                    r["grade_score"]    = gr["grade_score"]
                    r["grade_reasons"]  = gr["reasons"]
                    r["grade_warnings"] = gr["warnings"]
                    r["dow_phase"]      = gr.get("dow_phase", "UNCLEAR")
                    r["dow_confidence"] = gr.get("dow_confidence", "LOW")
                    r["dow_signals"]    = gr.get("dow_signals", [])
                    r["vwap"]           = gr.get("vwap", 0)
                    r["vwap_dist_pct"]  = gr.get("vwap_dist_pct", 0)
                    r["vwap_bias"]      = gr.get("vwap_bias", "AT")
                    r["vwap_strength"]  = gr.get("vwap_strength", "WEAK")
            except Exception as e:
                log.warning("cmd_coin grade error: " + str(e))
                r["grade"] = "B"

            # Grade / type guard
            if r.get("signal_type") == "WEAK" and r.get("grade") == "A":
                r["grade"] = "B"

        # ── Volume Profile ────────────────────────────────────────────────────
        try:
            from signal_engine import compute_volume_profile
            df_vp = r.get("df_4h")
            if df_vp is not None and len(df_vp) >= 20:
                vp = compute_volume_profile(df_vp)
                r["vp_poc"]          = vp.get("poc")
                r["vp_hvn_above"]    = vp.get("hvn_above")
                r["vp_hvn_below"]    = vp.get("hvn_below")
                r["vp_in_lvn"]       = vp.get("in_lvn", False)
                r["vp_adjustments"]  = []
                if vp.get("in_lvn") and r.get("grade") == "B":
                    r["grade"] = "A"
                    r.setdefault("grade_reasons", []).append("In LVN — fast move expected")
        except Exception as e:
            log.warning("cmd_coin VP error: " + str(e))

        # ── Relative strength vs BTC ──────────────────────────────────────────
        try:
            btc_val = _btc_change_cache.get("value")
            if btc_val is not None and symbol != "BTC/USDT:USDT":
                df_rs = r.get("df_4h")
                if df_rs is not None and len(df_rs) >= 24:
                    coin_chg = (float(df_rs["close"].iloc[-1]) - float(df_rs["close"].iloc[-24])) / float(df_rs["close"].iloc[-24]) * 100
                    r["relative_strength"] = round(coin_chg - btc_val, 2)
        except Exception:
            pass

        # ── VP-adjusted levels ────────────────────────────────────────────────
        from risk_manager import calc_levels_v2
        atr_pct = (r["atr"] / r["price"] * 100) if r["price"] > 0 else 1.0
        sl, tp1, tp2, vp_meta = calc_levels_v2(r["direction"], r["price"], r["atr"], atr_pct, r.get("df_4h"))
        r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
        r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
        r["vp_in_lvn"]      = vp_meta.get("vp_in_lvn", r.get("vp_in_lvn", False))

        ob_bias, ob_ratio   = get_order_book_bias(symbol)
        pos_usdt, contracts = calc_position_size(r["price"], sl)

        # ── Build message — full signal format identical to auto signals ───────
        regime_data = r.get("regime") if isinstance(r.get("regime"), dict) else {}
        regime_name = regime_data.get("regime", "UNKNOWN")
        adx_val     = round(float(regime_data.get("adx", 0)), 1)

        header  = "\U0001f50e *On-Demand Scan: " + coin + "* | " + datetime.now().strftime("%H:%M UTC") + "\n"
        header += "Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n"
        header += "1H Score: `" + str(r["score_4h"]) + "` | 4H Score: `" + str(r["score_1d"]) + "`\n"
        header += "RSI 1H: `" + str(round(r["rsi_4h"], 1)) + "` | RSI 4H: `" + str(round(r["rsi_1d"], 1)) + "`\n"
        header += "TF Agreement: " + ("\u2705 Yes" if r["tf_agree"] else "\u26a0 No") + "\n\n"

        full_msg = header + format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
        full_msg += format_ai_block(r.get("ai_result", {}))

        try:
            await update.message.reply_text(full_msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(full_msg, parse_mode=None)

    except Exception as e:
        await update.message.reply_text(
            "Could not scan " + coin + ".\n"
            "Make sure it has a USDT futures pair on Bitget.\nError: " + str(e)
        )


async def scalp_scan_coins():
    """Scan all coins using 15m+1H via Phase 1+2 pipeline for scalp trades."""
    scalp_results = []
    coins   = list(COINS)
    batches = [coins[i:i+BATCH_SIZE] for i in range(0, len(coins), BATCH_SIZE)]
    for batch in batches:
        batch_tasks = []
        for symbol in batch:
            if is_blacklisted(symbol):
                continue
            batch_tasks.append(_scalp_analyze_one(symbol))
        results = await asyncio.gather(*batch_tasks)
        for r in results:
            if r is not None:
                scalp_results.append(r)
        await asyncio.sleep(REQUEST_DELAY * BATCH_SIZE)
    return scalp_results

async def _scalp_analyze_one(symbol):
    """Run Phase 1+2 pipeline on a single coin using scalp timeframes (15m+1H)."""
    try:
        r = await run_full_pipeline(
            symbol, fetch_ohlcv_scalp, COIN_LABELS,
            ai_client, exchange, news_context
        )
        if not r or "direction" not in r:
            return None
        # ── Classify MOMENTUM vs REVERSAL vs WEAK ────────────────────────
        if r["direction"] not in (None, "NEUTRAL"):
            try:
                df_classify = r.get("df_4h")
                if df_classify is not None:
                    signal_type, type_conf, type_reason = classify_signal(
                        df_classify, r["direction"], r.get("abs_score", 0), None
                    )
                    r["signal_type"] = signal_type
                    r["type_conf"]   = type_conf
                    r["type_reason"] = type_reason
            except Exception as e:
                log.warning("classify_signal (scalp) failed for " + symbol + ": " + str(e))
                r["signal_type"] = "MOMENTUM"
        # ── Grade signal ──────────────────────────────────────────────────
        if r["direction"] not in (None, "NEUTRAL"):
            try:
                from signal_engine import grade_signal
                df_grade   = r.get("df_4h")  # for scalp this is 15m data
                df_4h_grade = fetch_ohlcv(symbol, "1h", 100)  # use 1H as HTF for scalp grade
                if df_grade is not None:
                    grade_result = grade_signal(
                        df_grade, df_4h_grade,
                        r["direction"],
                        r.get("abs_score", 0),
                        r.get("rsi_4h", 50),
                        r.get("funding", 0),
                    )
                    r["grade"]          = grade_result["grade"]
                    r["grade_score"]    = grade_result["grade_score"]
                    r["grade_reasons"]  = grade_result["reasons"]
                    r["grade_warnings"] = grade_result["warnings"]
                    r["dow_phase"]      = grade_result.get("dow_phase", "UNCLEAR")
                    r["dow_confidence"] = grade_result.get("dow_confidence", "LOW")
                    r["dow_signals"]    = grade_result.get("dow_signals", [])
                    r["vwap"]           = grade_result.get("vwap", 0)
                    r["vwap_dist_pct"]  = grade_result.get("vwap_dist_pct", 0)
                    r["vwap_bias"]      = grade_result.get("vwap_bias", "AT")
                    r["vwap_strength"]  = grade_result.get("vwap_strength", "WEAK")
            except Exception as e:
                log.warning("Scalp grade failed for " + symbol + ": " + str(e))
                r["grade"] = "B"

        # ── Volume Profile ────────────────────────────────────────────────
        try:
            from signal_engine import compute_volume_profile
            df_for_vp = r.get("df_4h")  # 15m data for scalp VP
            if df_for_vp is not None and len(df_for_vp) >= 20:
                vp = compute_volume_profile(df_for_vp)
                r["vp_poc"]       = vp.get("poc")
                r["vp_hvn_above"] = vp.get("hvn_above")
                r["vp_hvn_below"] = vp.get("hvn_below")
                r["vp_in_lvn"]    = vp.get("in_lvn", False)
                if vp.get("in_lvn") and r.get("grade") == "B":
                    r["grade"] = "A"
                    r.setdefault("grade_reasons", []).append("In LVN — fast scalp move expected")
        except Exception as e:
            log.warning("Scalp VP failed for " + symbol + ": " + str(e))

        # ── Relative Strength ─────────────────────────────────────────────
        try:
            btc_val = _btc_change_cache.get("value")
            if btc_val is not None and symbol != "BTC/USDT:USDT":
                df_rs = r.get("df_4h")
                if df_rs is not None and len(df_rs) >= 24:
                    coin_chg = (float(df_rs["close"].iloc[-1]) - float(df_rs["close"].iloc[-24])) / float(df_rs["close"].iloc[-24]) * 100
                    r["relative_strength"] = round(coin_chg - btc_val, 2)
        except Exception:
            pass

        # ── Grade / signal_type contradiction guard ──────────────────────
        if r.get("signal_type") == "WEAK" and r.get("grade") == "A":
            r["grade"] = "B"
            r.setdefault("grade_warnings", []).append("Grade capped at B — signal type is WEAK")

        # Tag as scalp timeframes for display
        r["tf_labels"] = ("15m", "1H")
        r["trade_type"] = "scalp"
        return r
    except Exception as e:
        log.error("Scalp scan error " + symbol + ": " + str(e))
        return None


async def cmd_scalp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Scalp scan using 15m+1H via Phase 1+2 — single best signal, paper trade auto-closes in 24hrs."""
    await update.message.reply_text("⚡ Scalp scan (15m+1H via Phase 1+2)... finding best setup...")
    try:
        scalp_results = await scalp_scan_coins()
        actionable = [r for r in scalp_results if r and r.get("direction") != "NEUTRAL"
                      and r.get("confidence") in ("HIGH","MEDIUM")
                      and r.get("signal_type", "REVERSAL") == "MOMENTUM"]
        if not actionable:
            await update.message.reply_text("⚡ No strong scalp signals right now. Try again in 15 minutes or use /swing.")
            return
        best = max(actionable, key=lambda x: x["abs_score"])

        # Use risk_gate for proper ATR-adjusted levels
        history = load_json(HISTORY_FILE, [])
        risk = risk_gate(
            symbol=best["symbol"],
            direction=best["direction"],
            price=best["price"],
            atr=best["atr"],
            quality_score=best.get("abs_score", 0),
            active_signals=active_signals,
            trade_history=history,
        )
        if not risk["approved"]:
            await update.message.reply_text("⚡ Best scalp signal blocked by risk gate: " + risk["reject_reason"])
            return

        sl        = risk["sl"]
        tp1       = risk["tp1"]
        tp2       = risk["tp2"]
        # Apply VP adjustment to scalp levels
        try:
            from risk_manager import calc_levels_v2
            _atr_pct = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
            sl, tp1, tp2, _ = calc_levels_v2(best["direction"], best["price"], best["atr"], _atr_pct, best.get("df_4h"))
        except Exception:
            pass
        pos_usdt  = risk["position_usdt"]
        contracts = risk["contracts"]
        ob_bias, ob_ratio = get_order_book_bias(best["symbol"])

        record_signal(best["symbol"], best["direction"], best["price"], sl, tp1, tp2,
                      best["confidence"], trade_type="scalp",
                      signal_type=best.get("signal_type", "MOMENTUM"),
                      grade=best.get("grade", "B"),
                      dow_phase=best.get("dow_phase", "UNCLEAR"),
                      vwap_bias=best.get("vwap_bias", "AT"))
        if paper_mode:
            existing = load_json(PAPER_FILE, [])
            already_open = any(t["symbol"] == best["symbol"] and t["status"] == "OPEN" and t.get("trade_type") == "scalp" for t in existing)
            if not already_open:
                open_paper_trade(best["symbol"], best["direction"], best["price"], sl, tp1, tp2,
                                 best["confidence"], trade_type="scalp",
                                 signal_type=best.get("signal_type", "MOMENTUM"))

        active_signals[best["symbol"]] = {
            "direction": best["direction"], "entry": best["price"],
            "sl": sl, "tp1": tp1, "tp2": tp2,
            "tp1_hit": False, "atr": best["atr"],
            "time": _time.time(),
            "trailing_extreme": best["price"],
            "trailing_stop": None,
            "trade_type": "scalp",
        }

        regime_data = best.get("regime") or {}
        regime_name = regime_data.get("regime", "UNKNOWN") if isinstance(regime_data, dict) else str(regime_data)
        header  = "⚡ *Scalp Signal (15m+1H) | " + datetime.now().strftime("%H:%M UTC") + "*\n"
        header += "Regime: `" + regime_name + "` | Act within 15 min | Auto-closes in 24hrs\n\n"
        msg = header + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
        msg += format_ai_block(best.get("ai_result", {}))
        await update.message.reply_text(msg, parse_mode="Markdown")
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
            from risk_manager import calc_levels_v2
            _atr_pct = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
            sl, tp1, tp2, _ = calc_levels_v2(best["direction"], best["price"], best["atr"], _atr_pct, best.get("df_4h"))
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            header  = "\U0001f319 *Swing Signal (1H+4H) | " + datetime.now().strftime("%H:%M UTC") + "*\n"
            header += "Overnight setup — hold until TP or SL\n\n"
            _swing_msg = header + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
            try:
                await update.message.reply_text(_swing_msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(_swing_msg, parse_mode=None)
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
        "/alpha \u2014 Scan Binance Alpha tokens on Bitget ($500k+ vol)\n"
        "/coupon \u2014 Toggle auto-scanner on 50 USDT voucher pairs (PI/DOGE/BGB/XRP/SUI/PEPE/SHIB)\n"
        "/btcstatus \u2014 Check BTC circuit breaker state\n"
        "/flip COIN LONG 5 high \u2014 Hunt flip entries (high/medium/low confidence)\n"
        "/flipping \u2014 Show active flip scalps\n"
        "/watch BTC SHORT 95000 swing high \u2014 Monitor trade (high/medium/low confidence)\n"
        "/unwatch BTC \u2014 Stop monitoring\n"
        "/watching \u2014 View monitored trades\n"
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
            from risk_manager import calc_levels_v2
            _atr_pct = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
            sl, tp1, tp2, _ = calc_levels_v2(best["direction"], best["price"], best["atr"], _atr_pct, best.get("df_4h"))
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            _scan_msg = "\U0001f3c6 *Best Signal: " + best["label"] + "*\n\n" + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
            try:
                await update.message.reply_text(_scan_msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(_scan_msg, parse_mode=None)
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
            from risk_manager import calc_levels_v2
            _atr_pct = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
            sl, tp1, tp2, _ = calc_levels_v2(best["direction"], best["price"], best["atr"], _atr_pct, best.get("df_4h"))
            ob_bias, ob_ratio = get_order_book_bias(best["symbol"])
            pos_usdt, contracts = calc_position_size(best["price"], sl)
            _best_msg = format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
            try:
                await update.message.reply_text(_best_msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(_best_msg, parse_mode=None)
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
            hours_left = max(0, round((14400 - (_time.time() - open_time)) / 3600, 1))  # 4h window
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
    # Extended history — Grade, Dow phase, coin performance
    history  = load_json(HISTORY_FILE, [])
    closed   = [s for s in history if s.get("outcome") in ("WIN","LOSS","BREAKEVEN")]
    total_cl = len(closed)

    if total_cl > 0:
        # Grade breakdown
        for grade in ("A", "B", "C"):
            g_trades = [s for s in closed if s.get("grade") == grade]
            if g_trades:
                g_wins = len([s for s in g_trades if s.get("outcome") == "WIN"])
                g_wr   = round(g_wins / len(g_trades) * 100, 1)
                g_avg  = round(sum(s.get("pnl_pct", 0) for s in g_trades) / len(g_trades), 1)
                # Store for display below
                pass  # handled inline

        # Dow phase breakdown
        for phase in ("ACCUMULATION", "PARTICIPATION", "DISTRIBUTION", "UNCLEAR"):
            p_trades = [s for s in closed if s.get("dow_phase") == phase]

        # Best coins
        coin_stats = {}
        for s in closed:
            lbl = s.get("label", s.get("symbol", "?").split("/")[0])
            if lbl not in coin_stats:
                coin_stats[lbl] = {"wins": 0, "total": 0, "pnl": 0}
            coin_stats[lbl]["total"] += 1
            coin_stats[lbl]["pnl"]   += s.get("pnl_pct", 0)
            if s.get("outcome") == "WIN":
                coin_stats[lbl]["wins"] += 1

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
    msg += "  Open signals: " + str(stats["open"]) + "\n"

    # ── Grade breakdown ───────────────────────────────────────────────────────
    if total_cl >= 5:
        msg += "\n\U0001f3af *By Grade:*\n"
        for grade, emoji in (("A", "\U0001f7e3"), ("B", "\U0001f7e2"), ("C", "\U0001f7e1")):
            g_trades = [s for s in closed if s.get("grade") == grade]
            if g_trades:
                g_wins = len([s for s in g_trades if s.get("outcome") == "WIN"])
                g_wr   = round(g_wins / len(g_trades) * 100, 1)
                g_avg  = round(sum(s.get("pnl_pct", 0) for s in g_trades) / len(g_trades), 1)
                msg += "  " + emoji + " Grade " + grade + ": " + str(len(g_trades)) + " trades | " + str(g_wr) + "% WR | Avg " + ("{:+.1f}".format(g_avg)) + "%\n"

    # ── Dow phase breakdown ───────────────────────────────────────────────────
    if total_cl >= 5:
        phase_data = []
        for phase, emoji in (("ACCUMULATION","\U0001f7e3"),("PARTICIPATION","\U0001f7e2"),("DISTRIBUTION","\U0001f7e1")):
            p_trades = [s for s in closed if s.get("dow_phase") == phase]
            if p_trades:
                p_wins = len([s for s in p_trades if s.get("outcome") == "WIN"])
                p_wr   = round(p_wins / len(p_trades) * 100, 1)
                phase_data.append((emoji, phase[:4], len(p_trades), p_wr))
        if phase_data:
            msg += "\n\U0001f4ca *By Dow Phase:*\n"
            for emoji, name, total_p, wr_p in phase_data:
                msg += "  " + emoji + " " + name + ": " + str(total_p) + " trades | " + str(wr_p) + "% WR\n"

    # ── Top performing coins ──────────────────────────────────────────────────
    if total_cl >= 5 and coin_stats:
        top = sorted(
            [(lbl, v) for lbl, v in coin_stats.items() if v["total"] >= 2],
            key=lambda x: x[1]["wins"] / x[1]["total"],
            reverse=True
        )[:5]
        if top:
            msg += "\n\U0001f3c6 *Top Coins (min 2 trades):*\n"
            for lbl, v in top:
                wr_c = round(v["wins"] / v["total"] * 100, 1)
                msg += "  \u2022 *" + lbl + "* " + str(v["total"]) + " trades | " + str(wr_c) + "% WR\n"

    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg, parse_mode=None)

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

            # ── Detect dominant regime across scanned coins ───────────────
            regime_votes = [r.get("regime", {}) for r in results if isinstance(r.get("regime"), dict)]
            trending_count = sum(1 for rv in regime_votes if rv.get("regime") in ("TRENDING_UP", "TRENDING_DOWN"))
            ranging_count  = sum(1 for rv in regime_votes if rv.get("regime") == "RANGING")
            market_ranging = ranging_count > trending_count  # majority RANGING = scalp mode

            # ── RANGING market: auto-fire scalp instead of swing ─────────
            if market_ranging and not actionable:
                log.info("Market is RANGING — triggering auto-scalp scan")
                scalp_results  = await scalp_scan_coins()
                scalp_actionable = [r for r in scalp_results
                                    if r and r.get("direction") != "NEUTRAL"
                                    and r.get("confidence") in ("HIGH", "MEDIUM")]
                if scalp_actionable:
                    best_scalp = max(scalp_actionable, key=lambda x: x["abs_score"])
                    key = best_scalp["symbol"] + "_SCALP_" + best_scalp["direction"]
                    last = last_signal.get(key)
                    if last and now - last["time"] < 14400:
                        log.info("Auto-scalp cooldown active for " + best_scalp["label"])
                    elif not is_blacklisted(best_scalp["symbol"]):
                        last_signal[key] = {"score": best_scalp["abs_score"], "time": now}
                        history  = load_json(HISTORY_FILE, [])
                        risk = risk_gate(
                            symbol=best_scalp["symbol"],
                            direction=best_scalp["direction"],
                            price=best_scalp["price"],
                            atr=best_scalp["atr"],
                            quality_score=best_scalp.get("abs_score", 0),
                            active_signals=active_signals,
                            trade_history=history,
                        )
                        if risk["approved"]:
                            # VP-adjusted levels for auto-scalp
                            try:
                                from risk_manager import calc_levels_v2 as _clv2
                                _ap = (best_scalp["atr"] / best_scalp["price"] * 100) if best_scalp["price"] > 0 else 1.0
                                sl, tp1, tp2, _vp_meta = _clv2(best_scalp["direction"], best_scalp["price"], best_scalp["atr"], _ap, best_scalp.get("df_4h"))
                                best_scalp["vp_adjustments"] = _vp_meta.get("vp_adjustments", [])
                                best_scalp["vp_poc"]         = _vp_meta.get("vp_poc") or best_scalp.get("vp_poc")
                                best_scalp["vp_in_lvn"]      = _vp_meta.get("vp_in_lvn", best_scalp.get("vp_in_lvn", False))
                            except Exception:
                                sl  = risk["sl"]
                                tp1 = risk["tp1"]
                                tp2 = risk["tp2"]
                            pos_usdt  = risk["position_usdt"]
                            contracts = risk["contracts"]
                            ob_bias, ob_ratio = get_order_book_bias(best_scalp["symbol"])
                            record_signal(best_scalp["symbol"], best_scalp["direction"],
                                          best_scalp["price"], sl, tp1, tp2,
                                          best_scalp["confidence"], trade_type="scalp",
                                          signal_type=best_scalp.get("signal_type", "MOMENTUM"),
                                          grade=best_scalp.get("grade", "B"),
                                          dow_phase=best_scalp.get("dow_phase", "UNCLEAR"),
                                          vwap_bias=best_scalp.get("vwap_bias", "AT"))
                            if paper_mode:
                                existing = load_json(PAPER_FILE, [])
                                already_open = any(
                                    t["symbol"] == best_scalp["symbol"]
                                    and t["status"] == "OPEN"
                                    for t in existing
                                )
                                if not already_open:
                                    open_paper_trade(
                                        best_scalp["symbol"], best_scalp["direction"],
                                        best_scalp["price"], sl, tp1, tp2,
                                        best_scalp["confidence"], trade_type="scalp",
                                        signal_type=best_scalp.get("signal_type", "MOMENTUM")
                                    )
                            active_signals[best_scalp["symbol"]] = {
                                "direction": best_scalp["direction"], "entry": best_scalp["price"],
                                "sl": sl, "tp1": tp1, "tp2": tp2,
                                "tp1_hit": False, "atr": best_scalp["atr"],
                                "time": _time.time(),
                                "trailing_extreme": best_scalp["price"],
                                "trailing_stop": None,
                                "trade_type": "scalp",
                            }
                            header  = "⚡ *Auto-Scalp Signal (RANGING Market) | " + datetime.now().strftime("%H:%M UTC") + "*\n"
                            header += "Market in consolidation — scalp mode activated\n\n"
                            perf = get_recent_perf()
                            msg = header + format_signal(best_scalp, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                            msg += format_ai_block(best_scalp.get("ai_result", {}))
                            if perf:
                                msg += "\n\U0001f4c8 _" + perf + "_"
                            await send_msg(app, msg)
                            log.info("Auto-scalp sent: " + best_scalp["label"] + " " + best_scalp["direction"])
                        else:
                            log.info("Auto-scalp risk gate blocked: " + risk["reject_reason"])
                else:
                    log.info("RANGING market — no strong scalp signals found")

            # ── Normal swing signals in TRENDING market ───────────────────
            # Check BTC circuit breaker before processing signals
            check_btc_circuit_breaker()
            # Limit to top 3 signals per scan cycle
            actionable = sorted(actionable, key=lambda x: x["abs_score"], reverse=True)[:3]
            for r in actionable:
                key = r["symbol"] + "_" + r["direction"]
                last = last_signal.get(key)
                # Skip if same coin+direction fired within last 4 hours — regardless of score
                if last and now - last["time"] < 14400:
                    continue
                # Skip if coin is blacklisted
                if is_blacklisted(r["symbol"]):
                    log.info("Blacklisted — skipping: " + r["label"])
                    continue
                # 2-candle confirmation — last 2 closed candles must agree with direction
                _df_c = r.get("df_4h")
                if _df_c is not None and not _two_candle_confirm(_df_c, r["direction"]):
                    log.info("2-candle confirm failed: " + r["label"] + " " + r["direction"])
                    continue
                # Block new LONGs if BTC circuit breaker is active
                if r["direction"] == "LONG" and btc_circuit_breaker["active"]:
                    log.info("Circuit breaker blocked LONG: " + r["label"] + " — " + btc_circuit_breaker["reason"])
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
                # VP-adjusted levels — override risk gate ATR levels with VP-snapped levels
                try:
                    from risk_manager import calc_levels_v2 as _clv2
                    _ap = (r["atr"] / r["price"] * 100) if r["price"] > 0 else 1.0
                    sl, tp1, tp2, _vp_meta = _clv2(r["direction"], r["price"], r["atr"], _ap, r.get("df_4h"))
                    r["vp_adjustments"] = _vp_meta.get("vp_adjustments", [])
                    r["vp_poc"]         = _vp_meta.get("vp_poc") or r.get("vp_poc")
                    r["vp_in_lvn"]      = _vp_meta.get("vp_in_lvn", r.get("vp_in_lvn", False))
                except Exception:
                    sl  = risk["sl"]
                    tp1 = risk["tp1"]
                    tp2 = risk["tp2"]
                pos_usdt  = risk["position_usdt"]
                contracts = risk["contracts"]
                ob_bias, ob_ratio = get_order_book_bias(r["symbol"])
               
                # Record signal
                record_signal(r["symbol"], r["direction"], r["price"], sl, tp1, tp2, r["confidence"],
                              signal_type=r.get("signal_type", "MOMENTUM"),
                              grade=r.get("grade", "B"),
                              dow_phase=r.get("dow_phase", "UNCLEAR"),
                              vwap_bias=r.get("vwap_bias", "AT"))

                # Open paper trade — only if no open trade already exists for this symbol
                if paper_mode:
                    existing = load_json(PAPER_FILE, [])
                    already_open = any(t["symbol"] == r["symbol"] and t["status"] == "OPEN" for t in existing)
                    if not already_open:
                        open_paper_trade(r["symbol"], r["direction"], r["price"], sl, tp1, tp2, r["confidence"], signal_type=r.get("signal_type", "MOMENTUM"))
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

                perf = get_recent_perf()
                msg = format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                msg += format_ai_block(r.get("ai_result", {}))
                if perf:
                    msg += "\n\U0001f4c8 _" + perf + "_"
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
                            msg += "Trailing stop set at: $" + "{:.4f}".format(trailing_stop) + " (6% buffer)\n"
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
                                # Use correct timeframe: 4H for swing, 1H for scalp
                                is_scalp_trade = sig.get("trade_type") == "scalp"
                                tf_fresh = "1h" if is_scalp_trade else "4h"
                                df_fresh  = fetch_ohlcv(symbol, tf_fresh, 50)
                                atr_fresh = ta.atr(df_fresh["high"], df_fresh["low"], df_fresh["close"], length=14)
                                new_atr   = float(atr_fresh.iloc[-1])
                                from risk_manager import calc_levels_v2
                                atr_pct = (new_atr / price * 100) if price > 0 else 1.0
                                _, _, new_tp2, _ = calc_levels_v2(direction, price, new_atr, atr_pct, df_fresh)
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

                    # Expire scalps after 4h, swings after 72h, or if paper trade closed
                    trades = load_json(PAPER_FILE, [])
                    trade_closed = any(t["symbol"] == symbol and t["status"] != "OPEN" for t in trades)
                    is_scalp   = sig.get("trade_type") == "scalp"
                    expiry_sec = 14400 if is_scalp else 259200
                    if trade_closed or _time.time() - sig["time"] > expiry_sec:
                        if symbol in active_signals:
                            del active_signals[symbol]
                            log.info("Removed from active_signals: " + label)

                except Exception as e:
                    log.error("Monitor error " + symbol + ": " + str(e))

            # ── Paper trade updates + closed alerts ────────────────────────
            if paper_mode:
                _, closed_now = update_paper_trades()
                for t in closed_now:
                    # Sync outcome back to signal_history.json
                    try:
                        hist = load_json(HISTORY_FILE, [])
                        for h in hist:
                            if (h.get("symbol") == t["symbol"]
                                    and h.get("outcome") == "OPEN"
                                    and h.get("direction") == t["direction"]):
                                h["outcome"] = t["status"]
                                h["pnl_pct"] = t["pnl_pct"]
                                break
                        save_json(HISTORY_FILE, hist)
                    except Exception as e:
                        log.error("History sync error: " + str(e))
                    # Update adaptive thresholds based on outcome
                    try:
                        from signal_engine import update_adaptive_thresholds
                        if t["status"] in ("WIN", "LOSS"):
                            update_adaptive_thresholds(t["status"])
                    except Exception as e:
                        log.warning("Adaptive threshold update error: " + str(e))
                    # Correct emoji for all outcomes
                    if t["status"] == "WIN":
                        emoji = "\U0001f7e2"
                    elif t["status"] == "LOSS":
                        emoji = "\U0001f534"
                    else:
                        emoji = "\U0001f7e1"  # yellow for BREAKEVEN
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




# ─── BINANCE ALPHA SCANNER ────────────────────────────────────────────────────
def fetch_binance_alpha_tokens() -> list:
    """Fetch Binance Alpha token list. Public API — no key needed."""
    import urllib.request as _ur, json as _j
    url = "https://www.binance.com/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
    try:
        req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _ur.urlopen(req, timeout=10) as r:
            data = _j.loads(r.read())
        if not data.get("success") or not data.get("data"):
            return []
        tokens = data["data"]
        tokens.sort(key=lambda x: float(x.get("volume24h", 0) or 0), reverse=True)
        log.info(f"Binance Alpha: fetched {len(tokens)} tokens")
        return tokens
    except Exception as e:
        log.error("Binance Alpha fetch error: " + str(e))
        return []


def refresh_alpha_tokens():
    """
    Cross-references Binance Alpha tokens with Bitget futures.
    Only keeps tokens with $500k+ 24H volume on Bitget (liquid enough to trade).
    Also flags tokens with extreme 24H gains as SHORT candidates.
    """
    global alpha_state
    try:
        alpha_tokens = fetch_binance_alpha_tokens()
        if not alpha_tokens:
            return

        # Fetch all Bitget futures markets once
        try:
            markets = exchange.load_markets()
            bitget_futures = set(markets.keys())
        except Exception as e:
            log.error("Bitget markets fetch error: " + str(e))
            return

        syms, lbls, vols = [], {}, {}
        checked = 0

        for token in alpha_tokens[:ALPHA_MAX_TOKENS]:
            sym_raw = token.get("symbol", "").upper().strip()
            if not sym_raw:
                continue
            candidate = sym_raw + "/USDT:USDT"
            # Skip if already in main COINS list
            if candidate in COINS:
                continue
            if candidate not in bitget_futures:
                continue

            # Volume filter — fetch 24H Bitget ticker for actual volume
            try:
                ticker  = exchange.fetch_ticker(candidate)
                vol_usd = float(ticker.get("quoteVolume", 0) or 0)
                if vol_usd < ALPHA_MIN_VOLUME:
                    log.info(f"Alpha skip {sym_raw}: volume ${vol_usd:,.0f} < ${ALPHA_MIN_VOLUME:,.0f}")
                    continue
                syms.append(candidate)
                lbls[candidate] = sym_raw
                vols[candidate] = vol_usd
                checked += 1
            except Exception:
                continue

        alpha_state["coins"]        = syms
        alpha_state["labels"]       = lbls
        alpha_state["volumes"]      = vols
        alpha_state["last_refresh"] = _time.time()
        log.info(f"Alpha state: {len(syms)} tokens passed $500k volume filter")
    except Exception as e:
        log.error("Alpha refresh error: " + str(e))


async def auto_alpha_refresh(app):
    """Refreshes Binance Alpha list every 6 hours. Sends startup alert."""
    log.info("Alpha refresh loop started")
    await asyncio.sleep(45)  # wait for bot startup
    refresh_alpha_tokens()

    if alpha_state["coins"]:
        msg  = "\U0001f536 *Binance Alpha Scanner Active*\n\n"
        msg += str(len(alpha_state["coins"])) + " Alpha tokens (>$500k vol) on Bitget:\n"
        for sym in alpha_state["coins"][:8]:
            lbl = alpha_state["labels"].get(sym, sym.split("/")[0])
            vol = alpha_state["volumes"].get(sym, 0)
            msg += "  \u2022 *" + lbl + "* — $" + "{:,.0f}".format(vol) + " vol\n"
        if len(alpha_state["coins"]) > 8:
            msg += "  \u2022 ... and " + str(len(alpha_state["coins"]) - 8) + " more\n"
        msg += "\nScanning every 10 min. Spike coins flagged for SHORT."
        await send_msg(app, msg)

    while True:
        await asyncio.sleep(ALPHA_REFRESH_HOURS * 3600)
        try:
            old_set = set(alpha_state["coins"])
            refresh_alpha_tokens()
            new_set = set(alpha_state["coins"])
            added   = new_set - old_set
            removed = old_set - new_set
            if added or removed:
                msg = "\U0001f536 *Alpha List Updated*\n"
                if added:
                    msg += "\u2795 " + ", ".join([alpha_state["labels"].get(s, s.split("/")[0]) for s in added]) + "\n"
                if removed:
                    msg += "\u2796 " + ", ".join([s.split("/")[0] for s in removed]) + "\n"
                msg += "Total: " + str(len(alpha_state["coins"])) + " tokens"
                await send_msg(app, msg)
        except Exception as e:
            log.error("Alpha refresh loop error: " + str(e))


async def auto_alpha_scan(app):
    """
    Scans Alpha tokens every 10 minutes through full Phase 1+2 pipeline.

    Special behaviour for spike coins:
    - If 24H gain >= 20% AND exhaustion signals fire → send SHORT alert
    - Normal LONG signals excluded if blow-off top detected
    - $500k+ volume already filtered at list refresh time
    """
    log.info("Alpha scan loop started")
    await asyncio.sleep(90)  # offset from main scan

    while True:
        await asyncio.sleep(ALPHA_SCAN_INTERVAL)
        if not alpha_state["coins"]:
            continue

        try:
            now    = _time.time()
            coins  = alpha_state["coins"]
            labels = alpha_state["labels"]

            batches = [coins[i:i+BATCH_SIZE] for i in range(0, len(coins), BATCH_SIZE)]
            results = []
            for batch in batches:
                br = await asyncio.gather(*[analyze_async(s) for s in batch])
                results.extend([r for r in br if r is not None])
                await asyncio.sleep(REQUEST_DELAY * BATCH_SIZE)

            check_btc_circuit_breaker()

            for r in results:
                symbol = r["symbol"]
                if now - alpha_state["last_signals"].get(symbol, 0) < ALPHA_COOLDOWN:
                    continue

                direction = r.get("direction", "NEUTRAL")
                quality   = r.get("abs_score", 0)
                grade     = r.get("grade", "B")

                # ── SHORT opportunity from exhaustion (spike coins) ───────────
                exh = r.get("quality", {}).get("exhaustion", {})
                if not exh:
                    # Try to get from result directly
                    exh = {}

                # Re-check exhaustion directly for alpha coins
                try:
                    from signal_engine import check_exhaustion
                    df_exh = r.get("df_4h")
                    if df_exh is not None:
                        exh = check_exhaustion(df_exh, "LONG", r.get("funding", 0))
                except Exception:
                    exh = {}

                # If SHORT opportunity from blow-off top
                # Alpha SHORT: only fire on confirmed exhaustion + significant pump
                if (exh.get("short_opp")
                        and exh.get("change_24h", 0) >= 20
                        and now - alpha_state["last_signals"].get(symbol, 0) >= ALPHA_COOLDOWN):
                    # Portfolio heat check
                    _a_heat = get_portfolio_heat(active_signals)
                    if not _a_heat["can_open"]:
                        log.info("Alpha SHORT blocked: " + labels.get(symbol, symbol) + " | " + _a_heat["reason"])
                        continue
                    # Dow Theory phase confirmation
                    _df_dow = r.get("df_4h")
                    if _df_dow is not None:
                        _dow_ok, _dow_phase, _dow_conf, _dow_reason = _dow_confirms_short(_df_dow)
                        if not _dow_ok:
                            log.info("Alpha SHORT blocked by Dow: " + labels.get(symbol, symbol) + " | " + _dow_reason)
                            continue
                        r["dow_phase"]      = _dow_phase
                        r["dow_confidence"] = _dow_conf
                    from risk_manager import calc_levels_v2
                    _ap = (r["atr"] * 1.5 / r["price"] * 100) if r["price"] > 0 else 1.0
                    sl, tp1, tp2, vp_meta = calc_levels_v2("SHORT", r["price"], r["atr"], _ap, r.get("df_4h"))
                    r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
                    r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
                    r["vp_in_lvn"]      = vp_meta.get("vp_in_lvn", False)
                    ob_bias, ob_ratio   = get_order_book_bias(symbol)
                    pos_usdt, contracts = calc_position_size(r["price"], sl)

                    # Override direction to SHORT
                    r_short = dict(r)
                    r_short["direction"]   = "SHORT"
                    r_short["signal_type"] = "MOMENTUM"
                    r_short["exhaustion"]  = exh

                    # ── Record signal + paper trade ───────────────────────────
                    record_signal(symbol, "SHORT", r["price"], sl, tp1, tp2,
                                  r.get("confidence","MEDIUM"),
                                  trade_type="swing", signal_type="MOMENTUM",
                                  grade=r.get("grade","B"),
                                  dow_phase=r.get("dow_phase","UNCLEAR"),
                                  vwap_bias=r.get("vwap_bias","AT"))
                    if paper_mode:
                        existing = load_json(PAPER_FILE, [])
                        if not any(t["symbol"] == symbol and t["status"] == "OPEN" for t in existing):
                            open_paper_trade(symbol, "SHORT", r["price"], sl, tp1, tp2,
                                             r.get("confidence","MEDIUM"),
                                             signal_type="MOMENTUM")
                    active_signals[symbol] = {
                        "direction": "SHORT", "entry": r["price"],
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "tp1_hit": False, "atr": r["atr"],
                        "time": now, "trailing_extreme": r["price"],
                        "trailing_stop": None, "trade_type": "swing",
                    }

                    _alpha_dow_label = r.get("dow_phase", "")
                    _alpha_dow_tag   = " | Dow: " + _alpha_dow_label if _alpha_dow_label and _alpha_dow_label != "UNCLEAR" else ""
                    header  = "\U0001f536\U0001f6a8 *Alpha SPIKE SHORT: " + labels.get(symbol, symbol.split("/")[0]) + "*\n"
                    header += "Blow-off top detected — consider SHORT" + _alpha_dow_tag + "\n"
                    header += "_" + exh.get("reason","") + "_\n\n"
                    perf = get_recent_perf()
                    msg  = header + format_signal(r_short, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                    msg += format_ai_block(r.get("ai_result", {}))
                    if perf:
                        msg += "\n\U0001f4c8 _" + perf + "_"
                    await send_msg(app, msg)
                    alpha_state["last_signals"][symbol] = now
                    log.info("Alpha SHORT paper trade opened: " + labels.get(symbol, symbol) + " | " + exh.get("reason",""))
                    continue

                # ── Normal LONG signal — strict gates for alpha tokens ────────
                # Grade A only, HIGH confidence only, abs_score >= 60
                # Alpha tokens are volatile — only take the very best setups
                if (direction == "LONG"
                        and r.get("confidence") == "HIGH"
                        and r.get("signal_type","REVERSAL") == "MOMENTUM"
                        and grade == "A"
                        and quality >= 70):

                    if btc_circuit_breaker["active"]:
                        continue

                    history = load_json(HISTORY_FILE, [])
                    risk = risk_gate(
                        symbol=symbol, direction=direction,
                        price=r["price"], atr=r["atr"],
                        quality_score=quality,
                        active_signals=active_signals,
                        trade_history=history,
                    )
                    if not risk["approved"]:
                        continue

                    from risk_manager import calc_levels_v2
                    _ap = (r["atr"] / r["price"] * 100) if r["price"] > 0 else 1.0
                    sl, tp1, tp2, vp_meta = calc_levels_v2(direction, r["price"], r["atr"], _ap, r.get("df_4h"))
                    r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
                    r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
                    r["vp_in_lvn"]      = vp_meta.get("vp_in_lvn", False)

                    ob_bias, ob_ratio   = get_order_book_bias(symbol)
                    pos_usdt, contracts = calc_position_size(r["price"], sl)

                    record_signal(symbol, direction, r["price"], sl, tp1, tp2,
                                  r["confidence"], signal_type="MOMENTUM",
                                  grade=grade, dow_phase=r.get("dow_phase","UNCLEAR"),
                                  vwap_bias=r.get("vwap_bias","AT"))
                    if paper_mode:
                        existing = load_json(PAPER_FILE, [])
                        if not any(t["symbol"] == symbol and t["status"] == "OPEN" for t in existing):
                            open_paper_trade(symbol, direction, r["price"], sl, tp1, tp2,
                                             r["confidence"], signal_type="MOMENTUM")
                    active_signals[symbol] = {
                        "direction": direction, "entry": r["price"],
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "tp1_hit": False, "atr": r["atr"],
                        "time": now, "trailing_extreme": r["price"], "trailing_stop": None,
                    }
                    header  = "\U0001f536 *Alpha Signal: " + labels.get(symbol, symbol.split("/")[0]) + "*\n"
                    header += "Vol: $" + "{:,.0f}".format(alpha_state["volumes"].get(symbol,0)) + " | Early-stage token\n\n"
                    perf    = get_recent_perf()
                    msg = header + format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                    msg += format_ai_block(r.get("ai_result", {}))
                    if perf:
                        msg += "\n\U0001f4c8 _" + perf + "_"
                    await send_msg(app, msg)
                    alpha_state["last_signals"][symbol] = now
                    log.info("Alpha signal: " + labels.get(symbol,symbol) + " " + direction)

        except Exception as e:
            log.error("Alpha scan error: " + str(e))


async def cmd_alpha(update, ctx):
    """/alpha — on-demand scan of Binance Alpha tokens on Bitget futures."""
    if not alpha_state["coins"]:
        await update.message.reply_text("\U0001f536 No Alpha tokens loaded. Fetching now...")
        refresh_alpha_tokens()
        if not alpha_state["coins"]:
            await update.message.reply_text("No Alpha tokens with $500k+ volume found on Bitget futures right now.")
            return

    await update.message.reply_text(
        "\U0001f536 Scanning " + str(len(alpha_state["coins"])) + " Alpha tokens ($500k+ vol)..."
    )
    try:
        batches = [alpha_state["coins"][i:i+BATCH_SIZE] for i in range(0, len(alpha_state["coins"]), BATCH_SIZE)]
        results = []
        for batch in batches:
            br = await asyncio.gather(*[analyze_async(s) for s in batch])
            results.extend([r for r in br if r is not None])
            await asyncio.sleep(REQUEST_DELAY * BATCH_SIZE)

        # Check exhaustion on each result
        spike_shorts = []
        normal_longs = []
        from signal_engine import check_exhaustion
        for r in results:
            df_e = r.get("df_4h")
            if df_e is not None:
                exh = check_exhaustion(df_e, "LONG", r.get("funding", 0))
                r["exhaustion"] = exh
                if exh.get("short_opp"):
                    spike_shorts.append(r)
                    continue
            if (r.get("direction") != "NEUTRAL"
                    and r.get("confidence") in ("HIGH","MEDIUM")
                    and r.get("signal_type","REVERSAL") == "MOMENTUM"
                    and r.get("grade","B") != "C"):
                normal_longs.append(r)

        msg = "\U0001f536 *Alpha Scan Results*\n\n"

        if spike_shorts:
            msg += "\U0001f6a8 *Spike SHORT Opportunities:*\n"
            for r in spike_shorts[:3]:
                lbl = alpha_state["labels"].get(r["symbol"], r["symbol"].split("/")[0])
                exh = r.get("exhaustion", {})
                msg += "  \U0001f534 *" + lbl + "* | " + str(exh.get("change_24h",0)) + "% pump | RSI " + str(exh.get("rsi",0)) + " | Vol " + str(exh.get("vol_ratio",0)) + "x\n"
            msg += "\n"

        if normal_longs:
            grade_order = {"A": 0, "B": 1, "C": 2}
            normal_longs.sort(key=lambda x: (grade_order.get(x.get("grade","B"),1), -x.get("abs_score",0)))
            best = normal_longs[0]
            from risk_manager import calc_levels_v2
            _ap = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
            sl, tp1, tp2, vp_meta = calc_levels_v2(best["direction"], best["price"], best["atr"], _ap, best.get("df_4h"))
            best["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
            best["vp_poc"]         = vp_meta.get("vp_poc") or best.get("vp_poc")
            ob_bias, ob_ratio      = get_order_book_bias(best["symbol"])
            pos_usdt, contracts    = calc_position_size(best["price"], sl)
            lbl = alpha_state["labels"].get(best["symbol"], best["symbol"].split("/")[0])
            header = "\U0001f536 *Best Alpha Signal: " + lbl + "*\n"
            header += "Vol: $" + "{:,.0f}".format(alpha_state["volumes"].get(best["symbol"],0)) + "\n\n"
            full_msg = msg + header + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
            try:
                await update.message.reply_text(full_msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(full_msg, parse_mode=None)
        else:
            if not spike_shorts:
                msg += "No signals on Alpha tokens right now."
            try:
                await update.message.reply_text(msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(msg, parse_mode=None)
    except Exception as e:
        await update.message.reply_text("Alpha scan error: " + str(e))


# ─── FLIP SCALP SYSTEM ────────────────────────────────────────────────────────
async def cmd_flip(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /flip COIN DIRECTION TARGET%
    Start hunting flip scalp entries on top of a swing position.

    Examples:
      /flip NEAR LONG 5     → alert when 15m shows local low + 2+ signals, target +5%
      /flip NEAR SHORT 5    → alert when 15m shows local high + 2+ signals, target +5%
      /flip NEAR stop       → stop flipping NEAR
      /flipping             → show all active flips

    The flip system monitors 15m chart every 2 minutes.
    Alert fires when 2+ of these agree:
      - RSI oversold (<32) for LONG flip, overbought (>68) for SHORT flip
      - EMA9 slope reversing direction
      - 3-candle structure: lower highs→higher close (LONG) or vice versa
      - Volume drying up into the reversal
    Includes volatility gate: pauses if ATR > 3x normal (too erratic to flip).
    Separate from /watch — flip scalps are independent small positions.
    """
    args = ctx.args
    if not args:
        await update.message.reply_text(
            "\U0001f504 *Flip Scalp System*\n\n"
            "Usage:\n"
            "  /flip COIN LONG 5   — hunt local lows, target +5%\n"
            "  /flip COIN SHORT 5  — hunt local highs, target +5%\n"
            "  /flip COIN stop     — stop flipping\n"
            "  /flipping           — show active flips\n\n"
            "_Flip scalps are separate small positions on top of your swing._",
            parse_mode="Markdown"
        )
        return

    coin = args[0].upper()
    symbol = coin + "/USDT:USDT"

    # Stop command
    if len(args) >= 2 and args[1].lower() == "stop":
        if symbol in flip_trades:
            del flip_trades[symbol]
            await update.message.reply_text("\u2705 Stopped flipping " + coin)
        else:
            await update.message.reply_text(coin + " is not being flipped.")
        return

    if len(args) < 3:
        await update.message.reply_text("Usage: /flip COIN LONG 5  or  /flip COIN SHORT 5")
        return

    direction = args[1].upper()
    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text("Direction must be LONG or SHORT.")
        return

    flip_conf = "medium"
    if len(args) >= 4 and args[3].lower() in ("high", "medium", "low"):
        flip_conf = args[3].lower()

    try:
        target_pct = float(args[2])
        if target_pct <= 0 or target_pct > 30:
            await update.message.reply_text("Target % must be between 1 and 30.")
            return
    except ValueError:
        await update.message.reply_text("Target must be a number, e.g. 5")
        return

    # Get current price as reference
    try:
        ticker    = exchange.fetch_ticker(symbol)
        ref_price = float(ticker["last"])
    except Exception as e:
        await update.message.reply_text("Could not fetch price for " + coin + ": " + str(e))
        return

    flip_min_signals = 3 if flip_conf == "high" else 2  # HIGH: only STRONG quality

    flip_trades[symbol] = {
        "coin":        coin,
        "label":       COIN_LABELS.get(symbol, coin),
        "direction":   direction,
        "target_pct":  target_pct,
        "confidence":  flip_conf,
        "min_signals": flip_min_signals,
        "ref_price":   ref_price,
        "start_time":  _time.time(),
        "last_flip":   0,
        "flip_count":  0,
        "last_price":  ref_price,
    }

    dir_emoji  = "\U0001f7e2" if direction == "LONG" else "\U0001f534"
    conf_label = {"high": "\U0001f6e1 HIGH — fires on STRONG signals only (3/4)", "medium": "\U0001f7e1 MEDIUM — standard (2/4)", "low": "\U0001f6a8 LOW — fires at first sign (2/4)"}
    await update.message.reply_text(
        "\U0001f504 *Flip Scalp Active: " + coin + "*\n\n"
        + dir_emoji + " Direction: " + direction + "\n"
        "\U0001f3af Target: +" + str(target_pct) + "% per flip\n"
        "\U0001f4b0 SL per flip: -" + str(round(target_pct * 0.4, 1)) + "% (auto)\n"
        "\U0001f4cd Ref price: $" + "{:.4f}".format(ref_price) + "\n"
        + conf_label.get(flip_conf, "") + "\n\n"
        "_Use /flip " + coin + " stop to cancel._",
        parse_mode="Markdown"
    )
    log.info("Flip started: " + coin + " " + direction + " " + str(target_pct) + "%")


async def cmd_flipping(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """/flipping — show all active flip scalps."""
    if not flip_trades:
        await update.message.reply_text("No active flip scalps. Use /flip COIN LONG 5 to start.")
        return

    msg = "\U0001f504 *Active Flip Scalps:*\n\n"
    now = _time.time()
    for symbol, f in flip_trades.items():
        try:
            price = float(exchange.fetch_ticker(symbol)["last"])
        except Exception:
            price = f["ref_price"]

        move_pct  = (price - f["ref_price"]) / f["ref_price"] * 100
        if f["direction"] == "SHORT":
            move_pct = -move_pct
        watched_m = round((now - f["start_time"]) / 60)
        last_m    = round((now - f["last_flip"]) / 60) if f["last_flip"] else None
        cd_left   = max(0, round((FLIP_COOLDOWN - (now - f["last_flip"])) / 60)) if f["last_flip"] else 0

        dir_e = "\U0001f7e2" if f["direction"] == "LONG" else "\U0001f534"
        msg  += dir_e + " *" + f["coin"] + "* " + f["direction"] + "\n"
        msg  += "  Target: +" + str(f["target_pct"]) + "% | Flips fired: " + str(f["flip_count"]) + "\n"
        msg  += "  Ref: $" + "{:.4f}".format(f["ref_price"]) + " | Now: $" + "{:.4f}".format(price) + "\n"
        msg  += "  Move: " + "{:+.1f}".format(move_pct) + "% | Watching " + str(watched_m) + "m\n"
        if cd_left > 0:
            msg += "  Cooldown: " + str(cd_left) + "m left\n"
        msg += "\n"

    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg, parse_mode=None)


def _check_flip_signals(df_15m: "pd.DataFrame", direction: str) -> tuple:
    """Thin wrapper — calls unified _check_reversal_signals in flip mode."""
    return _check_reversal_signals(df_15m, direction, mode="flip")


async def auto_flip(app):
    """
    Background loop — checks flip scalp setups every 2 minutes.
    Fires alert when 2+ signals agree on 15m.
    Includes volatility gate — pauses if ATR > 3x normal.
    """
    log.info("Flip loop started")
    while True:
        await asyncio.sleep(FLIP_CHECK_INTERVAL)
        if not flip_trades:
            continue

        for symbol, f in list(flip_trades.items()):
            try:
                now       = _time.time()
                coin      = f["coin"]
                direction = f["direction"]
                target    = f["target_pct"]

                # Cooldown gate
                if now - f["last_flip"] < FLIP_COOLDOWN:
                    continue

                # Fetch 15m data
                df_15m = fetch_ohlcv(symbol, "15m", 60)
                if df_15m is None or len(df_15m) < 20:
                    continue

                price = float(df_15m["close"].iloc[-1])
                flip_trades[symbol]["last_price"] = price

                # ── Volatility gate ───────────────────────────────────────────
                import pandas_ta as _ta
                atr_s = _ta.atr(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
                if atr_s is not None and len(atr_s.dropna()) >= 10:
                    atr_vals = atr_s.dropna().values
                    atr_now  = atr_vals[-1]
                    atr_avg  = float(atr_s.dropna().iloc[-10:].mean())
                    if atr_avg > 0 and atr_now > atr_avg * FLIP_MAX_ATR_MULT:
                        log.info("Flip " + coin + ": ATR " + str(round(atr_now/atr_avg,1)) + "x normal — pausing (too erratic)")
                        continue
                    atr_val = atr_now
                else:
                    atr_val = price * 0.01

                # ── Price moved enough from ref to be worth flipping ──────────
                ref   = f["ref_price"]
                # For LONG flip: price should have dipped enough to be a buy opportunity
                # For SHORT flip: price should have pumped enough to be a sell opportunity
                move_from_ref = (price - ref) / ref * 100
                if direction == "LONG" and move_from_ref > 1.0:
                    # Price is UP from ref — not a dip, skip
                    # Update ref to current if price moved significantly up
                    if move_from_ref > target * 0.5:
                        flip_trades[symbol]["ref_price"] = price
                        log.info("Flip " + coin + ": ref updated to " + str(round(price,4)))
                    continue
                if direction == "SHORT" and move_from_ref < -1.0:
                    # Price is DOWN from ref — not a pump, skip
                    if move_from_ref < -(target * 0.5):
                        flip_trades[symbol]["ref_price"] = price
                    continue

                # ── Check flip signals ────────────────────────────────────────
                signals, count, quality = _check_flip_signals(df_15m, direction)
                min_sigs = f.get("min_signals", 2)

                if count < min_sigs:
                    continue

                # ── Build flip alert ──────────────────────────────────────────
                atr_pct  = (atr_val / price * 100) if price > 0 else 1.0
                sl_pct   = round(target * 0.4, 1)  # SL = 40% of target
                tp_pct   = target

                if direction == "LONG":
                    sl_price  = round(price * (1 - sl_pct / 100), 4)
                    tp1_price = round(price * (1 + tp_pct / 2 / 100), 4)
                    tp2_price = round(price * (1 + tp_pct / 100), 4)
                    dir_emoji = "\U0001f7e2"
                else:
                    sl_price  = round(price * (1 + sl_pct / 100), 4)
                    tp1_price = round(price * (1 - tp_pct / 2 / 100), 4)
                    tp2_price = round(price * (1 - tp_pct / 100), 4)
                    dir_emoji = "\U0001f534"

                quality_emoji = "\U0001f7e2" if quality == "STRONG" else "\U0001f7e1"
                pos_usdt, contracts = calc_position_size(price, sl_price)

                msg  = "\U0001f504 *Flip Scalp: " + coin + "*\n\n"
                msg += dir_emoji + " Direction: *" + direction + "* | " + quality_emoji + " " + quality + " (" + str(count) + "/4 signals)\n"
                msg += "\U0001f4cd Entry zone: `$" + "{:.4f}".format(price) + "`\n"
                msg += "\U0001f6d1 SL: `$" + "{:.4f}".format(sl_price) + "` (-" + str(sl_pct) + "% / -" + str(round(sl_pct*LEVERAGE,1)) + "% at " + str(LEVERAGE) + "x)\n"
                msg += "\U0001f3af TP1: `$" + "{:.4f}".format(tp1_price) + "` (+" + str(round(tp_pct/2,1)) + "% / +" + str(round(tp_pct/2*LEVERAGE,1)) + "% at " + str(LEVERAGE) + "x)\n"
                msg += "\U0001f3af TP2: `$" + "{:.4f}".format(tp2_price) + "` (+" + str(tp_pct) + "% / +" + str(round(tp_pct*LEVERAGE,1)) + "% at " + str(LEVERAGE) + "x)\n"
                msg += "\U0001f4b0 Size: $" + "{:.2f}".format(pos_usdt) + " | Contracts: " + str(contracts) + "\n\n"
                msg += "*Flip Signals:*\n"
                for s in signals:
                    msg += "  \u2022 " + s + "\n"
                msg += "\n_This is a flip scalp on top of your swing — separate small position._\n"
                msg += "_Core swing unaffected. Close flip at TP or SL independently._"

                await send_msg(app, msg)
                flip_trades[symbol]["last_flip"]  = now
                flip_trades[symbol]["flip_count"] += 1
                # Update ref price to current after firing
                flip_trades[symbol]["ref_price"]  = price
                log.info("Flip alert: " + coin + " " + direction + " Q:" + quality + " signals:" + str(count))

            except Exception as e:
                log.error("Flip loop error " + symbol + ": " + str(e))


# ─── TOP GAINERS SCANNER ─────────────────────────────────────────────────────
def fetch_top_gainers() -> list:
    """
    Fetches top gaining USDT perpetual futures on Bitget by 24H % change.
    Filters out coins already in main COINS list and alpha list.
    Returns list of {symbol, label, change_24h, volume_24h} sorted by gain desc.
    """
    try:
        tickers = exchange.fetch_tickers()
        gainers = []
        for sym, t in tickers.items():
            if not sym.endswith("/USDT:USDT"):
                continue
            if sym in COINS:
                continue  # already in main scan
            change = float(t.get("percentage", 0) or 0)
            volume = float(t.get("quoteVolume", 0) or 0)
            if change >= GAINERS_MIN_CHANGE and volume >= 100000:  # min $100k vol
                gainers.append({
                    "symbol":     sym,
                    "label":      sym.split("/")[0],
                    "change_24h": round(change, 1),
                    "volume_24h": volume,
                })
        gainers.sort(key=lambda x: x["change_24h"], reverse=True)
        return gainers[:GAINERS_MAX_COINS]
    except Exception as e:
        log.error("fetch_top_gainers error: " + str(e))
        return []


async def auto_gainers_scan(app):
    """
    Scans top gainers every 30 minutes.

    Two signal paths for each gainer:
    1. If exhaustion signals fire (blow-off top) → SHORT alert
    2. If early breakout (no exhaustion, MOMENTUM signal) → LONG alert

    This catches coins like ORDI, SOON, BASED that the main scan misses
    because they're not in the top 40 by volume — but they're top gainers.
    """
    log.info("Top gainers scanner started")
    await asyncio.sleep(120)  # offset from other scans

    while True:
        await asyncio.sleep(GAINERS_SCAN_INTERVAL)
        try:
            now     = _time.time()
            gainers = fetch_top_gainers()
            if not gainers:
                continue

            log.info(f"Top gainers scan: {len(gainers)} coins ≥{GAINERS_MIN_CHANGE}% gain")

            for g in gainers:
                symbol     = g["symbol"]
                label      = g["label"]
                change_24h = g["change_24h"]

                # Cooldown per coin
                if now - gainers_state["last_signals"].get(symbol, 0) < GAINERS_COOLDOWN:
                    continue

                # Add temporarily to labels for pipeline
                original_label = COIN_LABELS.get(symbol)
                if not original_label:
                    COIN_LABELS[symbol] = label

                try:
                    r = await run_full_pipeline(
                        symbol, fetch_ohlcv, COIN_LABELS,
                        ai_client, exchange, news_context
                    )
                finally:
                    if not original_label and symbol in COIN_LABELS:
                        del COIN_LABELS[symbol]

                if not r or "direction" not in r or not r.get("price"):
                    continue

                price = r["price"]
                atr   = r.get("atr", price * 0.02)

                # Classify signal
                try:
                    df_c = r.get("df_4h")
                    if df_c is not None:
                        st, tc, tr = classify_signal(df_c, r["direction"], r.get("abs_score", 0), None)
                        r["signal_type"] = st
                except Exception:
                    r["signal_type"] = "MOMENTUM"

                # Grade + exhaustion check
                try:
                    from signal_engine import grade_signal, check_exhaustion, suggest_leverage
                    df_g = r.get("df_4h")
                    if df_g is not None:
                        gr = grade_signal(df_g, fetch_ohlcv(symbol, "4h", 100),
                                          r["direction"], r.get("abs_score", 0),
                                          r.get("rsi_4h", 50), r.get("funding", 0))
                        r["grade"]          = gr["grade"]
                        r["grade_score"]    = gr["grade_score"]
                        r["grade_reasons"]  = gr["reasons"]
                        r["grade_warnings"] = gr["warnings"]
                        r["dow_phase"]      = gr.get("dow_phase", "UNCLEAR")
                        r["vwap"]           = gr.get("vwap", 0)
                        r["vwap_dist_pct"]  = gr.get("vwap_dist_pct", 0)
                        r["vwap_bias"]      = gr.get("vwap_bias", "AT")
                        r["vwap_strength"]  = gr.get("vwap_strength", "WEAK")
                        # Leverage suggestion
                        lev = suggest_leverage(df_g, r.get("abs_score", 50), gr["grade"])
                        r["suggested_leverage"] = lev["suggested"]
                        r["max_safe_leverage"]  = lev["max_safe"]
                        r["leverage_reason"]    = lev["reason"]
                        # Exhaustion check on gainer
                        exh = check_exhaustion(df_g, "LONG", r.get("funding", 0))
                        r["exhaustion"] = exh
                except Exception as e:
                    log.warning("Gainers grade error " + label + ": " + str(e))
                    exh = {"block_long": False, "short_opp": False, "reason": ""}

                exh = r.get("exhaustion", {"block_long": False, "short_opp": False})

                # VP-adjusted levels
                from risk_manager import calc_levels_v2
                _ap = (atr / price * 100) if price > 0 else 1.0

                # ── Path 1: Blow-off top → SHORT alert ───────────────────────
                # Gainers SHORT: require strong pump (25%+) AND confirmed exhaustion
                if exh.get("short_opp") and change_24h >= 25:
                    # Portfolio heat check — don't open if at position limit
                    _g_heat = get_portfolio_heat(active_signals)
                    if not _g_heat["can_open"]:
                        log.info("Gainer SHORT blocked by heat: " + label + " | " + _g_heat["reason"])
                        continue
                    # Dow Theory phase confirmation
                    _df_dow_g = r.get("df_4h")
                    if _df_dow_g is not None:
                        _dow_ok_g, _dow_phase_g, _dow_conf_g, _dow_reason_g = _dow_confirms_short(_df_dow_g)
                        if not _dow_ok_g:
                            log.info("Gainer SHORT blocked by Dow: " + label + " | " + _dow_reason_g)
                            continue
                        r["dow_phase"]      = _dow_phase_g
                        r["dow_confidence"] = _dow_conf_g
                    _ap = (atr * 1.5 / price * 100) if price > 0 else 1.0  # wider for gainer volatility
                    sl, tp1, tp2, vp_meta = calc_levels_v2("SHORT", price, atr * 1.5, _ap, r.get("df_4h"))
                    r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
                    r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
                    r["direction"]      = "SHORT"
                    r["signal_type"]    = "MOMENTUM"
                    ob_bias, ob_ratio   = get_order_book_bias(symbol)
                    pos_usdt, contracts = calc_position_size(price, sl)

                    record_signal(symbol, "SHORT", price, sl, tp1, tp2,
                                  r.get("confidence", "MEDIUM"), trade_type="swing",
                                  signal_type="MOMENTUM", grade=r.get("grade", "B"),
                                  dow_phase=r.get("dow_phase", "UNCLEAR"),
                                  vwap_bias=r.get("vwap_bias", "AT"))
                    if paper_mode:
                        existing = load_json(PAPER_FILE, [])
                        if not any(t["symbol"] == symbol and t["status"] == "OPEN" for t in existing):
                            open_paper_trade(symbol, "SHORT", price, sl, tp1, tp2,
                                             r.get("confidence", "MEDIUM"), signal_type="MOMENTUM")
                    active_signals[symbol] = {
                        "direction": "SHORT", "entry": price,
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "tp1_hit": False, "atr": atr,
                        "time": now, "trailing_extreme": price, "trailing_stop": None,
                    }

                    _gain_dow_label = r.get("dow_phase", "")
                    _gain_dow_tag   = " | Dow: " + _gain_dow_label if _gain_dow_label and _gain_dow_label != "UNCLEAR" else ""
                    header  = "\U0001f3c6\U0001f6a8 *Top Gainer SHORT: " + label + "*\n"
                    header += "24H pump: *+" + str(change_24h) + "%*" + _gain_dow_tag + "\n"
                    header += "_" + exh.get("reason", "") + "_\n\n"
                    msg = header + format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                    msg += format_ai_block(r.get("ai_result", {}))
                    perf = get_recent_perf()
                    if perf:
                        msg += "\n\U0001f4c8 _" + perf + "_"
                    await send_msg(app, msg)
                    gainers_state["last_signals"][symbol] = now
                    log.info("Gainer SHORT: " + label + " +" + str(change_24h) + "%")

                # ── Path 2: Early breakout → LONG alert ──────────────────────
                elif (r.get("direction") == "LONG"
                      and r.get("confidence") == "HIGH"
                      and r.get("signal_type") == "MOMENTUM"
                      and r.get("grade") == "A"
                      and r.get("abs_score", 0) >= 70
                      and not exh.get("block_long")
                      and not btc_circuit_breaker["active"]):

                    history = load_json(HISTORY_FILE, [])
                    risk = risk_gate(symbol=symbol, direction="LONG",
                                     price=price, atr=atr,
                                     quality_score=r.get("abs_score", 0),
                                     active_signals=active_signals,
                                     trade_history=history)
                    if not risk["approved"]:
                        continue

                    sl, tp1, tp2, vp_meta = calc_levels_v2("LONG", price, atr, _ap, r.get("df_4h"))
                    r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
                    r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
                    ob_bias, ob_ratio   = get_order_book_bias(symbol)
                    pos_usdt, contracts = calc_position_size(price, sl)

                    record_signal(symbol, "LONG", price, sl, tp1, tp2,
                                  r.get("confidence", "MEDIUM"), trade_type="swing",
                                  signal_type="MOMENTUM", grade=r.get("grade", "B"),
                                  dow_phase=r.get("dow_phase", "UNCLEAR"),
                                  vwap_bias=r.get("vwap_bias", "AT"))
                    if paper_mode:
                        existing = load_json(PAPER_FILE, [])
                        if not any(t["symbol"] == symbol and t["status"] == "OPEN" for t in existing):
                            open_paper_trade(symbol, "LONG", price, sl, tp1, tp2,
                                             r.get("confidence", "MEDIUM"), signal_type="MOMENTUM")
                    active_signals[symbol] = {
                        "direction": "LONG", "entry": price,
                        "sl": sl, "tp1": tp1, "tp2": tp2,
                        "tp1_hit": False, "atr": atr,
                        "time": now, "trailing_extreme": price, "trailing_stop": None,
                    }

                    header  = "\U0001f3c6\U0001f525 *Top Gainer LONG: " + label + "*\n"
                    header += "24H gain: *+" + str(change_24h) + "%* — early breakout\n\n"
                    perf    = get_recent_perf()
                    msg = header + format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                    msg += format_ai_block(r.get("ai_result", {}))
                    if perf:
                        msg += "\n\U0001f4c8 _" + perf + "_"
                    await send_msg(app, msg)
                    gainers_state["last_signals"][symbol] = now
                    log.info("Gainer LONG: " + label + " +" + str(change_24h) + "%")

                await asyncio.sleep(REQUEST_DELAY * 2)  # rate limit between coins

        except Exception as e:
            log.error("Gainers scan error: " + str(e))

# ─── BTC CIRCUIT BREAKER ───────────────────────────────────────────────────────
def check_btc_circuit_breaker() -> dict:
    """
    Checks BTC condition and updates btc_circuit_breaker state.

    Triggers (block new LONGs) when EITHER:
      1. BTC 1H candle dropped >= 2% (fast dump detection)
      2. BTC 4H regime is TRENDING_DOWN with ADX > 25 (sustained downtrend)

    Resets (allow LONGs again) when ALL:
      1. BTC 1H drop < 1%
      2. BTC 1H RSI > 48
      3. BTC 4H regime is NOT TRENDING_DOWN

    Returns updated btc_circuit_breaker dict.
    """
    global btc_circuit_breaker
    try:
        from signal_engine import detect_regime
        import pandas_ta as _ta

        df_1h = fetch_ohlcv("BTC/USDT:USDT", "1h", 50)
        df_4h = fetch_ohlcv("BTC/USDT:USDT", "4h", 50)

        # 1H candle drop
        close_now  = float(df_1h["close"].iloc[-1])
        close_prev = float(df_1h["close"].iloc[-2])
        candle_chg = (close_now - close_prev) / close_prev * 100

        # 1H RSI
        rsi_s   = _ta.rsi(df_1h["close"], length=14)
        rsi_val = float(rsi_s.dropna().iloc[-1]) if rsi_s is not None and len(rsi_s.dropna()) > 0 else 50.0

        # 4H regime
        regime      = detect_regime(df_4h)
        regime_name = regime.get("regime", "RANGING")
        adx_val     = regime.get("adx", 0)

        btc_circuit_breaker["last_check"] = _time.time()

        # ── Check triggers ────────────────────────────────────────────────────
        fast_dump    = candle_chg <= -BTC_DUMP_THRESHOLD
        bear_regime  = regime_name == "TRENDING_DOWN" and adx_val > 25

        if fast_dump or bear_regime:
            reason = ""
            if fast_dump:
                reason += f"BTC 1H dropped {candle_chg:.1f}%"
            if bear_regime:
                reason += (" + " if reason else "") + f"4H regime TRENDING_DOWN (ADX:{adx_val:.0f})"
            btc_circuit_breaker["active"]       = True
            btc_circuit_breaker["reason"]       = reason
            btc_circuit_breaker["triggered_at"] = _time.time()
            log.info("BTC circuit breaker ACTIVE: " + reason)
            return btc_circuit_breaker

        # ── Check reset conditions ────────────────────────────────────────────
        if btc_circuit_breaker["active"]:
            drop_ok   = candle_chg > -BTC_RESET_DROP
            rsi_ok    = rsi_val > BTC_RESET_RSI
            regime_ok = regime_name != "TRENDING_DOWN"

            if drop_ok and rsi_ok and regime_ok:
                btc_circuit_breaker["active"]  = False
                btc_circuit_breaker["reason"]  = ""
                log.info(f"BTC circuit breaker RESET — RSI:{rsi_val:.1f} drop:{candle_chg:.1f}% regime:{regime_name}")

    except Exception as e:
        log.error("BTC circuit breaker check error: " + str(e))

    return btc_circuit_breaker


async def auto_btc_monitor(app):
    """
    Runs BTC circuit breaker check every 30 minutes.
    Sends Telegram alert when breaker activates or resets.
    """
    log.info("BTC circuit breaker monitor started")
    prev_state = False
    while True:
        await asyncio.sleep(BTC_CHECK_INTERVAL)
        try:
            state = check_btc_circuit_breaker()
            active = state["active"]

            # Alert on state change only
            if active and not prev_state:
                msg  = "\U0001f6a8 *BTC Circuit Breaker ACTIVE*\n\n"
                msg += "Reason: " + state["reason"] + "\n"
                msg += "New LONG signals are blocked until BTC stabilises.\n"
                msg += "Existing open trades are unaffected."
                await send_msg(app, msg)
                log.info("Circuit breaker alert sent")

            elif not active and prev_state:
                msg  = "\u2705 *BTC Circuit Breaker RESET*\n\n"
                msg += "BTC has stabilised. LONG signals are enabled again."
                await send_msg(app, msg)
                log.info("Circuit breaker reset alert sent")

            prev_state = active
        except Exception as e:
            log.error("BTC monitor error: " + str(e))


# ─── COUPON PAIRS ────────────────────────────────────────────────────────────
COUPON_PAIRS = [
    "PI/USDT:USDT",
    "DOGE/USDT:USDT",
    "BGB/USDT:USDT",
    "XRP/USDT:USDT",
    "SUI/USDT:USDT",
    "PEPE/USDT:USDT",
    "SHIB/USDT:USDT",
]
COUPON_LABELS = {
    "PI/USDT:USDT":   "PI",
    "DOGE/USDT:USDT": "DOGE",
    "BGB/USDT:USDT":  "BGB",
    "XRP/USDT:USDT":  "XRP",
    "SUI/USDT:USDT":  "SUI",
    "PEPE/USDT:USDT": "PEPE",
    "SHIB/USDT:USDT": "SHIB",
}
COUPON_COOLDOWN = 14400  # 4 hour cooldown per pair (same as main scan)


async def _scan_coupon_pair(symbol: str) -> dict | None:
    """
    Scan a single coupon pair through full pipeline.
    Returns result dict if MOMENTUM signal found, None otherwise.
    """
    try:
        old_label = COIN_LABELS.get(symbol)
        COIN_LABELS[symbol] = COUPON_LABELS[symbol]

        r = await run_full_pipeline(
            symbol, fetch_ohlcv, COIN_LABELS,
            ai_client, exchange, news_context
        )

        if not old_label:
            COIN_LABELS.pop(symbol, None)

        if not r or r.get("direction") in (None, "NEUTRAL"):
            return None

        # Classify
        try:
            df_c = r.get("df_4h")
            if df_c is not None:
                st, tc, tr = classify_signal(df_c, r["direction"], r.get("abs_score", 0), None)
                r["signal_type"] = st
                r["type_conf"]   = tc
                r["type_reason"] = tr
        except Exception:
            r["signal_type"] = "MOMENTUM"

        # Grade + VP
        try:
            from signal_engine import grade_signal, compute_volume_profile
            df_g  = r.get("df_4h")
            df_4g = fetch_ohlcv(symbol, "4h", 100)
            if df_g is not None:
                gr = grade_signal(df_g, df_4g, r["direction"],
                                  r.get("abs_score", 0), r.get("rsi_4h", 50), r.get("funding", 0))
                r["grade"]         = gr["grade"]
                r["grade_score"]   = gr["grade_score"]
                r["grade_reasons"] = gr["reasons"]
                r["grade_warnings"]= gr["warnings"]
                r["dow_phase"]     = gr.get("dow_phase", "UNCLEAR")
                r["dow_confidence"]= gr.get("dow_confidence", "LOW")
                r["dow_signals"]   = gr.get("dow_signals", [])
                r["vwap"]          = gr.get("vwap", 0)
                r["vwap_dist_pct"] = gr.get("vwap_dist_pct", 0)
                r["vwap_bias"]     = gr.get("vwap_bias", "AT")
                r["vwap_strength"] = gr.get("vwap_strength", "WEAK")
            if df_g is not None and len(df_g) >= 20:
                vp = compute_volume_profile(df_g)
                r["vp_poc"]         = vp.get("poc")
                r["vp_in_lvn"]      = vp.get("in_lvn", False)
                r["vp_adjustments"] = []
        except Exception:
            r["grade"] = "B"

        # Grade/type guard
        if r.get("signal_type") == "WEAK" and r.get("grade") == "A":
            r["grade"] = "B"

        # Only MOMENTUM, no Grade C, respect circuit breaker
        if r.get("signal_type") != "MOMENTUM":
            return None
        if r.get("grade") == "C":
            return None
        if r["direction"] == "LONG" and btc_circuit_breaker["active"]:
            return None

        return r

    except Exception as e:
        log.warning("Coupon scan error " + symbol + ": " + str(e))
        return None


async def auto_coupon_monitor(app):
    """
    Background loop — scans 7 coupon pairs every 5 minutes when active.
    Fires signal automatically when MOMENTUM found with 4H cooldown per pair.
    Toggle on/off via /coupon command.
    """
    log.info("Coupon monitor loop started (inactive until /coupon)")
    while True:
        await asyncio.sleep(300)  # check every 5 minutes

        if not coupon_monitor["active"]:
            continue

        try:
            now = _time.time()
            for symbol in COUPON_PAIRS:
                # Per-pair 4H cooldown
                last_fired = coupon_monitor["last_signals"].get(symbol, 0)
                if now - last_fired < COUPON_COOLDOWN:
                    continue

                r = await _scan_coupon_pair(symbol)
                if r is None:
                    continue

                # VP-adjusted levels
                from risk_manager import calc_levels_v2
                atr_pct = (r["atr"] / r["price"] * 100) if r["price"] > 0 else 1.0
                sl, tp1, tp2, vp_meta = calc_levels_v2(
                    r["direction"], r["price"], r["atr"], atr_pct, r.get("df_4h")
                )
                r["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
                r["vp_poc"]         = vp_meta.get("vp_poc") or r.get("vp_poc")
                r["vp_in_lvn"]      = vp_meta.get("vp_in_lvn", r.get("vp_in_lvn", False))

                ob_bias, ob_ratio   = get_order_book_bias(symbol)
                pos_usdt, contracts = calc_position_size(r["price"], sl)

                # Record + paper trade
                record_signal(symbol, r["direction"], r["price"], sl, tp1, tp2,
                              r["confidence"], signal_type=r.get("signal_type", "MOMENTUM"),
                              grade=r.get("grade", "B"),
                              dow_phase=r.get("dow_phase", "UNCLEAR"),
                              vwap_bias=r.get("vwap_bias", "AT"))
                if paper_mode:
                    existing = load_json(PAPER_FILE, [])
                    already_open = any(
                        t["symbol"] == symbol and t["status"] == "OPEN" for t in existing
                    )
                    if not already_open:
                        open_paper_trade(symbol, r["direction"], r["price"], sl, tp1, tp2,
                                         r["confidence"], signal_type=r.get("signal_type", "MOMENTUM"))

                active_signals[symbol] = {
                    "direction": r["direction"], "entry": r["price"],
                    "sl": sl, "tp1": tp1, "tp2": tp2,
                    "tp1_hit": False, "atr": r["atr"],
                    "time": now,
                    "trailing_extreme": r["price"],
                    "trailing_stop": None,
                }

                header  = "\U0001f3ab *Coupon Signal: " + COUPON_LABELS[symbol] + "* | " + datetime.now().strftime("%H:%M UTC") + "\n"
                header += "50 USDT boost active \u2014 use your position voucher on this trade\n\n"
                msg = header + format_signal(r, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
                msg += format_ai_block(r.get("ai_result", {}))

                await send_msg(app, msg)
                coupon_monitor["last_signals"][symbol] = now
                log.info("Coupon auto-signal: " + COUPON_LABELS[symbol] + " " + r["direction"] + " Q:" + str(r.get("abs_score", 0)))

        except Exception as e:
            log.error("Coupon monitor error: " + str(e))


async def cmd_coupon(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /coupon — Toggle auto-coupon scanner ON/OFF.

    First press: activates background scanner on 7 Bitget coupon pairs.
    Bot fires signals automatically every 5 min scan, 4H cooldown per pair.
    Second press: deactivates scanner and stops coupon signals.
    Use /coupon when you have used the voucher or it expires.

    50 USDT position boost voucher — valid until 2026-05-07.
    Valid pairs: PI, DOGE, BGB, XRP, SUI, PEPE, SHIB
    """
    # ── Coupon expiry check ──────────────────────────────────────────────────
    from datetime import date as _date
    expiry        = _date(2026, 5, 7)
    days_left     = (expiry - _date.today()).days
    expiry_str    = f"{days_left} days left (expires 2026-05-07)"
    if days_left <= 0:
        await update.message.reply_text(
            "\U0001f3ab *Coupon Expired*\n\n"
            "Your 50 USDT position voucher expired on 2026-05-07.\n"
            "Coupon monitor is disabled.",
            parse_mode="Markdown"
        )
        coupon_monitor["active"] = False
        return
    expiry_warning = " \u26a0 *" + str(days_left) + " days left!*" if days_left <= 7 else ""

    if coupon_monitor["active"]:
        # Toggle OFF
        coupon_monitor["active"] = False
        coupon_monitor["last_signals"] = {}
        await update.message.reply_text(
            "\U0001f3ab *Coupon Monitor: OFF*\n\n"
            "Auto-scanning stopped. Coupon signals will no longer fire.\n"
            "Press /coupon again to reactivate.\n"
            "_" + expiry_str + "_",
            parse_mode="Markdown"
        )
        log.info("Coupon monitor deactivated")
        return

    # Toggle ON — activate and do an immediate scan
    coupon_monitor["active"] = True
    await update.message.reply_text(
        "\U0001f3ab *Coupon Monitor: ON*" + expiry_warning + "\n\n"
        "Scanning PI, DOGE, BGB, XRP, SUI, PEPE, SHIB every 5 minutes.\n"
        "Signals fire automatically with 4H cooldown per pair.\n"
        "_" + expiry_str + "_\n\n"
        "Running immediate scan now...",
        parse_mode="Markdown"
    )
    log.info("Coupon monitor activated — running immediate scan")

    # Immediate scan on activation
    try:
        results = []
        for symbol in COUPON_PAIRS:
            r = await _scan_coupon_pair(symbol)
            if r is not None:
                results.append(r)

        if not results:
            await update.message.reply_text(
                "\U0001f3ab No MOMENTUM signals on coupon pairs right now.\n"
                "Bot will keep scanning every 5 minutes and fire when ready.",
                parse_mode="Markdown"
            )
            return

        # Pick best signal
        grade_order = {"A": 0, "B": 1, "C": 2}
        best = sorted(results, key=lambda x: (
            grade_order.get(x.get("grade", "B"), 1),
            -x.get("abs_score", 0)
        ))[0]

        from risk_manager import calc_levels_v2
        atr_pct = (best["atr"] / best["price"] * 100) if best["price"] > 0 else 1.0
        sl, tp1, tp2, vp_meta = calc_levels_v2(
            best["direction"], best["price"], best["atr"], atr_pct, best.get("df_4h")
        )
        best["vp_adjustments"] = vp_meta.get("vp_adjustments", [])
        best["vp_poc"]         = vp_meta.get("vp_poc") or best.get("vp_poc")
        best["vp_in_lvn"]      = vp_meta.get("vp_in_lvn", best.get("vp_in_lvn", False))

        ob_bias, ob_ratio   = get_order_book_bias(best["symbol"])
        pos_usdt, contracts = calc_position_size(best["price"], sl)

        # Record signal
        record_signal(best["symbol"], best["direction"], best["price"], sl, tp1, tp2,
                      best["confidence"], signal_type=best.get("signal_type", "MOMENTUM"),
                      grade=best.get("grade", "B"),
                      dow_phase=best.get("dow_phase", "UNCLEAR"),
                      vwap_bias=best.get("vwap_bias", "AT"))
        if paper_mode:
            existing = load_json(PAPER_FILE, [])
            already_open = any(t["symbol"] == best["symbol"] and t["status"] == "OPEN" for t in existing)
            if not already_open:
                open_paper_trade(best["symbol"], best["direction"], best["price"], sl, tp1, tp2,
                                 best["confidence"], signal_type=best.get("signal_type", "MOMENTUM"))

        active_signals[best["symbol"]] = {
            "direction": best["direction"], "entry": best["price"],
            "sl": sl, "tp1": tp1, "tp2": tp2,
            "tp1_hit": False, "atr": best["atr"],
            "time": _time.time(),
            "trailing_extreme": best["price"],
            "trailing_stop": None,
        }
        coupon_monitor["last_signals"][best["symbol"]] = _time.time()

        # Summary of all 7 pairs
        scanned = {r["symbol"]: r for r in results}
        summary = "\U0001f4cb *All 7 pairs scanned:*\n"
        for sym in COUPON_PAIRS:
            lbl = COUPON_LABELS[sym]
            if sym in scanned:
                m  = scanned[sym]
                g  = m.get("grade", "B")
                ge = {"A": "\U0001f7e3", "B": "\U0001f7e2", "C": "\U0001f7e1"}.get(g, "\U0001f7e2")
                summary += ge + " *" + lbl + "* " + m["direction"] + " | Q:" + str(m.get("abs_score", 0)) + " | " + g + "\n"
            else:
                summary += "\u26aa *" + lbl + "* — no signal\n"
        summary += "\n"

        header  = "\U0001f3ab *Coupon Signal: " + COUPON_LABELS.get(best["symbol"], best["symbol"].split("/")[0]) + "*\n"
        header += "50 USDT boost active \u2014 use your voucher on this trade\n\n"
        msg = header + summary + format_signal(best, sl, tp1, tp2, ob_bias, ob_ratio, pos_usdt, contracts)
        msg += format_ai_block(best.get("ai_result", {}))

        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            await update.message.reply_text(msg, parse_mode=None)

    except Exception as e:
        await update.message.reply_text("Coupon scan error: " + str(e))

async def cmd_btcstatus(update, ctx):
    """
    /btcstatus — shows current BTC circuit breaker state
    """
    check_btc_circuit_breaker()
    state  = btc_circuit_breaker
    active = state["active"]

    if active:
        triggered_ago = round((_time.time() - state["triggered_at"]) / 60)
        msg  = "\U0001f6a8 *BTC Circuit Breaker: ACTIVE*\n\n"
        msg += "New LONG signals are blocked.\n"
        msg += "Reason: " + state["reason"] + "\n"
        msg += "Active for: " + str(triggered_ago) + " minutes\n\n"
        msg += "_LONGs resume when BTC 1H RSI > 48, drop < 1%, and 4H not TRENDING DOWN._"
    else:
        msg  = "\u2705 *BTC Circuit Breaker: INACTIVE*\n\n"
        msg += "LONG signals are enabled normally."

    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg, parse_mode=None)


# ─── TRADE WATCH COMMANDS ──────────────────────────────────────────────────────
def fetch_price_any_exchange(symbol: str) -> float:
    """
    Fetch current price — tries Bitget first, falls back to MEXC public API.
    Allows watching MEXC futures positions not listed on Bitget.
    """
    # Try Bitget first
    try:
        return float(exchange.fetch_ticker(symbol)["last"])
    except Exception:
        pass
    # Fallback: MEXC public ticker API (no auth needed)
    try:
        import urllib.request as _ur, json as _j
        sym_clean = symbol.replace("/USDT:USDT", "USDT").replace("/", "")
        url = "https://contract.mexc.com/api/v1/contract/ticker?symbol=" + sym_clean + "_USDT"
        with _ur.urlopen(url, timeout=5) as r:
            data = _j.loads(r.read())
        if data.get("success") and data.get("data"):
            return float(data["data"].get("lastPrice", 0))
    except Exception:
        pass
    return 0.0


def fetch_ohlcv_mexc(symbol: str, timeframe: str = "1h", limit: int = 100):
    """
    Fetch OHLCV from MEXC for coins not on Bitget.
    Returns pandas DataFrame in same format as fetch_ohlcv().
    """
    try:
        import ccxt, pandas as _pd
        mexc = ccxt.mexc({"enableRateLimit": True})
        sym_mexc = symbol.replace(":USDT", "")  # MEXC uses BTC/USDT not BTC/USDT:USDT
        tf_map   = {"15m": "Min15", "1h": "Min60", "4h": "Hour4"}
        ohlcv    = mexc.fetch_ohlcv(sym_mexc, timeframe, limit=limit)
        df = _pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = _pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df
    except Exception as e:
        log.warning("MEXC OHLCV fetch error " + symbol + ": " + str(e))
        return None


async def cmd_watch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /watch BTC SHORT 95000
    Monitors a coin and alerts when a strong reversal appears against your trade.
    """
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "\U0001f440 *Watch Usage:*\n\n"
            "/watch BTC LONG 95000 swing\n"
            "/watch NEAR LONG 1.32 swing high 10 20\n"
            "  last 2 args = TP1% and TP2% targets\n\n"
            "Features: reversal warnings, trailing SL,\n"
            "profit targets, breakeven + time alerts.\n"
            "Stop: /unwatch BTC | Status: /watchstatus BTC",
            parse_mode="Markdown"
        )
        return
    coin      = args[0].upper()
    direction = args[1].upper()
    use_mexc  = any(a.lower() == "mexc" for a in args)
    symbol    = coin + "/USDT:USDT"

    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text("Direction must be LONG or SHORT.\nExample: /watch BTC SHORT 95000")
        return

    entry_price = None
    trade_type  = "swing"  # default to swing (safer/higher threshold)

    if len(args) >= 3:
        try:
            entry_price = float(args[2])
        except ValueError:
            await update.message.reply_text("Invalid entry price. Example: /watch BTC SHORT 95000 swing")
            return

    confidence = "medium"  # default

    if len(args) >= 4:
        if args[3].lower() in ("scalp", "swing"):
            trade_type = args[3].lower()
        elif args[3].lower() in ("high", "medium", "low"):
            confidence = args[3].lower()
        else:
            await update.message.reply_text(
                "4th arg must be trade type (scalp/swing) or confidence (high/medium/low).\n"
                "Example: /watch BTC SHORT 95000 swing high"
            )
            return

    if len(args) >= 5:
        if args[4].lower() in ("high", "medium", "low"):
            confidence = args[4].lower()

    # Optional TP targets: /watch NEAR LONG 1.32 swing medium 10 20
    tp1_pct = None
    tp2_pct = None
    for arg in args[5:]:
        try:
            val = float(arg)
            if tp1_pct is None:
                tp1_pct = val
            elif tp2_pct is None:
                tp2_pct = val
        except ValueError:
            pass

    quality_min = WATCH_QUALITY_MIN_SCALP if trade_type == "scalp" else WATCH_QUALITY_MIN_SWING

    if symbol in watched_trades:
        w = watched_trades[symbol]
        existing_dir = w["direction"]
        await update.message.reply_text(
            "👀 Already watching *" + coin + "* (" + existing_dir + ").\n\n"
            "Only one watch per coin to keep things stable.\n"
            "Use /unwatch " + coin + " first, then set the new watch.",
            parse_mode="Markdown"
        )
        return

    if len(watched_trades) >= WATCH_MAX:
        await update.message.reply_text(
            "⚠ Maximum " + str(WATCH_MAX) + " coins watched at once.\n"
            "Remove one with /unwatch before adding another."
        )
        return

    watched_trades[symbol] = {
        "label":              coin,
        "direction":          direction,
        "entry":              entry_price,
        "trade_type":         trade_type,
        "quality_min":        quality_min,
        "confidence":         confidence,
        "start_time":         _time.time(),
        "last_alert":         0,
        "last_early_alert":   0,
        # Trailing SL tracking
        "use_mexc":           use_mexc, # fetch price from MEXC if True
        "trail_sl":           None,   # current suggested SL level
        "trail_peak":         entry_price if entry_price else None,  # highest/lowest seen
        "trail_last_alert":   0,      # last time we sent a trail SL update
        "trail_step_pct":     None,   # None = dynamic (ATR-based), or fixed %
        # New features
        "tp1_pct":            tp1_pct,
        "tp2_pct":            tp2_pct,
        "tp1_hit":            False,
        "tp2_hit":            False,
        "be_alerted":         False,  # breakeven alert already sent
        "max_hold_hours":     72,     # time-based alert threshold (hours)
        "hold_alert_sent":    False,
    }

    entry_str  = " from $" + "{:.4f}".format(entry_price) if entry_price else ""
    conf_notes = {
        "high":   "\U0001f6e1 HIGH — 4H needs 4/7 signals. 1H noise suppressed. Best for trader signals.",
        "medium": "\U0001f7e1 MEDIUM — standard. 1H early warning + 4H confirmed.",
        "low":    "\U0001f6a8 LOW — maximum sensitivity. Warns at first 1H sign.",
    }
    trail_note = "\n\U0001f4cd Trailing SL active — dynamic ATR-based steps." if entry_price else ""
    mexc_note  = "\n\U0001f504 Fetching price from *MEXC* — coin not on Bitget." if use_mexc else ""
    tp_note = ""
    if tp1_pct:
        tp_note += "\n\U0001f3af TP1: +" + str(tp1_pct) + "%"
        if tp2_pct:
            tp_note += " | TP2: +" + str(tp2_pct) + "%"
        tp_note += " — will suggest Bitget trailing stop settings when hit."
    await update.message.reply_text(
        "\U0001f440 *Now watching " + coin + "*\n\n"
        "Direction: *" + direction + "* | Type: *" + trade_type.upper() + "*" + entry_str + "\n"
        + conf_notes.get(confidence, "") + trail_note + mexc_note + tp_note + "\n\n"
        "Status: /watchstatus " + coin + "\n"
        "Stop: /unwatch " + coin,
        parse_mode="Markdown"
    )
    log.info("Watch started: " + coin + " " + direction + entry_str)



async def cmd_watchstatus(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /watchstatus COIN — detailed status of a watched position.
    Shows: direction, entry, current price, PnL, trail SL, TP targets,
    hold time, next trail step, reversal cooldown.
    """
    args = ctx.args
    if not args:
        # Show all if no coin specified
        coins = list(watched_trades.keys())
        if not coins:
            await update.message.reply_text("No coins being watched.")
            return
        symbol = coins[0]
    else:
        symbol = args[0].upper() + "/USDT:USDT"

    if symbol not in watched_trades:
        coin = symbol.split("/")[0]
        await update.message.reply_text(
            coin + " is not being watched.\nUse /watching to see active watches."
        )
        return

    w         = watched_trades[symbol]
    direction = w["direction"]
    entry     = w.get("entry")
    label     = w["label"]
    now       = _time.time()
    dir_e     = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

    try:
        cur_price = fetch_price_any_exchange(symbol) if w.get("use_mexc") else float(exchange.fetch_ticker(symbol)["last"])
    except Exception:
        cur_price = 0

    hold_hrs = round((now - w["start_time"]) / 3600, 1)

    msg = "\U0001f4ca *Watch Status: " + label + "*\n\n"
    msg += dir_e + " *" + direction + "* | " + w.get("trade_type","swing").upper()
    msg += " | Conf: " + w.get("confidence","medium").upper() + "\n"
    msg += "Watching: *" + str(hold_hrs) + "h* / " + str(w.get("max_hold_hours",72)) + "h max\n\n"

    if entry and entry > 0:
        msg += "*Entry:* `$" + "{:.4f}".format(entry) + "`\n"
        if cur_price > 0:
            if direction == "LONG":
                cur_gain = (cur_price - entry) / entry * 100
            else:
                cur_gain = (entry - cur_price) / entry * 100
            pnl_lev = cur_gain * LEVERAGE
            pnl_e   = "\U0001f7e2" if cur_gain >= 0 else "\U0001f534"
            msg += "*Current:* `$" + "{:.4f}".format(cur_price) + "` " + pnl_e
            msg += " *" + "{:+.1f}".format(pnl_lev) + "%* at " + str(LEVERAGE) + "x\n\n"
        else:
            msg += "\n"

        # Trailing SL status
        trail_sl   = w.get("trail_sl")
        trail_peak = w.get("trail_peak")
        if trail_sl:
            msg += "\U0001f6e1 *Trail SL:* `$" + "{:.4f}".format(trail_sl) + "`\n"
        else:
            msg += "\U0001f6e1 Trail SL: not triggered yet\n"

        # TP status
        tp1 = w.get("tp1_pct")
        tp2 = w.get("tp2_pct")
        if tp1:
            tp1_hit = w.get("tp1_hit", False)
            tp1_e   = "\u2705" if tp1_hit else "\u23f3"
            tp1_price = round(entry * (1 + tp1/100), 4) if direction == "LONG"                         else round(entry * (1 - tp1/100), 4)
            msg += tp1_e + " TP1: *+" + str(tp1) + "%* → `$" + "{:.4f}".format(tp1_price) + "`"
            msg += " (HIT)\n" if tp1_hit else "\n"
        if tp2:
            tp2_hit = w.get("tp2_hit", False)
            tp2_e   = "\u2705" if tp2_hit else "\u23f3"
            tp2_price = round(entry * (1 + tp2/100), 4) if direction == "LONG"                         else round(entry * (1 - tp2/100), 4)
            msg += tp2_e + " TP2: *+" + str(tp2) + "%* → `$" + "{:.4f}".format(tp2_price) + "`"
            msg += " (HIT)\n" if tp2_hit else "\n"

        if tp1 or tp2:
            msg += "\n"

    # Reversal warning cooldowns
    l2_cd_left = max(0, round((1800 - (now - w.get("last_alert", 0))) / 60))
    l1_cd_left = max(0, round((3600 - (now - w.get("last_early_alert", 0))) / 60))
    if l2_cd_left > 0:
        msg += "\u26a0 4H reversal cooldown: " + str(l2_cd_left) + "m\n"
    if l1_cd_left > 0:
        msg += "\u26a1 1H warning cooldown: " + str(l1_cd_left) + "m\n"

    msg += "\n_/unwatch " + label + " to stop monitoring._"

    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg, parse_mode=None)

async def cmd_unwatch(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /unwatch BTC — stop watching a coin
    """
    args = ctx.args
    if not args:
        await update.message.reply_text("Usage: /unwatch BTC")
        return

    coin   = args[0].upper()
    symbol = coin + "/USDT:USDT"

    if symbol in watched_trades:
        del watched_trades[symbol]
        await update.message.reply_text("✅ Stopped watching *" + coin + "*.", parse_mode="Markdown")
        log.info("Watch stopped: " + coin)
    else:
        await update.message.reply_text(coin + " is not being watched.")


async def cmd_watching(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /watching — list all currently watched coins
    """
    if not watched_trades:
        await update.message.reply_text(
            "👀 No coins being watched.\n"
            "Start with: /watch BTC SHORT 95000"
        )
        return

    msg = "👀 *Currently Watching:*\n\n"
    now = _time.time()
    for symbol, w in watched_trades.items():
        try:
            price     = fetch_price_any_exchange(symbol) if w.get("use_mexc") else exchange.fetch_ticker(symbol)["last"]
            entry     = w.get("entry")
            direction = w["direction"]
            pnl_str   = ""
            if entry:
                pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
                pnl_lev = pnl_pct * LEVERAGE
                pnl_str = " | PnL: " + "{:+.1f}".format(pnl_lev) + "% at " + str(LEVERAGE) + "x"
        except Exception:
            price   = 0
            pnl_str = ""

        watched_mins = round((now - w["start_time"]) / 60)
        last_alert   = w.get("last_alert", 0)
        cooldown_left = max(0, round((WATCH_COOLDOWN - (now - last_alert)) / 60)) if last_alert else 0
        cooldown_str  = " | Cooldown: " + str(cooldown_left) + "m left" if cooldown_left > 0 else ""

        entry_str  = "$" + "{:.4f}".format(w["entry"]) if w.get("entry") else "not set"
        type_label = w.get("trade_type", "swing").upper()
        conf_label = w.get("confidence", "medium").upper()
        msg += "• *" + w["label"] + "* " + direction + " " + type_label + " [" + conf_label + "] | Entry: " + entry_str
        if price:
            msg += " | Now: $" + "{:.4f}".format(price)
        msg += pnl_str
        # Show current trailing SL
        trail_sl = w.get("trail_sl")
        if trail_sl:
            msg += " | Trail SL: `$" + "{:.4f}".format(trail_sl) + "`"
        msg += " | " + str(watched_mins) + "m" + cooldown_str + "\n"

    try:
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception:
        await update.message.reply_text(msg, parse_mode=None)


def _watch_pnl_str(entry, price, direction):
    """Helper — compute PnL string for watch alerts."""
    if not entry or not price:
        return ""
    pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
    pnl_lev = round(pnl_pct * LEVERAGE, 1)
    pnl_raw = round(pnl_pct, 2)
    return "PnL: " + "{:+.2f}".format(pnl_raw) + "% (" + "{:+.1f}".format(pnl_lev) + "% at " + str(LEVERAGE) + "x)\n"


def _check_reversal_signals(df: "pd.DataFrame", direction: str, mode: str = "watch") -> tuple:
    """
    Shared reversal detection used by both /watch (exit warnings) and /flip (entry signals).

    mode="watch" — detects reversal AGAINST an open position (7 checks, 1H/4H data)
    mode="flip"  — detects reversal TO ENTER a scalp (4 checks, 15m data)

    Both modes share the same 4 core checks. Watch adds 3 extra slow-reversal checks
    that are less relevant on 15m timeframes.

    Returns (signals: list[str], count: int, quality: str)
    quality: "STRONG" (3+), "MODERATE" (2), "WEAK" (<2)
    """
    import pandas_ta as _ta
    signals = []

    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]

        # ── 1. RSI ────────────────────────────────────────────────────────────
        rsi = _ta.rsi(close, length=14)
        if rsi is not None and len(rsi.dropna()) >= 3:
            rv = float(rsi.dropna().iloc[-1])
            if mode == "flip":
                # Flip: look for extreme RSI = local turning point
                if direction == "LONG" and rv < 35:
                    signals.append("RSI oversold (" + str(round(rv,1)) + ") — local low likely")
                elif direction == "SHORT" and rv > 65:
                    signals.append("RSI overbought (" + str(round(rv,1)) + ") — local high likely")
            else:
                # Watch: look for divergence = reversal against position
                if len(rsi.dropna()) >= 20:
                    rsi_v = rsi.dropna().values
                    pr_v  = close.values[-len(rsi_v):]
                    r_sl  = rsi_v[-20:]
                    p_sl  = pr_v[-20:]
                    if direction == "SHORT":
                        if p_sl[-1] < min(p_sl[:-1]) and r_sl[-1] > min(r_sl[:-1]):
                            signals.append("RSI bullish divergence")
                    else:
                        if p_sl[-1] > max(p_sl[:-1]) and r_sl[-1] < max(r_sl[:-1]):
                            signals.append("RSI bearish divergence")

        # ── 2. EMA9 slope reversing ───────────────────────────────────────────
        ema9 = _ta.ema(close, length=9)
        if ema9 is not None and len(ema9.dropna()) >= 6:
            ev         = ema9.dropna().values
            slope_now  = ev[-1] - ev[-3]
            slope_prev = ev[-4] - ev[-6]
            if direction == "LONG":
                if slope_prev < 0 and slope_now > slope_prev * 0.3:
                    signals.append("EMA9 slope reversing up — downswing losing steam")
                elif mode == "watch" and slope_prev > 0 and slope_now < slope_prev * 0.3:
                    signals.append("EMA9 slope flattening — uptrend losing steam")
            else:
                if slope_prev > 0 and slope_now < slope_prev * 0.3:
                    signals.append("EMA9 slope reversing down — upswing losing steam")
                elif mode == "watch" and slope_prev < 0 and slope_now > slope_prev * 0.3:
                    signals.append("EMA9 slope flattening — downtrend losing steam")

        # ── 3. Candle structure reversal ──────────────────────────────────────
        if len(close) >= 5:
            c = close.values
            h = high.values
            l = low.values
            if direction == "LONG":
                three_red = c[-4] < c[-5] and c[-3] < c[-4] and c[-2] < c[-3]
                reversal  = c[-1] > c[-2]
                hammer    = (c[-2] - l[-2]) > abs(c[-2] - c[-3]) * 1.5
                if three_red and reversal:
                    signals.append("3 red candles then bullish reversal candle")
                elif hammer and c[-1] > c[-2]:
                    signals.append("Hammer wick + current candle bullish")
            else:
                three_green = c[-4] > c[-5] and c[-3] > c[-4] and c[-2] > c[-3]
                reversal    = c[-1] < c[-2]
                shooting    = (h[-2] - c[-2]) > abs(c[-2] - c[-3]) * 1.5
                if three_green and reversal:
                    signals.append("3 green candles then bearish reversal candle")
                elif shooting and c[-1] < c[-2]:
                    signals.append("Shooting star wick + current candle bearish")

        # ── 4. Volume dry-up / exhaustion ────────────────────────────────────
        if len(volume) >= 6:
            v      = volume.values
            vol_ma = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
            if mode == "flip":
                declining = v[-3] > v[-2] > v[-1]
                low_vol   = v[-1] < vol_ma * 0.7
                if declining and low_vol:
                    signals.append("Volume drying up — current move exhausting")
            else:
                big_move     = len(close) >= 5 and abs(float(close.iloc[-4]) - float(close.iloc[-5])) / float(close.iloc[-5]) * 100 > 0.8
                vol_spike    = v[-4] > v[-10:-4].mean() * 1.5 if len(v) >= 10 else False
                vol_collapse = v[-1] < v[-4] * 0.5
                move_dir_ok  = (direction == "SHORT" and close.values[-4] < close.values[-5]) or                                (direction == "LONG"  and close.values[-4] > close.values[-5])
                if big_move and vol_spike and vol_collapse and move_dir_ok:
                    signals.append("Volume exhaustion on " + ("dump" if direction == "SHORT" else "pump"))

        # ── 5-7. Watch-only: slow reversal checks (not needed on 15m) ─────────
        if mode == "watch":
            # MACD crossover
            macd_df = _ta.macd(close, fast=12, slow=26, signal=9)
            if macd_df is not None and not macd_df.empty and len(macd_df) >= 3:
                dif_now  = float(macd_df.iloc[-1, 0])
                dea_now  = float(macd_df.iloc[-1, 2])
                dif_prev = float(macd_df.iloc[-2, 0])
                dea_prev = float(macd_df.iloc[-2, 2])
                if direction == "SHORT":
                    if dif_now > dea_now and dif_prev <= dea_prev:
                        signals.append("MACD bullish crossover")
                else:
                    if dif_now < dea_now and dif_prev >= dea_prev:
                        signals.append("MACD bearish crossover")

            # ROC momentum weakening
            roc = _ta.roc(close, length=10)
            if roc is not None and len(roc.dropna()) >= 5:
                roc_v = roc.dropna().values
                if direction == "SHORT":
                    if roc_v[-3] < -1.0 and roc_v[-1] > roc_v[-3] and roc_v[-1] > -0.5:
                        signals.append("Downward momentum weakening (ROC)")
                else:
                    if roc_v[-3] > 1.0 and roc_v[-1] < roc_v[-3] and roc_v[-1] < 0.5:
                        signals.append("Upward momentum weakening (ROC)")

            # Price vs EMA21 cross
            ema21 = _ta.ema(close, length=21)
            if ema21 is not None and len(ema21.dropna()) >= 3:
                ev21      = ema21.dropna().values
                price_now = float(close.iloc[-1])
                price_pre = float(close.iloc[-2])
                e21_now   = ev21[-1]
                e21_pre   = ev21[-2]
                if direction == "LONG":
                    if price_pre > e21_pre and price_now < e21_now:
                        signals.append("Price crossed below EMA21 — trend structure breaking")
                else:
                    if price_pre < e21_pre and price_now > e21_now:
                        signals.append("Price crossed above EMA21 — downtrend structure breaking")

            # 3-candle lower highs/lows (Dow structure)
            if len(high) >= 6:
                hv = high.values
                lv = low.values
                cv = close.values
                if direction == "LONG":
                    lh1 = hv[-2] < hv[-3]; lh2 = hv[-1] < hv[-2]
                    ll1 = lv[-2] < lv[-3]; ll2 = lv[-1] < lv[-2]
                    bear_closes = cv[-1] < cv[-2] < cv[-3]
                    if (lh1 and lh2 and ll1 and ll2) or (lh1 and lh2 and bear_closes):
                        signals.append("3 candles: lower highs + lower lows (slow bearish reversal)")
                else:
                    hh1 = hv[-2] > hv[-3]; hh2 = hv[-1] > hv[-2]
                    hl1 = lv[-2] > lv[-3]; hl2 = lv[-1] > lv[-2]
                    bull_closes = cv[-1] > cv[-2] > cv[-3]
                    if (hh1 and hh2 and hl1 and hl2) or (hh1 and hh2 and bull_closes):
                        signals.append("3 candles: higher highs + higher lows (slow bullish reversal)")

        # ── Supertrend — both modes: flip entry + watch exit ─────────────
        # Flip: price must be on the right side of Supertrend to enter
        # Watch: Supertrend flip against position = strong exit signal
        try:
            from signal_engine import compute_supertrend
            st_data = compute_supertrend(df)
            if mode == "flip":
                # For flip: only enter if Supertrend agrees with direction
                if direction == "LONG" and st_data["bullish"]:
                    signals.append("Supertrend bullish — flip LONG aligned")
                elif direction == "SHORT" and st_data["bearish"]:
                    signals.append("Supertrend bearish — flip SHORT aligned")
                # Fresh flip = strongest entry signal
                if st_data["just_flipped"]:
                    signals.append("Supertrend just flipped — fresh momentum signal")
            else:  # watch mode
                # For watch: Supertrend flipping AGAINST position = exit signal
                if direction == "LONG" and st_data["bearish"]:
                    if st_data["just_flipped"]:
                        signals.append("Supertrend flipped bearish — exit LONG signal")
                    else:
                        signals.append("Supertrend bearish — LONG opposing trend")
                elif direction == "SHORT" and st_data["bullish"]:
                    if st_data["just_flipped"]:
                        signals.append("Supertrend flipped bullish — exit SHORT signal")
                    else:
                        signals.append("Supertrend bullish — SHORT opposing trend")
        except Exception:
            pass

    except Exception as e:
        log.warning("Reversal signal check error (" + mode + "): " + str(e))

    count   = len(signals)
    quality = "STRONG" if count >= 3 else "MODERATE" if count == 2 else "WEAK"
    return signals, count, quality

async def _check_early_warning(df: "pd.DataFrame", direction: str) -> tuple:
    """Thin wrapper — calls unified _check_reversal_signals in watch mode."""
    sigs, count, _ = _check_reversal_signals(df, direction, mode="watch")
    return sigs, count

async def auto_watch(app):
    """
    Background loop — checks watched coins every 2 minutes.
    Uses 7 early warning signals — alert fires when 2+ agree.
    Two-level early warning system:

    SWING trades:
      Level 1 — 1H early warning  (⚡): 2/7 signals on 1H → tighten SL, prepare to exit
      Level 2 — 4H confirmed (⚠):  2/7 signals on 4H → reversal confirmed, close trade
      Level 1 cooldown: 60 min | Level 2 cooldown: 30 min
      Level 2 always checked first — if it fires, Level 1 is skipped that cycle

    SCALP trades:
      Single level on 15m — 2/7 signals → close trade
      Cooldown: 30 min

    7 checks (4 original + 3 new slow-reversal detectors):
      Original: RSI divergence, Volume exhaustion, MACD crossover, ROC weakening
      New:      EMA9 slope change, Price vs EMA21 cross, Candle structure (lower highs/lows)
    Alert fires at 2+ signals — catches both fast AND slow reversals.
    """
    log.info("Watch loop started")
    while True:
        await asyncio.sleep(WATCH_INTERVAL)
        if not watched_trades:
            continue
        for symbol, w in list(watched_trades.items()):
            try:
                now       = _time.time()
                direction = w["direction"]
                entry     = w.get("entry")
                label     = w["label"]
                is_swing  = w.get("trade_type", "swing") == "swing"

                import pandas_ta as _ta

                # ── Skip if symbol not available on chosen exchange ───────────
                if not w.get("use_mexc"):
                    try:
                        exchange.market(symbol)
                    except Exception:
                        log.warning("Watch skip — symbol not on Bitget: " + symbol + " (add 'mexc' flag to use MEXC)")
                        continue

                # ── Skip gracefully if symbol not available on exchange ────────
                if not w.get("use_mexc"):
                    try:
                        exchange.market(symbol)
                    except Exception:
                        log.warning("Watch skip — symbol not on Bitget: " + label + ". Use /watch ... mexc flag.")
                        continue

                # ── All position-tracking alerts (run every cycle) ────────────
                entry      = w.get("entry")
                trail_peak = w.get("trail_peak")
                trail_sl   = w.get("trail_sl")
                trail_cd   = 1800  # 30 min between trail SL alerts

                if entry and entry > 0:
                    try:
                        # Use MEXC data if flagged, else Bitget
                        if w.get("use_mexc"):
                            _df_trail = fetch_ohlcv_mexc(symbol, "1h", 20)
                            if _df_trail is None or len(_df_trail) == 0:
                                _df_trail = fetch_ohlcv(symbol, "1h", 20)
                        else:
                            _df_trail = fetch_ohlcv(symbol, "1h", 20)
                        if _df_trail is None or len(_df_trail) == 0:
                            raise ValueError("No OHLCV data available")
                        cur_price = float(_df_trail["close"].iloc[-1])

                        # ── Dynamic step size (ATR-based) ─────────────────────
                        _atr_s = _ta.atr(_df_trail["high"], _df_trail["low"], _df_trail["close"], length=14)
                        if _atr_s is not None and len(_atr_s.dropna()) > 0:
                            _atr_v   = float(_atr_s.dropna().iloc[-1])
                            _atr_pct = (_atr_v / cur_price * 100) if cur_price > 0 else 2.0
                            # Step = 2x ATR%, clamped 3%-10%
                            step_pct = max(3.0, min(10.0, round(_atr_pct * 2, 1)))
                        else:
                            step_pct = 5.0

                        # ── Update peak price in our favour ───────────────────
                        if direction == "LONG":
                            new_peak = max(trail_peak or entry, cur_price)
                            gain_pct = (new_peak - entry) / entry * 100
                            cur_gain = (cur_price - entry) / entry * 100
                        else:
                            new_peak = min(trail_peak or entry, cur_price)
                            gain_pct = (entry - new_peak) / entry * 100
                            cur_gain = (entry - cur_price) / entry * 100
                        watched_trades[symbol]["trail_peak"] = new_peak

                        pnl_lev = round(cur_gain * LEVERAGE, 1)
                        dir_e   = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

                        # ── Trailing SL update (Supertrend-aware) ─────────────
                        if gain_pct >= step_pct:
                            # Try Supertrend line as SL first — smarter than fixed %
                            use_st  = False
                            st_line = 0.0
                            try:
                                from signal_engine import compute_supertrend
                                st_data = compute_supertrend(_df_trail)
                                st_line = st_data["value"]
                                st_dir  = st_data["direction"]
                                use_st  = (
                                    st_line > 0 and
                                    ((direction == "LONG"  and st_dir == 1
                                      and entry < st_line < cur_price) or
                                     (direction == "SHORT" and st_dir == -1
                                      and cur_price < st_line < entry))
                                )
                            except Exception:
                                pass

                            lock_pct = gain_pct * 0.5
                            if use_st:
                                new_sl   = round(st_line, 6)
                                sl_src   = "Supertrend line (adaptive)"
                            else:
                                new_sl   = round(entry * (1 + lock_pct/100), 4) if direction == "LONG" \
                                           else round(entry * (1 - lock_pct/100), 4)
                                sl_src   = "ATR-based (locks " + "{:.1f}".format(lock_pct) + "% of gain)"

                            sl_improved = (
                                trail_sl is None or
                                (direction == "LONG"  and new_sl > trail_sl * 1.005) or
                                (direction == "SHORT" and new_sl < trail_sl * 0.995)
                            )
                            if sl_improved and now - w.get("trail_last_alert", 0) >= trail_cd:
                                watched_trades[symbol]["trail_sl"]         = new_sl
                                watched_trades[symbol]["trail_last_alert"] = now
                                msg  = "\U0001f4cd *Trail SL Update: " + label + "*\n\n"
                                msg += dir_e + " *" + direction + "* from `$" + "{:.4f}".format(entry) + "`\n"
                                msg += "Current: `$" + "{:.4f}".format(cur_price) + "` | PnL: *" + "{:+.1f}".format(pnl_lev) + "%* at " + str(LEVERAGE) + "x\n\n"
                                msg += "\U0001f6e1 *Move SL to: `$" + "{:.4f}".format(new_sl) + "`*\n"
                                msg += "_" + sl_src + "_\n\n"
                                msg += "\U0001f4cc *Bitget Trailing Stop settings:*\n"
                                msg += "  Callback Rate: *" + "{:.1f}".format(step_pct) + "%*\n"
                                msg += "  Activation Price: `$" + "{:.4f}".format(cur_price) + "`\n"
                                msg += "_Position → TP/SL → Trailing Stop_"
                                await send_msg(app, msg)
                                log.info("Trail SL: " + label + " SL→$" + str(new_sl) + " (" + ("ST" if use_st else "ATR") + ")")


                        # ── Profit target alerts ──────────────────────────────
                        tp1_pct = w.get("tp1_pct")
                        tp2_pct = w.get("tp2_pct")
                        if tp1_pct and not w.get("tp1_hit") and cur_gain >= tp1_pct:
                            watched_trades[symbol]["tp1_hit"] = True
                            callback = max(3.0, min(10.0, round(step_pct, 1)))
                            tp1_price = round(entry * (1 + tp1_pct/100), 4) if direction == "LONG"                                         else round(entry * (1 - tp1_pct/100), 4)
                            msg  = "\U0001f3af *TP1 Hit: " + label + "*\n\n"
                            msg += dir_e + " *+" + "{:.1f}".format(cur_gain) + "%* (" + "{:+.1f}".format(pnl_lev) + "% at " + str(LEVERAGE) + "x)\n"
                            msg += "Entry: `$" + "{:.4f}".format(entry) + "` → Now: `$" + "{:.4f}".format(cur_price) + "`\n\n"
                            msg += "\U0001f4cc *Set Bitget Trailing Stop now:*\n"
                            msg += "  Activation Price: `$" + "{:.4f}".format(tp1_price) + "`\n"
                            msg += "  Callback Rate: *" + str(callback) + "%*\n"
                            msg += "_Position → TP/SL → Trailing Stop_\n\n"
                            if tp2_pct:
                                msg += "\U0001f3af Next target: +" + str(tp2_pct) + "%"
                            await send_msg(app, msg)
                            log.info("TP1 hit: " + label + " +" + str(round(cur_gain,1)) + "%")

                        if tp2_pct and not w.get("tp2_hit") and cur_gain >= tp2_pct:
                            watched_trades[symbol]["tp2_hit"] = True
                            msg  = "\U0001f3af\U0001f3af *TP2 Hit: " + label + "*\n\n"
                            msg += dir_e + " *+" + "{:.1f}".format(cur_gain) + "%* (" + "{:+.1f}".format(pnl_lev) + "% at " + str(LEVERAGE) + "x)\n"
                            msg += "Entry: `$" + "{:.4f}".format(entry) + "` → Now: `$" + "{:.4f}".format(cur_price) + "`\n\n"
                            msg += "\U0001f4a1 Consider closing the full position or moving SL very tight."
                            await send_msg(app, msg)
                            log.info("TP2 hit: " + label + " +" + str(round(cur_gain,1)) + "%")

                        # ── Breakeven alert ───────────────────────────────────
                        # When price returns to entry after being in profit ≥ step_pct
                        if not w.get("be_alerted") and gain_pct >= step_pct:
                            at_be = (direction == "LONG"  and cur_price <= entry * 1.005) or                                     (direction == "SHORT" and cur_price >= entry * 0.995)
                            if at_be:
                                watched_trades[symbol]["be_alerted"] = True
                                msg  = "\u26a0 *Breakeven Alert: " + label + "*\n\n"
                                msg += dir_e + " Price returned to entry: `$" + "{:.4f}".format(cur_price) + "`\n"
                                msg += "Move SL to entry *now* to protect from loss."
                                await send_msg(app, msg)
                                log.info("Breakeven alert: " + label)

                        # ── Time-based alert ──────────────────────────────────
                        max_hours = w.get("max_hold_hours", 72)
                        hold_hrs  = (now - w["start_time"]) / 3600
                        if not w.get("hold_alert_sent") and hold_hrs >= max_hours:
                            watched_trades[symbol]["hold_alert_sent"] = True
                            msg  = "\U0001f552 *Hold Time Alert: " + label + "*\n\n"
                            msg += "Position open *" + str(round(hold_hrs, 1)) + " hours*\n"
                            msg += dir_e + " *" + direction + "* from `$" + "{:.4f}".format(entry) + "`\n"
                            msg += "Current PnL: *" + "{:+.1f}".format(pnl_lev) + "%* at " + str(LEVERAGE) + "x\n\n"
                            msg += "\U0001f4a1 Review your trade — no TP hit after " + str(max_hours) + "h.\n"
                            msg += "_Consider tightening SL or closing if thesis changed._"
                            await send_msg(app, msg)
                            log.info("Hold time alert: " + label + " " + str(round(hold_hrs,1)) + "h")

                    except Exception as e:
                        log.warning("Position tracking error " + label + ": " + str(e))

                # ── Scalp: single level on 15m only ──────────────────────────
                if not is_swing:
                    if now - w.get("last_alert", 0) < WATCH_COOLDOWN:
                        continue
                    df_scalp = fetch_ohlcv(symbol, "15m", 100)
                    price    = float(df_scalp["close"].iloc[-1])
                    sigs, count = await _check_early_warning(df_scalp, direction)
                    if count < 2:
                        continue
                    pnl_str   = _watch_pnl_str(entry, price, direction)
                    sigs_str  = "\n".join(["  \u2022 " + s for s in sigs])
                    entry_str = "$" + "{:.4f}".format(entry) if entry else "not set"
                    msg  = "\u26a0 *Reversal Warning: " + label + "* (SCALP)\n\n"
                    msg += "Your position: *" + direction + "* from " + entry_str + "\n"
                    msg += "Current price: `$" + "{:.4f}".format(price) + "`\n"
                    msg += pnl_str
                    msg += "\nSignals (" + str(count) + "/7 agree):\n" + sigs_str + "\n"
                    msg += "\n\U0001f4a1 Consider closing your *" + direction + "* position.\n"
                    msg += "_Use /unwatch " + label + " to stop monitoring._"
                    try:
                        await send_msg(app, msg)
                    except Exception:
                        await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=msg, parse_mode=None)
                    watched_trades[symbol]["last_alert"] = now
                    log.info("Scalp watch alert: " + label + " | " + str(count) + "/4: " + ", ".join(sigs))
                    continue

                # ── Swing: two-level system ───────────────────────────────────
                # Level 1 — Early warning on 1H (get ready, tighten SL)
                # Level 2 — Confirmed reversal on 4H (close the trade)
                # Separate cooldowns so Level 1 doesn't block Level 2

                # ── Confidence level thresholds ───────────────────────────────
                conf          = w.get("confidence", "medium")
                l2_min        = WATCH_CONF_HIGH_L2_MIN if conf == "high" else WATCH_CONF_MED_L2_MIN
                suppress_l1   = (conf == "high")  # HIGH: never fire 1H noise warnings

                # ── Level 2 first: 4H confirmed reversal ─────────────────────
                l2_cooldown_ok = now - w.get("last_alert", 0) >= WATCH_COOLDOWN
                if l2_cooldown_ok:
                    df_4h   = fetch_ohlcv_mexc(symbol, "4h", 100) if w.get("use_mexc") else fetch_ohlcv(symbol, "4h", 100)
                    if df_4h is None or len(df_4h) == 0:
                        continue
                    price   = float(df_4h["close"].iloc[-1])
                    sigs_4h, count_4h = await _check_early_warning(df_4h, direction)

                    # HIGH confidence: also verify 4H trend has actually broken
                    if conf == "high" and count_4h >= l2_min:
                        try:
                            regime_4h = detect_regime(df_4h)
                            trend_ok  = regime_4h["regime"] == "RANGING" or                                         (direction == "LONG"  and regime_4h["regime"] == "TRENDING_DOWN") or                                         (direction == "SHORT" and regime_4h["regime"] == "TRENDING_UP")
                            if not trend_ok:
                                log.info("HIGH conf watch: 4H trend intact for " + label + " — suppressing warning")
                                count_4h = 0  # suppress — 4H trend not yet broken
                        except Exception:
                            pass

                    if count_4h >= l2_min:
                        pnl_str   = _watch_pnl_str(entry, price, direction)
                        sigs_str  = "\n".join(["  \u2022 " + s for s in sigs_4h])
                        entry_str = "$" + "{:.4f}".format(entry) if entry else "not set"
                        conf_tag  = " [" + conf.upper() + "]" if conf != "medium" else ""
                        msg  = "\u26a0 *Reversal Warning: " + label + "* (4H CONFIRMED" + conf_tag + ")\n\n"
                        msg += "Your position: *" + direction + "* from " + entry_str + "\n"
                        msg += "Current price: `$" + "{:.4f}".format(price) + "`\n"
                        msg += pnl_str
                        msg += "\n4H signals (" + str(count_4h) + "/7 agree):\n" + sigs_str + "\n"
                        msg += "\n\U0001f6d1 *Strong reversal confirmed — consider closing your " + direction + " now.*\n"
                        msg += "_Use /unwatch " + label + " to stop monitoring._"
                        try:
                            await send_msg(app, msg)
                        except Exception:
                            await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=msg, parse_mode=None)
                        watched_trades[symbol]["last_alert"]       = now
                        watched_trades[symbol]["last_early_alert"] = now
                        log.info("4H confirmed alert: " + label + " conf=" + conf + " | " + str(count_4h) + "/7")
                        continue  # skip Level 1 this cycle

                # ── Level 1: 1H early warning ─────────────────────────────────
                # HIGH confidence: 1H warnings suppressed entirely — only 4H fires
                if suppress_l1:
                    continue

                l1_cooldown_ok = now - w.get("last_early_alert", 0) >= WATCH_EARLY_COOLDOWN
                if l1_cooldown_ok:
                    df_1h   = fetch_ohlcv_mexc(symbol, "1h", 100) if w.get("use_mexc") else fetch_ohlcv(symbol, "1h", 100)
                    if df_1h is None or len(df_1h) == 0:
                        continue
                    price   = float(df_1h["close"].iloc[-1])
                    sigs_1h, count_1h = await _check_early_warning(df_1h, direction)
                    l1_min  = 3 if conf == "low" else 2  # LOW: same as medium, but fires faster
                    if count_1h >= l1_min:
                        pnl_str   = _watch_pnl_str(entry, price, direction)
                        sigs_str  = "\n".join(["  \u2022 " + s for s in sigs_1h])
                        entry_str = "$" + "{:.4f}".format(entry) if entry else "not set"
                        msg  = "\u26a1 *Early Warning: " + label + "* (1H forming)\n\n"
                        msg += "Your position: *" + direction + "* from " + entry_str + "\n"
                        msg += "Current price: `$" + "{:.4f}".format(price) + "`\n"
                        msg += pnl_str
                        msg += "\n1H signals (" + str(count_1h) + "/7 agree):\n" + sigs_str + "\n"
                        msg += "\n\U0001f4a1 Reversal forming on 1H — consider tightening SL.\n"
                        msg += "_Waiting for 4H confirmation..._"
                        try:
                            await send_msg(app, msg)
                        except Exception:
                            await app.bot.send_message(chat_id=TELEGRAM_CHAT, text=msg, parse_mode=None)
                        watched_trades[symbol]["last_early_alert"] = now
                        log.info("1H early warning: " + label + " conf=" + conf + " | " + str(count_1h) + "/7")

            except Exception as e:
                log.error("Watch loop error " + symbol + ": " + str(e))

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
    app.add_handler(CommandHandler("alpha",     cmd_alpha))
    app.add_handler(CommandHandler("coupon",    cmd_coupon))
    app.add_handler(CommandHandler("btcstatus", cmd_btcstatus))
    app.add_handler(CommandHandler("flip",     cmd_flip))
    app.add_handler(CommandHandler("flipping", cmd_flipping))
    app.add_handler(CommandHandler("watch",    cmd_watch))
    app.add_handler(CommandHandler("unwatch",  cmd_unwatch))
    app.add_handler(CommandHandler("watching",   cmd_watching))
    app.add_handler(CommandHandler("watchstatus", cmd_watchstatus))

    async def error_handler(update, context):
        log.error("Telegram error: " + str(context.error))
    app.add_error_handler(error_handler)
    
    async def post_init(application):
        asyncio.create_task(auto_scan(application))
        asyncio.create_task(auto_news(application))
        asyncio.create_task(auto_monitor(application))
        asyncio.create_task(auto_coin_refresh(application))
        asyncio.create_task(auto_weekly_report(application))
        asyncio.create_task(auto_price_alerts(application))
        asyncio.create_task(auto_watch(application))
        asyncio.create_task(auto_btc_monitor(application))
        asyncio.create_task(auto_coupon_monitor(application))
        asyncio.create_task(auto_flip(application))
        asyncio.create_task(auto_gainers_scan(application))
        asyncio.create_task(auto_alpha_refresh(application))
        asyncio.create_task(auto_alpha_scan(application))

    app.post_init = post_init
    log.info("Phyrobot starting — top " + str(TOP_COINS) + " coins | 1H+4H")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()