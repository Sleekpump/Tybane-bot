"""
Phyrobot — Multi-Coin Signal Bot
Timeframes: 4H + 1D | 40 coins dynamic | Score-based signals
"""

import os, json, re, logging, asyncio, feedparser
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

# ─── STARTUP VALIDATION ────────────────────────────────────────────────────────
_missing = [v for v in ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID"] if not os.getenv(v)]
if _missing:
    raise EnvironmentError(f"Missing required env vars: {', '.join(_missing)}. Check Railway config.")
if not GROQ_KEY:
    logging.getLogger(__name__).warning("GROQ_API_KEY not set — news analysis disabled")

# ─── AUTHORIZATION DECORATOR ───────────────────────────────────────────────────
from functools import wraps

def owner_only(func):
    """Restricts command to bot owner only. Silently ignores all other senders."""
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        if update.message is None:
            return
        if str(update.message.chat_id) != str(TELEGRAM_CHAT):
            log.warning(f"Unauthorized: chat_id={update.message.chat_id} tried /{func.__name__}")
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

# ─── STATE ─────────────────────────────────────────────────────────────────────
COINS               = []
COIN_LABELS         = {}
last_signal         = {}
_refine_last_call   = {}   # {user_id: timestamp} — prevents /refine spam (10s cooldown)
active_signals      = {}
reentry_cooldown    = {}  # {symbol: timestamp} — prevents re-entry spam
watched_trades      = {}  # {symbol: {direction, entry, start_time, last_alert}} — /watch command
btc_circuit_breaker = {   # BTC dump protection — tiered response (elevated / full)
    "active":       False,
    "tier":         "none",   # "none" | "elevated" | "full"
    "triggered_at": 0,
    "reason":       "",
    "last_check":   0,
}
# BTC 4H market regime cache — updated by check_btc_circuit_breaker (runs every 30 min).
# Used for signal invalidation: if BTC flips to TRENDING_DOWN while a LONG is open,
# the macro context that validated the setup is gone.
_btc_market_regime_cache = {
    "regime":     "RANGING",   # TRENDING_UP / TRENDING_DOWN / RANGING
    "adx":        0.0,
    "updated_at": 0,
}
coupon_monitor = {        # Auto-coupon scanner state
    "active":      False, # toggle via /coupon
    "last_signals": {},   # {symbol: timestamp} cooldown tracker
}
flip_trades = {}          # Active flip scalps: {symbol: {direction, ref_price, target_pct, entry, ...}}
FLIP_CHECK_INTERVAL = 120  # check every 2 minutes
FLIP_COOLDOWN       = 1800 # 30 min between flips on same coin
FLIP_MAX_ATR_MULT   = 3.0  # pause flipping if ATR > 3x normal (too erratic)

_manual_prices = {}       # Manual price overrides
_oi_cache = {}            # Rolling OI cache {symbol: prev_oi} for change detection

# BTC dominance mode — controls full vs simplified pipeline
_btc_dom_state = {
    "dominance":    None,   # last fetched BTC dominance %
    "mode":         "simplified",  # "full" or "simplified"
    "last_fetch":   0,
    "last_alert":   0,
}
BTC_DOM_FULL_THRESHOLD       = 52.0  # below this → full pipeline (altseason)
BTC_DOM_SIMPLIFIED_THRESHOLD = 55.0  # above this → simplified pipeline (BTC season)
BTC_DOM_FETCH_INTERVAL       = 1800  # fetch every 30 min

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
HISTORY_FILE        = "signal_history.json"
PAPER_FILE          = "paper_trades.json"
WATCH_FILE          = "watched_trades.json"
COUNTER_SCAN_FILE   = "counter_scan_state.json"
BLACKLIST_FILE = "blacklist.json"
ALERTS_FILE    = "price_alerts.json"

def load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default

def save_json(path, data):
    """Atomic write — prevents file corruption if bot crashes mid-write."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"Save error {path}: {e}")
        try:
            os.remove(tmp)
        except Exception:
            pass


def update_trade_state(symbol: str, updates: dict):
    """
    Centralized trade state mutation — single source of truth.

    Applies updates to:
      1. active_signals[symbol]  — runtime dict (in-memory)
      2. Matching OPEN paper trade in PAPER_FILE — persistent copy

    Use everywhere instead of direct active_signals[symbol][key] = val
    for trade-lifecycle fields: tp1_hit, trailing_stop, trailing_extreme, tp2, sl
    """
    if symbol in active_signals:
        active_signals[symbol].update(updates)
    try:
        _pt_list = load_json(PAPER_FILE, [])
        _changed = False
        for _pt in _pt_list:
            if _pt.get("symbol") == symbol and _pt.get("status") == "OPEN":
                _pt.update(updates)
                _changed = True
        if _changed:
            save_json(PAPER_FILE, _pt_list)
    except Exception as _uts_e:
        log.error(f"update_trade_state error ({symbol}): {_uts_e}")
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
    """Fetch OHLCV with error handling. Returns None on failure — all callers must check."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw or len(raw) < 5:
            log.warning(f"fetch_ohlcv: insufficient data for {symbol} {timeframe}")
            return None
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.dropna()
        return df if len(df) >= 5 else None
    except Exception as e:
        log.warning(f"fetch_ohlcv error {symbol} {timeframe}: {e}")
        return None

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

        # ── Live funding rate — stamp onto result so exhaustion/derivatives
        # checks in the scan path see real data (not hardcoded 0.0) ─────────
        try:
            _fr_raw = exchange.fetch_funding_rate(symbol)
            # Store in raw Bitget scale (e.g. 0.0001 = 0.01%)
            # check_exhaustion thresholds are calibrated to this scale
            result["funding"] = float(_fr_raw.get("fundingRate", 0.0))
        except Exception:
            result["funding"] = 0.0   # non-fatal — exhaustion just won't fire

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
                df_grade     = result.get("df_1h")   # FIX: 1H is LTF (was using "df_4h" which was already mislabelled)
                df_4h_grade  = result.get("df_4h")   # FIX: no longer re-fetches 4H (was a redundant extra API call)
                if df_grade is not None and df_4h_grade is not None:
                    grade_result = grade_signal(
                        df_grade, df_4h_grade,
                        result["direction"],
                        result.get("abs_score", 0),
                        result.get("rsi_1h", 50),   # FIX: use 1H RSI (was rsi_4h which stored 1H data anyway)
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
                            result.get("df_1h") or result.get("df_4h"),  # FIX: df_ltf was NameError
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

        # ── 15m entry timing check ────────────────────────────────────────────
        try:
            from signal_engine import check_15m_entry_quality, detect_order_blocks, detect_bos_choch, detect_liquidity_pools
            df_15m = result.get("df_15m")
            df_1h  = result.get("df_1h")
            if df_15m is not None and len(df_15m) >= 20 and result["direction"] not in (None, "NEUTRAL"):
                # Pass SMC data so entry check uses structural levels not just EMA9
                _scan_ob   = None
                _scan_bos  = None
                try:
                    if df_1h is not None:
                        _ob_scan  = detect_order_blocks(df_1h)
                        _scan_ob  = (_ob_scan.get("demand_ob") if result["direction"] == "LONG"
                                     else _ob_scan.get("supply_ob"))
                        _bos_scan = detect_bos_choch(df_1h)
                        _liq_scan = detect_liquidity_pools(df_1h)
                        _scan_bos = {**_bos_scan,
                                     "sweep_bullish": _liq_scan.get("sweep_bullish", False),
                                     "sweep_bearish": _liq_scan.get("sweep_bearish", False)}
                except Exception:
                    pass
                entry_check = check_15m_entry_quality(
                    df_15m, result["direction"],
                    demand_ob=_scan_ob,
                    bos_data=_scan_bos,
                )
                result["entry_action"]   = entry_check["action"]
                result["entry_reason"]   = entry_check["reason"]
                result["entry_pullback"] = entry_check.get("pullback_target")
                result["rsi_15m"]        = entry_check["rsi_15m"]
            else:
                result["entry_action"]   = "NEUTRAL"
                result["entry_reason"]   = "15m data unavailable"
                result["entry_pullback"] = None
        except Exception as e:
            log.warning(f"15m entry check error for {symbol}: {e}")
            result["entry_action"] = "NEUTRAL"

        # ── Shakeout detection ────────────────────────────────────────────────
        try:
            from signal_engine import detect_shakeout
            df_15m = result.get("df_15m")
            df_1h  = result.get("df_1h")
            df_4h  = result.get("df_4h")
            if df_15m is not None and df_1h is not None:
                sk = detect_shakeout(df_15m, df_1h, df_4h, symbol)
                result["shakeout"]              = sk["is_shakeout"]
                result["shakeout_confidence"]   = sk["confidence"]
                result["shakeout_score"]        = sk["score"]
                result["shakeout_signals"]      = sk["signals"]
                result["shakeout_invalidation"] = sk["invalidation"]
                result["shakeout_pump_target"]  = sk["pump_target"]
                result["shakeout_flush_low"]    = sk["flush_low"]
                # If a shakeout is detected, upgrade direction to LONG
                # and flag it separately so signal formatting can label it
                if sk["is_shakeout"] and result["direction"] in (None, "NEUTRAL", "SHORT"):
                    result["direction"]       = "LONG"
                    result["shakeout_signal"] = True
                    result["type_reason"]     = (
                        f"SHAKEOUT detected ({sk['confidence']}, {sk['score']}/8) — "
                        f"crimepump setup | flush low ${sk['flush_low']:.4f}"
                    )
            else:
                result["shakeout"] = False
        except Exception as e:
            log.warning(f"Shakeout detection error for {symbol}: {e}")
            result["shakeout"] = False

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

        # ── Derivatives context classification ───────────────────────────
        # Fetches OI (current + previous snapshot) and runs the classifier.
        # Stores result so signal gates and format_signal can use it.
        # Fails silently — missing OI data = NEUTRAL context, no blocking.
        try:
            from signal_engine import classify_derivatives_context
            _oi_now  = None
            _oi_prev = None
            try:
                _oi_data = exchange.fetch_open_interest(symbol)
                _oi_now  = float(_oi_data["openInterestAmount"]) if _oi_data else None
                # Previous OI: stored in a rolling cache per symbol
                _oi_prev = _oi_cache.get(symbol)
                if _oi_now:
                    _oi_cache[symbol] = _oi_now
            except Exception:
                pass
            _change_24h = abs(result.get("relative_strength", 0)) + abs(_btc_change_cache.get("value", 0))
            # Use coin's own 24h change from df if available
            try:
                _df_c24 = result.get("df_4h")
                if _df_c24 is not None and len(_df_c24) >= 24:
                    _c24 = float(_df_c24["close"].iloc[-1])
                    _c24_prev = float(_df_c24["close"].iloc[-24])
                    _change_24h = (_c24 - _c24_prev) / _c24_prev * 100 if _c24_prev > 0 else 0
            except Exception:
                pass
            deriv_ctx = classify_derivatives_context(
                funding_rate=result.get("funding", 0),
                oi_now=_oi_now,
                oi_prev=_oi_prev,
                change_24h=_change_24h,
            )
            result["deriv_ctx"]       = deriv_ctx["state"]
            result["deriv_block_long"]= deriv_ctx["block_long"]
            result["deriv_boost_short"]= deriv_ctx["boost_short"]
            result["deriv_reason"]    = deriv_ctx["reason"]
            result["deriv_confidence"]= deriv_ctx["confidence"]
        except Exception as e:
            log.warning("Derivatives context error " + symbol + ": " + str(e))
            result["deriv_ctx"]        = "NEUTRAL"
            result["deriv_block_long"] = False
            result["deriv_boost_short"]= False

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
        if btc_df is not None and len(btc_df) >= 25:
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
                and r.get("grade", "B") != "C"   # exclude Grade C
                and (
                    _btc_dom_state["mode"] != "full"  # simplified: B+ ok
                    or r.get("grade") == "A"           # full: A only
                )
                # Derivatives gate: block LONG if squeeze or crowded context
                and not (
                    r["direction"] == "LONG"
                    and r.get("deriv_block_long", False)
                )]

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
from collections import deque as _dq
_SEEN_MAX = 500
seen_headlines = set()
_seen_order    = _dq(maxlen=_SEEN_MAX)

def _add_headline(title: str):
    if title in seen_headlines:
        return
    if len(seen_headlines) >= _SEEN_MAX:
        seen_headlines.discard(_seen_order[0])
    seen_headlines.add(title)
    _seen_order.append(title)
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
                    _add_headline(title)
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
        text = re.sub(r"```(?:json)?\s*", "", text).strip()  # handle all fence variants
        data = json.loads(text)
        return data.get("sentiment", "NEUTRAL"), data.get("key_points", [])
    except Exception as e:
        log.error("News analysis error: " + str(e))
        return "NEUTRAL", []

# ─── POSITION SIZING ───────────────────────────────────────────────────────────
def get_structural_tp(
    direction: str,
    entry_price: float,
    bos_data: dict,
    atr_value: float,
    trade_type: str = "swing",
) -> tuple:
    """
    Pattern 3 — CHoCH vs BOS changes the TP target.

    BOS (continuation): price broke structure in trend direction.
      TP1 = next swing high (LONG) / swing low (SHORT) from bos_data
      TP2 = 2.0x the BOS impulse distance

    CHoCH (reversal): first break against the trend — higher R:R but lower probability.
      TP1 = origin of the last impulsive leg (where the move started from)
      TP2 = 1.618 Fibonacci extension of the CHoCH impulse

    Falls back to ATR-based targets if BOS data unavailable.

    Returns (tp1_price, tp2_price, signal_type_label)
    """
    try:
        is_choch  = bos_data and (bos_data.get("choch_bullish") or bos_data.get("choch_bearish"))
        is_bos    = bos_data and (bos_data.get("bos_bullish") or bos_data.get("bos_bearish"))
        bos_level = bos_data.get("last_bos_level", 0) if bos_data else 0
        sw_high   = bos_data.get("swing_high_lvl", 0) if bos_data else 0
        sw_low    = bos_data.get("swing_low_lvl", 0)  if bos_data else 0

        if is_choch and bos_level > 0:
            # CHoCH reversal — TP1 at origin of last impulsive leg
            impulse_dist = abs(entry_price - bos_level)
            if direction == "LONG":
                tp1 = round(entry_price + impulse_dist, 8)
                tp2 = round(entry_price + impulse_dist * 1.618, 8)
            else:
                tp1 = round(entry_price - impulse_dist, 8)
                tp2 = round(entry_price - impulse_dist * 1.618, 8)
            label = "CHoCH reversal"

        elif is_bos and sw_high > 0 and sw_low > 0:
            # BOS continuation — TP1 at next swing level
            if direction == "LONG":
                tp1 = round(sw_high, 8)
                tp2 = round(entry_price + abs(sw_high - sw_low) * 1.5, 8)
            else:
                tp1 = round(sw_low, 8)
                tp2 = round(entry_price - abs(sw_high - sw_low) * 1.5, 8)
            label = "BOS continuation"

        else:
            # Fallback: ATR-based
            _m1, _m2 = (1.2, 2.2) if trade_type == "scalp" else (2.5, 4.5)
            if direction == "LONG":
                tp1 = round(entry_price + atr_value * _m1, 8)
                tp2 = round(entry_price + atr_value * _m2, 8)
            else:
                tp1 = round(entry_price - atr_value * _m1, 8)
                tp2 = round(entry_price - atr_value * _m2, 8)
            label = "ATR-based"

        # Sanity check — TPs must be on the right side of entry
        if direction == "LONG" and (tp1 <= entry_price or tp2 <= entry_price):
            raise ValueError(f"Invalid LONG TP: tp1={tp1} tp2={tp2} entry={entry_price}")
        if direction == "SHORT" and (tp1 >= entry_price or tp2 >= entry_price):
            raise ValueError(f"Invalid SHORT TP: tp1={tp1} tp2={tp2} entry={entry_price}")

        # TP ordering — TP2 must always be further from entry than TP1.
        # BOS continuation formula can produce TP2 < TP1 when swing range
        # is smaller than (TP1 - entry), so we enforce order explicitly.
        if direction == "LONG" and tp2 < tp1:
            # Recalculate TP2 as entry + 1.5 × (TP1 - entry)
            tp2 = round(entry_price + (tp1 - entry_price) * 1.5, 8)
        if direction == "SHORT" and tp2 > tp1:
            tp2 = round(entry_price - (entry_price - tp1) * 1.5, 8)

        return tp1, tp2, label

    except Exception as e:
        log.warning(f"get_structural_tp error: {e}")
        _m1, _m2 = (1.2, 2.2) if trade_type == "scalp" else (2.5, 4.5)
        if direction == "LONG":
            return (round(entry_price + atr_value * _m1, 8),
                    round(entry_price + atr_value * _m2, 8), "ATR-based")
        return (round(entry_price - atr_value * _m1, 8),
                round(entry_price - atr_value * _m2, 8), "ATR-based")


def calc_position_size(entry, sl):
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)
    sl_distance = abs(entry - sl) / entry
    if sl_distance < 1e-8:   # float equality == 0 is unsafe — epsilon guard prevents div-by-zero
        return 0, 0
    position_usdt = round(risk_amount / sl_distance, 2)
    contracts     = round(position_usdt * LEVERAGE / entry, 4)
    return position_usdt, contracts

TRAILING_BUFFER       = 0.08   # swing trailing buffer — 8% below/above price after TP1
TRAILING_BUFFER_SCALP = 0.04   # scalp trailing buffer — 4% (tighter, scalps resolve fast)

def record_signal(symbol, direction, entry, sl, tp1, tp2, confidence,
                  trade_type="swing", signal_type="MOMENTUM",
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

    grade_a_closed    = [s for s in closed if s.get("grade") == "A"]
    grade_b_closed    = [s for s in closed if s.get("grade") == "B"]
    grade_c_closed    = [s for s in closed if s.get("grade") == "C"]
    shakeout_closed   = [s for s in closed if s.get("shakeout")]
    accum_closed      = [s for s in closed if s.get("accumulating")]
    enter_now_closed  = [s for s in closed if s.get("entry_action") == "ENTER"]
    wait_ignored      = [s for s in closed if s.get("entry_action") == "WAIT"]

    return {
        "scalp":       calc_stats(scalp_closed),
        "swing":       calc_stats(swing_closed),
        "momentum":    calc_stats(momentum_closed),
        "reversal":    calc_stats(reversal_closed),
        "grade_a":     calc_stats(grade_a_closed),
        "grade_b":     calc_stats(grade_b_closed),
        "grade_c":     calc_stats(grade_c_closed),
        "shakeout":    calc_stats(shakeout_closed),
        "accumul":     calc_stats(accum_closed),
        "entry_now":   calc_stats(enter_now_closed),
        "entry_wait":  calc_stats(wait_ignored),
        "overall":     calc_stats(closed),
        "open":        len([s for s in history if s["outcome"] == "OPEN"]),
    }

# ─── PAPER TRADING ─────────────────────────────────────────────────────────────
paper_mode = False

def open_paper_trade(symbol, direction, entry, sl, tp1, tp2, confidence,
                     trade_type="swing", signal_type="MOMENTUM",
                     grade="B", dow_phase="UNCLEAR", shakeout=False,
                     entry_action="NEUTRAL", accumulating=False,
                     leverage=None):
    """
    Opens a paper trade with full metadata for post-test analysis.
    Stores grade, signal type, shakeout flag, entry timing, and leverage
    so /paper_stats can break down performance by each dimension.
    """
    trades   = load_json(PAPER_FILE, [])
    _lev     = leverage if leverage is not None else LEVERAGE
    trade = {
        "id":             len(trades) + 1,
        "symbol":         symbol,
        "label":          COIN_LABELS.get(symbol, symbol.split("/")[0]),
        "direction":      direction,
        "entry":          entry,
        "sl":             sl,
        "tp1":            tp1,
        "tp2":            tp2,
        "confidence":     confidence,
        "time":           _time.strftime("%Y-%m-%d %H:%M"),
        "open_timestamp": _time.time(),
        "trade_type":     trade_type,
        "signal_type":    signal_type,
        "grade":          grade,          # A / B / C
        "dow_phase":      dow_phase,      # ACCUMULATION / PARTICIPATION / DISTRIBUTION / UNCLEAR
        "shakeout":       shakeout,       # True if crimepump shakeout signal
        "entry_action":   entry_action,   # ENTER / WAIT / NEUTRAL (15m timing at entry)
        "accumulating":   accumulating,   # True if accumulation scanner flagged this
        "leverage":       _lev,           # actual leverage used
        "status":         "OPEN",
        "tp1_hit":        False,
        "trailing_stop":  None,     # set by update_trade_state when TP1 hits
        "trailing_extreme": None,   # highest/lowest price reached after TP1
        "btc_regime":     _btc_market_regime_cache.get("regime", "RANGING"),  # BTC 4H regime at entry
        "pnl_pct":        0,
        "pnl_usdt":       0,
        "original_signal": {
            "entry": entry, "sl": sl, "tp1": tp1, "tp2": tp2,
            "confidence": confidence, "direction": direction,
            "trade_type": trade_type, "signal_type": signal_type,
            "grade": grade, "dow_phase": dow_phase,
            "shakeout": shakeout, "entry_action": entry_action,
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
            _t_lev    = t.get("leverage", LEVERAGE)  # use stored leverage, not global

            # Auto-close scalp trades after 4 hours
            if t.get("trade_type") == "scalp":
                open_time = t.get("open_timestamp", _time.time())
                if _time.time() - open_time > 14400:  # 4 hours
                    pnl_pct = (price - entry) / entry * 100 * _t_lev if direction == "LONG" else (entry - price) / entry * 100 * _t_lev
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
                pnl_pct = (price - entry) / entry * 100 * _t_lev

                # Check trailing stop first (only after TP1 hit)
                # Read trailing state from paper trade itself — NOT active_signals,
                # which may already be deleted when trailing fires (desync bug fix).
                trailing_stop = None
                if t.get("tp1_hit"):
                    trailing_stop = t.get("trailing_stop") or None

                if trailing_stop and price <= trailing_stop:
                    t["status"]   = "WIN"   # trailing stop after TP1 = profit
                    t["pnl_pct"]  = round((trailing_stop - entry) / entry * 100 * _t_lev, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price <= t["sl"]:
                    # LONG LOSS — SL < entry, so result is negative
                    t["status"]   = "LOSS"
                    t["pnl_pct"]  = round((t["sl"] - entry) / entry * 100 * _t_lev, 2)  # negative
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price >= t["tp2"]:
                    # LONG WIN — TP2 > entry, so result is positive
                    t["status"]   = "WIN"
                    t["pnl_pct"]  = round((t["tp2"] - entry) / entry * 100 * _t_lev, 2)  # positive
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                else:
                    t["pnl_pct"]  = round(pnl_pct, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * pnl_pct / 100, 2)
                    if not t["tp1_hit"] and price >= t["tp1"]:
                        t["tp1_hit"] = True
            else:
                pnl_pct = (entry - price) / entry * 100 * _t_lev

                # Check trailing stop first (only after TP1 hit)
                # Read trailing state from paper trade itself — NOT active_signals,
                # which may already be deleted when trailing fires (desync bug fix).
                trailing_stop = None
                if t.get("tp1_hit"):
                    trailing_stop = t.get("trailing_stop") or None

                if trailing_stop and price >= trailing_stop:
                    t["status"]   = "WIN"   # trailing stop after TP1 = profit
                    t["pnl_pct"]  = round((entry - trailing_stop) / entry * 100 * _t_lev, 2)
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price >= t["sl"]:
                    # SHORT LOSS — SL > entry, so (entry - sl) is negative
                    t["status"]   = "LOSS"
                    t["pnl_pct"]  = round((entry - t["sl"]) / entry * 100 * _t_lev, 2)  # negative
                    t["pnl_usdt"] = round(ACCOUNT_SIZE * RISK_PCT / 100 * t["pnl_pct"] / 100, 2)
                    closed_now.append(t)
                elif price <= t["tp2"]:
                    # SHORT WIN — TP2 < entry, so (entry - tp2) is positive
                    t["status"]   = "WIN"
                    t["pnl_pct"]  = round((entry - t["tp2"]) / entry * 100 * _t_lev, 2)  # positive
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
    # Shakeout tag — shown prominently at the top of shakeout signals
    shakeout_tag = ""
    if r.get("shakeout_signal") or r.get("shakeout"):
        sk_conf = r.get("shakeout_confidence", "")
        sk_sc   = r.get("shakeout_score", 0)
        sk_low  = r.get("shakeout_flush_low", 0)
        shakeout_tag = "\n⚡ *SHAKEOUT* (" + str(sk_conf) + ", " + str(sk_sc) + "/8) — flush $" + "{:.4f}".format(sk_low) + " reclaimed\n"

    msg  = emoji + " *" + r["label"] + " Signal | " + datetime.now().strftime("%H:%M UTC") + "*\n"
    if shakeout_tag:
        msg += shakeout_tag
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
    msg += tf_emoji + " " + tf_label[0] + ": `" + str(r["score_4h"]) + "` | " + tf_label[1] + ": `" + str(r["score_1d"]) + "` | Funding: `" + "{:.3f}".format(r["funding"]*100) + "%`\n"
    # Derivatives context — always shown
    _dctx    = r.get("deriv_ctx", "NEUTRAL")
    _dreason = r.get("deriv_reason", "")
    _dctx_emoji = {
        "ORGANIC":  "\U0001f7e2",
        "SQUEEZE":  "\U0001f534",
        "CROWDED":  "\U0001f7e0",
        "NEUTRAL":  "\u26aa",
    }.get(_dctx, "\u26aa")
    if _dctx != "NEUTRAL" and _dreason:
        msg += _dctx_emoji + " Derivatives: *" + _dctx + "* — _" + _dreason + "_\n"
    else:
        msg += _dctx_emoji + " Derivatives: *" + _dctx + "* (funding neutral, no strong positioning)\n"
    msg += "\n"

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

    _sig_lev = r.get("suggested_leverage", LEVERAGE)
    msg += "*How to trade (Bitget " + str(_sig_lev) + "x Futures):*\n"
    msg += "  1\ufe0f\u20e3 Futures \u2192 " + r["label"] + "USDT \u2192 " + str(_sig_lev) + "x\n"
    msg += "  2\ufe0f\u20e3 " + dir_text + " at market\n"
    msg += "  3\ufe0f\u20e3 Set levels below\n\n"

    msg += "*Levels:*\n"
    msg += "  \U0001f7e1 Entry: `$" + "{:.4f}".format(r["price"]) + "`\n"
    _sig_lev = r.get("suggested_leverage", LEVERAGE)  # use signal-specific leverage
    msg += "  \U0001f534 SL:  `$" + "{:.4f}".format(sl) + "` (-" + "{:.1f}".format(sl_pct) + "% / -" + "{:.0f}".format(sl_pct*_sig_lev) + "% at " + str(_sig_lev) + "x)\n"
    msg += "  \U0001f3af TP1: `$" + "{:.4f}".format(tp1) + "` (+" + "{:.1f}".format(tp1_pct) + "% / +" + "{:.0f}".format(tp1_pct*_sig_lev) + "% at " + str(_sig_lev) + "x)\n"
    msg += "  \U0001f3af TP2: `$" + "{:.4f}".format(tp2) + "` (+" + "{:.1f}".format(tp2_pct) + "% / +" + "{:.0f}".format(tp2_pct*_sig_lev) + "% at " + str(_sig_lev) + "x)\n\n"

    msg += "*Position Sizing (" + str(RISK_PCT) + "% risk / $" + str(ACCOUNT_SIZE) + " account):*\n"
    msg += "  Size: $" + str(pos_usdt) + " | Contracts: " + str(contracts) + "\n"
    # Dynamic leverage suggestion
    sug_lev  = r.get("suggested_leverage")
    max_lev  = r.get("max_safe_leverage")
    lev_rsn  = r.get("leverage_reason", "")
    # Always show leverage — even when it matches default 10x
    if sug_lev:
        lev_emoji = "\U0001f7e2" if sug_lev >= 8 else "\U0001f7e1" if sug_lev >= 5 else "\U0001f534"
        msg += "  " + lev_emoji + " Suggested leverage: *" + str(sug_lev) + "x* (max safe: " + str(max_lev) + "x)\n"
        if lev_rsn:
            msg += "  _" + lev_rsn + "_\n"
    else:
        msg += "  \U0001f7e1 Suggested leverage: *10x* (default — no ATR data)\n"
    msg += "\n"

    msg += "*Key Signals:*\n"
    # Separate direction-supporting signals from opposing ones.
    # Previously all signals were shown together — causing contradictions like
    # "LONG Grade A" with "MBI bright red / Lower highs + lower lows" below it.
    _direction = r.get("direction", "LONG")
    _bearish_kw = {"bearish", "overbought", "falling", "lower high", "lower low",
                   "bright red", "death", "dump", "drop", "down", "short"}
    _bullish_kw = {"bullish", "oversold", "rising", "higher high", "higher low",
                   "bright green", "golden", "pump", "up", "long", "cross above"}

    supporting_sigs = []
    opposing_sigs   = []
    for s in r["signals"]:
        s_low = s.lower()
        is_bearish = any(k in s_low for k in _bearish_kw)
        is_bullish = any(k in s_low for k in _bullish_kw)
        if _direction == "LONG":
            if is_bullish and not is_bearish:
                supporting_sigs.append(s)
            elif is_bearish:
                opposing_sigs.append(s)
            else:
                supporting_sigs.append(s)  # neutral signals go with supporting
        else:
            if is_bearish and not is_bullish:
                supporting_sigs.append(s)
            elif is_bullish:
                opposing_sigs.append(s)
            else:
                supporting_sigs.append(s)

    # Filter low-level EMA9 proximity lines — they downgrade perceived intelligence.
    # EMA9 context is already captured in momentum/timing blocks above.
    _ema9_noise = {"price near 15m ema9", "near 15m ema9", "price at 15m ema9"}
    supporting_sigs = [s for s in supporting_sigs
                       if not any(n in s.lower() for n in _ema9_noise)]

    for s in supporting_sigs[:5]:
        msg += "  \u2022 " + s + "\n"
    if opposing_sigs:
        msg += "  \u26a0 *Opposing:* " + " | ".join(opposing_sigs[:2]) + "\n"

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

    # ── 15m entry timing display ─────────────────────────────────────────────
    entry_action   = r.get("entry_action", "NEUTRAL")
    entry_reason   = r.get("entry_reason", "")
    entry_pullback = r.get("entry_pullback")
    if entry_action == "WAIT":
        msg += f"\n⏳ *Entry timing: WAIT*\n_{entry_reason}_"
        if entry_pullback:
            msg += f"\nPullback target: `${entry_pullback:.4f}`"
        msg += "\n"
    elif entry_action == "ENTER":
        msg += f"\n✅ *Entry timing: NOW* — _{entry_reason}_\n"

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


def fetch_ohlcv_smart(symbol: str, timeframe: str = "1h", limit: int = 100):
    """
    Fetch OHLCV — tries Bitget first, falls back to MEXC automatically.
    Returns (df, exchange_used) tuple. df is None if both fail.
    """
    df = fetch_ohlcv(symbol, timeframe, limit)
    if df is not None and len(df) >= 5:
        return df, "bitget"
    # Bitget failed or coin not listed — try MEXC
    try:
        df = fetch_ohlcv_mexc(symbol, timeframe, limit)
        if df is not None and len(df) >= 5:
            return df, "mexc"
    except Exception as e:
        log.warning(f"fetch_ohlcv_smart MEXC fallback failed for {symbol} {timeframe}: {e}")
    return None, None


@owner_only
async def cmd_limit(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /limit SYMBOL DIRECTION PRICE [LEVERAGE]
    Pre-plans a limit order entry at a FUTURE price before price gets there.

    Unlike /refine (which analyses current price timing), /limit:
    - Accepts any price — including prices below/above current market
    - Computes structural confluence AT that price level
    - Shows whether a demand/supply OB, FVG, or swing level exists near that price
    - Gives SL, TP1, TP2 and /watch command pre-filled — ready to paste when filled

    Usage:
      /limit HIVE LONG 0.060        — plan a long limit at $0.060
      /limit SKYAI SHORT 0.90 15    — plan a short limit at $0.90 at 15x
    """
    args = ctx.args or []
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: /limit SYMBOL DIRECTION PRICE [LEVERAGE]\n"
            "Example: /limit HIVE LONG 0.060\n\n"
            "Plans a limit entry at a future price before it gets there."
        )
        return

    raw_symbol = args[0].upper()
    direction  = args[1].upper()
    try:
        limit_price = float(args[2])
    except ValueError:
        await update.message.reply_text("Invalid price. Use a number e.g. 0.060")
        return

    leverage   = int(args[3]) if len(args) >= 4 else LEVERAGE
    _type_arg  = args[4].lower() if len(args) >= 5 else ""
    if _type_arg in ("scalp", "s"):
        trade_type = "scalp"
    elif _type_arg in ("swing", "w"):
        trade_type = "swing"
    else:
        trade_type = "scalp" if leverage > 15 else "swing"

    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text("Direction must be LONG or SHORT")
        return

    symbol = raw_symbol + "/USDT:USDT"
    exchange_note = ""
    use_mexc = False
    try:
        exchange.market(symbol)
    except Exception:
        symbol = raw_symbol + "/USDT"
        exchange_note = " (MEXC spot)"
        use_mexc = True

    await update.message.reply_text(
        f"📍 Planning limit entry for {raw_symbol} {direction} @ ${limit_price}..."
    )

    try:
        import pandas_ta as _pta
        from signal_engine import (
            detect_order_blocks, detect_bos_choch, detect_liquidity_pools,
            detect_regime
        )

        # Fetch data
        df_1h = fetch_ohlcv(symbol, "1h", 200)
        df_4h = fetch_ohlcv(symbol, "4h", 100)
        if df_1h is None or df_4h is None or len(df_1h) < 50:
            await update.message.reply_text("Insufficient data for " + raw_symbol)
            return

        price_now = float(df_1h["close"].iloc[-1])
        gap_pct   = (limit_price - price_now) / price_now * 100

        # ── SMC analysis at limit price level ─────────────────────────────────
        ob   = detect_order_blocks(df_1h)
        bos  = detect_bos_choch(df_1h)
        liq  = detect_liquidity_pools(df_1h)

        # What structural level exists near the limit price? (within 3%)
        tol = 0.03
        confluence = []
        zone_quality = 0  # 0=none 1=weak 2=good 3=strong

        # Check demand/supply OB
        if direction == "LONG" and ob.get("demand_ob"):
            d = ob["demand_ob"]
            if abs(d["high"] - limit_price) / limit_price < tol:
                confluence.append(f"Demand OB top at ${d['high']:.4f} (entry zone)")
                zone_quality += 2
            elif abs(d["low"] - limit_price) / limit_price < tol:
                confluence.append(f"Demand OB bottom at ${d['low']:.4f} (tight — SL nearby)")
                zone_quality += 1
        elif direction == "SHORT" and ob.get("supply_ob"):
            s = ob["supply_ob"]
            if abs(s["low"] - limit_price) / limit_price < tol:
                confluence.append(f"Supply OB bottom at ${s['low']:.4f} (entry zone)")
                zone_quality += 2
            elif abs(s["high"] - limit_price) / limit_price < tol:
                confluence.append(f"Supply OB top at ${s['high']:.4f} (tight — SL nearby)")
                zone_quality += 1

        # Check BOS/CHOCH swing levels
        if bos.get("swing_low_lvl") and direction == "LONG":
            if abs(bos["swing_low_lvl"] - limit_price) / limit_price < tol:
                confluence.append(f"Recent swing low at ${bos['swing_low_lvl']:.4f}")
                zone_quality += 1
        if bos.get("swing_high_lvl") and direction == "SHORT":
            if abs(bos["swing_high_lvl"] - limit_price) / limit_price < tol:
                confluence.append(f"Recent swing high at ${bos['swing_high_lvl']:.4f}")
                zone_quality += 1

        # Check liquidity pool levels
        if liq.get("equal_lows_level") and direction == "LONG":
            if abs(liq["equal_lows_level"] - limit_price) / limit_price < tol:
                confluence.append(f"Sell-side liquidity pool at ${liq['equal_lows_level']:.4f} (stops here)")
                zone_quality += 2
        if liq.get("equal_highs_level") and direction == "SHORT":
            if abs(liq["equal_highs_level"] - limit_price) / limit_price < tol:
                confluence.append(f"Buy-side liquidity pool at ${liq['equal_highs_level']:.4f} (stops here)")
                zone_quality += 2

        # BOS/CHOCH structure context
        struct = bos.get("structure", "NEUTRAL")
        if direction == "LONG" and struct == "BULLISH":
            confluence.append("Structure BULLISH (HH+HL) — limit long aligns with trend")
            zone_quality += 1
        elif direction == "LONG" and struct == "BEARISH":
            confluence.append("Structure BEARISH (LH+LL) — limit long is counter-trend")
        elif direction == "SHORT" and struct == "BEARISH":
            confluence.append("Structure BEARISH (LH+LL) — limit short aligns with trend")
            zone_quality += 1

        # Zone rating
        if zone_quality >= 4:
            zone_label = "Strong zone — multiple confluences"
            zone_icon  = "🟢"
        elif zone_quality >= 2:
            zone_label = "Good zone — at least one structural level"
            zone_icon  = "🟡"
        elif zone_quality >= 1:
            zone_label = "Weak zone — marginal structural support"
            zone_icon  = "🟠"
        else:
            zone_label = "No structural confluence — arbitrary level"
            zone_icon  = "🔴"

        # ── SL calculation from limit price ───────────────────────────────────
        sl_price = None
        try:
            if direction == "LONG":
                if ob.get("demand_ob") and ob["demand_ob"].get("active"):
                    # SL = just below OB bottom
                    sl_price = round(ob["demand_ob"]["low"] * 0.990, 6)
                elif trade_type == "scalp":
                    sl_price = round(float(df_1h["low"].tail(8).min()) * 0.995, 6)
                else:
                    sl_price = round(float(df_4h["low"].tail(10).min()) * 0.99, 6)
            else:
                if ob.get("supply_ob") and ob["supply_ob"].get("active"):
                    sl_price = round(ob["supply_ob"]["high"] * 1.010, 6)
                elif trade_type == "scalp":
                    sl_price = round(float(df_1h["high"].tail(8).max()) * 1.005, 6)
                else:
                    sl_price = round(float(df_4h["high"].tail(10).max()) * 1.01, 6)
        except Exception:
            pass

        # ── TP calculation from limit price ───────────────────────────────────
        tp1_price = tp2_price = tp1_pct = tp2_pct = None
        try:
            _atr_df = df_4h if trade_type == "swing" else df_1h
            _atr    = float(_pta.atr(
                _atr_df["high"], _atr_df["low"], _atr_df["close"], length=14
            ).dropna().iloc[-1])
            _m1, _m2 = (2.5, 4.5) if trade_type == "swing" else (1.2, 2.2)
            if direction == "LONG":
                tp1_price = round(limit_price + _atr * _m1, 6)
                tp2_price = round(limit_price + _atr * _m2, 6)
            else:
                tp1_price = round(limit_price - _atr * _m1, 6)
                tp2_price = round(limit_price - _atr * _m2, 6)
            tp1_pct = round(abs(tp1_price - limit_price) / limit_price * 100, 1)
            tp2_pct = round(abs(tp2_price - limit_price) / limit_price * 100, 1)
        except Exception:
            pass

        # ── Build message ─────────────────────────────────────────────────────
        _type_label = "Scalp" if trade_type == "scalp" else "Swing"
        msg  = f"📍 *{raw_symbol} {direction} Limit Plan* — {_type_label}{exchange_note}\n"
        msg += f"Limit price: ${limit_price} | Now: ${price_now:.4f} ({gap_pct:+.1f}%)\n"

        if gap_pct > 0 and direction == "LONG":
            msg += f"Price needs to drop {abs(gap_pct):.1f}% to fill this limit.\n"
        elif gap_pct < 0 and direction == "LONG":
            msg += f"Price already below limit — set at market or revise up.\n"
        elif gap_pct < 0 and direction == "SHORT":
            msg += f"Price needs to rise {abs(gap_pct):.1f}% to fill this limit.\n"

        msg += f"\n{zone_icon} *Zone quality: {zone_label}*\n"
        if confluence:
            for c in confluence:
                msg += f"  - {c}\n"
        else:
            msg += "  No structural level detected near this price.\n"
            msg += "  Consider adjusting to align with OB or liquidity zone.\n"

        msg += "\n*Limit levels:*\n"
        msg += f"🎯 Entry: ${limit_price}\n"

        if sl_price:
            sl_pct = round(abs(sl_price - limit_price) / limit_price * 100, 1)
            _sl_tf = "1H" if trade_type == "scalp" else "4H"
            msg += f"🛑 SL ({_sl_tf}): ${sl_price} ({sl_pct}% away)\n"

        if tp1_price and tp2_price:
            msg += f"✅ TP1: ${tp1_price:.4f} (+{tp1_pct}%)\n"
            msg += f"✅ TP2: ${tp2_price:.4f} (+{tp2_pct}%)\n"

        # Pre-filled watch command — paste when limit fills
        _watch_tp = f" {tp1_pct} {tp2_pct}" if (tp1_pct and tp2_pct) else ""
        msg += f"\nWhen filled, paste this:\n"
        msg += f"`/watch {raw_symbol} {direction} {limit_price} {trade_type} {leverage}{_watch_tp}`"

        # Refine suggestion if zone is strong
        if zone_quality >= 2:
            msg += f"\n\nWhen price reaches ${limit_price}, run:\n"
            msg += f"`/refine {raw_symbol} {direction} {limit_price} {leverage}`"

        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            _plain = msg.replace("*", "").replace("`", "").replace("_", "")
            await update.message.reply_text(_plain)

    except Exception as e:
        log.error(f"cmd_limit error: {e}", exc_info=True)
        await update.message.reply_text("Limit plan error: " + str(e)[:200])


@owner_only
async def cmd_refine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /refine SYMBOL DIRECTION PRICE [LEVERAGE]
    Top trader signal refinement — checks if NOW is a good entry or if you should wait.

    Examples:
      /refine BAN LONG 0.085
      /refine CROSS SHORT 0.111 8
    """
    args = ctx.args
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: /refine SYMBOL LONG|SHORT PRICE [LEVERAGE]\n"
            "Example: /refine BAN LONG 0.085 8",
            parse_mode="Markdown"
        )
        return

    raw_symbol = args[0].upper()
    direction  = args[1].upper()
    try:
        entry_price = float(args[2])
    except ValueError:
        await update.message.reply_text("Invalid price. Use a number e.g. 0.085")
        return
    leverage   = int(args[3]) if len(args) >= 4 else LEVERAGE
    # Detect trade type — scalp or swing
    # Explicit 5th arg takes priority: /refine BTC LONG 95000 10 scalp
    # Otherwise: leverage > 15 = scalp, anything else = swing (default)
    # Previously >=10 → scalp, which meant the 10x default always showed scalp.
    # Most refine usage is for swing planning — default should be swing.
    _type_arg  = args[4].lower() if len(args) >= 5 else ""
    if _type_arg in ("scalp", "s"):
        trade_type = "scalp"
    elif _type_arg in ("swing", "w"):
        trade_type = "swing"
    else:
        trade_type = "scalp" if leverage > 15 else "swing"

    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text("Direction must be LONG or SHORT")
        return

    symbol = next((s for s in COIN_LABELS if s.split("/")[0].upper() == raw_symbol), None)
    if symbol is None:
        symbol = f"{raw_symbol}/USDT:USDT"

    # ── Cooldown BEFORE analysing — stop spam before any work is done ────────
    _uid = update.effective_user.id if update.effective_user else 0
    _now_r = _time.time()
    if _now_r - _refine_last_call.get(_uid, 0) < 10:
        await update.message.reply_text("⏳ Please wait a moment before running /refine again.")
        return
    _refine_last_call[_uid] = _now_r

    await update.message.reply_text("🔍 Analysing entry for " + raw_symbol + " " + direction + " @ $" + str(entry_price) + "...")

    try:
        import pandas_ta as _pta
        from signal_engine import (check_15m_entry_quality, grade_signal,
                                   detect_accumulation_setup, detect_shakeout)

        exchange_note = ""
        df_15m, _exch_15 = fetch_ohlcv_smart(symbol, "15m", 100)
        df_1h,  _exch_1h = fetch_ohlcv_smart(symbol, "1h",  200)
        df_4h,  _        = fetch_ohlcv_smart(symbol, "4h",  100)
        if _exch_1h == "mexc":
            exchange_note = " (via MEXC)"

        if df_1h is None or df_4h is None:
            await update.message.reply_text("Could not fetch data for that symbol. Check the ticker.")
            return

        price_now      = float(df_1h["close"].iloc[-1])
        price_diff_pct = (price_now - entry_price) / entry_price * 100

        # ── RSI helper ────────────────────────────────────────────────────
        def _rsi_val(df, length=14):
            try:
                s = _pta.rsi(df["close"], length=length)
                return round(float(s.dropna().iloc[-1]), 1) if s is not None and len(s.dropna()) > 0 else 50.0
            except Exception:
                return 50.0

        rsi_1h_v = _rsi_val(df_1h)

        # ── Price gap warning + forced WAIT ──────────────────────────────────
        # When price has moved significantly ABOVE a LONG entry (or BELOW a SHORT),
        # the original entry is no longer valid. Force WAIT and recalculate levels
        # from the pullback zone, not the stale entry price.
        # Threshold: 10% gap forces WAIT (was 5% warning only — contradicted ENTER verdict)
        gap_warning    = ""
        force_wait_gap = False
        abs_gap        = abs(price_diff_pct)

        if abs_gap >= 5.0:
            if direction == "LONG" and price_diff_pct > 5.0:
                if abs_gap >= 10.0:
                    force_wait_gap = True
                    gap_warning = (f"Price moved +{abs_gap:.1f}% above your entry — "
                                   f"original entry ${ entry_price} is no longer valid. "
                                   f"Levels below calculated from pullback zone, not ${entry_price}.")
                else:
                    gap_warning = (f"Price +{abs_gap:.1f}% above entry — "
                                   f"entry timing reflects NOW. Consider waiting for pullback.")
            elif direction == "LONG" and price_diff_pct < -5.0:
                gap_warning = (f"Price -{abs_gap:.1f}% below intended entry — "
                               f"you may get a better fill than ${entry_price} right now.")
            elif direction == "SHORT" and price_diff_pct < -5.0:
                if abs_gap >= 10.0:
                    force_wait_gap = True
                    gap_warning = (f"Price dropped -{abs_gap:.1f}% from your entry — "
                                   f"chasing here is risky. Levels calculated from bounce zone.")
                else:
                    gap_warning = (f"Price -{abs_gap:.1f}% below entry — "
                                   f"timing reflects NOW. Chasing a dump is risky.")
            elif direction == "SHORT" and price_diff_pct > 5.0:
                gap_warning = (f"Price +{abs_gap:.1f}% above SHORT entry — "
                               f"may be a better entry zone now than ${entry_price}.")

        # ── Trend strength check (ADX) — override WAIT in strong trends ──
        # If ADX > 28 and price is already moving in the signal direction,
        # a "WAIT for pullback" verdict is wrong — pullbacks may never come in momentum moves.
        strong_trend_override = False
        trend_strength_note   = ""
        try:
            adx_df = _pta.adx(df_15m["high"], df_15m["low"], df_15m["close"], length=14)
            if adx_df is not None and not adx_df.empty:
                _adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
                _dmp_col = [c for c in adx_df.columns if c.startswith("DMP_")]
                _dmn_col = [c for c in adx_df.columns if c.startswith("DMN_")]
                if _adx_col and _dmp_col and _dmn_col:
                    adx_val = float(adx_df[_adx_col[0]].iloc[-1])
                    dmp_val = float(adx_df[_dmp_col[0]].iloc[-1])
                    dmn_val = float(adx_df[_dmn_col[0]].iloc[-1])
                if adx_val > 28:
                    if direction == "LONG" and dmp_val > dmn_val:
                        strong_trend_override = True
                        trend_strength_note   = ("Strong trend active (ADX " + "{:.0f}".format(adx_val)
                                                 + ") — momentum favours LONG, pullback may not come.")
                    elif direction == "SHORT" and dmn_val > dmp_val:
                        strong_trend_override = True
                        trend_strength_note   = ("Strong trend active (ADX " + "{:.0f}".format(adx_val)
                                                 + ") — momentum favours SHORT, don't wait for bounce.")
        except Exception as adx_e:
            log.warning(f"Refine ADX error: {adx_e}")

        # ── 15m entry timing ──────────────────────────────────────────────
        entry_check = {"action": "NEUTRAL", "reason": "15m data unavailable",
                       "rsi_15m": 50.0, "pullback_target": None}
        if df_15m is not None and len(df_15m) >= 20:
            # Pass SMC data so pullback target uses structural levels (OB → FVG → swing)
            # rather than just EMA9. _ob_data and _fvg_levels computed below if not yet done.
            try:
                from signal_engine import detect_order_blocks, detect_bos_choch, detect_liquidity_pools
                _ob_r      = detect_order_blocks(df_1h)
                _ob_for_pb = _ob_r.get("demand_ob") if direction == "LONG" else _ob_r.get("supply_ob")
                # BOS/CHOCH + liquidity sweep — passed into 15m check for smarter verdict
                _bos_r  = detect_bos_choch(df_1h)
                _liq_r  = detect_liquidity_pools(df_1h)
                # Merge sweep signals into bos_data dict
                _bos_for_15m = {**_bos_r,
                                 "sweep_bullish": _liq_r.get("sweep_bullish", False),
                                 "sweep_bearish": _liq_r.get("sweep_bearish", False)}
            except Exception:
                _ob_for_pb   = None
                _bos_for_15m = None
            # Reuse FVG levels computed later — compute early if not yet done
            try:
                _fvg_for_pb = []
                for _df_f, _tf_l in [(df_15m, "15m"), (df_1h, "1H")]:
                    if _df_f is None or len(_df_f) < 5:
                        continue
                    _hs = _df_f["high"].values; _ls = _df_f["low"].values
                    for _i in range(1, len(_hs) - 1):
                        if _hs[_i-1] < _ls[_i+1]:
                            _mid = (_hs[_i-1] + _ls[_i+1]) / 2
                            if float(_df_f["close"].iloc[-1]) > _ls[_i+1]:
                                _fvg_for_pb.append({"type":"BULL","mid":_mid,"tf":_tf_l,
                                                    "top":_ls[_i+1],"bot":_hs[_i-1]})
                        if _ls[_i-1] > _hs[_i+1]:
                            _mid = (_ls[_i-1] + _hs[_i+1]) / 2
                            if float(_df_f["close"].iloc[-1]) < _hs[_i+1]:
                                _fvg_for_pb.append({"type":"BEAR","mid":_mid,"tf":_tf_l,
                                                    "top":_ls[_i-1],"bot":_hs[_i+1]})
            except Exception:
                _fvg_for_pb = []
            entry_check = check_15m_entry_quality(df_15m, direction,
                                                   demand_ob=_ob_for_pb,
                                                   fvg_levels=_fvg_for_pb,
                                                   bos_data=_bos_for_15m)

        # Strong trend override — flip WAIT/CONDITIONAL/NEUTRAL to ENTER
        # when ADX confirms momentum, BUT NOT when gap forced the WAIT —
        # a 15% gap means entry is stale regardless of trend strength.
        if (not force_wait_gap and strong_trend_override
                and entry_check["action"] in ("WAIT", "CONDITIONAL", "NEUTRAL")):
            entry_check["action"] = "ENTER"
            entry_check["reason"] = trend_strength_note

        # ── Grade the setup ───────────────────────────────────────────────
        # funding_rate must be initialized here — it's used by compute_signal_quality
        # and is fetched live later in the block below. Default 0.0 until fetched.
        funding_rate     = 0.0   # raw Bitget scale — matches check_exhaustion thresholds
        funding_rate_pct = 0.0   # percentage scale — for display only
        funding_warn     = ""
        grade_result = None
        try:
            from signal_engine import compute_signal_quality
            # Compute real quality score — was hardcoded at 60.0 making grade unreliable
            _sq = compute_signal_quality(df_1h, df_4h, symbol,
                                            funding_rate=funding_rate)
            _real_score = _sq.get("quality_score", 60.0)
            grade_result = grade_signal(df_1h, df_4h, direction,
                                        _real_score, rsi_1h_v, 0.0)
        except Exception as ge:
            log.warning(f"refine grade error: {ge}")

        # ── Accumulation check ────────────────────────────────────────────
        acc_result = detect_accumulation_setup(df_1h, df_4h, symbol)

        # ── Shakeout check ────────────────────────────────────────────────
        sk_result = {"is_shakeout": False, "score": 0, "confidence": "LOW",
                     "signals": [], "flush_low": 0, "pump_target": 0, "invalidation": 0}
        try:
            if df_15m is not None and direction == "LONG":
                sk_result = detect_shakeout(df_15m, df_1h, df_4h, symbol)
        except Exception as sk_e:
            log.warning(f"Refine shakeout error: {sk_e}")

        # ── FVG detection ─────────────────────────────────────────────────
        # Scan 15m and 1H for open Fair Value Gaps near current price.
        # FVGs are imbalance zones where price moved so fast it skipped levels.
        # For LONG: FVGs below price = support / re-entry zones
        # For SHORT: FVGs above price = resistance / entry zones
        fvg_levels = []
        try:
            for _df_fvg, _tf_label in [(df_15m, "15m"), (df_1h, "1H")]:
                if _df_fvg is None or len(_df_fvg) < 5:
                    continue
                highs  = _df_fvg["high"].values
                lows   = _df_fvg["low"].values
                closes = _df_fvg["close"].values
                # FVG: candle[i-1].high < candle[i+1].low (bullish gap)
                #      candle[i-1].low  > candle[i+1].high (bearish gap)
                for i in range(1, len(highs) - 1):
                    bull_fvg = highs[i-1] < lows[i+1]   # gap up — bullish FVG
                    bear_fvg = lows[i-1]  > highs[i+1]  # gap down — bearish FVG
                    if bull_fvg:
                        fvg_mid = (highs[i-1] + lows[i+1]) / 2
                        # Only include unfilled: current price must be above the gap
                        if price_now > lows[i+1]:
                            fvg_levels.append({"type": "BULL", "top": lows[i+1],
                                               "bot": highs[i-1], "mid": fvg_mid, "tf": _tf_label})
                    if bear_fvg:
                        fvg_mid = (lows[i-1] + highs[i+1]) / 2
                        # Only include unfilled: current price must be below the gap
                        if price_now < highs[i+1]:
                            fvg_levels.append({"type": "BEAR", "top": lows[i-1],
                                               "bot": highs[i+1], "mid": fvg_mid, "tf": _tf_label})

            # Sort and pick the 3 most relevant to direction
            if direction == "LONG":
                # Bull FVGs below price = support zones
                relevant = sorted(
                    [f for f in fvg_levels if f["type"] == "BULL" and f["mid"] < price_now],
                    key=lambda x: x["mid"], reverse=True)[:3]
            else:
                # Bear FVGs above price = resistance zones
                relevant = sorted(
                    [f for f in fvg_levels if f["type"] == "BEAR" and f["mid"] > price_now],
                    key=lambda x: x["mid"])[:3]

            fvg_levels = relevant
        except Exception as fvg_e:
            log.warning(f"Refine FVG error: {fvg_e}")
            fvg_levels = []

        # ── Live funding rate fetch — updates the funding_rate initialized above ──
        try:
            _fr = exchange.fetch_funding_rate(symbol)
            funding_rate     = float(_fr.get("fundingRate", 0.0))   # raw: 0.0001 = 0.01%
            funding_rate_pct = funding_rate * 100                    # display: 0.01%
            if direction == "LONG" and funding_rate_pct > 0.1:
                funding_warn = (f"⚠ Funding: +{funding_rate_pct:.3f}% — perp traders heavily long, "
                                f"smart money may fade this move")
            elif direction == "SHORT" and funding_rate_pct < -0.1:
                funding_warn = (f"⚠ Funding: {funding_rate_pct:.3f}% — perp traders heavily short, "
                                f"squeeze risk if price pushes up")
            elif abs(funding_rate_pct) > 0.05:
                funding_warn = f"Funding: {funding_rate_pct:+.3f}% (elevated)"
        except Exception:
            pass   # funding not available on this exchange/symbol — skip silently

        # ── BTC trend context ─────────────────────────────────────────────────
        # If BTC is in a confirmed downtrend and you're longing an alt,
        # probability drops — say so clearly.
        btc_context = ""
        try:
            from signal_engine import detect_regime as _detect_regime
            _btc_df4h = fetch_ohlcv("BTC/USDT:USDT", "4h", 50)
            if _btc_df4h is not None and len(_btc_df4h) >= 20:
                _btc_regime = _detect_regime(_btc_df4h)
                _btc_r      = _btc_regime.get("regime", "RANGING")
                _btc_adx    = _btc_regime.get("adx", 0)
                if direction == "LONG" and _btc_r == "TRENDING_DOWN" and _btc_adx > 22:
                    btc_context = (f"⚠ BTC 4H: TRENDING DOWN (ADX {_btc_adx:.0f}) — "
                                   f"longing alts against BTC downtrend reduces probability")
                elif direction == "SHORT" and _btc_r == "TRENDING_UP" and _btc_adx > 22:
                    btc_context = (f"⚠ BTC 4H: TRENDING UP (ADX {_btc_adx:.0f}) — "
                                   f"shorting alts against BTC uptrend reduces probability")
                elif _btc_r == "RANGING":
                    btc_context = f"BTC 4H: RANGING (ADX {_btc_adx:.0f}) — alt moves are independent"
        except Exception:
            pass   # BTC fetch failed — skip silently

        # ── Format message ────────────────────────────────────────────────
        def _c(s):
            return str(s).replace("*","").replace("`","").replace("_"," ").replace("[","").replace("]","").replace("✦","").replace("—","-")

        # ── OB reference — MUST be defined before _pullback_entry uses it ────
        _ob_ref   = None
        _ob_label = ""
        try:
            if _ob_r:
                if direction == "LONG" and _ob_r.get("demand_ob") and _ob_r["demand_ob"].get("active"):
                    _ob_ref   = _ob_r["demand_ob"]
                    _ob_label = "Demand OB"
                elif direction == "SHORT" and _ob_r.get("supply_ob") and _ob_r["supply_ob"].get("active"):
                    _ob_ref   = _ob_r["supply_ob"]
                    _ob_label = "Supply OB"
        except Exception:
            pass

        # ── SL — OB-aware, AFTER _ob_ref is defined ──────────────────────────
        # OB-based SL = just below OB bottom (LONG) / just above OB top (SHORT)
        # Tighter and more logical than swing-based — invalidation is structurally clear.
        sl_price = None
        try:
            if sk_result["is_shakeout"] and sk_result["invalidation"] > 0 and direction == "LONG":
                sl_price = round(sk_result["invalidation"], 6)
            elif _ob_ref:
                if direction == "LONG":
                    sl_price = round(_ob_ref["low"] * 0.993, 6)   # just below OB bottom
                else:
                    sl_price = round(_ob_ref["high"] * 1.007, 6)  # just above OB top
            elif acc_result["is_accumulating"] and acc_result["confidence"] == "HIGH":
                if direction == "LONG":
                    sl_price = round(float(df_1h["low"].tail(10).min()) * 0.988, 6)
                else:
                    sl_price = round(float(df_1h["high"].tail(10).max()) * 1.012, 6)
            elif trade_type == "scalp":
                if direction == "LONG":
                    sl_price = round(float(df_1h["low"].tail(8).min()) * 0.995, 6)
                else:
                    sl_price = round(float(df_1h["high"].tail(8).max()) * 1.005, 6)
            else:
                if direction == "LONG":
                    sl_price = round(float(df_4h["low"].tail(10).min()) * 0.99, 6)
                else:
                    sl_price = round(float(df_4h["high"].tail(10).max()) * 1.01, 6)
        except Exception:
            pass

        # ── _pullback_entry — MUST be defined before TP calc uses it ─────────
        # Default: use original entry price. Override when gap forced WAIT.
        _pullback_entry = entry_price

        # ── Force WAIT when price moved 10%+ from stale entry ────────────────
        if force_wait_gap:
            action      = "WAIT"
            action_icon = "⏳"
            if _ob_ref and direction == "LONG":
                _pullback_entry = _ob_ref["high"]
                entry_check["pullback_target"] = round(_ob_ref["high"], 6)
                entry_check["reason"] = (
                    f"Price ran +{abs_gap:.1f}% - wait for pullback to demand OB "
                    f"${_ob_ref['low']:.4f}-${_ob_ref['high']:.4f}"
                )
            elif _ob_ref and direction == "SHORT":
                _pullback_entry = _ob_ref["low"]
                entry_check["pullback_target"] = round(_ob_ref["low"], 6)
                entry_check["reason"] = (
                    f"Price dropped -{abs_gap:.1f}% - wait for bounce to supply OB "
                    f"${_ob_ref['low']:.4f}-${_ob_ref['high']:.4f}"
                )
            else:
                _pullback_entry = price_now
                entry_check["reason"] = (
                    f"Price moved {abs_gap:.1f}% from entry - no structural OB found, "
                    f"use chart to identify pullback zone"
                )
        else:
            # Normal path — action already set by entry_check
            action      = entry_check["action"]
            action_icon = {"ENTER": "✅", "WAIT": "⏳",
                           "CONDITIONAL": "🟡", "NEUTRAL": "⚪"}.get(action, "⚪")

        # ── TP1 / TP2 — Pattern 3: CHoCH vs BOS adjusts targets ─────────────
        # CHoCH reversal → TP at origin of impulse leg (higher R:R, lower prob)
        # BOS continuation → TP at next swing level (lower R:R, higher prob)
        # Fallback: ATR-based
        tp1_price = tp2_price = None
        tp1_pct_r = tp2_pct_r = None
        _tp_signal_type = "ATR-based"
        try:
            if trade_type == "scalp":
                _atr_tp = float(ta.atr(df_1h["high"], df_1h["low"],
                                       df_1h["close"], length=14).dropna().iloc[-1])
            else:
                _atr_tp = float(ta.atr(df_4h["high"], df_4h["low"],
                                       df_4h["close"], length=14).dropna().iloc[-1])
            _calc_from  = _pullback_entry
            tp1_price, tp2_price, _tp_signal_type = get_structural_tp(
                direction   = direction,
                entry_price = _calc_from,
                bos_data    = _bos_for_15m,   # already computed above
                atr_value   = _atr_tp,
                trade_type  = trade_type,
            )
            tp1_price = round(tp1_price, 6)
            tp2_price = round(tp2_price, 6)
            tp1_pct_r = round(abs(tp1_price - _calc_from) / _calc_from * 100, 1)
            tp2_pct_r = round(abs(tp2_price - _calc_from) / _calc_from * 100, 1)
        except Exception as _tp_e:
            log.warning(f"Refine TP calc error: {_tp_e}")

        # ── Pattern 2: POI density at the entry zone ──────────────────────────
        # Score how many structure layers overlap at this price (OB+FVG+struct+liq)
        # Single OB = weak. OB+FVG+swing low = Strong POI.
        _poi = {"density": 0, "label": "No confluence", "layers": []}
        try:
            from signal_engine import detect_poi_density
            _poi = detect_poi_density(df_1h, _pullback_entry, direction)
        except Exception:
            pass

        # ── Pattern 1: Sweep status at this level ────────────────────────────
        # Was the liquidity at this zone already swept and recovered?
        # Swept+recovered within 2 bars = cleanest entry (stop hunt complete)
        _sweep_status = ""
        try:
            _liq_r = detect_liquidity_pools(df_1h)
            _sba   = _liq_r.get("sweep_bars_ago")
            if direction == "LONG" and _liq_r.get("sweep_bullish"):
                _sba_txt = f"{_sba} bar ago" if _sba == 1 else f"{_sba} bars ago"
                _sweep_status = f"Sell-side liquidity swept and recovered ({_sba_txt}) — stop hunt complete ✦"
            elif direction == "SHORT" and _liq_r.get("sweep_bearish"):
                _sba_txt = f"{_sba} bar ago" if _sba == 1 else f"{_sba} bars ago"
                _sweep_status = f"Buy-side liquidity swept and recovered ({_sba_txt}) — stop hunt complete ✦"
            elif direction == "LONG" and _liq_r.get("sell_side_liq"):
                _ssl = _liq_r.get("equal_lows_level", 0)
                if _ssl > 0 and abs(_ssl - _pullback_entry) / _pullback_entry < 0.03:
                    _sweep_status = f"Sell-side liquidity at ${_ssl:.4f} not yet swept — pool below, wait for sweep"
            elif direction == "SHORT" and _liq_r.get("buy_side_liq"):
                _bsl = _liq_r.get("equal_highs_level", 0)
                if _bsl > 0 and abs(_bsl - _pullback_entry) / _pullback_entry < 0.03:
                    _sweep_status = f"Buy-side liquidity at ${_bsl:.4f} not yet swept — pool above, wait for sweep"
        except Exception:
            pass

        # ── Opposing direction check ──────────────────────────────────────────
        # Run grade on the other side so user knows which direction is cleaner
        _opp_dir   = "SHORT" if direction == "LONG" else "LONG"
        _opp_grade = None
        _opp_phase = None
        try:
            if grade_result and df_1h is not None and df_4h is not None:
                _opp_gr    = grade_signal(df_1h, df_4h, _opp_dir, _real_score, rsi_1h_v, 0.0)
                _opp_grade = _opp_gr.get("grade")
                _opp_phase = _opp_gr.get("dow_phase", "")
        except Exception:
            pass

        # ── Build ONE decisive context line ───────────────────────────────────
        if sk_result["is_shakeout"] and sk_result["confidence"] in ("HIGH", "MEDIUM"):
            context_line = "⚡ Shakeout detected — flush complete, continuation likely"
        elif acc_result["is_accumulating"] and acc_result["confidence"] == "HIGH":
            context_line = "📦 Accumulation active — smart money absorbing"
        elif grade_result:
            _g = grade_result["grade"]
            _dow = grade_result.get("dow_phase", "")
            _icon = "🥇" if _g == "A" else ("🥈" if _g == "B" else "🥉")
            context_line = f"{_icon} Grade {_g}" + (f" | {_dow}" if _dow and _dow != "UNCLEAR" else "")
            if grade_result.get("warnings"):
                context_line += f" ⚠ {grade_result['warnings'][0]}"
        else:
            context_line = ""

        # Opposing direction verdict — one line telling user which side wins
        opp_note = ""
        if _opp_grade and grade_result:
            _grade_rank = {"A": 3, "B": 2, "C": 1}
            _phase_rank = {"ACCUMULATION": 3, "PARTICIPATION": 2,
                           "DISTRIBUTION": 1, "UNCLEAR": 0}
            _this_g  = _grade_rank.get(grade_result["grade"], 0)
            _opp_g   = _grade_rank.get(_opp_grade, 0)
            _this_p  = _phase_rank.get(grade_result.get("dow_phase", ""), 0)
            _opp_p   = _phase_rank.get(_opp_phase or "", 0)
            if _this_g > _opp_g:
                opp_note = (f"✅ {direction} Grade {grade_result['grade']} > "
                            f"{_opp_dir} Grade {_opp_grade} — you have the right side")
            elif _opp_g > _this_g:
                opp_note = (f"⚠ {_opp_dir} Grade {_opp_grade} is stronger than "
                            f"this {direction} Grade {grade_result['grade']} — check your direction")
            elif _this_p > _opp_p:
                opp_note = (f"✅ {direction} ({grade_result.get('dow_phase','')}) beats "
                            f"{_opp_dir} ({_opp_phase}) on phase — {direction} favoured")
            elif _opp_p > _this_p:
                opp_note = (f"⚠ {_opp_dir} ({_opp_phase}) phase is more favourable "
                            f"— double-check your direction")
            else:
                # Same grade and phase — warn that both sides look equal
                # This usually means the market is at a decision point
                opp_note = (f"⚠ Both {direction} and {_opp_dir} score Grade "
                            f"{grade_result['grade']} | {grade_result.get('dow_phase','?')} "
                            f"— market is at a decision point, use structure to decide")

        # ── Message — narrative-driven, not indicator-dump ──────────────────
        # Structure: Header → Conviction → Market Context → Structure →
        #            Momentum → Risks → Levels → Sizing → Watch
        _type_label = "📈 Scalp" if trade_type == "scalp" else "📊 Swing"
        msg  = f"🔍 *{raw_symbol} {direction}* — {_type_label}{exchange_note}\n"
        msg += f"Entry: ${entry_price} | Now: ${price_now:.4f} ({price_diff_pct:+.1f}%)\n"

        if gap_warning:
            msg += f"⚠ {gap_warning}\n"

        # ── 1. SKIP detection — evaluate before conviction tier ──────────────
        # A SKIP is an explicit "don't take this trade" verdict with reasons.
        # It fires when 2+ independent conditions oppose the trade simultaneously.
        # Even on SKIP, a pullback entry is computed so the user knows where to
        # re-enter if conditions improve.
        _skip_reasons  = []
        _grade_val     = grade_result.get("grade", "B") if grade_result else "B"
        _gs_now        = (_sq.get("group_scores", {}) if "_sq" in locals() and _sq else {})

        # Direction alignment check — quality score belongs to the DETECTED direction.
        # If the detected direction opposes the user's requested direction, the score
        # of e.g. 85 means "strong SHORT", not "strong LONG". Using it for LONG
        # conviction would show HIGH CONVICTION when all groups voted against you.
        _detected_dir = (_sq.get("direction", direction)
                         if "_sq" in locals() and _sq else direction)
        _dir_aligned  = (_detected_dir == direction or _detected_dir == "NEUTRAL")

        if "_real_score" in locals() and _dir_aligned:
            _score_for_tier = _real_score
        elif "_real_score" in locals() and not _dir_aligned:
            # Score is for the opposite direction — treat as 0 for requested direction
            _score_for_tier = 0.0
        else:
            _score_for_tier = 60.0

        # ── SKIP condition evaluation ─────────────────────────────────────────
        # Hard SKIP: signal was rejected by compute_signal_quality pipeline.
        # Categorical rejections (volatility, candle quality, exhaustion, quality
        # below threshold) don't need a second condition — one is enough.
        _sq_rejected   = ("_sq" in locals() and _sq
                          and bool(_sq.get("reject_reason", "")))
        _reject_reason = (_sq.get("reject_reason", "")
                          if "_sq" in locals() and _sq else "")

        # Condition 0: direction mismatch (already set above in _skip_reasons)

        # Condition 1: quality score too low (rejected or near-rejected)
        if _score_for_tier < 40:
            _skip_reasons.append(f"Quality {_score_for_tier:.0f}/100 — no narrative consensus")

        # Condition 2: Grade C (late/weak entry)
        if _grade_val == "C":
            _skip_reasons.append("Grade C — late entry, unfavourable R:R")

        # Condition 3: all 3 groups are flat or absent (none voted for direction).
        # Empty group_scores means signal was rejected before groups were computed.
        _all_flat = all(abs(_gs_now.get(g, 0)) <= 0.12
                        for g in ("trend", "momentum", "structure"))
        if not _gs_now:
            # Groups never computed — filtered out before scoring
            _skip_reasons.append("Signal rejected at analysis gate — no group votes computed")
        elif _all_flat:
            _skip_reasons.append("All groups neutral — market has no directional bias")

        # Condition 4: BTC circuit breaker full block on a LONG
        if (direction == "LONG"
                and btc_circuit_breaker.get("active")
                and btc_circuit_breaker.get("tier") == "full"):
            _skip_reasons.append("BTC circuit breaker active — sustained 4H downtrend")

        # Condition 5: BTC 4H market regime directly opposes direction
        _cur_btc_r = _btc_market_regime_cache.get("regime", "RANGING")
        if direction == "LONG" and _cur_btc_r == "TRENDING_DOWN":
            _skip_reasons.append(f"BTC 4H trending DOWN — macro against LONG")
        elif direction == "SHORT" and _cur_btc_r == "TRENDING_UP":
            _skip_reasons.append(f"BTC 4H trending UP — macro against SHORT")

        # Condition 6: severe funding rate conflict
        if funding_warn and any(w in funding_warn.lower()
                                for w in ("extreme", "overcrowded", "block", "flush")):
            _skip_reasons.append(funding_warn)

        # Hard SKIP: pipeline rejection is definitive — no second condition needed.
        # Soft SKIP: needs 2+ conditions to avoid false positives on borderline setups.
        _is_skip = _sq_rejected or len(_skip_reasons) >= 2

        # Prepend pipeline rejection reason so user sees WHY it was blocked
        if _sq_rejected and _reject_reason:
            if not any(_reject_reason[:30] in r for r in _skip_reasons):
                _skip_reasons.insert(0, f"Pipeline: {_reject_reason}")

        # ── Pullback entry for SKIP / WAIT / CONDITIONAL ──────────────────────
        # Always compute a pullback suggestion when the bot says don't enter now.
        # Priority: 1) OB zone  2) existing pullback_target  3) ATR-based level
        _skip_pullback = None
        if _is_skip or action in ("WAIT", "CONDITIONAL", "NEUTRAL"):
            _skip_pullback = entry_check.get("pullback_target")
            if not _skip_pullback and _ob_ref:
                _skip_pullback = (_ob_ref["high"] if direction == "LONG"
                                  else _ob_ref["low"])
            if not _skip_pullback and price_now > 0:
                # ATR-based fallback: compute inline — atr_pct not in scope here
                try:
                    _atr_s   = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
                    _atr_abs = float(_atr_s.dropna().iloc[-1]) if _atr_s is not None and len(_atr_s.dropna()) > 0 else price_now * 0.02
                    _skip_pullback = round(
                        (price_now - _atr_abs * 1.5) if direction == "LONG"
                        else (price_now + _atr_abs * 1.5), 6
                    )
                except Exception:
                    pass  # skip pullback stays None — not fatal

        # ── 1. Conviction tier ────────────────────────────────────────────────
        if _is_skip:
            _conviction_label = "🔴 SKIP THIS TRADE"
        elif action == "ENTER":
            if _score_for_tier >= 75:
                _conviction_label = "🟢 HIGH CONVICTION"
            elif _score_for_tier >= 55:
                _conviction_label = "🟡 MODERATE CONVICTION"
            else:
                _conviction_label = "🟠 CONDITIONAL SETUP"
        elif action == "WAIT":
            _conviction_label = "⏳ WAIT — Setup valid, await better entry"
        elif action == "CONDITIONAL":
            _conviction_label = "🟠 CONDITIONAL — Mixed signals, wait for zone"
        else:
            _conviction_label = "⚪ NEUTRAL — No strong timing signal"

        msg += f"\n*{_conviction_label}*\n"

        # ── SKIP reasons block ────────────────────────────────────────────────
        if _is_skip:
            for _sr in _skip_reasons:
                msg += f"• {_sr}\n"
            if _skip_pullback:
                _pb_dist_pct = abs(price_now - _skip_pullback) / price_now * 100 if price_now > 0 else 0
                msg += f"\n*Better entry if conditions improve:*\n"
                msg += f"👀 Watch: ${_skip_pullback:.4f} ({_pb_dist_pct:.1f}% from now)\n"
                # What needs to change for a valid entry
                if _score_for_tier < 40:
                    msg += "Needs: Quality > 55 + at least 2 group votes\n"
                if _grade_val == "C":
                    msg += "Needs: Price pull back to OB / demand zone\n"
                if _all_flat:
                    msg += "Needs: Momentum or structure group to confirm direction\n"
                _watch_pb = _skip_pullback
                _watch_tp_sk = f" {tp1_pct_r} {tp2_pct_r}" if (tp1_pct_r and tp2_pct_r) else ""
                msg += f"`/watch {raw_symbol} {direction} {_watch_pb:.4f} {trade_type} {leverage}{_watch_tp_sk}`\n"

        # ── Confidence decomposition — shows WHY the score is what it is ──────
        # Weighted breakdown of the 3 meta-groups so the user sees which pillars
        # are strong (+) vs weak/opposing (-), not just a raw number.
        # Group scores are relative to the DETECTED direction. If detected
        # direction != requested direction, scores are flipped so the user
        # always reads + as "supports your trade" and - as "opposes your trade".
        if "_sq" in locals() and _sq:
            _gs = _sq.get("group_scores", {})
            if _gs:
                _dir_flip = (not _dir_aligned and _detected_dir != "NEUTRAL")
                def _grp_line(name, val, icon):
                    # Group scores are in absolute terms: positive=bullish, negative=bearish.
                    # For SHORT trades, negative scores support the trade — flip for display
                    # so + always means "supports your trade" regardless of direction.
                    if direction == "SHORT":
                        display_val = -val if not _dir_flip else val
                    else:
                        display_val = -val if _dir_flip else val
                    arrow = "+" if display_val > 0.12 else ("-" if display_val < -0.12 else "~")
                    bar   = "▓▓▓" if abs(display_val) > 0.35 else ("▓▓" if abs(display_val) > 0.18 else "▓")
                    return f"{icon} {arrow}{bar} {name} ({display_val:+.2f})"

                _decomp_lines = [
                    _grp_line("Trend",     _gs.get("trend",     0), "📈"),
                    _grp_line("Momentum",  _gs.get("momentum",  0), "⚡"),
                    _grp_line("Structure", _gs.get("structure", 0), "🏗"),
                ]
                _score_display = _score_for_tier  # already direction-adjusted
                msg += f"Quality: `{_score_display:.0f}/100`\n"
                if not _dir_aligned:
                    msg += f"⚠ Analysis detected {_detected_dir} — scores shown relative to your {direction}\n"
                for _dl in _decomp_lines:
                    msg += f"`{_dl}`\n"

        # ── 2. Market Context — what is happening narratively ────────────────
        # BTC regime + any HTF context line. This is the "why" before the "what".
        _ctx_lines = []
        if btc_context:
            _ctx_lines.append(btc_context)
        if context_line:
            _ctx_lines.append(context_line)
        if opp_note:
            _ctx_lines.append(opp_note)
        if _ctx_lines:
            msg += "\n*Market Context:*\n"
            for _cl in _ctx_lines:
                msg += f"{_cl}\n"

        # ── 3. Structure — SMC basis for the setup ────────────────────────────
        _struct_lines = []
        if _sweep_status:
            _struct_lines.append(f"💧 {_sweep_status}")
        _poi_icon = "🟢" if _poi["density"] >= 3 else ("🟡" if _poi["density"] == 2 else "🔴")
        if _poi["density"] > 0:
            _struct_lines.append(f"{_poi_icon} Zone: {_poi['label']} ({_poi['density']} layers: {chr(44).join(_poi['layers'])})")
        else:
            _struct_lines.append(f"{_poi_icon} Zone: Single level — limited structural confluence")
        if _struct_lines:
            msg += "\n*Structure:*\n"
            for _sl in _struct_lines:
                msg += f"{_sl}\n"

        # ── 4. Momentum — 15m entry timing ────────────────────────────────────
        # Strip raw EMA9 proximity fragments from the reason string.
        # entry_check["reason"] is a pipe-joined string built in signal_engine,
        # so we filter at the segment level before displaying.
        _ec_reason_raw = entry_check.get("reason", "")
        _ema9_fragments = {"price near 15m ema9", "near 15m ema9", "price at 15m ema9",
                           "clean zone", "within 0.1%", "within 0.2%", "within 0.3%"}
        _ec_parts = [
            seg.strip() for seg in _ec_reason_raw.split("|")
            if not any(n in seg.lower() for n in _ema9_fragments)
        ]
        _ec_reason = " | ".join(_ec_parts).strip(" |")
        if _ec_reason:
            msg += f"\n*Momentum:*\n{_ec_reason}\n"

        # ── 5. Risks — everything that could go wrong ─────────────────────────
        _risk_lines = []
        if grade_result:
            _vwap_dist = grade_result.get("vwap_dist_pct", 0)
            if _vwap_dist and abs(_vwap_dist) > 5.0:
                _vwap_side = "above" if _vwap_dist > 0 else "below"
                if direction == "LONG" and _vwap_dist > 10.0:
                    _risk_lines.append(f"⚠ VWAP: Price {abs(_vwap_dist):.1f}% above — premium to fair value")
                elif direction == "SHORT" and _vwap_dist < -10.0:
                    _risk_lines.append(f"⚠ VWAP: Price {abs(_vwap_dist):.1f}% below — bounce risk")
                else:
                    _risk_lines.append(f"VWAP: Price {abs(_vwap_dist):.1f}% {_vwap_side} ${grade_result.get('vwap', 0):.4f}")
        if funding_warn:
            _risk_lines.append(funding_warn)
        if _risk_lines:
            msg += "\n*Risks:*\n"
            for _rl in _risk_lines:
                msg += f"{_rl}\n"

        # ── 6. Levels ─────────────────────────────────────────────────────────
        _ref = _pullback_entry
        msg += "\n*Levels:*\n"

        if _is_skip and _skip_pullback:
            # On SKIP, show the pullback zone as the only actionable level
            msg += f"🎯 Watch zone: ${_skip_pullback:.4f} (do not enter at current price)\n"
        elif entry_check.get("pullback_target") and action in ("WAIT", "CONDITIONAL"):
            _pb = entry_check["pullback_target"]
            msg += f"🎯 Pullback entry: ${_pb:.4f}"
            if force_wait_gap:
                msg += " (OB top — where buyers stepped in)"
            msg += "\n"
        else:
            msg += f"🎯 Entry: ${_ref:.4f} (now)\n"

        if _ob_ref:
            _at_zone = _ob_r.get("price_at_demand_ob") or _ob_r.get("price_at_supply_ob")
            if direction == "LONG":
                msg += (f"📍 Demand OB: ${_ob_ref['high']:.4f} (entry zone)"
                        f" | ${_ob_ref['low']:.4f} (SL below this)"
                        f"{'  at zone now' if _at_zone else ''}\n")
            else:
                msg += (f"📍 Supply OB: ${_ob_ref['low']:.4f} (entry zone)"
                        f" | ${_ob_ref['high']:.4f} (SL above this)"
                        f"{'  at zone now' if _at_zone else ''}\n")

        if sl_price:
            sl_pct = abs(sl_price - _ref) / _ref * 100 if _ref > 0 else 0
            if _ob_ref:
                _sl_label = "OB bottom" if direction == "LONG" else "OB top"
            elif trade_type == "scalp":
                _sl_label = "1H structure"
            else:
                _sl_label = "4H structure"
            msg += f"🛑 SL ({_sl_label}): ${sl_price} ({sl_pct:.1f}% from entry)\n"

        if tp1_price and tp2_price:
            # Pattern 3: Show which signal type drove the TP calculation
            msg += f"✅ TP1: ${tp1_price:.4f} (+{tp1_pct_r}%)\n"
            msg += f"✅ TP2: ${tp2_price:.4f} (+{tp2_pct_r}%) [{_tp_signal_type}]\n"

        # ── Position sizing — suppressed on SKIP (not actionable) ──────────
        try:
            if not _is_skip and sl_price and sl_price > 0 and _pullback_entry > 0:
                _sizing = calc_trade_sizing(
                    entry_price     = _pullback_entry,
                    sl_price        = sl_price,
                    leverage        = leverage,
                    trade_type      = trade_type,
                    amount_override = amount_override,
                    tp1_price       = tp1_price,
                    tp2_price       = tp2_price,
                )
                _sz_label = (f"${_sizing['amount']} override"
                             if _sizing["is_override"]
                             else f"${_sizing['amount']} {trade_type}")
                msg += f"\n💰 *Sizing ({_sz_label}):*\n"
                msg += f"  Contracts: {_sizing['contracts']} {raw_symbol} | Size: ${_sizing['pos_value']} at {leverage}x\n"
                if _sizing["loss_at_sl"] is not None:
                    msg += f"  Loss if SL: -${_sizing['loss_at_sl']}"
                if _sizing["gain_at_tp1"] is not None:
                    msg += f" | Gain TP1: +${_sizing['gain_at_tp1']}"
                if _sizing["gain_at_tp2"] is not None:
                    msg += f" | Gain TP2: +${_sizing['gain_at_tp2']}"
                msg += "\n"
        except Exception as _sz_e:
            log.warning(f"Sizing calc error: {_sz_e}")

        # ── Watch command ─────────────────────────────────────────────────────
        _watch_entry = entry_check.get("pullback_target") or _ref
        _watch_tp    = f" {tp1_pct_r} {tp2_pct_r}" if (tp1_pct_r and tp2_pct_r) else ""
        msg += f"\n`/watch {raw_symbol} {direction} {_watch_entry} {trade_type} {leverage}{_watch_tp}`"

        # ── Inline limit plan — only when WAIT/CONDITIONAL (not SKIP) ─────
        _pb_target = entry_check.get("pullback_target")
        if not _is_skip and action in ("WAIT", "CONDITIONAL") and _pb_target and _pb_target > 0:
            msg += f"\n\n*Limit Plan at ${_pb_target:.4f}:*\n"

            # POI quality at the pullback target specifically
            _pb_poi = _poi  # already computed at _pullback_entry
            _pb_poi_icon = "🟢" if _pb_poi["density"] >= 3 else ("🟡" if _pb_poi["density"] == 2 else "🔴")
            msg += f"{_pb_poi_icon} Zone quality: {_pb_poi['label']}"
            if _pb_poi["layers"]:
                msg += f" ({', '.join(_pb_poi['layers'])})"
            msg += "\n"

            # Sweep advice
            if _sweep_status and "swept and recovered" in _sweep_status:
                msg += "💧 Stop hunt already complete — zone is primed\n"
            elif _sweep_status and "not yet swept" in _sweep_status:
                msg += "💧 Liquidity not yet swept — wait for the sweep before entering\n"
            else:
                msg += "💧 No sweep detected — enter at zone, SL below OB bottom\n"

            # Pre-filled watch command for the limit
            msg += f"\nSet this now:\n`/watch {raw_symbol} {direction} {_pb_target:.4f} {trade_type} {leverage}{_watch_tp}`"
            msg += f"\nRun when price arrives:\n`/refine {raw_symbol} {direction} {_pb_target:.4f} {leverage}`"

        # ── Counter scalp — only when distance >= 5% and good RR ────────────
        if (action in ("WAIT", "CONDITIONAL")
                and _ob_ref and _pb_target
                and price_now > 0 and _pb_target > 0):
            _counter_dist_pct = abs(price_now - _pb_target) / price_now * 100
            if _counter_dist_pct >= 5.0:
                _counter_dir = "SHORT" if direction == "LONG" else "LONG"
                try:
                    if _counter_dir == "SHORT":
                        _c_sl     = round(float(df_1h["high"].tail(5).max()) * 1.012, 4)
                        _c_sl_pct = round((_c_sl - price_now) / price_now * 100, 1)
                    else:
                        _c_sl     = round(float(df_1h["low"].tail(5).min()) * 0.988, 4)
                        _c_sl_pct = round((price_now - _c_sl) / price_now * 100, 1)
                    _c_tp    = round(_pb_target, 4)
                    _c_tp_pct = round(_counter_dist_pct, 1)
                    _c_rr    = _c_tp_pct / _c_sl_pct if _c_sl_pct > 0 else 0
                    if _c_rr >= 1.5:
                        msg += "\n\n---"
                        msg += "\n⚡ *While you wait - Counter Scalp*"
                        msg += (f"\n{_counter_dist_pct:.1f}% to pullback entry — "
                                f"tradeable as a {'short' if _counter_dir=='SHORT' else 'long'} scalp.\n")
                        msg += f"Entry: ${price_now:.4f} | TP: ${_c_tp} ({_c_tp_pct:.1f}%) | "
                        msg += f"SL: ${_c_sl} ({_c_sl_pct:.1f}%) | RR: {_c_rr:.1f}:1\n"
                        msg += (f"When TP hits - flip to {direction} at ${_c_tp} "
                                f"with SL below OB at ${_ob_ref['low']:.4f}\n")
                        msg += f"`/refine {raw_symbol} {_counter_dir} {price_now:.4f} {leverage} scalp`"
                except Exception:
                    pass

        # ── Send the message ──────────────────────────────────────────────────
        # Try Markdown first, fall back to plain text if parse fails.
        # Previously this reply_text was missing entirely — message was built
        # but never sent, causing infinite "Analysing..." with no response.
        _refine_last_call[_uid] = _time.time()   # update cooldown on successful completion
        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            # Strip markdown characters and retry as plain text
            _plain = (msg.replace("*", "").replace("`", "")
                         .replace("_", "").replace("[", "").replace("]", ""))
            await update.message.reply_text(_plain)

    except Exception as e:
        log.error(f"cmd_refine error: {e}", exc_info=True)
        await update.message.reply_text("Refinement error: " + str(e)[:200])
@owner_only
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

        # ── Symbol resolution — try multiple exchange formats ─────────────────
        # Coins not on Bitget perpetuals (e.g. RAVE, IP) fail with KeyError 'rsi_1d'
        # because fetch_ohlcv returns None and analyze_v2 returns minimal dict.
        # Try USDT:USDT first (Bitget perp), then bare USDT (spot), then MEXC perp.
        def _try_fetch(sym):
            """Return True if symbol has enough data on the primary exchange."""
            try:
                df = fetch_ohlcv(sym, "1h", 50)
                return df is not None and len(df) >= 20
            except Exception:
                return False

        resolved_symbol = symbol
        if not _try_fetch(symbol):
            # Try spot format
            _spot = coin + "/USDT"
            if _try_fetch(_spot):
                resolved_symbol = _spot
                log.info(f"cmd_coin: {coin} not on Bitget perp — using spot {_spot}")
            else:
                # Try MEXC perp via fetch_ohlcv_mexc if available
                try:
                    _mx_df = fetch_ohlcv_mexc(symbol, "1h", 50)
                    if _mx_df is not None and len(_mx_df) >= 20:
                        resolved_symbol = symbol  # keep symbol, flag as MEXC
                        log.info(f"cmd_coin: {coin} using MEXC data")
                    else:
                        await update.message.reply_text(
                            f"Could not find {coin} on any exchange.\n"
                            f"Tried: Bitget USDT perp, Bitget spot, MEXC perp.\n"
                            f"Check the ticker is correct."
                        )
                        return
                except Exception:
                    await update.message.reply_text(
                        f"Could not find {coin} — not available on Bitget or MEXC.\n"
                        f"Check the ticker is correct (e.g. /coin BTC not /coin BTCUSDT)."
                    )
                    return

        symbol = resolved_symbol

        # Temporarily add to labels if not in watchlist
        original_label = COIN_LABELS.get(symbol)
        if not original_label:
            COIN_LABELS[symbol] = coin

        r = await run_full_pipeline(symbol, fetch_ohlcv, COIN_LABELS, ai_client, exchange, news_context)

        if not original_label and symbol in COIN_LABELS:
            del COIN_LABELS[symbol]

        if not r or "direction" not in r:
            regime      = r.get("regime", {}) if r else {}
            regime_name = regime.get("regime", "UNKNOWN") if isinstance(regime, dict) else str(regime)
            await update.message.reply_text("\u26a0 No signal for " + coin + " right now.\nRegime: " + regime_name)
            return

        # ── Early exit for NEUTRAL — show diagnostic, no levels ──────────────
        if r.get("direction") in (None, "NEUTRAL"):
            regime_data   = r.get("regime", {}) if isinstance(r.get("regime"), dict) else {}
            regime_name   = regime_data.get("regime", "UNKNOWN")
            adx_val       = round(float(regime_data.get("adx", 0)), 1)
            reject_reason = r.get("quality", {}).get("reject_reason", "") if isinstance(r.get("quality"), dict) else ""
            cur_price_n   = r.get("price", 0)
            msg  = "\U0001f50e *On-Demand Scan: " + coin + "* | " + datetime.now().strftime("%H:%M UTC") + "\n\n"
            if cur_price_n and cur_price_n > 0:
                msg += "\U0001f4b0 *Current price: `$" + "{:.6f}".format(cur_price_n) + "`*\n"
            msg += "\U0001f534 *No tradeable signal right now.*\n\n"
            msg += "Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n"
            msg += "Score: `" + str(r.get("abs_score", 0)) + "` | Confidence: `" + r.get("confidence", "LOW") + "`\n"
            if reject_reason:
                msg += "\n\U0001f4cb *Why:* " + reject_reason + "\n"
            msg += "\n_The narrative pipeline requires groups to agree with zero active opposition._\n"
            msg += "_Try again when regime changes or use /watch to monitor for a setup._"
            try:
                await update.message.reply_text(msg, parse_mode="Markdown")
            except Exception:
                await update.message.reply_text(msg, parse_mode=None)
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
                # LTF = 1H (where structure forms), HTF = 4H (trend context)
                # Was passing df_4h as LTF — HTF confluence check agreed with itself.
                # FIX: `r.get("df_X") or fetch_ohlcv(...)` causes "DataFrame is ambiguous"
                _raw_1h_c = r.get("df_1h")
                _raw_4h_c = r.get("df_4h")
                df_grade  = _raw_1h_c if (_raw_1h_c is not None and not _raw_1h_c.empty) \
                            else fetch_ohlcv(symbol, "1h", 200)
                df_4h_htf = _raw_4h_c if (_raw_4h_c is not None and not _raw_4h_c.empty) \
                            else fetch_ohlcv(symbol, "4h", 100)
                if df_grade is not None and df_4h_htf is not None:
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

        # ── Suggested leverage ────────────────────────────────────────────────
        try:
            from signal_engine import suggest_leverage
            _lev_df = r.get("df_4h")
            if _lev_df is not None:
                _lev = suggest_leverage(_lev_df, r.get("abs_score", 50), r.get("grade", "B"))
                r["suggested_leverage"] = _lev["suggested"]
                r["max_safe_leverage"]  = _lev["max_safe"]
                r["leverage_reason"]    = _lev["reason"]
        except Exception as e:
            log.warning("cmd_coin leverage error: " + str(e))

        # ── Derivatives context ───────────────────────────────────────────────
        try:
            from signal_engine import classify_derivatives_context
            _oi_now = None
            try:
                _oi_data = exchange.fetch_open_interest(symbol)
                _oi_now  = float(_oi_data["openInterestAmount"]) if _oi_data else None
            except Exception:
                pass
            _c24 = 0.0
            try:
                _df_c = r.get("df_4h")
                if _df_c is not None and len(_df_c) >= 24:
                    _c24 = (float(_df_c["close"].iloc[-1]) - float(_df_c["close"].iloc[-24])) / float(_df_c["close"].iloc[-24]) * 100
            except Exception:
                pass
            _dctx = classify_derivatives_context(
                funding_rate=r.get("funding", 0),
                oi_now=_oi_now,
                oi_prev=_oi_cache.get(symbol),
                change_24h=_c24,
            )
            r["deriv_ctx"]        = _dctx["state"]
            r["deriv_block_long"] = _dctx["block_long"]
            r["deriv_boost_short"]= _dctx["boost_short"]
            r["deriv_reason"]     = _dctx["reason"]
        except Exception as e:
            log.warning("cmd_coin deriv error: " + str(e))
            r["deriv_ctx"] = "NEUTRAL"

        # ── Swing vs Scalp assessment ─────────────────────────────────────────
        # Gives honest opinion on whether this coin suits swing or scalp
        try:
            _df_assess = r.get("df_4h")
            _atr_pct_a = (r["atr"] / r["price"] * 100) if r.get("price", 0) > 0 else 2.0
            _regime_a  = r.get("regime", {}).get("regime", "RANGING") if isinstance(r.get("regime"), dict) else "RANGING"
            _grade_a   = r.get("grade", "B")
            _adx_a     = float(r.get("regime", {}).get("adx", 0)) if isinstance(r.get("regime"), dict) else 0

            # Swing: trending regime + ADX>25 + ATR not extreme + Grade A/B
            # Scalp: any regime + high ATR + quick entry/exit
            _swing_score  = 0
            _scalp_score  = 0
            _swing_notes  = []
            _scalp_notes  = []

            if _regime_a == "TRENDING_UP" or _regime_a == "TRENDING_DOWN":
                _swing_score += 2; _swing_notes.append("trending regime")
            elif _regime_a == "RANGING":
                _scalp_score += 2; _scalp_notes.append("ranging market suits scalp")

            if _adx_a >= 25:
                _swing_score += 1; _swing_notes.append(f"ADX {_adx_a:.0f} confirms trend")
            elif _adx_a < 20:
                _scalp_score += 1; _scalp_notes.append(f"ADX {_adx_a:.0f} weak trend")

            if _atr_pct_a > 4.0:
                _scalp_score += 2; _scalp_notes.append(f"ATR {_atr_pct_a:.1f}% — high volatility suits scalp")
            elif _atr_pct_a < 1.5:
                _swing_score += 1; _swing_notes.append(f"ATR {_atr_pct_a:.1f}% — stable, suits swing")

            if _grade_a == "A":
                _swing_score += 1; _swing_notes.append("Grade A — early entry suits swing")

            if _swing_score > _scalp_score:
                _trade_rec = "⚡ *Best as SWING* — " + ", ".join(_swing_notes[:2])
            elif _scalp_score > _swing_score:
                _trade_rec = "⚡ *Best as SCALP* — " + ", ".join(_scalp_notes[:2])
            else:
                _trade_rec = "⚡ *Either works* — borderline setup"

            r["trade_recommendation"] = _trade_rec
        except Exception:
            r["trade_recommendation"] = ""

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
        if r.get("price") and r["price"] > 0:
            header += "\U0001f4b0 *Current: `$" + "{:.6f}".format(r["price"]) + "`*\n"
        header += "Regime: `" + regime_name + "` | ADX: `" + str(adx_val) + "`\n"
        header += "1H Score: `" + str(r.get("score_4h", r.get("score_1h", 0))) + "` | 4H Score: `" + str(r.get("score_1d", r.get("score_4h", 0))) + "`\n"
        # rsi_1d was never in analyze_v2 return dict — caused KeyError on every coin
        _rsi_1h_disp = round(r.get("rsi_1h", r.get("rsi_4h", 50)), 1)
        _rsi_4h_disp = round(r.get("rsi_4h", 50), 1)
        header += "RSI 1H: `" + str(_rsi_1h_disp) + "` | RSI 4H: `" + str(_rsi_4h_disp) + "`\n"
        header += "TF Agreement: " + ("\u2705 Yes" if r.get("tf_agree") else "\u26a0 No") + "\n"
        if r.get("trade_recommendation"):
            header += r["trade_recommendation"] + "\n"
        header += "\n"

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


@owner_only
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


@owner_only
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


@owner_only
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


@owner_only
async def cmd_whale(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args   = ctx.args
    coin   = args[0].upper() if args else "BTC"
    symbol = coin + "/USDT:USDT"
    await update.message.reply_text("\U0001f433 Checking whale activity for " + coin + "...")
    try:
        # Volume spike check
        df = fetch_ohlcv(symbol, "1h", 50)
        if df is None or len(df) < 20:
            await update.message.reply_text("⚠️ Could not fetch data for that symbol.")
            return
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


@owner_only
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


@owner_only
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
        "/counterscan \u2014 Find coins extended from OB — counter scalp setups\n"
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
        "/domstatus \u2014 BTC dominance + pipeline mode (full/simplified)\n"
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


@owner_only
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

@owner_only
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

@owner_only
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

@owner_only
async def cmd_coins(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    labels = list(COIN_LABELS.values())
    msg    = "\U0001f4cb *Watchlist (" + str(len(labels)) + " coins)*\n\n"
    msg   += " \u00b7 ".join(labels)
    await update.message.reply_text(msg, parse_mode="Markdown")

@owner_only
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


@owner_only
async def cmd_paper_refine(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /paper_refine SYMBOL LONG|SHORT PRICE [LEVERAGE]

    Manually opens a paper trade from a top trader call.
    Runs full refinement analysis first (15m timing, grade, shakeout, accumulation)
    and records all metadata so you can track how well the refinement system works.

    Examples:
      /paper_refine BAN LONG 0.085
      /paper_refine CROSS SHORT 0.111 8
    """
    args = ctx.args
    if len(args) < 3:
        await update.message.reply_text(
            "Usage: /paper_refine SYMBOL LONG|SHORT PRICE [LEVERAGE]\n"
            "Example: /paper_refine BAN LONG 0.085 8",
            parse_mode="Markdown"
        )
        return

    raw_symbol = args[0].upper()
    direction  = args[1].upper()
    try:
        entry_price = float(args[2])
    except ValueError:
        await update.message.reply_text("Invalid price.")
        return
    leverage = int(args[3]) if len(args) >= 4 else LEVERAGE

    if direction not in ("LONG", "SHORT"):
        await update.message.reply_text("Direction must be LONG or SHORT.")
        return

    if not paper_mode:
        await update.message.reply_text(
            "⚠️ Paper mode is OFF. Enable it first with /paper\n"
            "This command only records paper trades, not real ones.",
            parse_mode="Markdown"
        )
        return

    symbol = next((s for s in COIN_LABELS if s.split("/")[0].upper() == raw_symbol), None)
    if symbol is None:
        symbol = f"{raw_symbol}/USDT:USDT"

    await update.message.reply_text(
        f"🔍 Analysing *{raw_symbol} {direction}* @ `${entry_price}` before opening paper trade...",
        parse_mode="Markdown"
    )

    try:
        from signal_engine import (check_15m_entry_quality, grade_signal,
                                   detect_accumulation_setup, detect_shakeout,
                                   compute_signal_quality)
        import pandas_ta as ta

        df_15m, _    = fetch_ohlcv_smart(symbol, "15m", 100)
        df_1h,  _src_p = fetch_ohlcv_smart(symbol, "1h",  200)
        df_4h,  _    = fetch_ohlcv_smart(symbol, "4h",  100)

        if df_1h is None or df_4h is None:
            await update.message.reply_text(
                f"⚠️ Could not fetch data for {raw_symbol} on Bitget or MEXC. Check the ticker."
            )
            return

        exchange_note = " (via MEXC)" if _src_p == "mexc" else ""
        price_now    = float(df_1h["close"].iloc[-1])
        atr_s        = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
        atr          = float(atr_s.dropna().iloc[-1]) if atr_s is not None and len(atr_s.dropna()) > 0 else price_now * 0.02
        atr_pct      = atr / price_now * 100

        # ── 15m entry check — with SMC context for accurate pullback targets ──
        # Previously called without demand_ob/fvg/bos — pullback always fell
        # through to EMA9 fallback. Now passes same structural data as live /refine.
        entry_check = {"action": "NEUTRAL", "reason": "No 15m data", "rsi_15m": 50.0, "pullback_target": None}
        if df_15m is not None and len(df_15m) >= 20:
            try:
                from signal_engine import detect_order_blocks, detect_bos_choch, detect_liquidity_pools
                _pr_ob_r   = detect_order_blocks(df_1h)
                _pr_bos_r  = detect_bos_choch(df_1h)
                _pr_liq_r  = detect_liquidity_pools(df_1h)
                _pr_ob_ref = (_pr_ob_r.get("demand_ob") if direction == "LONG"
                              else _pr_ob_r.get("supply_ob"))
                # FVG levels from 15m
                _pr_fvg = []
                highs   = df_15m["high"].values
                lows    = df_15m["low"].values
                closes  = df_15m["close"].values
                for _fi in range(2, len(df_15m)):
                    if lows[_fi] > highs[_fi - 2]:
                        _pr_fvg.append({"type": "BULL", "mid": (lows[_fi] + highs[_fi - 2]) / 2})
                    elif highs[_fi] < lows[_fi - 2]:
                        _pr_fvg.append({"type": "BEAR", "mid": (highs[_fi] + lows[_fi - 2]) / 2})
                _pr_bos_for_15m = {
                    "bos_bullish":   _pr_bos_r.get("bos_bullish",   False),
                    "bos_bearish":   _pr_bos_r.get("bos_bearish",   False),
                    "choch_bullish": _pr_bos_r.get("choch_bullish", False),
                    "choch_bearish": _pr_bos_r.get("choch_bearish", False),
                    "sweep_bullish": _pr_liq_r.get("sweep_bullish", False),
                    "sweep_bearish": _pr_liq_r.get("sweep_bearish", False),
                }
                entry_check = check_15m_entry_quality(
                    df_15m, direction,
                    demand_ob  = _pr_ob_ref,
                    fvg_levels = _pr_fvg,
                    bos_data   = _pr_bos_for_15m,
                )
            except Exception as _pr_ec_e:
                log.warning(f"paper_refine entry_check SMC error: {_pr_ec_e}")
                entry_check = check_15m_entry_quality(df_15m, direction)

        # ── Grade — use real computed quality score, not hardcoded 60.0 ──────
        grade_result = None
        grade_str    = "B"
        dow_phase    = "UNCLEAR"
        try:
            rsi_s = ta.rsi(df_1h["close"], length=14)
            rsi_v = float(rsi_s.dropna().iloc[-1]) if rsi_s is not None and len(rsi_s.dropna()) > 0 else 50.0
            # Compute real quality score — same pipeline as live /refine
            try:
                _qr        = compute_signal_quality(df_1h, df_4h, symbol)
                # Note: funding not fetched in paper_refine path — exhaustion uses 0.0
                _real_score = _qr.get("quality_score", 60.0)
            except Exception:
                _real_score = 60.0
            grade_result = grade_signal(df_1h, df_4h, direction, _real_score, rsi_v, 0.0)
            grade_str = grade_result["grade"]
            dow_phase = grade_result.get("dow_phase", "UNCLEAR")
        except Exception as ge:
            log.warning(f"paper_refine grade error: {ge}")

        # ── Accumulation ──────────────────────────────────────────────────────
        acc = detect_accumulation_setup(df_1h, df_4h, symbol)

        # ── Shakeout ──────────────────────────────────────────────────────────
        sk = {"is_shakeout": False, "score": 0, "confidence": "LOW",
              "signals": [], "flush_low": 0, "pump_target": 0, "invalidation": 0}
        if df_15m is not None and direction == "LONG":
            try:
                sk = detect_shakeout(df_15m, df_1h, df_4h, symbol)
            except Exception:
                pass

        # ── SL/TP from structure ──────────────────────────────────────────────
        from risk_manager import calc_levels_v2
        sl, tp1, tp2, _ = calc_levels_v2(direction, entry_price, atr, atr_pct, df_4h)
        if sk["is_shakeout"] and sk["invalidation"] > 0 and direction == "LONG":
            sl = max(sl, sk["invalidation"])  # use tighter flush-low SL

        # ── Open paper trade with full metadata ───────────────────────────────
        trade_id = open_paper_trade(
            symbol, direction, entry_price, sl, tp1, tp2,
            confidence="MEDIUM",
            trade_type="swing",
            signal_type="MOMENTUM",
            grade=grade_str,
            dow_phase=dow_phase,
            shakeout=sk["is_shakeout"],
            entry_action=entry_check["action"],
            accumulating=acc["is_accumulating"],
            leverage=leverage,
        )

        # ── Format response ───────────────────────────────────────────────────
        action      = entry_check["action"]
        action_icon = "✅" if action == "ENTER" else ("⏳" if action == "WAIT" else "➡️")
        grade_icon  = {"A": "🥇", "B": "🥈", "C": "🥉"}.get(grade_str, "🥈")
        sl_pct      = abs(entry_price - sl) / entry_price * 100
        tp1_pct     = abs(tp1 - entry_price) / entry_price * 100
        tp2_pct     = abs(tp2 - entry_price) / entry_price * 100
        liq         = round(entry_price * (1 - 0.9/leverage), 4) if direction == "LONG" else round(entry_price * (1 + 0.9/leverage), 4)

        def _clean(s):
            """Strip all special chars for plain text Telegram message."""
            return str(s).replace("*","").replace("`","").replace("_"," ").replace("[","").replace("]","").replace("✦","").replace("—","-")

        # Build as plain text — avoids ALL Markdown entity parsing issues
        msg  = f"📝 Paper Trade #{trade_id} Opened{exchange_note}\n"
        msg += f"{raw_symbol} {direction} @ ${entry_price} | {leverage}x\n\n"
        msg += f"{action_icon} 15m Entry: {action}\n"
        msg += _clean(entry_check['reason']) + "\n"
        if entry_check.get("pullback_target"):
            msg += f"Pullback target: ${entry_check['pullback_target']:.4f}\n"
        msg += f"\n{grade_icon} Grade {grade_str} | Dow: {_clean(dow_phase)}\n"
        if grade_result and grade_result.get("reasons"):
            msg += "✔ " + _clean(" | ".join(grade_result["reasons"][:2])) + "\n"
        if grade_result and grade_result.get("warnings"):
            msg += "⚠ " + _clean(" | ".join(grade_result["warnings"][:1])) + "\n"
        if sk["is_shakeout"]:
            msg += f"\n⚡ Shakeout ({_clean(sk['confidence'])}, {sk['score']}/8)\n"
            msg += f"Flush: ${sk['flush_low']:.4f} | Target: ${sk['pump_target']:.4f}\n"
        if acc["is_accumulating"]:
            msg += f"\n📦 Accumulation ({_clean(acc['confidence'])}, {acc['score']}/7)\n"
        msg += f"\n📌 SL: ${sl:.4f} ({sl_pct:.1f}%)"
        msg += f" | TP1: ${tp1:.4f} ({tp1_pct:.1f}%)"
        msg += f" | TP2: ${tp2:.4f} ({tp2_pct:.1f}%)\n"
        msg += f"💥 Liquidation at {leverage}x: ${liq:.4f}\n"
        msg += "\nTrack with /portfolio — check /paper_stats for performance"

        await update.message.reply_text(msg)  # no parse_mode = plain text, never crashes

    except Exception as e:
        log.error(f"paper_refine error: {e}", exc_info=True)
        await update.message.reply_text(f"⚠️ Error: {str(e)[:200]}")


@owner_only
async def cmd_paper_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /paper_stats — detailed breakdown of paper trade performance by signal type.

    Shows win rate and avg PnL broken down by:
    - Grade A vs B vs C
    - Signal type: Momentum vs Reversal
    - Shakeout signals
    - Accumulation signals
    - 15m entry: ENTER vs WAIT (did waiting help?)
    - Scalp vs Swing
    """
    wr = get_win_rate()
    if not wr:
        await update.message.reply_text("No closed paper trades yet. Let signals run and re-check.")
        return

    def fmt(stats, label):
        if stats["total"] == 0:
            return f"  {label}: no data\n"
        return (f"  {label}: {stats['win_rate']}% WR | "
                f"{stats['wins']}W {stats['losses']}L | "
                f"Avg {stats['avg_pnl']:+.1f}%\n")

    msg  = "📊 *Paper Trade Stats*\n\n"
    msg += f"*Overall:* {wr['overall']['win_rate']}% WR | "
    msg += f"{wr['overall']['wins']}W {wr['overall']['losses']}L | "
    msg += f"Open: {wr['open']}\n\n"

    msg += "*By Grade:*\n"
    msg += fmt(wr["grade_a"],  "🥇 Grade A (early)")
    msg += fmt(wr["grade_b"],  "🥈 Grade B (confirmed)")
    msg += fmt(wr["grade_c"],  "🥉 Grade C (late)")

    msg += "\n*By Signal Type:*\n"
    msg += fmt(wr["momentum"], "⚡ Momentum")
    msg += fmt(wr["reversal"], "🔄 Reversal")
    msg += fmt(wr["shakeout"], "⚡ Shakeout")
    msg += fmt(wr["accumul"],  "📦 Accumulation")

    msg += "\n*By 15m Entry Timing:*\n"
    msg += fmt(wr["entry_now"],  "✅ ENTER (timing good)")
    msg += fmt(wr["entry_wait"], "⏳ WAIT (entered anyway)")

    msg += "\n*By Trade Type:*\n"
    msg += fmt(wr["scalp"], "📈 Scalp")
    msg += fmt(wr["swing"], "📊 Swing")

    msg += "\n_Use this to see which signal types actually work for you._"

    await update.message.reply_text(msg, parse_mode="Markdown")

@owner_only
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
        msg += "Total PnL: $" + "{:+.2f}".format(total_pnl) + "\n"

        # ── Grade breakdown — shows which grades are worth taking ─────────
        if len(closed_t) >= 3:
            msg += "\n*By Grade:*\n"
            for grade, emoji in (("A", "🟣"), ("B", "🟢"), ("C", "🟡")):
                g_trades = [t for t in closed_t if t.get("grade") == grade]
                if g_trades:
                    g_wins = len([t for t in g_trades if t.get("status") == "WIN"])
                    g_wr   = round(g_wins / len(g_trades) * 100)
                    g_avg  = round(sum(t.get("pnl_pct", 0) for t in g_trades) / len(g_trades), 1)
                    msg += f"  {emoji} Grade {grade}: {len(g_trades)} trades | {g_wr}% WR | Avg {g_avg:+.1f}%\n"

        # ── Dow phase breakdown ───────────────────────────────────────────
        if len(closed_t) >= 3:
            phase_lines = []
            for phase, ph_emoji in (("ACCUMULATION","🟣"),("PARTICIPATION","🟢"),("DISTRIBUTION","🟡")):
                p_trades = [t for t in closed_t if t.get("dow_phase") == phase]
                if p_trades:
                    p_wins = len([t for t in p_trades if t.get("status") == "WIN"])
                    p_wr   = round(p_wins / len(p_trades) * 100)
                    phase_lines.append(f"  {ph_emoji} {phase[:4]}: {len(p_trades)} | {p_wr}% WR")
            if phase_lines:
                msg += "\n*By Phase:*\n" + "\n".join(phase_lines) + "\n"

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
        _sc_lev = trade.get("leverage", LEVERAGE)  # FIX: was r.get() — r not defined here
        msg += "  \U0001f534 SL:  `$" + "{:.4f}".format(sl) + "` (-" + "{:.1f}".format(sl_pct) + "% / -" + "{:.0f}".format(sl_pct*_sc_lev) + "% at " + str(_sc_lev) + "x)\n"
        msg += "  \U0001f3af TP1: `$" + "{:.4f}".format(tp1) + "` (+" + "{:.1f}".format(tp1_pct) + "% / +" + "{:.0f}".format(tp1_pct*_sc_lev) + "% at " + str(_sc_lev) + "x)\n"
        msg += "  \U0001f3af TP2: `$" + "{:.4f}".format(tp2) + "` (+" + "{:.1f}".format(tp2_pct) + "% / +" + "{:.0f}".format(tp2_pct*_sc_lev) + "% at " + str(_sc_lev) + "x)\n\n"
        msg += "*Current Status:*\n"
        current_emoji = "\U0001f7e2" if trade["pnl_usdt"] >= 0 else "\U0001f534"
        msg += current_emoji + " " + trade["status"] + " | PnL: " + "{:+.1f}".format(trade["pnl_pct"]) + "% ($" + "{:+.2f}".format(trade["pnl_usdt"]) + ")"
        await query.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await query.message.reply_text("Error: " + str(e))


@owner_only
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

@owner_only
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

@owner_only
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

@owner_only
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
                # 2-candle confirmation — runs in BOTH modes.
                # Previously skipped in simplified mode — but ranging/BTC-season
                # conditions produce MORE single-candle spikes, not fewer.
                # Removing the confirm when conditions are noisier was backwards.
                _df_c = r.get("df_4h")
                if _df_c is not None and not _two_candle_confirm(_df_c, r["direction"]):
                    log.info("2-candle confirm failed: " + r["label"] + " " + r["direction"])
                    continue

                # ── 15m timing gate ───────────────────────────────────────────
                # If 15m says WAIT (overbought, MACD crossed against, extended),
                # only fire if Grade A — strong structure overrides bad timing.
                # Grade B/C signals defer for one cycle to avoid entering at the
                # top of a local push, which was the main source of late entries.
                _entry_action = r.get("entry_action", "NEUTRAL")
                _grade        = r.get("grade", "B")
                if _entry_action == "WAIT" and _grade != "A":
                    log.info(
                        f"15m timing gate deferred {r['label']} {r['direction']} "
                        f"(entry=WAIT grade={_grade}) — will retry next cycle"
                    )
                    # Keep in last_signal so it can re-fire next cycle without cooldown
                    # but don't fire this cycle
                    continue

                # Tiered BTC circuit breaker gate
                if r["direction"] == "LONG" and btc_circuit_breaker["active"]:
                    _cb_tier = btc_circuit_breaker.get("tier", "full")
                    if _cb_tier == "full":
                        # 4H sustained bear trend — no new longs at all
                        log.info("Circuit breaker FULL blocked LONG: " + r["label"])
                        continue
                    elif _cb_tier == "elevated":
                        # Single candle dump — raise quality bar, allow clean SMC setups
                        _cb_score_ok = r.get("abs_score", 0) >= 75
                        _cb_grade_ok = r.get("grade") == "A"
                        # Check coin's own SMC structure
                        try:
                            from signal_engine import detect_bos_choch, detect_order_blocks
                            _cb_bos = detect_bos_choch(r["df_4h"]) if r.get("df_4h") is not None else {}
                            _cb_ob  = detect_order_blocks(r["df_4h"]) if r.get("df_4h") is not None else {}
                            _cb_smc_ok = (
                                _cb_bos.get("bos_bullish") or
                                _cb_bos.get("choch_bullish") or
                                _cb_ob.get("price_at_demand_ob")
                            )
                        except Exception:
                            _cb_smc_ok = False
                        if not (_cb_score_ok and _cb_grade_ok and _cb_smc_ok):
                            log.info(
                                f"Circuit breaker ELEVATED blocked LONG: {r['label']} "
                                f"score={r.get('abs_score',0)} grade={r.get('grade','?')} "
                                f"smc_ok={_cb_smc_ok}"
                            )
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
                        open_paper_trade(r["symbol"], r["direction"], r["price"], sl, tp1, tp2, r["confidence"],
                                        signal_type=r.get("signal_type","MOMENTUM"),
                                        grade=r.get("grade","B"), dow_phase=r.get("dow_phase","UNCLEAR"),
                                        shakeout=r.get("shakeout",False), entry_action=r.get("entry_action","NEUTRAL"),
                                        accumulating=r.get("is_accumulating",False), leverage=r.get("suggested_leverage",LEVERAGE))
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
                    # Lazily stamp btc_market_regime on signals missing the field.
                    # Uses the actual BTC 4H market regime (TRENDING_UP/DOWN/RANGING),
                    # NOT the pipeline mode (simplified/full) which is unrelated to
                    # individual signal validity.
                    if "btc_regime" not in sig:
                        active_signals[symbol]["btc_regime"] = _btc_market_regime_cache.get("regime", "RANGING")

                    # TP1 hit
                    if not sig["tp1_hit"]:
                        if (direction == "LONG" and price >= tp1) or (direction == "SHORT" and price <= tp1):
                            # Scalp gets tighter trailing buffer — resolves fast, lock gains quicker
                            _tbuf = (TRAILING_BUFFER_SCALP
                                     if sig.get("trade_type") == "scalp"
                                     else TRAILING_BUFFER)
                            trailing_stop = (round(price * (1 - _tbuf), 6)
                                              if direction == "LONG"
                                              else round(price * (1 + _tbuf), 6))
                            update_trade_state(symbol, {
                                "tp1_hit":          True,
                                "trailing_stop":    trailing_stop,
                                "trailing_extreme": price,
                            })
                            _tbuf_pct = int(_tbuf * 100)
                            msg  = "\U0001f3af *TP1 Hit: " + label + "*\n\n"
                            msg += "Price: $" + "{:.4f}".format(price) + "\n"
                            msg += "Action: Close 50% of your position\n"
                            msg += "Move SL to breakeven: $" + "{:.4f}".format(entry) + "\n"
                            msg += "Trailing stop set at: $" + "{:.4f}".format(trailing_stop) + " (" + str(_tbuf_pct) + "% buffer)\n"
                            msg += "Remaining 50% protected — stop follows price"
                            await send_msg(app, msg)

                    # Trailing stop management after TP1 hit
                    elif sig["tp1_hit"] and sig.get("trailing_stop"):
                        trailing_stop    = sig["trailing_stop"]
                        trailing_extreme = sig["trailing_extreme"]

                        # Update trailing extreme and move stop — only in favorable direction
                        _tbuf_r = (TRAILING_BUFFER_SCALP
                                   if sig.get("trade_type") == "scalp"
                                   else TRAILING_BUFFER)
                        if direction == "LONG" and price > trailing_extreme:
                            new_extreme = price
                            new_stop    = round(price * (1 - _tbuf_r), 6)
                            # FIX: only move stop UP for LONG, never down
                            if new_stop > trailing_stop:
                                update_trade_state(symbol, {
                                    "trailing_extreme": new_extreme,
                                    "trailing_stop":    new_stop,
                                })
                                log.info("Trailing stop moved up for " + label + ": $" + str(new_stop))

                        elif direction == "SHORT" and price < trailing_extreme:
                            new_extreme = price
                            new_stop    = round(price * (1 + _tbuf_r), 6)
                            # FIX: only move stop DOWN for SHORT, never up
                            if new_stop < trailing_stop:
                                update_trade_state(symbol, {
                                    "trailing_extreme": new_extreme,
                                    "trailing_stop":    new_stop,
                                })
                                log.info("Trailing stop moved down for " + label + ": $" + str(new_stop))

                        # Check if trailing stop triggered
                        trailing_hit = (direction == "LONG" and price <= trailing_stop) or                                        (direction == "SHORT" and price >= trailing_stop)
                        if trailing_hit:
                            pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
                            msg  = "\U0001f6d1 *Trailing Stop Hit: " + label + "*\n\n"
                            msg += "Price: $" + "{:.4f}".format(price) + "\n"
                            msg += "Action: Close remaining 50% now\n"
                            msg += "Entry was: $" + "{:.4f}".format(entry) + "\n"
                            _mon_lev = sig.get("leverage", LEVERAGE)
                            msg += "Approx PnL on remainder: " + "{:+.1f}".format(pnl_pct * _mon_lev) + "% at " + str(_mon_lev) + "x"
                            await send_msg(app, msg)
                            del active_signals[symbol]

                        # Auto TP2 adjustment — FIX: only move TP2 in favorable direction
                        else:
                            try:
                                # Use correct timeframe: 4H for swing, 1H for scalp
                                is_scalp_trade = sig.get("trade_type") == "scalp"
                                tf_fresh = "1h" if is_scalp_trade else "4h"
                                df_fresh  = fetch_ohlcv(symbol, tf_fresh, 50)
                                if df_fresh is None or len(df_fresh) < 15:
                                    continue
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

                    # ── Signal expiration — 3 invalidation conditions ──────────
                    # 1. Time expiry: scalp >4h, swing >72h
                    # 2. Candle expiry: too many bars elapsed since signal (stale context)
                    # 3. BTC regime flip: if BTC regime changed since signal fired,
                    #    the macro context that validated the setup no longer exists
                    # 4. Paper trade closed externally
                    trades = load_json(PAPER_FILE, [])
                    trade_closed = any(t["symbol"] == symbol and t["status"] != "OPEN" for t in trades)
                    is_scalp     = sig.get("trade_type") == "scalp"
                    expiry_sec   = 14400 if is_scalp else 259200
                    sig_age_sec  = _time.time() - sig.get("time", _time.time())

                    # Candle-based expiry — swing only (scalp already covered by 4h time limit).
                    # 18 × 4H bars = 72h matches the swing time expiry and provides a
                    # meaningful structural check: if 18 4H candles have closed since
                    # the signal, the setup context has completely refreshed.
                    candles_elapsed = 0
                    candle_expired  = False
                    if not is_scalp:
                        candles_elapsed = int(sig_age_sec / 14400)
                        candle_expired  = candles_elapsed > 18

                    # BTC market regime flip invalidation.
                    # Compares the BTC 4H market regime at signal creation vs now.
                    # Only invalidates LONG when BTC flips to TRENDING_DOWN (genuine
                    # macro reversal), and SHORT when BTC flips to TRENDING_UP.
                    # Cache is updated every 30 min by auto_btc_monitor.
                    btc_regime_flipped = False
                    sig_btc_regime     = sig.get("btc_regime")
                    current_btc_regime = _btc_market_regime_cache.get("regime", "RANGING")
                    cache_age          = _time.time() - _btc_market_regime_cache.get("updated_at", 0)
                    if (sig_btc_regime and cache_age < 7200):  # only if cache is fresh (<2h)
                        direction_check = sig.get("direction", "LONG")
                        if direction_check == "LONG" and current_btc_regime == "TRENDING_DOWN" and sig_btc_regime != "TRENDING_DOWN":
                            btc_regime_flipped = True
                        elif direction_check == "SHORT" and current_btc_regime == "TRENDING_UP" and sig_btc_regime != "TRENDING_UP":
                            btc_regime_flipped = True

                    expired = trade_closed or (_time.time() - sig.get("time", 0) > expiry_sec) or candle_expired
                    if expired:
                        expire_reason = ("paper closed" if trade_closed
                                         else f"candle limit ({candles_elapsed}/{candle_limit})" if candle_expired
                                         else "time limit")
                        if symbol in active_signals:
                            del active_signals[symbol]
                            log.info(f"Signal expired ({expire_reason}): {label}")
                    elif btc_regime_flipped and not sig.get("tp1_hit"):
                        # Regime flip invalidation — only before TP1 (already in profit = let it run)
                        if symbol in active_signals:
                            del active_signals[symbol]
                            log.info(f"Signal invalidated — BTC flipped to {current_btc_regime} ({sig_btc_regime} → {current_btc_regime}): {label}")
                            try:
                                await send_msg(app, f"⚠️ *Signal Invalidated: {label}*\nBTC 4H regime flipped to {current_btc_regime} — macro context changed")
                            except Exception:
                                pass

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
                # Derivatives context boosts alpha SHORT threshold
                _deriv_boosts_short = r.get("deriv_boost_short", False)
                _alpha_short_threshold = 15 if _deriv_boosts_short else 20  # lower threshold when derivatives confirm

                if (exh.get("short_opp")
                        and exh.get("change_24h", 0) >= _alpha_short_threshold
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
                    # Exhaustion SHORTs inherit abs_score from the original coin analysis
                    # which is often 0 (NEUTRAL direction). Set a real score based on how
                    # many exhaustion signals fired — Score:0 was showing on valid setups.
                    _exh_signals = exh.get("blowoff_signals", 0)
                    _exh_score   = min(85, 50 + _exh_signals * 9)   # 50-85 based on signal count
                    r_short["abs_score"]  = _exh_score
                    r_short["score"]      = -_exh_score
                    r_short["confidence"] = "HIGH" if _exh_score >= 68 else "MEDIUM"

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

                # ── Normal LONG signal — mode-aware grade gate ────────────────
                # Alpha tokens stay at Grade A always — too volatile for Grade B
                _alpha_min_grade = "A"
                if (direction == "LONG"
                        and r.get("confidence") == "HIGH"
                        and r.get("signal_type","REVERSAL") == "MOMENTUM"
                        and grade == _alpha_min_grade
                        and quality >= 70
                        and not r.get("deriv_block_long", False)):  # no LONG in squeeze/crowded

                    # Tiered circuit breaker — mirrors auto_scan logic
                    if btc_circuit_breaker["active"]:
                        _cb_tier = btc_circuit_breaker.get("tier", "full")
                        if _cb_tier == "full":
                            continue
                        elif _cb_tier == "elevated":
                            _cb_score_ok = quality >= 75
                            _cb_grade_ok = grade == "A"
                            try:
                                from signal_engine import detect_bos_choch, detect_order_blocks
                                _cb_bos = detect_bos_choch(r["df_4h"]) if r.get("df_4h") is not None else {}
                                _cb_ob  = detect_order_blocks(r["df_4h"]) if r.get("df_4h") is not None else {}
                                _cb_smc_ok = (
                                    _cb_bos.get("bos_bullish") or
                                    _cb_bos.get("choch_bullish") or
                                    _cb_ob.get("price_at_demand_ob")
                                )
                            except Exception:
                                _cb_smc_ok = False
                            if not (_cb_score_ok and _cb_grade_ok and _cb_smc_ok):
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


@owner_only
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
@owner_only
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


@owner_only
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
                _gn_lev = r.get("suggested_leverage", LEVERAGE)
                msg += "\U0001f6d1 SL: `$" + "{:.4f}".format(sl_price) + "` (-" + str(sl_pct) + "% / -" + str(round(sl_pct*_gn_lev,1)) + "% at " + str(_gn_lev) + "x)\n"
                msg += "\U0001f3af TP1: `$" + "{:.4f}".format(tp1_price) + "` (+" + str(round(tp_pct/2,1)) + "% / +" + str(round(tp_pct/2*_gn_lev,1)) + "% at " + str(_gn_lev) + "x)\n"
                msg += "\U0001f3af TP2: `$" + "{:.4f}".format(tp2_price) + "` (+" + str(tp_pct) + "% / +" + str(round(tp_pct*_gn_lev,1)) + "% at " + str(_gn_lev) + "x)\n"
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
                    # LTF = 1H (where structure forms), HTF = 4H (trend context)
                    # Previously both were df_4h — grade_signal's HTF check always agreed
                    # with itself, inflating gainer grades artificially.
                    # FIX: `r.get("df_1h") or fetch_ohlcv(...)` causes "DataFrame is ambiguous"
                    # because pandas DataFrames cannot be used in boolean `or` expressions.
                    # Must use explicit None check.
                    _raw_df1h = r.get("df_1h")
                    df_g_ltf  = _raw_df1h if (_raw_df1h is not None and not _raw_df1h.empty) \
                                else fetch_ohlcv(symbol, "1h", 200)
                    df_g_htf  = r.get("df_4h")
                    if df_g_ltf is not None and df_g_htf is not None:
                        gr = grade_signal(df_g_ltf, df_g_htf,
                                          r["direction"], r.get("abs_score", 0),
                                          r.get("rsi_1h", 50), r.get("funding", 0))
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
                        lev = suggest_leverage(df_g_ltf, r.get("abs_score", 50), gr["grade"])
                        r["suggested_leverage"] = lev["suggested"]
                        r["max_safe_leverage"]  = lev["max_safe"]
                        r["leverage_reason"]    = lev["reason"]
                        # Exhaustion check — MUST run on 1H data, not 4H.
                        # 4H data undersamples the move: a 1H blow-off is invisible on 4H
                        # until the next candle closes, meaning exhaustion fires too late.
                        exh = check_exhaustion(df_g_ltf, "LONG", r.get("funding", 0))
                        r["exhaustion"] = exh
                    else:
                        exh = {"block_long": False, "short_opp": False, "reason": ""}
                except Exception as e:
                    log.warning("Gainers grade error " + label + ": " + str(e))
                    exh = {"block_long": False, "short_opp": False, "reason": ""}

                exh = r.get("exhaustion", {"block_long": False, "short_opp": False})

                # VP-adjusted levels
                from risk_manager import calc_levels_v2
                _ap = (atr / price * 100) if price > 0 else 1.0

                # ── Path 1: Blow-off top → SHORT alert ───────────────────────
                # Gainers SHORT: threshold lowered if derivatives confirm (SQUEEZE/CROWDED)
                _g_deriv_boost  = r.get("deriv_boost_short", False)
                _g_short_thresh = 20 if _g_deriv_boost else 25
                if exh.get("short_opp") and change_24h >= _g_short_thresh:
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
                    # Fix Score:0 on gainers SHORT — set score from exhaustion signal count
                    _g_exh_score = min(85, 50 + exh.get("blowoff_signals", 0) * 9)
                    r["abs_score"]  = _g_exh_score
                    r["score"]      = -_g_exh_score
                    r["confidence"] = "HIGH" if _g_exh_score >= 68 else "MEDIUM"
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
                                             r.get("confidence","MEDIUM"), signal_type="MOMENTUM",
                                             grade=r.get("grade","B"), dow_phase=r.get("dow_phase","UNCLEAR"),
                                             leverage=r.get("suggested_leverage",LEVERAGE))
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
                      and r.get("confidence") in ("HIGH", "MEDIUM")
                      and r.get("signal_type") == "MOMENTUM"
                      and (r.get("grade") == "A" or (
                           _btc_dom_state["mode"] == "simplified"
                           and r.get("grade") == "B"
                           and r.get("abs_score", 0) >= 75))
                      and r.get("abs_score", 0) >= 70
                      and not exh.get("block_long")
                      and not r.get("deriv_block_long", False)  # derivatives gate
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
                                             r.get("confidence","MEDIUM"), signal_type="MOMENTUM",
                                             grade=r.get("grade","B"), dow_phase=r.get("dow_phase","UNCLEAR"),
                                             leverage=r.get("suggested_leverage",LEVERAGE))
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

                # ── Path 3: Shakeout → pre-pump LONG alert ────────────────
                elif not exh.get("block_long") and not btc_circuit_breaker["active"]:
                    try:
                        from signal_engine import detect_shakeout
                        df_15m_g = fetch_ohlcv(symbol, "15m", 100)
                        # FIX: `r.get("df_1h") or fetch_ohlcv(...)` causes "DataFrame is ambiguous"
                        # — must use explicit None/empty check, not Python or.
                        _raw_1h_sk = r.get("df_1h")
                        df_1h_g = _raw_1h_sk if (_raw_1h_sk is not None and not _raw_1h_sk.empty) \
                                  else fetch_ohlcv(symbol, "1h", 200)
                        df_4h_g  = r.get("df_4h")
                        if df_15m_g is not None and df_1h_g is not None:
                            sk = detect_shakeout(df_15m_g, df_1h_g, df_4h_g, symbol)
                            if sk["is_shakeout"] and sk["confidence"] in ("HIGH", "MEDIUM"):
                                # Check portfolio heat
                                _sk_heat = get_portfolio_heat(active_signals)
                                if not _sk_heat["can_open"]:
                                    log.info(f"Shakeout LONG blocked by heat: {label}")
                                else:
                                    price_sk  = sk["entry_zone"] or r["price"]
                                    atr_sk    = r.get("atr", price_sk * 0.02)
                                    _ap_sk    = (atr_sk / price_sk * 100) if price_sk > 0 else 1.0
                                    sl_sk, tp1_sk, tp2_sk, vp_sk = calc_levels_v2(
                                        "LONG", price_sk, atr_sk, _ap_sk, df_4h_g)
                                    # Use flush low as hard SL if tighter
                                    flush_sl = sk["invalidation"]
                                    if flush_sl > 0 and flush_sl > sl_sk:
                                        sl_sk = flush_sl
                                    ob_bias, ob_ratio = get_order_book_bias(symbol)
                                    pos_usdt, contracts = calc_position_size(price_sk, sl_sk)

                                    record_signal(
                                        symbol, "LONG", price_sk, sl_sk, tp1_sk, tp2_sk,
                                        "MEDIUM", trade_type="swing",
                                        signal_type="MOMENTUM", grade=r.get("grade", "B"),
                                        dow_phase=r.get("dow_phase", "UNCLEAR"),
                                        vwap_bias=r.get("vwap_bias", "AT"))
                                    if paper_mode:
                                        existing = load_json(PAPER_FILE, [])
                                        if not any(t["symbol"] == symbol and t["status"] == "OPEN"
                                                   for t in existing):
                                            open_paper_trade(
                                                symbol, "LONG", price_sk, sl_sk, tp1_sk, tp2_sk,
                                                "MEDIUM", signal_type="MOMENTUM",
                                                grade=r.get("grade","B"),
                                                shakeout=True,
                                                entry_action="ENTER",
                                                leverage=r.get("suggested_leverage", LEVERAGE))
                                    active_signals[symbol] = {
                                        "direction": "LONG", "entry": price_sk,
                                        "sl": sl_sk, "tp1": tp1_sk, "tp2": tp2_sk,
                                        "tp1_hit": False, "atr": atr_sk,
                                        "time": now, "trailing_extreme": price_sk,
                                        "trailing_stop": None,
                                    }
                                    gainers_state["last_signals"][symbol] = now

                                    sk_sigs = " | ".join(sk["signals"][:2])
                                    msg  = "\u26a1\U0001f3af *Shakeout LONG: " + label + "*\n"
                                    msg += "Confidence: *" + sk["confidence"] + "* (score " + str(sk["score"]) + "/8)\n"
                                    msg += "Flush low: `$" + "{:.4f}".format(sk["flush_low"]) + "` \u2014 reclaimed\n"
                                    msg += "Pump target: `$" + "{:.4f}".format(sk["pump_target"]) + "`\n"
                                    msg += "Invalidation (hard SL): `$" + "{:.4f}".format(sl_sk) + "`\n"
                                    msg += "_" + sk_sigs + "_\n\n"
                                    msg += format_signal(r, sl_sk, tp1_sk, tp2_sk,
                                                         ob_bias, ob_ratio, pos_usdt, contracts)
                                    await send_msg(app, msg)
                                    log.info(f"Shakeout LONG: {label} | score {sk['score']}/8 | confidence {sk['confidence']}")
                    except Exception as sk_err:
                        log.warning(f"Shakeout path error for {label}: {sk_err}")

                await asyncio.sleep(REQUEST_DELAY * 2)  # rate limit between coins

        except Exception as e:
            log.error("Gainers scan error: " + str(e))


# ─── COUNTER SCALP SCANNER ───────────────────────────────────────────────────
# Coins that have run 8%+ in the last 4H and are now sitting extended above
# a clean demand OB. These are the counter scalp setups — short the extension
# back to the OB zone. Runs every 45 minutes, alerts directly to Telegram.
counter_scan_state = load_json(COUNTER_SCAN_FILE, {})  # persisted — survives Railway restarts

async def _run_counter_scan(app, triggered_by_command=False):
    """
    Core counter scan logic — shared by auto_counter_scan and /counterscan command.

    For each coin in COIN_LABELS:
      1. Fetch 1H + 4H OHLCV
      2. Check 4H change — must be ≥ 8% up (for LONG→SHORT counter)
         or ≥ 8% down (for SHORT→LONG counter)
      3. Detect demand/supply OB on 1H
      4. Check price is ≥ 5% away from OB zone
      5. Calculate counter trade: entry=now, TP=OB zone, SL=recent swing extreme
      6. Require R:R ≥ 1.5
      7. Alert if all conditions met

    Alert format:
      ⚡ Counter Scalp: COIN (SHORT)
      Coin ran +14.5% — now extended above demand zone
      Entry: $X | TP: $Y (Z%) | SL: $W (V%) | RR: N:1
      Zone: OB,FVG (Strong POI — 3 layers)
      /refine COIN SHORT X 10 scalp
    """
    from signal_engine import detect_order_blocks, detect_poi_density
    now     = _time.time()
    found   = 0
    checked = 0
    alerts  = []

    for symbol, label in list(COIN_LABELS.items()):
        # Per-coin 4H cooldown — don't spam the same setup
        if not triggered_by_command:
            if now - counter_scan_state.get(symbol, 0) < 14400:
                continue

        try:
            # Try Bitget first, fall back to MEXC for coins not listed there
            df_1h = fetch_ohlcv(symbol, "1h", 60)
            df_4h = fetch_ohlcv(symbol, "4h", 24)
            if df_1h is None or len(df_1h) < 20:
                try:
                    df_1h = fetch_ohlcv_mexc(symbol, "1h", 60)
                except Exception:
                    df_1h = None
            if df_4h is None or len(df_4h) < 6:
                try:
                    df_4h = fetch_ohlcv_mexc(symbol, "4h", 24)
                except Exception:
                    df_4h = None
            if df_1h is None or df_4h is None or len(df_1h) < 20 or len(df_4h) < 6:
                continue

            checked += 1
            price_now    = float(df_1h["close"].iloc[-1])
            price_4h_ago = float(df_4h["close"].iloc[-5])   # ~20H ago on 4H
            price_1h_ago = float(df_1h["close"].iloc[-5])   # 4 x 1H bars ago
            if price_4h_ago <= 0 or price_1h_ago <= 0:
                continue

            # Dual-window change detection
            change_4h = (price_now - price_4h_ago) / price_4h_ago * 100
            change_1h = (price_now - price_1h_ago) / price_1h_ago * 100

            # ── Check for LONG→SHORT counter (coin pumped, fade back to OB) ──
            if change_4h >= 5.0 or change_1h >= 4.0:
                ob_data = detect_order_blocks(df_1h)
                ob_ref  = ob_data.get("demand_ob")
                if not ob_ref or not ob_ref.get("active"):
                    continue

                ob_top  = ob_ref.get("high", 0)
                ob_bot  = ob_ref.get("low", 0)
                if ob_top <= 0 or ob_top >= price_now:
                    continue

                dist_pct = (price_now - ob_top) / price_now * 100
                if dist_pct < 3.0:
                    continue

                # Zone quality gate — Moderate POI or better (2+ layers)
                _poi = detect_poi_density(df_1h, ob_top, "LONG")
                if _poi.get("density", 0) < 2:
                    continue

                # ── Fix 1: RSI momentum filter ────────────────────────────────
                # If 1H RSI > 72, momentum is too strong for a counter SHORT.
                # BSB-type setups (28% pump with RSI 80+) will keep going.
                try:
                    _rsi_s = ta.rsi(df_1h["close"], length=14)
                    _rsi_v = float(_rsi_s.dropna().iloc[-1]) if _rsi_s is not None and len(_rsi_s.dropna()) > 0 else 50.0
                except Exception:
                    _rsi_v = 50.0
                if _rsi_v > 72:
                    log.info(f"Counter scan: {label} SHORT blocked — RSI {_rsi_v:.0f} too high, momentum too strong")
                    continue

                # ── Fix 2: Volume expansion filter ───────────────────────────
                # If current volume > 2.5x 20-period average, institutional
                # momentum is driving the move — counter fade is too risky.
                try:
                    _vol_now = float(df_1h["volume"].iloc[-1])
                    _vol_ma  = float(df_1h["volume"].rolling(20).mean().iloc[-1])
                    _vol_ratio = _vol_now / _vol_ma if _vol_ma > 0 else 1.0
                except Exception:
                    _vol_ratio = 1.0
                if _vol_ratio > 2.5:
                    log.info(f"Counter scan: {label} SHORT blocked — volume {_vol_ratio:.1f}x MA, momentum expansion")
                    continue

                # ── Fix 3: ATR-based SL (replaces fixed 5-bar high * 1.015) ──
                # Fixed % buffer on a spike candle puts SL barely above entry.
                # ATR * 1.5 gives a realistic stop based on current volatility.
                try:
                    _atr_s   = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
                    _atr_val = float(_atr_s.dropna().iloc[-1]) if _atr_s is not None and len(_atr_s.dropna()) > 0 else price_now * 0.015
                    c_sl     = round(price_now + _atr_val * 1.5, 6)
                except Exception:
                    c_sl     = round(float(df_1h["high"].tail(5).max()) * 1.015, 6)
                c_sl_pct = (c_sl - price_now) / price_now * 100
                if c_sl_pct <= 0:
                    continue

                c_tp     = round(ob_top, 6)
                c_tp_pct = dist_pct
                c_rr     = round(c_tp_pct / c_sl_pct, 1)
                if c_rr < 1.5:
                    continue

                _zone_label = _poi.get("label", "OB zone")
                _zone_icon  = ("🟢" if _poi.get("density", 0) >= 3
                               else ("🟡" if _poi.get("density", 0) >= 2 else "🔴"))

                alerts.append({
                    "symbol":     symbol,
                    "label":      label,
                    "direction":  "SHORT",
                    "base_dir":   "LONG",
                    "change_pct": change_4h,
                    "price":      price_now,
                    "c_tp":       c_tp,
                    "c_tp_pct":   round(c_tp_pct, 1),
                    "c_sl":       c_sl,
                    "c_sl_pct":   round(c_sl_pct, 1),
                    "c_rr":       c_rr,
                    "ob_bot":     ob_bot,
                    "zone_label": _zone_label,
                    "zone_icon":  _zone_icon,
                    "ob_range":   f"${ob_top:.4f}–${ob_bot:.4f}",
                    "decoupled":  False,
                })
                found += 1

            # ── Check for SHORT→LONG counter (coin dumped, bounce back to OB) ─
            elif change_4h <= -5.0 or change_1h <= -4.0:
                ob_data = detect_order_blocks(df_1h)
                ob_ref  = ob_data.get("supply_ob")
                if not ob_ref or not ob_ref.get("active"):
                    continue

                ob_bot  = ob_ref.get("low", 0)
                ob_top  = ob_ref.get("high", 0)
                if ob_bot <= 0 or ob_bot <= price_now:
                    continue

                dist_pct = (ob_bot - price_now) / price_now * 100
                if dist_pct < 3.0:
                    continue

                # Zone quality gate — Moderate POI or better (2+ layers)
                _poi_pre = detect_poi_density(df_1h, ob_bot, "SHORT")
                if _poi_pre.get("density", 0) < 2:
                    continue

                # ── Fix 1: RSI momentum filter (mirrored for dump) ───────────
                # If 1H RSI < 28, momentum is too bearish for a counter LONG.
                try:
                    _rsi_s = ta.rsi(df_1h["close"], length=14)
                    _rsi_v = float(_rsi_s.dropna().iloc[-1]) if _rsi_s is not None and len(_rsi_s.dropna()) > 0 else 50.0
                except Exception:
                    _rsi_v = 50.0
                if _rsi_v < 28:
                    log.info(f"Counter scan: {label} LONG blocked — RSI {_rsi_v:.0f} too low, dump momentum too strong")
                    continue

                # ── Fix 2: Volume expansion filter ───────────────────────────
                try:
                    _vol_now   = float(df_1h["volume"].iloc[-1])
                    _vol_ma    = float(df_1h["volume"].rolling(20).mean().iloc[-1])
                    _vol_ratio = _vol_now / _vol_ma if _vol_ma > 0 else 1.0
                except Exception:
                    _vol_ratio = 1.0
                if _vol_ratio > 2.5:
                    log.info(f"Counter scan: {label} LONG blocked — volume {_vol_ratio:.1f}x MA, dump momentum expansion")
                    continue

                # ── Fix 3: ATR-based SL ───────────────────────────────────────
                try:
                    _atr_s   = ta.atr(df_1h["high"], df_1h["low"], df_1h["close"], length=14)
                    _atr_val = float(_atr_s.dropna().iloc[-1]) if _atr_s is not None and len(_atr_s.dropna()) > 0 else price_now * 0.015
                    c_sl     = round(price_now - _atr_val * 1.5, 6)
                except Exception:
                    c_sl     = round(float(df_1h["low"].tail(5).min()) * 0.985, 6)
                c_sl_pct = (price_now - c_sl) / price_now * 100
                if c_sl_pct <= 0:
                    continue

                c_tp     = round(ob_bot, 6)
                c_tp_pct = dist_pct
                c_rr     = round(c_tp_pct / c_sl_pct, 1)
                if c_rr < 1.5:
                    continue

                # ── BTC circuit breaker gate (LONG counters only) ─────────────
                # Full block = sustained BTC 4H downtrend. LONG counter setups
                # during this period almost always fail /refine. No point alerting.
                #
                # Decoupling exception: coin up 3%+ on 1H while BTC is down →
                # genuine independence from macro. Let it through.
                _is_decoupled = False
                if (btc_circuit_breaker.get("active")
                        and btc_circuit_breaker.get("tier") == "full"):
                    try:
                        _btc_1h = fetch_ohlcv("BTC/USDT:USDT", "1h", 5)
                        if _btc_1h is not None and len(_btc_1h) >= 5:
                            _btc_chg_1h = ((float(_btc_1h["close"].iloc[-1]) -
                                            float(_btc_1h["close"].iloc[-4])) /
                                           float(_btc_1h["close"].iloc[-4]) * 100)
                            if change_1h >= 3.0 and _btc_chg_1h < 0:
                                _is_decoupled = True
                    except Exception:
                        pass  # BTC fetch failed — be conservative, block the alert

                    if not _is_decoupled:
                        log.info(f"Counter scan: {label} LONG blocked — circuit breaker active, no decoupling")
                        continue

                _poi        = _poi_pre
                _zone_label = _poi.get("label", "Supply OB zone")
                _zone_icon  = ("🟢" if _poi.get("density", 0) >= 3
                               else ("🟡" if _poi.get("density", 0) >= 2 else "🔴"))

                alerts.append({
                    "symbol":     symbol,
                    "label":      label,
                    "direction":  "LONG",
                    "base_dir":   "SHORT",
                    "change_pct": change_4h,
                    "price":      price_now,
                    "c_tp":       c_tp,
                    "c_tp_pct":   round(c_tp_pct, 1),
                    "c_sl":       c_sl,
                    "c_sl_pct":   round(c_sl_pct, 1),
                    "c_rr":       c_rr,
                    "ob_bot":     ob_top,
                    "zone_label": _zone_label,
                    "zone_icon":  _zone_icon,
                    "ob_range":   f"${ob_bot:.4f}–${ob_top:.4f}",
                    "decoupled":  _is_decoupled,
                })
                found += 1

        except Exception as _ce:
            log.warning(f"Counter scan error {label}: {_ce}")

        await asyncio.sleep(REQUEST_DELAY)

    # ── Send alerts ───────────────────────────────────────────────────────────
    if not alerts:
        if triggered_by_command:
            return "⚡ *Counter Scalp Scan*\n\nNo setups found right now.\nLooking for: 5%+ extended (4H) or 4%+ (1H) from a Moderate+ OB zone with R:R ≥ 1.5."
        return None

    # Sort by R:R descending — best setups first
    alerts.sort(key=lambda x: x["c_rr"], reverse=True)

    if triggered_by_command:
        msg = f"⚡ *Counter Scalp Scan* — {found} setup{'s' if found > 1 else ''} found\n\n"
    else:
        msg = f"⚡ *Counter Scalp Alert* — {found} setup{'s' if found > 1 else ''}\n\n"

    for a in alerts[:5]:   # cap at 5 alerts per scan
        change_str = f"+{a['change_pct']:.1f}%" if a['change_pct'] > 0 else f"{a['change_pct']:.1f}%"
        verb       = "pumped" if a["direction"] == "SHORT" else "dumped"
        decouple_note = " ⚡ decoupled from BTC" if a.get("decoupled") else ""
        msg += f"{'—' * 20}\n"
        msg += f"⚡ *{a['label']} {a['direction']}* (counter scalp{decouple_note})\n"
        msg += f"Coin {verb} {change_str} — extended from {a['base_dir']} zone\n"
        msg += f"Entry: `${a['price']:.4f}` | TP: `${a['c_tp']:.4f}` ({a['c_tp_pct']:.1f}%)\n"
        msg += f"SL: `${a['c_sl']:.4f}` ({a['c_sl_pct']:.1f}%) | RR: {a['c_rr']:.1f}:1\n"
        msg += f"{a['zone_icon']} Zone: {a['zone_label']} — {a['ob_range']}\n"
        msg += f"When TP hits → flip to {a['base_dir']} from zone\n"
        msg += (f"`/refine {a['label']} {a['direction']} "
                f"{a['price']:.4f} 10 scalp`\n\n")
        if not triggered_by_command:
            counter_scan_state[a["symbol"]] = now

    # Persist cooldowns so Railway restarts don't re-alert same setups
    if not triggered_by_command:
        save_json(COUNTER_SCAN_FILE, counter_scan_state)

    return msg


async def auto_counter_scan(app):
    """Runs _run_counter_scan every 45 minutes and pushes alerts to Telegram."""
    log.info("Counter scalp scanner started")
    await asyncio.sleep(180)   # offset from other scans at startup
    while True:
        await asyncio.sleep(2700)   # 45 minutes
        try:
            msg = await _run_counter_scan(app, triggered_by_command=False)
            if msg:
                await send_msg(app, msg)
                log.info("Counter scan alert sent")
        except Exception as e:
            log.error(f"auto_counter_scan error: {e}")


@owner_only
async def cmd_counterscan(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /counterscan — manual trigger of counter scalp scan.
    Scans all coins immediately, ignoring cooldowns.
    """
    await update.message.reply_text("⚡ Scanning for counter scalp setups...")
    try:
        msg = await _run_counter_scan(app=ctx.application, triggered_by_command=True)
        if not msg:
            await update.message.reply_text("⚡ Counter Scalp Scan\n\nNo setups found.")
            return
        try:
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception:
            _plain = msg.replace("*", "").replace("`", "").replace("_", "")
            await update.message.reply_text(_plain)
    except Exception as e:
        await update.message.reply_text(f"Counter scan error: {e}")
async def fetch_btc_dominance() -> float:
    """
    Fetch BTC market dominance % from CoinGecko public API.
    No API key required.
    """
    try:
        import urllib.request as _ur, json as _j
        url = "https://api.coingecko.com/api/v3/global"
        req = _ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _ur.urlopen(req, timeout=10) as r:
            data = _j.loads(r.read())
        dom = float(data["data"]["market_cap_percentage"]["btc"])
        log.info(f"BTC dominance: {dom:.1f}%")
        return dom
    except Exception as e:
        log.warning(f"BTC dominance fetch error: {e}")
        return _btc_dom_state.get("dominance") or 58.0  # fallback to last known


async def auto_dominance_monitor(app):
    """
    Background loop — checks BTC dominance every 30 minutes.
    Switches pipeline between full and simplified mode:

    Full mode   (dominance < 52%): Supertrend gate active, Grade A required
    Transition  (52-55%):          Simplified but log the approach
    Simplified  (dominance > 55%): Supertrend disabled, Grade B allowed

    Sends Telegram alert when mode switches.
    """
    log.info("Dominance monitor started")
    await asyncio.sleep(30)  # small offset from startup

    while True:
        try:
            now = _time.time()
            dom = await fetch_btc_dominance()

            if dom <= 0:
                await asyncio.sleep(BTC_DOM_FETCH_INTERVAL)
                continue

            prev_mode = _btc_dom_state["mode"]
            prev_dom  = _btc_dom_state.get("dominance") or dom

            _btc_dom_state["dominance"]  = dom
            _btc_dom_state["last_fetch"] = now

            # Determine new mode
            if dom < BTC_DOM_FULL_THRESHOLD:
                new_mode = "full"
            elif dom > BTC_DOM_SIMPLIFIED_THRESHOLD:
                new_mode = "simplified"
            else:
                new_mode = prev_mode  # in transition zone — keep current mode

            _btc_dom_state["mode"] = new_mode

            # Apply to signal engine
            from signal_engine import set_pipeline_mode
            set_pipeline_mode(new_mode == "full")

            # Alert on mode switch
            if new_mode != prev_mode and now - _btc_dom_state.get("last_alert", 0) > 3600:
                _btc_dom_state["last_alert"] = now
                if new_mode == "full":
                    msg  = "\U0001f7e2 *Pipeline: FULL MODE*\n\n"
                    msg += f"BTC dominance dropped to *{dom:.1f}%* (was {prev_dom:.1f}%)\n"
                    msg += f"Threshold: below {BTC_DOM_FULL_THRESHOLD}%\n\n"
                    msg += "Supertrend gate active. Grade A signals only.\n"
                    msg += "_Altcoin season conditions forming._"
                else:
                    msg  = "\U0001f7e1 *Pipeline: SIMPLIFIED MODE*\n\n"
                    msg += f"BTC dominance at *{dom:.1f}%* (was {prev_dom:.1f}%)\n"
                    msg += f"Threshold: above {BTC_DOM_SIMPLIFIED_THRESHOLD}%\n\n"
                    msg += "Supertrend disabled. Grade B signals allowed.\n"
                    msg += "_BTC season — simplified scanner active._"
                await send_msg(app, msg)
                log.info(f"Pipeline mode switched: {prev_mode} → {new_mode} | BTC dom={dom:.1f}%")

        except Exception as e:
            log.error(f"Dominance monitor error: {e}")

        await asyncio.sleep(BTC_DOM_FETCH_INTERVAL)

# ─── BTC CIRCUIT BREAKER ───────────────────────────────────────────────────────
def check_btc_circuit_breaker() -> dict:
    """
    Checks BTC condition and updates btc_circuit_breaker state.

    TIERED response — previously this was a binary full-block on all LONGs
    which killed valid setups on alts that hold structure while BTC corrects.

    Tier 1 — ELEVATED (fast_dump only: single 1H candle -2%):
      LONGs still allowed but quality bar is raised:
        - quality_score must be >= 75 (was 65)
        - grade must be A (B not allowed)
        - coin's own SMC structure must be BULLISH or have active demand OB
      This still fires on clean structure retests — it just removes marginal setups.

    Tier 2 — FULL BLOCK (bear_regime: 4H TRENDING_DOWN + ADX > 25):
      No new LONGs at all. A sustained confirmed downtrend on BTC 4H means
      the macro context is against longs across the board.

    Reset — simplified to 2 conditions (previously 3 which kept breaker stuck):
      drop < 1%  AND  RSI > 48
      Regime check removed — regime lags too much, would keep breaker on
      through the first 4H candle of a recovery.

    Returns updated btc_circuit_breaker dict with "tier" key:
      "none" | "elevated" | "full"
    """
    global btc_circuit_breaker
    try:
        from signal_engine import detect_regime
        import pandas_ta as _ta

        df_1h = fetch_ohlcv("BTC/USDT:USDT", "1h", 50)
        df_4h = fetch_ohlcv("BTC/USDT:USDT", "4h", 50)
        if df_1h is None or df_4h is None or len(df_1h) < 3:
            return btc_circuit_breaker   # no data — return unchanged state

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

        # Update global cache so monitor loop can use real BTC market regime
        # for signal invalidation (not pipeline mode which is a different thing)
        global _btc_market_regime_cache
        _btc_market_regime_cache["regime"]     = regime_name
        _btc_market_regime_cache["adx"]        = adx_val
        _btc_market_regime_cache["updated_at"] = _time.time()

        btc_circuit_breaker["last_check"] = _time.time()

        # ── Tier classification ───────────────────────────────────────────────
        fast_dump   = candle_chg <= -BTC_DUMP_THRESHOLD
        bear_regime = regime_name == "TRENDING_DOWN" and adx_val > 25

        if bear_regime:
            # Tier 2: sustained 4H downtrend — full block
            reason = f"4H regime TRENDING_DOWN (ADX:{adx_val:.0f})"
            if fast_dump:
                reason = f"BTC 1H dropped {candle_chg:.1f}% + " + reason
            btc_circuit_breaker["active"]       = True
            btc_circuit_breaker["tier"]         = "full"
            btc_circuit_breaker["reason"]       = reason
            btc_circuit_breaker["triggered_at"] = _time.time()
            log.info("BTC circuit breaker FULL BLOCK: " + reason)
            return btc_circuit_breaker

        elif fast_dump:
            # Tier 1: single candle spike — raise quality bar, don't kill alts with clean structure
            reason = f"BTC 1H dropped {candle_chg:.1f}% — elevated quality threshold"
            btc_circuit_breaker["active"]       = True
            btc_circuit_breaker["tier"]         = "elevated"
            btc_circuit_breaker["reason"]       = reason
            btc_circuit_breaker["triggered_at"] = _time.time()
            log.info("BTC circuit breaker ELEVATED: " + reason)
            return btc_circuit_breaker

        # ── Reset check ───────────────────────────────────────────────────────
        # Simplified to 2 conditions — old triple-condition (drop + RSI + regime) kept
        # the breaker active for hours through early recovery because 4H regime lags.
        if btc_circuit_breaker["active"]:
            drop_ok = candle_chg > -BTC_RESET_DROP
            rsi_ok  = rsi_val > BTC_RESET_RSI
            if drop_ok and rsi_ok:
                btc_circuit_breaker["active"] = False
                btc_circuit_breaker["tier"]   = "none"
                btc_circuit_breaker["reason"] = ""
                log.info(f"BTC circuit breaker RESET — RSI:{rsi_val:.1f} drop:{candle_chg:.1f}%")

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
                _tier = state.get("tier", "full")
                if _tier == "full":
                    msg  = "🚨 *BTC Circuit Breaker — FULL BLOCK*\n\n"
                    msg += "Reason: " + state["reason"] + "\n"
                    msg += "All new LONG signals are blocked — sustained 4H bear trend confirmed.\n"
                    msg += "_Existing open trades are unaffected._"
                else:
                    msg  = "⚠️ *BTC Circuit Breaker — ELEVATED*\n\n"
                    msg += "Reason: " + state["reason"] + "\n"
                    msg += "LONG bar raised: score ≥ 75 + Grade A + confirmed SMC structure required.\n"
                    msg += "_Clean demand OB retests and CHOCH setups can still fire._"
                await send_msg(app, msg)
                log.info("Circuit breaker alert sent: tier=" + _tier)

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


@owner_only
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


@owner_only
async def cmd_domstatus(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /domstatus — show current BTC dominance and pipeline mode
    """
    dom  = _btc_dom_state.get("dominance")
    mode = _btc_dom_state.get("mode", "simplified")
    last = _btc_dom_state.get("last_fetch", 0)
    age  = round((_time.time() - last) / 60) if last else None

    mode_emoji = "\U0001f7e2" if mode == "full" else "\U0001f7e1"
    dom_str    = f"{dom:.1f}%" if dom else "fetching..."
    age_str    = f"{age}m ago" if age is not None else "not yet fetched"

    msg  = "\U0001f4ca *BTC Dominance Status*\n\n"
    msg += f"Dominance: *{dom_str}* (updated {age_str})\n"
    msg += mode_emoji + f" Mode: *{mode.upper()}*\n\n"

    if mode == "full":
        msg += "\U0001f7e2 Full pipeline active:\n"
        msg += "  • Supertrend 4H gate blocks opposing signals\n"
        msg += "  • Grade A signals only\n"
        msg += "  • 2-candle confirmation required\n"
    else:
        msg += "\U0001f7e1 Simplified pipeline active:\n"
        msg += "  • Supertrend disabled\n"
        msg += "  • Grade B signals allowed\n"
        msg += "  • No 2-candle confirmation\n\n"
        if dom and dom < 55:
            msg += f"_Approaching transition zone ({BTC_DOM_FULL_THRESHOLD}%-{BTC_DOM_SIMPLIFIED_THRESHOLD}%)_"

    await update.message.reply_text(msg, parse_mode="Markdown")

@owner_only
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

    Symbol format: "AIA/USDT:USDT" → coin = "AIA" → MEXC: "AIA_USDT"
    """
    # Try Bitget first
    try:
        price = float(exchange.fetch_ticker(symbol)["last"])
        if price > 0:
            return price
    except Exception:
        pass

    # Extract base coin: "AIA/USDT:USDT" → "AIA"
    coin = symbol.split("/")[0].upper()

    # Try MEXC futures public API (no auth needed)
    try:
        import urllib.request as _ur, json as _j
        url = "https://contract.mexc.com/api/v1/contract/ticker?symbol=" + coin + "_USDT"
        with _ur.urlopen(url, timeout=8) as r:
            data = _j.loads(r.read())
        if data.get("success") and data.get("data"):
            price = float(data["data"].get("lastPrice", 0))
            if price > 0:
                log.info("MEXC price for " + coin + ": $" + str(price))
                return price
    except Exception as e:
        log.warning("MEXC futures price error " + coin + ": " + str(e))

    # Try MEXC spot
    try:
        import urllib.request as _ur, json as _j
        url = "https://api.mexc.com/api/v3/ticker/price?symbol=" + coin + "USDT"
        with _ur.urlopen(url, timeout=8) as r:
            data = _j.loads(r.read())
        price = float(data.get("price", 0))
        if price > 0:
            log.info("MEXC spot price for " + coin + ": $" + str(price))
            return price
    except Exception as e:
        log.warning("MEXC spot price error " + coin + ": " + str(e))

    # Try Binance spot (broad coverage, no auth needed)
    try:
        import urllib.request as _ur, json as _j
        url = "https://api.binance.com/api/v3/ticker/price?symbol=" + coin + "USDT"
        with _ur.urlopen(url, timeout=8) as r:
            data = _j.loads(r.read())
        price = float(data.get("price", 0))
        if price > 0:
            log.info("Binance spot price for " + coin + ": $" + str(price))
            return price
    except Exception as e:
        log.warning("Binance price error " + coin + ": " + str(e))

    # Try KuCoin (covers many small-cap coins)
    try:
        import urllib.request as _ur, json as _j
        url = "https://api.kucoin.com/api/v1/market/orderbook/level1?symbol=" + coin + "-USDT"
        with _ur.urlopen(url, timeout=8) as r:
            data = _j.loads(r.read())
        if data.get("code") == "200000" and data.get("data"):
            price = float(data["data"].get("price", 0))
            if price > 0:
                log.info("KuCoin price for " + coin + ": $" + str(price))
                return price
    except Exception as e:
        log.warning("KuCoin price error " + coin + ": " + str(e))

    # Check manual price override (set via /setprice command)
    manual = _manual_prices.get(coin.upper())
    if manual and manual.get("price", 0) > 0:
        age_mins = (_time.time() - manual.get("ts", 0)) / 60
        log.info("Using manual price for " + coin + ": $" + str(manual["price"]) + " (set " + str(round(age_mins)) + "m ago)")
        return float(manual["price"])

    return 0.0


def fetch_ohlcv_mexc(symbol: str, timeframe: str = "1h", limit: int = 100):
    """
    Fetch OHLCV from MEXC for coins not on Bitget.
    Uses MEXC public REST API directly — no ccxt auth needed.
    Returns pandas DataFrame in same format as fetch_ohlcv().
    """
    import pandas as _pd, urllib.request as _ur, json as _j

    coin = symbol.split("/")[0].upper()

    # MEXC futures kline API interval mapping
    tf_map = {"1m": "Min1", "5m": "Min5", "15m": "Min15",
              "30m": "Min30", "1h": "Min60", "4h": "Hour4",
              "1d": "Day1"}
    interval = tf_map.get(timeframe, "Min60")

    # Try MEXC futures kline API
    try:
        url = ("https://contract.mexc.com/api/v1/contract/kline/" + coin + "_USDT"
               + "?interval=" + interval + "&limit=" + str(limit))
        with _ur.urlopen(url, timeout=10) as r:
            data = _j.loads(r.read())
        if data.get("success") and data.get("data"):
            rows = data["data"]
            # MEXC futures kline: [time, open, high, low, close, vol, ...]
            records = []
            for row in rows:
                records.append({
                    "timestamp": _pd.to_datetime(int(row[0]), unit="s"),
                    "open":  float(row[1]),
                    "high":  float(row[2]),
                    "low":   float(row[3]),
                    "close": float(row[4]),
                    "volume":float(row[5]) if len(row) > 5 else 0.0,
                })
            df = _pd.DataFrame(records).set_index("timestamp")
            log.info("MEXC OHLCV " + coin + " " + timeframe + ": " + str(len(df)) + " candles")
            return df
    except Exception as e:
        log.warning("MEXC futures kline error " + coin + ": " + str(e))

    # Fallback: MEXC spot kline
    try:
        url = ("https://api.mexc.com/api/v3/klines?symbol=" + coin + "USDT"
               + "&interval=" + timeframe.replace("h", "h").replace("m", "m")
               + "&limit=" + str(limit))
        with _ur.urlopen(url, timeout=10) as r:
            data = _j.loads(r.read())
        # Spot kline: [openTime, open, high, low, close, vol, ...]
        records = []
        for row in data:
            records.append({
                "timestamp": _pd.to_datetime(int(row[0]), unit="ms"),
                "open":  float(row[1]),
                "high":  float(row[2]),
                "low":   float(row[3]),
                "close": float(row[4]),
                "volume":float(row[5]),
            })
        df = _pd.DataFrame(records).set_index("timestamp")
        log.info("MEXC spot OHLCV " + coin + " " + timeframe + ": " + str(len(df)) + " candles")
        return df
    except Exception as e:
        log.warning("MEXC spot kline error " + coin + ": " + str(e))

    return None



@owner_only
async def cmd_setprice(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """
    /setprice COIN PRICE — manually set current price for a watched coin
    when all exchange APIs fail to fetch it.

    Example: /setprice 4USDT 0.0042
    The price is stored and used until a live API fetch succeeds.
    """
    args = ctx.args
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /setprice COIN PRICE\n"
            "Example: /setprice 4USDT 0.0042\n\n"
            "Use when the bot shows 'Price unavailable' for a MEXC coin."
        )
        return

    coin = args[0].upper()
    try:
        price = float(args[1])
        if price <= 0:
            raise ValueError()
    except ValueError:
        await update.message.reply_text("Invalid price. Example: /setprice 4USDT 0.0042")
        return

    _manual_prices[coin] = {"price": price, "ts": _time.time()}

    # Update trail_peak if watching this coin and no peak set yet
    for sym, w in watched_trades.items():
        if w["label"].upper() == coin:
            if not w.get("trail_peak") or w.get("trail_peak") == w.get("entry"):
                watched_trades[sym]["trail_peak"] = price
                log.info("Updated trail_peak for " + coin + " to $" + str(price))

    await update.message.reply_text(
        "\U0001f4cd *Manual price set: " + coin + "* = `$" + "{:.6f}".format(price) + "`\n\n"
        "_Will be used until a live exchange price is fetched.\n"
        "Trail SL and TP alerts will now work normally._",
        parse_mode="Markdown"
    )
    log.info("Manual price set: " + coin + " = $" + str(price))

@owner_only
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

    # Optional args from position 5 onward: leverage, TP1%, TP2%
    # Format: /watch NEAR LONG 1.32 swing medium 5 10 20
    # where 5=leverage, 10=TP1%, 20=TP2% (leverage detected as <=50, TPs as order they appear)
    tp1_pct    = None
    tp2_pct    = None
    watch_lev  = LEVERAGE  # default from env
    numerics   = []
    for arg in args[5:]:
        if arg.lower() in ("mexc",):
            continue  # skip flag args
        try:
            numerics.append(float(arg))
        except ValueError:
            pass

    # Heuristic: first number <= 50 and looks like leverage (integer or x suffix)
    lev_args = [a for a in args[5:] if a.rstrip("xX").replace(".","").isdigit()
                and float(a.rstrip("xX")) <= 50]
    if lev_args:
        watch_lev = int(float(lev_args[0].rstrip("xX")))
        numerics  = [v for v in numerics if v != float(lev_args[0].rstrip("xX"))]

    if len(numerics) >= 1:
        tp1_pct = numerics[0]
    if len(numerics) >= 2:
        tp2_pct = numerics[1]

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
        "leverage":           watch_lev,
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
    lev_note   = "\n\u26a1 Leverage: *" + str(watch_lev) + "x*"  # always shown so user can confirm
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
        + conf_notes.get(confidence, "") + trail_note + lev_note + mexc_note + tp_note + "\n\n"
        "Status: /watchstatus " + coin + "\n"
        "Stop: /unwatch " + coin,
        parse_mode="Markdown"
    )
    save_json("watched_trades.json", {k: {f: v for f, v in w.items()
             if f not in ("trail_sl","trail_peak","tp1_hit","tp2_hit",
                          "be_alerted","hold_alert_sent","last_alert",
                          "last_early_alert","trail_last_alert")}
             for k, w in watched_trades.items()})
    log.info("Watch started: " + coin + " " + direction + entry_str)



@owner_only
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
        # Always use fetch_price_any_exchange — it tries Bitget first then
        # falls back to MEXC/Binance. Handles MEXC-only coins (like ARCSOL)
        # even when the watch was created without use_mexc flag.
        cur_price = fetch_price_any_exchange(symbol) or 0
    except Exception:
        cur_price = 0

    hold_hrs = round((now - w["start_time"]) / 3600, 1)

    msg = "\U0001f4ca *Watch Status: " + label + "*\n\n"
    msg += dir_e + " *" + direction + "* | " + w.get("trade_type","swing").upper()
    msg += " | Conf: " + w.get("confidence","medium").upper() + "\n"
    msg += "Watching: *" + str(hold_hrs) + "h* / " + str(w.get("max_hold_hours",72)) + "h max\n\n"

    if entry and entry > 0:
        msg += "*Entry:* `$" + "{:.4f}".format(entry) + "`\n"
        if cur_price and cur_price > 0:
            if direction == "LONG":
                cur_gain = (cur_price - entry) / entry * 100
            else:
                cur_gain = (entry - cur_price) / entry * 100
            pnl_lev = cur_gain * w.get("leverage", LEVERAGE)   # use per-watch leverage, not global
            pnl_e   = "\U0001f7e2" if cur_gain >= 0 else "\U0001f534"
            msg += "*Current:* `$" + "{:.4f}".format(cur_price) + "` " + pnl_e
            msg += " *" + "{:+.1f}".format(pnl_lev) + "%* at " + str(w.get("leverage", LEVERAGE)) + "x\n\n"
        else:
            msg += "_Price unavailable — check the exchange feed for this coin._\n\n"

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

@owner_only
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
        save_json("watched_trades.json", {k: {f: v for f, v in w.items()
                 if f not in ("trail_sl","trail_peak","tp1_hit","tp2_hit",
                              "be_alerted","hold_alert_sent","last_alert",
                              "last_early_alert","trail_last_alert")}
                 for k, w in watched_trades.items()})
        await update.message.reply_text("✅ Stopped watching *" + coin + "*.", parse_mode="Markdown")
        log.info("Watch stopped: " + coin)
    else:
        await update.message.reply_text(coin + " is not being watched.")


@owner_only
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
        # Assign safe defaults BEFORE the try so they are always defined even if
        # fetch_ticker throws. Previously direction/entry were inside the try — a
        # network error caused NameError on `direction` below, killing the entire
        # loop silently and leaving the user with no /watching response.
        entry     = w.get("entry")
        direction = w["direction"]
        price     = 0
        pnl_str   = ""
        try:
            _raw_price = fetch_price_any_exchange(symbol)
            price = float(_raw_price) if _raw_price else 0
            if entry and price > 0:
                _w_disp_lev = w.get("leverage", LEVERAGE)
                pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
                pnl_lev = pnl_pct * _w_disp_lev
                pnl_str = " | PnL: " + "{:+.1f}".format(pnl_lev) + "% at " + str(_w_disp_lev) + "x"
        except Exception:
            pass  # price stays 0, pnl_str stays "" — loop continues safely

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


def _watch_pnl_str(entry, price, direction, lev=None):
    """Helper — compute PnL string for watch alerts."""
    if not entry or not price:
        return ""
    _lev    = lev if lev is not None else LEVERAGE
    pnl_pct = (price - entry) / entry * 100 if direction == "LONG" else (entry - price) / entry * 100
    pnl_lev = round(pnl_pct * _lev, 1)
    pnl_raw = round(pnl_pct, 2)
    return "PnL: " + "{:+.2f}".format(pnl_raw) + "% (" + "{:+.1f}".format(pnl_lev) + "% at " + str(_lev) + "x)\n"


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
                _wmc = [c for c in macd_df.columns if c.startswith("MACD_")]
                _wms = [c for c in macd_df.columns if c.startswith("MACDs_")]
                if _wmc and _wms:
                    dif_now  = float(macd_df[_wmc[0]].iloc[-1])
                    dea_now  = float(macd_df[_wms[0]].iloc[-1])
                    dif_prev = float(macd_df[_wmc[0]].iloc[-2])
                    dea_prev = float(macd_df[_wms[0]].iloc[-2])
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

                # ── Detect exchange availability ──────────────────────────────
                # use_mexc flag set explicitly → use MEXC path
                # No flag → check Bitget. If not on Bitget, auto-detect and
                # silently switch to MEXC so coins like ARCSOL work without
                # the user needing to re-create the watch with 'mexc' flag.
                _use_mexc = w.get("use_mexc", False)
                if not _use_mexc:
                    try:
                        exchange.market(symbol)
                    except Exception:
                        # Not on Bitget — auto-enable MEXC path and persist the flag
                        _use_mexc = True
                        watched_trades[symbol]["use_mexc"] = True
                        log.info(f"Watch auto-switched to MEXC: {label} not on Bitget")

                # ── All position-tracking alerts (run every cycle) ────────────
                entry      = w.get("entry")
                trail_peak = w.get("trail_peak")
                trail_sl   = w.get("trail_sl")
                # Cooldown: scalp = 10 min (can complete in 30 min), swing = 30 min
                trail_cd   = 600 if not is_swing else 1800

                if entry and entry > 0:
                    try:
                        # ── Fetch price and OHLCV ─────────────────────────────
                        _df_trail = None
                        cur_price = 0.0
                        _w_lev    = w.get("leverage", LEVERAGE)   # hoisted — used by scalp block too

                        if _use_mexc:
                            # Try MEXC OHLCV first
                            _df_trail = fetch_ohlcv_mexc(symbol, "1h", 20)
                            if _df_trail is not None and len(_df_trail) > 0:
                                cur_price = float(_df_trail["close"].iloc[-1])
                            # If OHLCV failed, use price API directly
                            if cur_price <= 0:
                                cur_price = fetch_price_any_exchange(symbol)
                                _df_trail = None  # no candle data — use fixed step
                                log.info("Trail SL " + label + ": using direct price (OHLCV unavailable), price=$" + str(cur_price))
                        else:
                            _df_trail = fetch_ohlcv(symbol, "1h", 20)
                            if _df_trail is not None and len(_df_trail) > 0:
                                cur_price = float(_df_trail["close"].iloc[-1])
                            # Fallback for Bitget coins where OHLCV fails
                            if cur_price <= 0:
                                cur_price = fetch_price_any_exchange(symbol)

                        if cur_price <= 0:
                            log.warning("Trail SL " + label + ": could not fetch price, skipping")
                            continue

                        # ── Dynamic step size (ATR-based or estimated) ────────
                        if _df_trail is not None and len(_df_trail) >= 14:
                            _atr_s = _ta.atr(_df_trail["high"], _df_trail["low"], _df_trail["close"], length=14)
                            if _atr_s is not None and len(_atr_s.dropna()) > 0:
                                _atr_v   = float(_atr_s.dropna().iloc[-1])
                                _atr_pct = (_atr_v / cur_price * 100) if cur_price > 0 else 2.0
                                step_pct = max(3.0, min(10.0, round(_atr_pct * 2, 1)))
                            else:
                                step_pct = 5.0
                        else:
                            # No candle data — estimate step from price movement so far
                            rough_move = abs(cur_price - entry) / entry * 100 if entry > 0 else 5.0
                            step_pct   = max(3.0, min(8.0, round(rough_move * 0.3, 1)))
                            log.info("Trail SL " + label + ": estimated step_pct=" + str(step_pct) + "% (no candles)")

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

                        _w_lev  = w.get("leverage", LEVERAGE)
                        pnl_lev = round(cur_gain * _w_lev, 1)
                        dir_e   = "\U0001f7e2" if direction == "LONG" else "\U0001f534"

                        # ── Trailing SL update (Supertrend-aware) ─────────────
                        # FIX: previously activated only when gain_pct >= step_pct.
                        # step_pct can be up to 10% (ATR*2 on volatile alts) — meant
                        # trail only started after a 10% gain, then locked only 45% of it.
                        # Now: activate at 3% gain minimum. step_pct is kept ONLY for
                        # the Bitget callback rate suggestion — separate from activation.
                        TRAIL_ACTIVATE_PCT = 3.0  # start tracking at 3% gain
                        if gain_pct >= TRAIL_ACTIVATE_PCT:
                            # ── ZLEMA trend check — don't lock aggressively if trend intact
                            zlema_still_bullish = True
                            try:
                                if _df_trail is not None and len(_df_trail) >= 80:
                                    from signal_engine import compute_zlema
                                    _zl = compute_zlema(_df_trail, length=70, band_multiplier=1.2)
                                    if direction == "LONG"  and _zl["bearish"]: zlema_still_bullish = False
                                    if direction == "SHORT" and _zl["bullish"]: zlema_still_bullish = False
                            except Exception:
                                pass

                            # ── Tiered lock — prevents early closure on big trends ──
                            # Small gain: lock 60% (protect early profits)
                            # Mid gain:   lock 40% (give room to breathe)
                            # Large gain: lock 25% (let trend run, only protect against crash)
                            # ZLEMA still in our favour: reduce lock by extra 10% (more room)
                            # Lock percentages — lower early lock to avoid noise-triggered
                            # closures on volatile alts. Trail must be meaningful.
                            if gain_pct < 8:
                                lock_pct = gain_pct * 0.45   # FIX: was 0.60 — too aggressive
                            elif gain_pct < 20:
                                lock_pct = gain_pct * 0.40
                            else:
                                lock_pct = gain_pct * 0.28

                            if zlema_still_bullish and gain_pct >= 12:
                                lock_pct *= 0.80  # trend intact — give 20% more room (was 15%)
                                trend_note = " (ZLEMA trend intact — relaxed lock)"
                            else:
                                trend_note = ""

                            # Try Supertrend as SL when candles available
                            use_st  = False
                            st_line = 0.0
                            try:
                                if _df_trail is not None and len(_df_trail) >= 15:
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

                            if use_st:
                                new_sl = round(st_line, 6)
                                sl_src = "Supertrend line (adaptive)" + trend_note
                            else:
                                new_sl = round(entry * (1 + lock_pct/100), 4) if direction == "LONG" \
                                         else round(entry * (1 - lock_pct/100), 4)
                                sl_src = ("Tiered lock " + "{:.1f}".format(lock_pct) +
                                          "% of " + "{:.1f}".format(gain_pct) + "% gain" + trend_note)

                            sl_improved = (
                                trail_sl is None or
                                (direction == "LONG"  and new_sl > trail_sl * 1.005) or
                                (direction == "SHORT" and new_sl < trail_sl * 0.995)
                            )
                            _w_lev = w.get("leverage", LEVERAGE)
                            if sl_improved and now - w.get("trail_last_alert", 0) >= trail_cd:
                                watched_trades[symbol]["trail_sl"]         = new_sl
                                watched_trades[symbol]["trail_last_alert"] = now
                                pnl_lev_w = round(cur_gain * _w_lev, 1)
                                msg  = "\U0001f4cd *Trail SL Update: " + label + "*\n\n"
                                msg += dir_e + " *" + direction + "* from `$" + "{:.4f}".format(entry) + "`\n"
                                msg += "Current: `$" + "{:.4f}".format(cur_price) + "` | PnL: *" + "{:+.1f}".format(pnl_lev_w) + "%* at " + str(_w_lev) + "x\n\n"
                                msg += "\U0001f6e1 *Move SL to: `$" + "{:.4f}".format(new_sl) + "`*\n"
                                msg += "_" + sl_src + "_\n\n"
                                msg += "\U0001f4cc *Bitget Trailing Stop settings:*\n"
                                msg += "  Callback Rate: *" + "{:.1f}".format(step_pct) + "%*\n"
                                msg += "  Activation Price: `$" + "{:.4f}".format(cur_price) + "`\n"
                                msg += "_Position → TP/SL → Trailing Stop_"
                                await send_msg(app, msg)
                                log.info("Trail SL: " + label + " SL→$" + str(new_sl))


                        # ── Profit target alerts ──────────────────────────────
                        tp1_pct = w.get("tp1_pct")
                        tp2_pct = w.get("tp2_pct")
                        if tp1_pct and not w.get("tp1_hit") and cur_gain >= tp1_pct:
                            watched_trades[symbol]["tp1_hit"] = True
                            callback = max(3.0, min(10.0, round(step_pct, 1)))
                            tp1_price = round(entry * (1 + tp1_pct/100), 4) if direction == "LONG"                                         else round(entry * (1 - tp1_pct/100), 4)
                            msg  = "\U0001f3af *TP1 Hit: " + label + "*\n\n"
                            msg += dir_e + " *+" + "{:.1f}".format(cur_gain) + "%* (" + "{:+.1f}".format(pnl_lev) + "% at " + str(_w_lev) + "x)\n"
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
                            msg += dir_e + " *+" + "{:.1f}".format(cur_gain) + "%* (" + "{:+.1f}".format(pnl_lev) + "% at " + str(_w_lev) + "x)\n"
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
                            msg += "Current PnL: *" + "{:+.1f}".format(pnl_lev) + "%* at " + str(_w_lev) + "x\n\n"
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
                    if df_scalp is None or len(df_scalp) < 20:
                        continue
                    price    = float(df_scalp["close"].iloc[-1])
                    sigs, count = await _check_early_warning(df_scalp, direction)
                    if count < 2:
                        continue
                    pnl_str   = _watch_pnl_str(entry, price, direction, lev=_w_lev)
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
                # Level 0 — Opposing BOS invalidation on 1H (structural breach)
                # Level 1 — Early warning on 1H (get ready, tighten SL)
                # Level 2 — Confirmed reversal on 4H (close the trade)
                # Separate cooldowns so Level 1 doesn't block Level 2

                # ── Confidence level thresholds ───────────────────────────────
                conf          = w.get("confidence", "medium")
                l2_min        = WATCH_CONF_HIGH_L2_MIN if conf == "high" else WATCH_CONF_MED_L2_MIN
                suppress_l1   = (conf == "high")  # HIGH: never fire 1H noise warnings

                # ── Level 0: Opposing BOS check (swing only) ──────────────────
                # Detects a structural swing breach on 1H that invalidates the
                # trade thesis regardless of how many reversal signals have fired.
                # Fires independently — doesn't wait for cooldown or signal count.
                #
                # Rules:
                #   LONG: a 1H candle closes below the most recent 1H swing low
                #         AND that breach is > 0.5 ATR (filters wick noise)
                #   SHORT: mirror — closes above most recent 1H swing high
                #
                # On fire: kills active_signals entry, marks paper trade CLOSED,
                #          sends a distinct alert (not a reversal warning).
                # Does NOT unwatch — user stays monitored until they /unwatch.
                # Only runs if no BOS alert was sent in the last 4H (one per setup).
                _bos_cooldown_ok = now - w.get("last_bos_alert", 0) >= 14400
                if _bos_cooldown_ok and not w.get("tp1_hit"):
                    try:
                        _df_bos = (fetch_ohlcv_mexc(symbol, "1h", 60)
                                   if _use_mexc
                                   else fetch_ohlcv(symbol, "1h", 60))
                        if _df_bos is not None and len(_df_bos) >= 20:
                            _bos_close  = float(_df_bos["close"].iloc[-1])
                            _bos_high   = _df_bos["high"]
                            _bos_low    = _df_bos["low"]

                            # ATR for noise filter — breach must exceed 0.5 ATR
                            _bos_atr_s  = ta.atr(_bos_high, _bos_low,
                                                  _df_bos["close"], length=14)
                            _bos_atr    = (float(_bos_atr_s.dropna().iloc[-1])
                                           if _bos_atr_s is not None
                                           and len(_bos_atr_s.dropna()) > 0
                                           else _bos_close * 0.015)

                            # Swing pivots: look back 5–20 bars, exclude last 2
                            # (current forming candle + last closed — too recent)
                            _pivot_window = _df_bos.iloc[-22:-2]
                            _swing_low  = float(_pivot_window["low"].min())
                            _swing_high = float(_pivot_window["high"].max())

                            _bos_fired  = False
                            _bos_msg    = ""

                            if (direction == "LONG"
                                    and _bos_close < _swing_low
                                    and (_swing_low - _bos_close) > _bos_atr * 0.5):
                                _bos_fired = True
                                _breach    = round(_swing_low - _bos_close, 6)
                                _bos_msg   = (
                                    f"🚨 *Structure Broken: {label}*\n\n"
                                    f"1H opposing BOS confirmed — trade thesis invalidated.\n\n"
                                    f"Swing low ${_swing_low:.4f} breached by ${_breach:.6f} "
                                    f"({round(_breach / _swing_low * 100, 2)}%)\n"
                                    f"Current price: `${_bos_close:.4f}`\n\n"
                                    f"📌 *The buyers that justified this LONG did not step in.*\n"
                                    f"Consider closing manually — SL may be next.\n\n"
                                    f"_/unwatch {label} to stop monitoring_"
                                )

                            elif (direction == "SHORT"
                                    and _bos_close > _swing_high
                                    and (_bos_close - _swing_high) > _bos_atr * 0.5):
                                _bos_fired = True
                                _breach    = round(_bos_close - _swing_high, 6)
                                _bos_msg   = (
                                    f"🚨 *Structure Broken: {label}*\n\n"
                                    f"1H opposing BOS confirmed — trade thesis invalidated.\n\n"
                                    f"Swing high ${_swing_high:.4f} breached by ${_breach:.6f} "
                                    f"({round(_breach / _swing_high * 100, 2)}%)\n"
                                    f"Current price: `${_bos_close:.4f}`\n\n"
                                    f"📌 *The sellers that justified this SHORT did not hold.*\n"
                                    f"Consider closing manually — SL may be next.\n\n"
                                    f"_/unwatch {label} to stop monitoring_"
                                )

                            if _bos_fired:
                                # Invalidate the active signal and paper trade
                                update_trade_state(symbol, {"status": "BOS_INVALIDATED"})
                                if symbol in active_signals:
                                    del active_signals[symbol]
                                # Mark paper trade closed at current price
                                _pt_list = load_json(PAPER_FILE, [])
                                for _pt in _pt_list:
                                    if (_pt.get("symbol") == symbol
                                            and _pt.get("status") == "OPEN"):
                                        _pt["status"]       = "CLOSED"
                                        _pt["close_price"]  = _bos_close
                                        _pt["close_reason"] = "BOS invalidation"
                                        _pt["close_time"]   = now
                                save_json(PAPER_FILE, _pt_list)
                                watched_trades[symbol]["last_bos_alert"] = now
                                log.info(f"BOS invalidation: {label} {direction} — swing breached")
                                try:
                                    await send_msg(app, _bos_msg)
                                except Exception:
                                    await app.bot.send_message(
                                        chat_id=TELEGRAM_CHAT,
                                        text=_bos_msg, parse_mode=None
                                    )
                    except Exception as _bos_e:
                        log.warning(f"BOS check error {label}: {_bos_e}")

                # ── Level 2 first: 4H confirmed reversal ─────────────────────
                l2_cooldown_ok = now - w.get("last_alert", 0) >= WATCH_COOLDOWN
                df_4h = None  # must be initialised here — referenced below regardless of cooldown state
                if l2_cooldown_ok:
                    df_4h = fetch_ohlcv_mexc(symbol, "4h", 100) if _use_mexc else fetch_ohlcv(symbol, "4h", 100)
                    if df_4h is None or len(df_4h) == 0:
                        # OHLCV unavailable — skip reversal check, trail SL already ran above
                        log.info("Reversal check skipped for " + label + " — no 4H data")
                        l2_cooldown_ok = False  # prevent entering reversal block
                    else:
                        price = float(df_4h["close"].iloc[-1])
                if l2_cooldown_ok and df_4h is not None and len(df_4h) > 0:
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
                        pnl_str   = _watch_pnl_str(entry, price, direction, lev=_w_lev)
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
                    df_1h   = fetch_ohlcv_mexc(symbol, "1h", 100) if _use_mexc else fetch_ohlcv(symbol, "1h", 100)
                    if df_1h is None or len(df_1h) == 0:
                        continue
                    price   = float(df_1h["close"].iloc[-1])
                    sigs_1h, count_1h = await _check_early_warning(df_1h, direction)
                    l1_min  = 1 if conf == "low" else 2  # LOW: fires at 1 signal (max sensitivity); MEDIUM: 2
                    if count_1h >= l1_min:
                        pnl_str   = _watch_pnl_str(entry, price, direction, lev=_w_lev)
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


async def supervised_task(coro_fn, app, name: str):
    """Wraps a background loop with crash detection and auto-restart."""
    RESTART_DELAY = 30
    while True:
        try:
            log.info(f"Task started: {name}")
            await coro_fn(app)
            log.warning(f"Task {name} returned normally (unexpected)")
        except asyncio.CancelledError:
            log.info(f"Task {name} cancelled")
            return
        except Exception as e:
            log.error(f"Task {name} crashed: {e}", exc_info=True)
            try:
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT,
                    text=f"⚠️ *Task crashed: {name}*\n`{str(e)[:200]}`\n_Restarting in {RESTART_DELAY}s..._",
                    parse_mode="Markdown",
                )
            except Exception:
                pass
        log.info(f"Task {name} restarting in {RESTART_DELAY}s...")
        await asyncio.sleep(RESTART_DELAY)

async def auto_weekly_report(app):
    log.info("Weekly report loop started")
    while True:
        now = datetime.utcnow()
        days_ahead = (6 - now.weekday()) % 7          # 0 if today is Sunday
        target = now.replace(hour=19, minute=0, second=0, microsecond=0)
        target = target + timedelta(days=days_ahead)
        if target <= now:                              # FIX: Sunday after 19:00 → next week
            target += timedelta(days=7)
        wait_seconds = (target - now).total_seconds()
        log.info(f"Weekly report in {wait_seconds/3600:.1f}h (next Sunday 19:00 UTC)")
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

    # ── Restore watched trades from disk ─────────────────────────────────────
    # watched_trades.json is written on every /watch and /unwatch.
    # Without this, every bot restart silently clears all active watches.
    global watched_trades
    _saved_watches = load_json(WATCH_FILE, {})
    for _sym, _w in _saved_watches.items():
        watched_trades[_sym] = {
            **_w,
            # Reset runtime-only state — these don't survive a restart meaningfully
            "last_alert":       0,
            "last_early_alert": 0,
            "trail_sl":         None,
            "trail_peak":       _w.get("entry"),
            "trail_last_alert": 0,
            "tp1_hit":          False,
            "tp2_hit":          False,
            "be_alerted":       False,
            "hold_alert_sent":  False,
        }
    if watched_trades:
        log.info(f"Restored {len(watched_trades)} watched trade(s) from {WATCH_FILE}: "
                 + ", ".join(v["label"] for v in watched_trades.values()))

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
    app.add_handler(CommandHandler("refine",       cmd_refine))
    app.add_handler(CommandHandler("limit",        cmd_limit))
    app.add_handler(CommandHandler("paper_refine", cmd_paper_refine))
    app.add_handler(CommandHandler("paper_stats",  cmd_paper_stats))
    app.add_handler(CallbackQueryHandler(callback_signal_detail, pattern="^sig_"))
    app.add_handler(CommandHandler("scalp",        cmd_scalp))
    app.add_handler(CommandHandler("swing",        cmd_swing))
    app.add_handler(CommandHandler("counterscan",  cmd_counterscan))
    app.add_handler(CommandHandler("blacklist",  cmd_blacklist))
    app.add_handler(CommandHandler("whale",      cmd_whale))
    app.add_handler(CommandHandler("alert",      cmd_alert))
    app.add_handler(CommandHandler("backtest", lambda u, c: cmd_backtest(u, c, exchange, COIN_LABELS)))
    app.add_handler(CommandHandler("risk",     lambda u, c: cmd_risk(u, c, active_signals, load_json(HISTORY_FILE, []))))
    app.add_handler(CommandHandler("alpha",     cmd_alpha))
    app.add_handler(CommandHandler("coupon",    cmd_coupon))
    app.add_handler(CommandHandler("btcstatus",  cmd_btcstatus))
    app.add_handler(CommandHandler("domstatus",  cmd_domstatus))
    app.add_handler(CommandHandler("flip",     cmd_flip))
    app.add_handler(CommandHandler("flipping", cmd_flipping))
    app.add_handler(CommandHandler("watch",    cmd_watch))
    app.add_handler(CommandHandler("unwatch",  cmd_unwatch))
    app.add_handler(CommandHandler("watching",   cmd_watching))
    app.add_handler(CommandHandler("setprice",  cmd_setprice))
    app.add_handler(CommandHandler("watchstatus", cmd_watchstatus))

    async def error_handler(update, context):
        log.error("Telegram error: " + str(context.error))
    app.add_error_handler(error_handler)
    
    async def post_init(application):
        # ── Startup reconciliation ────────────────────────────────────────────
        # Runs before background tasks start. Prevents orphaned signals,
        # duplicate watches, and incorrect TP state after Railway restarts.
        try:
            _reconcile_issues = []

            # 1. Reload persisted paper trades — mark any OPEN trade whose
            #    symbol is missing from active_signals as ORPHANED so the
            #    monitor loop doesn't try to update it via a None signal.
            _pt_list = load_json(PAPER_FILE, [])
            _orphaned = 0
            for _pt in _pt_list:
                if (_pt.get("status") == "OPEN"
                        and _pt.get("symbol") not in active_signals):
                    _pt["status"]        = "ORPHANED"
                    _pt["close_reason"]  = "Bot restarted — signal lost"
                    _orphaned += 1
            if _orphaned:
                save_json(PAPER_FILE, _pt_list)
                _reconcile_issues.append(f"{_orphaned} orphaned paper trade(s) closed")

            # 2. Deduplicate watched_trades — keep only the latest entry per symbol
            #    (duplicates accumulate if bot restarts mid-alert)
            _seen_wt = {}
            _dupes   = []
            for _wt_sym, _wt_data in list(watched_trades.items()):
                # watched_trades is keyed by symbol already — duplicates shouldn't
                # exist by design, but if the same symbol appears twice (edge case
                # from concurrent writes) keep the newer entry.
                _wt_time = _wt_data.get("start_time", 0)
                if _wt_sym in _seen_wt:
                    _prev_time = watched_trades[_seen_wt[_wt_sym]].get("start_time", 0)
                    if _wt_time >= _prev_time:
                        _dupes.append(_seen_wt[_wt_sym])
                        _seen_wt[_wt_sym] = _wt_sym
                    else:
                        _dupes.append(_wt_sym)
                else:
                    _seen_wt[_wt_sym] = _wt_sym
            for _d in _dupes:
                watched_trades.pop(_d, None)
                _reconcile_issues.append(f"Duplicate watch removed: {_d}")

            # 3. Validate TP state on surviving active signals — if price is
            #    already past TP1 but tp1_hit is False, correct it silently.
            for _sym, _sig in list(active_signals.items()):
                try:
                    _px = exchange.fetch_ticker(_sym)["last"]
                    _dir = _sig.get("direction", "LONG")
                    _tp1 = _sig.get("tp1", 0)
                    if not _sig.get("tp1_hit") and _tp1 > 0:
                        if (_dir == "LONG" and _px >= _tp1) or (_dir == "SHORT" and _px <= _tp1):
                            _tbuf_rc = (TRAILING_BUFFER_SCALP
                                        if _sig.get("trade_type") == "scalp"
                                        else TRAILING_BUFFER)
                            _trail = (round(_px * (1 - _tbuf_rc), 6) if _dir == "LONG"
                                      else round(_px * (1 + _tbuf_rc), 6))
                            update_trade_state(_sym, {
                                "tp1_hit":          True,
                                "trailing_stop":    _trail,
                                "trailing_extreme": _px,
                            })
                            _reconcile_issues.append(f"TP1 state corrected: {_sym}")
                except Exception:
                    pass  # price fetch can fail — non-fatal

            if _reconcile_issues:
                log.info("Startup reconciliation: " + " | ".join(_reconcile_issues))
            else:
                log.info("Startup reconciliation: clean — no issues found")

        except Exception as _rec_e:
            log.error(f"Startup reconciliation error: {_rec_e}")

        # ── Background tasks ──────────────────────────────────────────────────
        _tasks = [
            ("auto_scan",              auto_scan),
            ("auto_news",              auto_news),
            ("auto_monitor",           auto_monitor),
            ("auto_coin_refresh",      auto_coin_refresh),
            ("auto_weekly_report",     auto_weekly_report),
            ("auto_price_alerts",      auto_price_alerts),
            ("auto_watch",             auto_watch),
            ("auto_btc_monitor",       auto_btc_monitor),
            ("auto_coupon_monitor",    auto_coupon_monitor),
            ("auto_flip",              auto_flip),
            ("auto_dominance_monitor", auto_dominance_monitor),
            ("auto_gainers_scan",      auto_gainers_scan),
            ("auto_alpha_refresh",     auto_alpha_refresh),
            ("auto_alpha_scan",        auto_alpha_scan),
            ("auto_counter_scan",      auto_counter_scan),
        ]
        for _name, _fn in _tasks:
            asyncio.create_task(supervised_task(_fn, application, _name))
        log.info(f"Started {len(_tasks)} supervised background tasks")

    app.post_init = post_init
    log.info("Phyrobot starting — top " + str(TOP_COINS) + " coins | 1H+4H")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()