"""
Phyrobot — Phase 1: Smarter Scoring & Signal Filtering
────────────────────────────────────────────────────────
Key upgrades:
  1. Weighted indicator scoring (not all indicators equal)
  2. Regime detection (trending vs ranging — different rules apply)
  3. Confluence gate — signal only fires if 3+ independent groups agree
  4. Volatility filter — skip coins in erratic/choppy conditions
  5. Trend alignment filter — higher TF must agree before signal fires
  6. Signal quality score (0-100) replaces raw integer score
  7. False positive suppression — recent candle quality check
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
import logging

log = logging.getLogger(__name__)


# ─── PIPELINE MODE FLAGS (controlled by bot.py via BTC dominance) ────────────
# Full mode (dominance < 52%): Supertrend gate active, strict grading
# Simplified mode (dominance >= 55%): Supertrend disabled, Grade B allowed
_SUPERTREND_GATE_ACTIVE = False  # default: simplified (safer for BTC-season)

def set_pipeline_mode(full_mode: bool):
    """Called by bot.py when BTC dominance crosses thresholds."""
    global _SUPERTREND_GATE_ACTIVE
    _SUPERTREND_GATE_ACTIVE = full_mode
    log.info("Pipeline mode: " + ("FULL (ST gate active)" if full_mode else "SIMPLIFIED (ST disabled)"))

# ─── INDICATOR WEIGHTS ────────────────────────────────────────────────────────
# Each indicator group has a max contribution to the final score.
# Groups are INDEPENDENT — confluence requires agreement across groups, not just stacking.
WEIGHTS = {
    "trend":      30,   # EMA alignment, MA cross, price vs MA
    "momentum":   25,   # RSI, MACD, Rate of Change
    "structure":  20,   # Support/resistance, breakout, swing structure
    "volume":     15,   # Volume confirmation, OBV, volume spikes
    "oscillator": 10,   # StochRSI, CCI, Williams %R (lower weight — lagging)
}

# Minimum groups that must agree for a signal to fire (confluence gate)
CONFLUENCE_MIN = 4

# Minimum quality score to emit a signal
QUALITY_THRESHOLD_HIGH   = 65
QUALITY_THRESHOLD_MEDIUM = 50

# ATR multiplier thresholds
VOLATILITY_MAX_ATR_PCT = 8.0   # skip coin if ATR% > 8% (too erratic)
VOLATILITY_MIN_ATR_PCT = 0.3   # skip coin if ATR% < 0.3% (dead/no movement)


# ─── REGIME DETECTION ────────────────────────────────────────────────────────
def detect_regime(df: pd.DataFrame) -> dict:
    """
    Classify the current market regime.
    Returns: {"regime": "TRENDING_UP" | "TRENDING_DOWN" | "RANGING", "adx": float, "strength": float}

    ADX > 25 = trending, ADX < 20 = ranging.
    We use ADX + EMA slope to confirm direction.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    try:
        adx_df  = ta.adx(high, low, close, length=14)
        adx_val = float(adx_df.iloc[-1, 0]) if adx_df is not None and not adx_df.empty else 20.0
        dmp     = float(adx_df.iloc[-1, 1]) if adx_df is not None and not adx_df.empty else 0
        dmn     = float(adx_df.iloc[-1, 2]) if adx_df is not None and not adx_df.empty else 0
    except Exception:
        adx_val, dmp, dmn = 20.0, 0, 0

    # EMA slope over last 5 candles
    ema50 = ta.ema(close, length=50)
    slope = 0.0
    if ema50 is not None and len(ema50.dropna()) >= 6:
        vals  = ema50.dropna().values
        slope = (vals[-1] - vals[-5]) / vals[-5] * 100  # % change over 5 candles

    if adx_val >= 25:
        if dmp > dmn and slope > 0:
            regime = "TRENDING_UP"
        elif dmn > dmp and slope < 0:
            regime = "TRENDING_DOWN"
        else:
            regime = "TRENDING_UP" if dmp > dmn else "TRENDING_DOWN"
    elif adx_val >= 20 and dmn > dmp and slope < 0:
        # Mild downtrend (ADX 20-25) — classify as TRENDING_DOWN for SHORT signals
        # Many bear market coins have ADX 20-24, not 25+, but are clearly downtrending
        regime = "TRENDING_DOWN"
    else:
        regime = "RANGING"

    return {"regime": regime, "adx": round(adx_val, 1), "slope": round(slope, 3)}


# ─── VOLATILITY FILTER ───────────────────────────────────────────────────────
def volatility_check(df: pd.DataFrame) -> dict:
    """
    Returns {"pass": bool, "atr_pct": float, "reason": str}
    Coins with extreme volatility or no movement are skipped.
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    price = float(close.iloc[-1])

    atr    = ta.atr(high, low, close, length=14)
    atr_v  = float(atr.iloc[-1]) if atr is not None else 0
    atr_pct = (atr_v / price * 100) if price > 0 else 0

    if atr_pct > VOLATILITY_MAX_ATR_PCT:
        return {"pass": False, "atr_pct": round(atr_pct, 2), "reason": f"ATR too high ({atr_pct:.1f}%) — erratic"}
    if atr_pct < VOLATILITY_MIN_ATR_PCT:
        return {"pass": False, "atr_pct": round(atr_pct, 2), "reason": f"ATR too low ({atr_pct:.2f}%) — no movement"}

    return {"pass": True, "atr_pct": round(atr_pct, 2), "reason": "OK"}


# ─── GROUP SCORERS ────────────────────────────────────────────────────────────

def score_trend_group(df: pd.DataFrame, regime: dict) -> tuple[float, list[str]]:
    """
    Score: EMA alignment, MA50/200, price vs key MAs.
    Returns (raw_score -1.0 to +1.0, signals[])
    Higher weight in trending regime, lower in ranging.
    """
    close  = df["close"]
    price  = float(close.iloc[-1])
    points = 0
    total  = 0
    sigs   = []

    # EMA 9/21 — most responsive trend signal
    ema9  = ta.ema(close, length=9)
    ema21 = ta.ema(close, length=21)
    if ema9 is not None and ema21 is not None and len(ema9.dropna()) >= 3:
        e9  = ema9.dropna().values
        e21 = ema21.dropna().values
        if len(e9) >= 2 and len(e21) >= 2:
            cross_up   = e9[-1] > e21[-1] and e9[-2] <= e21[-2]
            cross_down = e9[-1] < e21[-1] and e9[-2] >= e21[-2]
            above      = e9[-1] > e21[-1]
            total += 3
            if cross_up:
                points += 3; sigs.append("EMA 9/21 bullish cross ✦")
            elif cross_down:
                points -= 3; sigs.append("EMA 9/21 bearish cross ✦")
            elif above:
                points += 1; sigs.append("EMA 9 > EMA 21")
            else:
                points -= 1; sigs.append("EMA 9 < EMA 21")

    # EMA 50 — medium trend
    ema50 = ta.ema(close, length=50)
    if ema50 is not None and len(ema50.dropna()) >= 2:
        e50 = float(ema50.dropna().iloc[-1])
        total += 2
        if price > e50:
            points += 2; sigs.append(f"Price above EMA50 (${e50:.4f})")
        else:
            points -= 2; sigs.append(f"Price below EMA50 (${e50:.4f})")

    # MA 50/200 golden/death cross
    ma50  = ta.sma(close, length=50)
    ma200 = ta.sma(close, length=200)
    if ma50 is not None and ma200 is not None:
        m50  = ma50.dropna()
        m200 = ma200.dropna()
        if len(m50) > 0 and len(m200) > 0:
            m50_v  = float(m50.iloc[-1])
            m200_v = float(m200.iloc[-1])
            total += 2
            if m50_v > m200_v:
                points += 2; sigs.append("Golden cross (MA50 > MA200)")
            else:
                points -= 2; sigs.append("Death cross (MA50 < MA200)")

    # In a ranging regime, trend signals are less reliable — reduce weight
    if regime["regime"] == "RANGING":
        points = int(points * 0.5)

    raw = points / total if total > 0 else 0
    return max(-1.0, min(1.0, raw)), sigs


def score_momentum_group(df: pd.DataFrame, regime: dict) -> tuple[float, list[str]]:
    """
    Score: RSI, MACD, RSI divergence, Rate of Change.
    In ranging regime, oscillators are more reliable — boost weight.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    points = 0
    total  = 0
    sigs   = []

    # RSI — dynamic thresholds based on regime
    rsi = ta.rsi(close, length=14)
    if rsi is not None and len(rsi.dropna()) >= 3:
        rsi_v    = float(rsi.dropna().iloc[-1])
        rsi_prev = float(rsi.dropna().iloc[-2])
        total   += 3

        if regime["regime"] == "RANGING":
            # In ranging: use tighter extremes (40/60)
            if rsi_v < 38:
                points += 3; sigs.append(f"RSI oversold in range ({rsi_v:.1f})")
            elif rsi_v > 62:
                points -= 3; sigs.append(f"RSI overbought in range ({rsi_v:.1f})")
            elif rsi_v < 50 and rsi_v > rsi_prev:
                points += 1; sigs.append(f"RSI recovering ({rsi_v:.1f})")
            elif rsi_v > 50 and rsi_v < rsi_prev:
                points -= 1; sigs.append(f"RSI declining ({rsi_v:.1f})")
        else:
            # In trend: look for momentum continuation
            if rsi_v < 35:
                points += 3; sigs.append(f"RSI oversold ({rsi_v:.1f})")
            elif rsi_v > 65:
                points -= 3; sigs.append(f"RSI overbought ({rsi_v:.1f})")
            elif 45 < rsi_v < 65 and regime["regime"] == "TRENDING_UP":
                points += 2; sigs.append(f"RSI bullish momentum zone ({rsi_v:.1f})")
            elif 35 < rsi_v < 55 and regime["regime"] == "TRENDING_DOWN":
                points -= 2; sigs.append(f"RSI bearish momentum zone ({rsi_v:.1f})")

        # RSI divergence (last 20 candles)
        prices_w = close.tail(20).values
        rsis_w   = rsi.tail(20).values
        if len(prices_w) >= 20 and len(rsis_w) >= 20:
            if prices_w[-1] < min(prices_w[:-1]) and rsis_w[-1] > min(rsis_w[:-1]):
                points += 3; total += 3; sigs.append("Bullish RSI divergence ✦")
            elif prices_w[-1] > max(prices_w[:-1]) and rsis_w[-1] < max(rsis_w[:-1]):
                points -= 3; total += 3; sigs.append("Bearish RSI divergence ✦")

    # MACD — only crossovers score high, histogram alone scores low
    macd_df = ta.macd(close, fast=12, slow=26, signal=9)
    if macd_df is not None and not macd_df.empty and len(macd_df) >= 3:
        macd_line = float(macd_df.iloc[-1, 0])
        sig_line  = float(macd_df.iloc[-1, 2])
        hist_now  = float(macd_df.iloc[-1, 1])
        hist_prev = float(macd_df.iloc[-2, 1])
        macd_prev = float(macd_df.iloc[-2, 0])
        sig_prev  = float(macd_df.iloc[-2, 2])
        total    += 3

        crossover_bull = macd_line > sig_line and macd_prev <= sig_prev
        crossover_bear = macd_line < sig_line and macd_prev >= sig_prev

        if crossover_bull:
            points += 3; sigs.append("MACD bullish crossover ✦")
        elif crossover_bear:
            points -= 3; sigs.append("MACD bearish crossover ✦")
        elif hist_now > 0 and hist_now > hist_prev:
            points += 1; sigs.append("MACD histogram rising")
        elif hist_now < 0 and hist_now < hist_prev:
            points -= 1; sigs.append("MACD histogram falling")

    # Rate of Change
    roc = ta.roc(close, length=10)
    if roc is not None and len(roc.dropna()) >= 1:
        roc_v = float(roc.dropna().iloc[-1])
        total += 1
        if roc_v > 2.0:
            points += 1; sigs.append(f"Positive momentum ROC ({roc_v:.1f}%)")
        elif roc_v < -2.0:
            points -= 1; sigs.append(f"Negative momentum ROC ({roc_v:.1f}%)")

    raw = points / total if total > 0 else 0
    return max(-1.0, min(1.0, raw)), sigs


def score_structure_group(df: pd.DataFrame, regime: dict) -> tuple[float, list[str]]:
    """
    Score: Support/resistance proximity, breakout confirmation, swing structure.
    Most reliable in trending regimes.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    price  = float(close.iloc[-1])
    volume = df["volume"]
    points = 0
    total  = 0
    sigs   = []

    # Support / Resistance (20 candle lookback)
    support_20    = float(low.tail(20).min())
    resistance_20 = float(high.tail(20).max())
    total        += 2

    near_support    = abs(price - support_20) / price < 0.012
    near_resistance = abs(price - resistance_20) / price < 0.012

    if near_support:
        points += 2; sigs.append(f"Near support ${support_20:.4f}")
    elif near_resistance:
        points -= 2; sigs.append(f"Near resistance ${resistance_20:.4f}")

    # Breakout / Breakdown with volume confirmation
    vol_avg     = float(volume.tail(10).mean())
    vol_confirm = float(volume.iloc[-1]) > vol_avg * 1.3

    if price > resistance_20 and vol_confirm:
        points += 3; total += 3; sigs.append(f"Breakout above ${resistance_20:.4f} ✦")
    elif price < support_20 and vol_confirm:
        points -= 3; total += 3; sigs.append(f"Breakdown below ${support_20:.4f} ✦")

    # Swing structure (Higher Highs / Higher Lows or Lower Highs / Lower Lows)
    if len(high) >= 15:
        swing_highs, swing_lows = [], []
        for i in range(2, min(15, len(high) - 2)):
            if high.iloc[-i] > high.iloc[-i-1] and high.iloc[-i] > high.iloc[-i+1]:
                swing_highs.append(float(high.iloc[-i]))
            if low.iloc[-i] < low.iloc[-i-1] and low.iloc[-i] < low.iloc[-i+1]:
                swing_lows.append(float(low.iloc[-i]))

        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            total += 2
            hh = swing_highs[0] > swing_highs[1]
            hl = swing_lows[0] > swing_lows[1]
            lh = swing_highs[0] < swing_highs[1]
            ll = swing_lows[0] < swing_lows[1]
            if hh and hl:
                points += 2; sigs.append("Higher highs + higher lows ✦")
            elif lh and ll:
                points -= 2; sigs.append("Lower highs + lower lows ✦")

    # Candlestick patterns — weighted higher for reversal signals
    try:
        o = float(df["open"].iloc[-1])
        h = float(df["high"].iloc[-1])
        l = float(df["low"].iloc[-1])
        c = float(df["close"].iloc[-1])
        body        = abs(c - o)
        candle_rng  = h - l
        prev_o      = float(df["open"].iloc[-2])
        prev_c      = float(df["close"].iloc[-2])

        if candle_rng > 0 and body > 0:
            upper_wick = h - max(o, c)
            lower_wick = min(o, c) - l

            # Hammer — bullish
            if lower_wick > body * 2 and upper_wick < body * 0.5 and c > o:
                points += 2; total += 2; sigs.append("Hammer candle — bullish ✦")
            # Shooting star — bearish
            elif upper_wick > body * 2 and lower_wick < body * 0.5 and c < o:
                points -= 2; total += 2; sigs.append("Shooting star — bearish ✦")
            # Bullish engulfing
            elif prev_c < prev_o and c > o and o < prev_c and c > prev_o:
                points += 2; total += 2; sigs.append("Bullish engulfing candle ✦")
            # Bearish engulfing
            elif prev_c > prev_o and c < o and o > prev_c and c < prev_o:
                points -= 2; total += 2; sigs.append("Bearish engulfing candle ✦")
    except Exception:
        pass

    raw = points / total if total > 0 else 0
    return max(-1.0, min(1.0, raw)), sigs


def score_volume_group(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Score: Volume trend, OBV trend, volume exhaustion.
    Volume confirms or invalidates price action.
    """
    close  = df["close"]
    volume = df["volume"]
    price  = float(close.iloc[-1])
    points = 0
    total  = 0
    sigs   = []

    # Volume spike with direction
    vol_ma = float(volume.rolling(20).mean().iloc[-1])
    vol_now = float(volume.iloc[-1])
    if vol_now > vol_ma * 1.5:
        total += 2
        if price > float(close.iloc[-2]):
            points += 2; sigs.append(f"High volume bullish candle ({vol_now/vol_ma:.1f}x)")
        else:
            points -= 2; sigs.append(f"High volume bearish candle ({vol_now/vol_ma:.1f}x)")

    # OBV trend — 10 candle slope
    try:
        obv = ta.obv(close, volume)
        if obv is not None:
            obv_clean = obv.fillna(0)
            if len(obv_clean) >= 10:
                obv_vals  = obv_clean.values[-10:]
                obv_slope = np.polyfit(range(len(obv_vals)), obv_vals, 1)[0]
                total += 2
                if obv_slope > 0:
                    points += 2; sigs.append("OBV rising (accumulation)")
                else:
                    points -= 2; sigs.append("OBV falling (distribution)")
    except Exception:
        pass

    # Volume exhaustion — price moved but volume dried up (reversal warning)
    vol_series   = volume.tail(10).values
    price_series = close.tail(10).values
    recent_pump  = price_series[-1] > price_series[-5] * 1.04
    recent_dump  = price_series[-1] < price_series[-5] * 0.96
    vol_declining = vol_series[-1] < vol_series[-5:].mean() * 0.65

    if vol_declining:
        total += 2
        if recent_pump:
            points -= 2; sigs.append("Volume exhaustion on pump ⚠")
        elif recent_dump:
            points += 2; sigs.append("Volume exhaustion on dump ⚠")

    raw = points / total if total > 0 else 0
    return max(-1.0, min(1.0, raw)), sigs


def score_oscillator_group(df: pd.DataFrame) -> tuple[float, list[str]]:
    """
    Score: StochRSI, CCI, Williams %R.
    Lower weight — used as tie-breakers only.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    points = 0
    total  = 0
    sigs   = []

    # StochRSI
    stoch = ta.stochrsi(close, length=14)
    if stoch is not None and not stoch.empty:
        try:
            sk = float(stoch.iloc[-1, 0])
            sd = float(stoch.iloc[-1, 1])
            total += 1
            if sk < 20 and sd < 20:
                points += 1; sigs.append(f"StochRSI oversold (K:{sk:.0f})")
            elif sk > 80 and sd > 80:
                points -= 1; sigs.append(f"StochRSI overbought (K:{sk:.0f})")
        except Exception:
            pass

    # CCI
    cci = ta.cci(high, low, close, length=20)
    if cci is not None and len(cci.dropna()) >= 1:
        cci_v = float(cci.dropna().iloc[-1])
        total += 1
        if cci_v < -100:
            points += 1; sigs.append(f"CCI oversold ({cci_v:.0f})")
        elif cci_v > 100:
            points -= 1; sigs.append(f"CCI overbought ({cci_v:.0f})")

    # Williams %R
    willr = ta.willr(high, low, close, length=14)
    if willr is not None and len(willr.dropna()) >= 1:
        wr = float(willr.dropna().iloc[-1])
        total += 1
        if wr < -80:
            points += 1; sigs.append(f"Williams %R oversold ({wr:.0f})")
        elif wr > -20:
            points -= 1; sigs.append(f"Williams %R overbought ({wr:.0f})")

    raw = points / total if total > 0 else 0
    return max(-1.0, min(1.0, raw)), sigs


# ─── CANDLE QUALITY CHECK ────────────────────────────────────────────────────
def candle_quality_check(df: pd.DataFrame, direction: str) -> dict:
    """
    Check the last 3 candles for quality.
    A LONG signal should not fire after 3 big red candles in a row with no tail.
    Returns {"pass": bool, "reason": str}
    """
    try:
        recent = df.tail(4)
        closes = recent["close"].values
        opens  = recent["open"].values
        bodies = [abs(closes[i] - opens[i]) for i in range(len(closes))]

        if direction == "LONG":
            # Don't buy into 3 consecutive strong bearish candles
            consecutive_red = all(closes[i] < opens[i] for i in range(-3, 0))
            strong_bodies   = sum(bodies[-3:]) / 3 > bodies[0] * 0.8
            if consecutive_red and strong_bodies:
                return {"pass": False, "reason": "3 consecutive strong bearish candles — wait for reversal confirmation"}
        else:
            # Don't short into 3 consecutive strong bullish candles
            consecutive_green = all(closes[i] > opens[i] for i in range(-3, 0))
            strong_bodies     = sum(bodies[-3:]) / 3 > bodies[0] * 0.8
            if consecutive_green and strong_bodies:
                return {"pass": False, "reason": "3 consecutive strong bullish candles — wait for exhaustion confirmation"}
    except Exception:
        pass

    return {"pass": True, "reason": "OK"}


# ─── TREND ALIGNMENT FILTER ──────────────────────────────────────────────────
def trend_alignment_filter(score_htf: float, score_ltf: float) -> dict:
    """
    Higher timeframe must not strongly disagree with lower timeframe.
    If HTF is strongly bearish (< -0.3) but LTF is bullish → reject LONG.
    Returns {"pass": bool, "penalty": float, "reason": str}
    """
    conflict = (score_htf < -0.3 and score_ltf > 0.2) or \
               (score_htf > 0.3 and score_ltf < -0.2)

    if conflict:
        return {
            "pass": False,
            "multiplier": 0.5,
            "reason": f"HTF/LTF conflict (HTF:{score_htf:.2f} vs LTF:{score_ltf:.2f})"
        }

    # Alignment bonus
    aligned = (score_htf > 0.2 and score_ltf > 0.2) or \
              (score_htf < -0.2 and score_ltf < -0.2)
    bonus = 1.2 if aligned else 1.0

    return {"pass": True, "multiplier": bonus, "reason": "Timeframes aligned" if aligned else "Neutral alignment"}


# ─── MAIN SCORING ENGINE ─────────────────────────────────────────────────────
def compute_signal_quality(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    symbol: str = "",
    funding_rate: float = 0.0,
    rsi_val: float = 50.0,
) -> dict:
    """
    REWRITTEN — Three-group narrative pipeline.

    Instead of 5 groups voting independently and scores cancelling each other,
    three meta-groups must all agree. If any one opposes the direction, the
    signal is rejected. No mathematical compensation between groups.

    Three meta-groups:
      TREND     — Are we in the right trend? (EMA alignment, ADX, regime)
      MOMENTUM  — Is momentum confirming? (RSI, MACD, ROC, OBV slope)
      STRUCTURE — Is market structure valid? (volume, BB, candle quality)

    Each group returns: AGREE / NEUTRAL / OPPOSE for a given direction.
    Signal fires only when: AGREE >= 2 AND OPPOSE == 0

    Returns same dict structure as before for full bot.py compatibility.
    """
    result = {
        "quality_score":    0,
        "direction":        "NEUTRAL",
        "confidence":       "LOW",
        "regime":           {},
        "confluence_groups": 0,
        "group_scores":     {},
        "signals":          [],
        "filters":          {},
        "passed":           False,
        "reject_reason":    "",
    }

    # ── Fast fail: volatility ─────────────────────────────────────────────────
    vol_check = volatility_check(df_ltf)
    result["filters"]["volatility"] = vol_check
    if not vol_check["pass"]:
        result["reject_reason"] = vol_check["reason"]
        return result

    # ── Layer 1: Market Regime ────────────────────────────────────────────────
    regime      = detect_regime(df_htf)
    regime_name = regime.get("regime", "RANGING")
    result["regime"] = regime

    # ── Layer 2: Three meta-group votes ──────────────────────────────────────
    all_signals = []

    # --- GROUP A: TREND ---
    # Uses existing score_trend_group — but we read its direction vote only
    trend_s,  trend_sigs   = score_trend_group(df_ltf, regime)
    trend_htf, _           = score_trend_group(df_htf, regime)
    all_signals.extend(trend_sigs)
    # Combined LTF + HTF trend vote
    trend_combined = trend_s * 0.6 + trend_htf * 0.4
    if   trend_combined >  0.12: trend_vote = "LONG"
    elif trend_combined < -0.12: trend_vote = "SHORT"
    else:                        trend_vote = "NEUTRAL"

    # --- GROUP B: MOMENTUM ---
    mom_s,   mom_sigs   = score_momentum_group(df_ltf, regime)
    osc_s,   osc_sigs   = score_oscillator_group(df_ltf)
    all_signals.extend(mom_sigs + osc_sigs)
    mom_combined = mom_s * 0.6 + osc_s * 0.4
    if   mom_combined >  0.12: mom_vote = "LONG"
    elif mom_combined < -0.12: mom_vote = "SHORT"
    else:                      mom_vote = "NEUTRAL"

    # --- GROUP C: STRUCTURE ---
    struct_s, struct_sigs = score_structure_group(df_ltf, regime)
    vol_s,    vol_sigs    = score_volume_group(df_ltf)
    all_signals.extend(struct_sigs + vol_sigs)
    struct_combined = struct_s * 0.5 + vol_s * 0.5
    if   struct_combined >  0.10: struct_vote = "LONG"
    elif struct_combined < -0.10: struct_vote = "SHORT"
    else:                         struct_vote = "NEUTRAL"

    votes = [trend_vote, mom_vote, struct_vote]
    result["group_scores"] = {
        "trend":     trend_combined,
        "momentum":  mom_combined,
        "structure": struct_combined,
        # Keep legacy keys for compatibility
        "volume":    vol_s,
        "oscillator": osc_s,
    }
    result["signals"] = all_signals

    # ── Layer 3: Narrative check — all groups must tell the same story ────────
    long_agrees  = votes.count("LONG")
    short_agrees = votes.count("SHORT")
    long_opposes = votes.count("SHORT")   # groups opposing LONG
    short_opposes= votes.count("LONG")    # groups opposing SHORT

    # Determine raw direction from vote majority
    if   long_agrees  >= 2: raw_dir = "LONG"
    elif short_agrees >= 2: raw_dir = "SHORT"
    else:
        result["reject_reason"] = f"No narrative consensus — votes: T={trend_vote} M={mom_vote} S={struct_vote}"
        return result

    # Hard rule: zero opposition allowed
    if raw_dir == "LONG"  and long_opposes  > 0:
        result["reject_reason"] = f"Narrative conflict: {long_opposes} group(s) oppose LONG — {votes}"
        return result
    if raw_dir == "SHORT" and short_opposes > 0:
        result["reject_reason"] = f"Narrative conflict: {short_opposes} group(s) oppose SHORT — {votes}"
        return result

    # ── Regime gate ────────────────────────────────────────────────────────────
    if raw_dir == "LONG"  and regime_name == "TRENDING_DOWN":
        result["reject_reason"] = "LONG blocked — regime TRENDING_DOWN"
        return result
    if raw_dir == "SHORT" and regime_name == "TRENDING_UP":
        result["reject_reason"] = "SHORT blocked — regime TRENDING_UP"
        return result
    if regime_name == "RANGING":
        result["reject_reason"] = "Blocked — RANGING regime, no swing trades"
        return result

    # ── Exhaustion / blow-off filter ──────────────────────────────────────────
    exhaustion = check_exhaustion(df_ltf, raw_dir, funding_rate)
    result["exhaustion"] = exhaustion
    if raw_dir == "LONG" and exhaustion["block_long"]:
        result["reject_reason"] = "LONG blocked — blow-off top: " + exhaustion["reason"]
        result["direction"] = "NEUTRAL"
        if exhaustion["short_opp"]:
            result["short_opportunity"] = True
            result["short_reason"]      = exhaustion["reason"]
        return result

    # ── Candle quality ────────────────────────────────────────────────────────
    cq = candle_quality_check(df_ltf, raw_dir)
    result["filters"]["candle_quality"] = cq
    if not cq["pass"]:
        result["reject_reason"] = cq["reason"]
        return result

    # ── Quality score — based on agreement strength not count ─────────────────
    # Unanimous (3/3): high quality. Majority (2/3): standard.
    agree_count   = long_agrees if raw_dir == "LONG" else short_agrees
    # Strength of each group's vote (how far from neutral)
    trend_str     = abs(trend_combined)
    mom_str       = abs(mom_combined)
    struct_str    = abs(struct_combined)
    avg_strength  = (trend_str + mom_str + struct_str) / 3

    # Base: 0-80 from group strength
    base_score = min(80, avg_strength * 200)
    # Bonus: unanimous agreement
    unanimous_bonus = 15 if agree_count == 3 else 0
    # Bonus: trending regime
    regime_bonus = 5 if regime_name != "RANGING" else 0
    # HTF trend alignment bonus
    htf_align = 5 if ((raw_dir == "LONG" and trend_htf > 0.1) or
                      (raw_dir == "SHORT" and trend_htf < -0.1)) else 0

    quality = min(100, base_score + unanimous_bonus + regime_bonus + htf_align)
    quality = max(0, quality)

    result["direction"]         = raw_dir
    result["quality_score"]     = round(quality, 1)
    result["confluence_groups"] = agree_count
    result["filters"]["tf_alignment"] = {"pass": True, "multiplier": 1.0,
                                          "reason": f"Narrative aligned {agree_count}/3 groups"}

    if   quality >= QUALITY_THRESHOLD_HIGH:   confidence = "HIGH"
    elif quality >= QUALITY_THRESHOLD_MEDIUM: confidence = "MEDIUM"
    else:
        result["reject_reason"] = f"Quality score too low ({quality:.1f} < {QUALITY_THRESHOLD_MEDIUM})"
        return result

    result["confidence"] = confidence
    result["passed"]     = True
    return result


# ─── DROP-IN REPLACEMENT for score_timeframe ─────────────────────────────────
def score_timeframe_v2(df: pd.DataFrame, regime: dict = None) -> tuple:
    """
    Quality-weighted, regime-aware scorer.
    Returns (score, signals, support, resistance, rsi_val, atr_val, price)
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    price  = float(close.iloc[-1])

    support    = float(low.tail(20).min())
    resistance = float(high.tail(20).max())

    rsi     = ta.rsi(close, length=14)
    rsi_val = float(rsi.dropna().iloc[-1]) if rsi is not None and len(rsi.dropna()) > 0 else 50.0
    atr     = ta.atr(high, low, close, length=14)
    atr_val = float(atr.dropna().iloc[-1]) if atr is not None and len(atr.dropna()) > 0 else 0.0

    if regime is None:
        regime = detect_regime(df)

    trend_s,  trend_sigs    = score_trend_group(df, regime)
    mom_s,    mom_sigs      = score_momentum_group(df, regime)
    struct_s, struct_sigs   = score_structure_group(df, regime)
    vol_s,    vol_sigs      = score_volume_group(df)
    osc_s,    osc_sigs      = score_oscillator_group(df)

    weighted = (
        trend_s  * WEIGHTS["trend"] +
        mom_s    * WEIGHTS["momentum"] +
        struct_s * WEIGHTS["structure"] +
        vol_s    * WEIGHTS["volume"] +
        osc_s    * WEIGHTS["oscillator"]
    )
    score   = int(weighted / 5)
    signals = trend_sigs + mom_sigs + struct_sigs + vol_sigs + osc_sigs
    return score, signals, support, resistance, rsi_val, atr_val, price


def analyze_v2(symbol: str, fetch_ohlcv_fn, coin_labels: dict) -> dict:
    """
    Full Phase 1 analysis — called by ai_validator.run_full_pipeline().
    Fixed: single return with complete result dict.
    """
    df_1h = fetch_ohlcv_fn(symbol, "1h", 200)
    df_4h = fetch_ohlcv_fn(symbol, "4h", 100)

    rsi_series  = ta.rsi(df_1h["close"], length=14)
    rsi_current = float(rsi_series.dropna().iloc[-1]) if rsi_series is not None and len(rsi_series.dropna()) > 0 else 50.0

    quality_result = compute_signal_quality(df_1h, df_4h, symbol, funding_rate=0.0, rsi_val=rsi_current)

    regime    = quality_result.get("regime") or detect_regime(df_4h)
    score_4h, signals_4h, support, resistance, rsi_4h, atr_4h, price = score_timeframe_v2(df_1h, regime)
    score_1d, signals_1d, _,       _,          rsi_1d, atr_1d, _     = score_timeframe_v2(df_4h, regime)

    direction  = quality_result.get("direction", "NEUTRAL")
    confidence = quality_result.get("confidence", "LOW")
    label      = coin_labels.get(symbol, symbol.split("/")[0])

    return {
        "symbol":      symbol,
        "label":       label,
        "direction":   direction,
        "confidence":  confidence,
        "score":       quality_result.get("quality_score", 0),
        "abs_score":   quality_result.get("quality_score", 0),
        "score_4h":    score_4h,
        "score_1d":    score_1d,
        "tf_agree":    quality_result.get("filters", {}).get("tf_alignment", {}).get("pass", False),
        "signals":     quality_result.get("signals", [])[:10],
        "price":       price,
        "support":     support,
        "resistance":  resistance,
        "rsi_4h":      rsi_4h,
        "rsi_1d":      rsi_1d,
        "atr":         atr_4h,
        "funding":     0,
        "df_4h":       df_1h,
        "tf_labels":   ("1H", "4H"),
        "regime":      regime,
        "quality":     quality_result,
        "signal_type": "REVERSAL",
        "type_conf":   confidence,
        "type_reason": quality_result.get("reject_reason", "") or
                       f"{quality_result.get('confluence_groups', 0)} groups confluent | Q:{quality_result.get('quality_score', 0)}",
    }


# ─── DERIVATIVES CONTEXT CLASSIFIER ─────────────────────────────────────────
def classify_derivatives_context(
    funding_rate: float,
    oi_now: float = None,
    oi_prev: float = None,
    change_24h: float = 0.0,
) -> dict:
    """
    Classifies the derivatives market context for a coin before scoring runs.
    Uses funding rate + OI change to determine the character of current positioning.

    Four states:
      ORGANIC   — OI rising, funding neutral/slightly positive.
                  Real demand. Signals fire normally.
      SQUEEZE   — OI rising, funding deeply negative during price pump.
                  Forced short liquidations. LONG entry risky, reversal likely.
      CROWDED   — OI elevated, funding very positive, price extended.
                  Overcrowded longs. Flush risk. LONG blocked, SHORT elevated.
      NEUTRAL   — OI flat/falling or data unavailable.
                  No strong positioning. Use simplified scoring.

    Returns:
    {
        "state":        str,   # ORGANIC / SQUEEZE / CROWDED / NEUTRAL
        "block_long":   bool,  # True if LONG should be blocked
        "boost_short":  bool,  # True if SHORT opportunity elevated
        "reason":       str,   # human-readable explanation
        "confidence":   str,   # HIGH / MEDIUM / LOW
    }
    """
    # Default when data unavailable
    no_data = {
        "state": "NEUTRAL", "block_long": False, "boost_short": False,
        "reason": "No OI data — neutral context", "confidence": "LOW"
    }

    # Funding thresholds
    FUNDING_VERY_NEGATIVE = -0.008   # deeply negative — shorts crowded / squeeze
    FUNDING_NEGATIVE      = -0.003   # mildly negative
    FUNDING_POSITIVE      = 0.010    # healthy positive
    FUNDING_VERY_POSITIVE = 0.025    # overcrowded longs

    # OI change (if available)
    oi_rising   = False
    oi_falling  = False
    oi_change_pct = 0.0
    if oi_now and oi_prev and oi_prev > 0:
        oi_change_pct = (oi_now - oi_prev) / oi_prev * 100
        oi_rising     = oi_change_pct > 3.0   # 3%+ rise = meaningful
        oi_falling    = oi_change_pct < -3.0

    fr = funding_rate  # shorthand

    # ── SQUEEZE: OI rising + funding deeply negative + price pumping ──────────
    # Classic short squeeze. The pump is driven by forced liquidations not demand.
    # LONG entry here is chasing a move that reverses once shorts are exhausted.
    if fr <= FUNDING_VERY_NEGATIVE and change_24h >= 5.0:
        if oi_rising:
            return {
                "state":       "SQUEEZE",
                "block_long":  True,
                "boost_short": True,  # good SHORT setup once pump exhausts
                "reason":      f"Short squeeze: funding {fr*100:.3f}% + OI rising {oi_change_pct:+.1f}% + price +{change_24h:.1f}%",
                "confidence":  "HIGH",
            }
        # Funding very negative even without OI data = likely squeeze
        elif oi_now is None:
            return {
                "state":       "SQUEEZE",
                "block_long":  True,
                "boost_short": False,  # can't confirm without OI
                "reason":      f"Funding deeply negative ({fr*100:.3f}%) during pump — likely squeeze",
                "confidence":  "MEDIUM",
            }

    # ── CROWDED: High positive funding + extended price ───────────────────────
    # Too many longs piled in. Funding punishes longs. Flush risk is elevated.
    if fr >= FUNDING_VERY_POSITIVE:
        return {
            "state":       "CROWDED",
            "block_long":  True,
            "boost_short": True,
            "reason":      f"Crowded longs: funding {fr*100:.3f}% — flush risk HIGH",
            "confidence":  "HIGH",
        }
    if fr >= FUNDING_POSITIVE and oi_rising:
        return {
            "state":       "CROWDED",
            "block_long":  True,
            "boost_short": False,
            "reason":      f"Longs crowding: funding {fr*100:.3f}% + OI rising — proceed with caution",
            "confidence":  "MEDIUM",
        }

    # ── ORGANIC: OI rising + neutral-to-mild positive funding ─────────────────
    # Real demand. New positions opening without excessive leverage cost.
    # Best environment for LONG signals.
    if oi_rising and FUNDING_NEGATIVE < fr < FUNDING_POSITIVE:
        return {
            "state":       "ORGANIC",
            "block_long":  False,
            "boost_short": False,
            "reason":      f"Organic demand: OI +{oi_change_pct:.1f}% + funding {fr*100:.3f}% balanced",
            "confidence":  "HIGH",
        }

    # ── OI falling: unwinding positions ──────────────────────────────────────
    if oi_falling:
        return {
            "state":       "NEUTRAL",
            "block_long":  False,
            "boost_short": False,
            "reason":      f"OI declining {oi_change_pct:.1f}% — positions unwinding, reduced conviction",
            "confidence":  "LOW",
        }

    # Default: funding mild, no OI signal
    return {
        "state":       "NEUTRAL",
        "block_long":  False,
        "boost_short": False,
        "reason":      f"Neutral context: funding {fr*100:.3f}%",
        "confidence":  "LOW",
    }


def check_exhaustion(
    df: pd.DataFrame,
    direction: str,
    funding_rate: float = 0.0,
) -> dict:
    """
    Detects blow-off tops (LONG block) and capitulation dumps (SHORT opportunity).

    Blow-off top — block LONG when ALL of:
      - 24H gain >= 20% (massive single-day pump)
      - RSI >= 72 (overbought)
      - Volume >= 2.5x 20-period average (crowd chasing)
      - Price >= 8% above EMA50 (overextended)

    Exhaustion SHORT opportunity — flag SHORT when:
      - 24H gain >= 25% followed by volume collapsing (classic pump-and-dump top)
      - OR funding rate >= 0.05% (crowded longs — squeeze incoming)
      - OR RSI >= 78 AND price >= 12% above EMA50 (extreme overextension)

    Returns:
    {
      "block_long":    bool,   # True = do not fire LONG signal
      "short_opp":     bool,   # True = consider SHORT instead
      "reason":        str,    # human-readable explanation
      "severity":      str,    # "HIGH" | "MEDIUM" | "NONE"
    }
    """
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        price  = float(close.iloc[-1])

        block_long  = False
        short_opp   = False
        reasons     = []

        # ── 24H price change ─────────────────────────────────────────────────
        change_24h = 0.0
        if len(close) >= 24:
            price_24h_ago = float(close.iloc[-24])
            if price_24h_ago > 0:
                change_24h = (price - price_24h_ago) / price_24h_ago * 100

        # ── RSI ───────────────────────────────────────────────────────────────
        rsi_s   = ta.rsi(close, length=14)
        rsi_val = float(rsi_s.dropna().iloc[-1]) if rsi_s is not None and len(rsi_s.dropna()) > 0 else 50.0

        # ── Volume vs average ─────────────────────────────────────────────────
        vol_ma  = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())
        vol_now = float(volume.iloc[-1])
        vol_ratio = vol_now / vol_ma if vol_ma > 0 else 1.0

        # ── EMA50 extension ───────────────────────────────────────────────────
        ema50   = ta.ema(close, length=50)
        ext_pct = 0.0
        if ema50 is not None and len(ema50.dropna()) > 0:
            e50     = float(ema50.dropna().iloc[-1])
            ext_pct = (price - e50) / e50 * 100 if e50 > 0 else 0.0

        # ── Volume collapse after spike (pump-and-dump top) ───────────────────
        vol_collapsing = False
        if len(volume) >= 6:
            v = volume.values
            peak_vol     = max(v[-6:-1])
            current_vol  = v[-1]
            vol_collapsing = current_vol < peak_vol * 0.4 and peak_vol > vol_ma * 2.0

        # ── Blow-off top detection — block LONG ───────────────────────────────
        # FIX: lowered to 2/4 signals but requires >= 15% 24H gain (prevents false positives)
        # This catches BASED (10%+ dump after pump) and ORDI/SOON type blow-offs
        blowoff_signals = 0
        if change_24h >= 15:
            blowoff_signals += 1
            reasons.append(f"24H pump +{change_24h:.1f}%")
        if rsi_val >= 70:
            blowoff_signals += 1
            reasons.append(f"RSI overbought ({rsi_val:.0f})")
        if vol_ratio >= 2.0:
            blowoff_signals += 1
            reasons.append(f"Volume {vol_ratio:.1f}x average — crowd chasing")
        if ext_pct >= 6:
            blowoff_signals += 1
            reasons.append(f"Price {ext_pct:.1f}% above EMA50 — overextended")

        if blowoff_signals >= 2 and change_24h >= 15:
            block_long = True
            reasons.append(f"→ Blow-off top ({blowoff_signals}/4 signals + {change_24h:.1f}% pump)")
        elif blowoff_signals >= 4:
            block_long = True
            reasons.append(f"→ Extreme overextension ({blowoff_signals}/4 signals)")

        # ── SHORT opportunity — lowered thresholds to catch BASED/ORDI type tops ──
        short_signals = 0
        if change_24h >= 15 and vol_collapsing:
            short_signals += 2
            reasons.append(f"Pump {change_24h:.1f}% + volume collapsing — distribution top")
        if funding_rate >= 0.03:
            short_signals += 1
            reasons.append(f"Funding {funding_rate*100:.3f}% — longs crowded")
        if funding_rate >= 0.05:
            short_signals += 1
            reasons.append(f"Funding {funding_rate*100:.3f}% — squeeze risk HIGH")
        if rsi_val >= 75 and ext_pct >= 8:
            short_signals += 2
            reasons.append(f"RSI {rsi_val:.0f} + {ext_pct:.1f}% above EMA50 — extreme overextension")
        elif rsi_val >= 70 and ext_pct >= 5:
            short_signals += 1
            reasons.append(f"RSI {rsi_val:.0f} overbought + {ext_pct:.1f}% above EMA50")
        if change_24h >= 20:
            short_signals += 1
            reasons.append(f"20%+ single-day pump — blow-off territory")

        if short_signals >= 2:
            short_opp  = True
            block_long = True


        # ── Severity ──────────────────────────────────────────────────────────
        if short_opp:
            severity = "HIGH"
        elif block_long:
            severity = "MEDIUM"
        else:
            severity = "NONE"

        return {
            "block_long": block_long,
            "short_opp":  short_opp,
            "reason":     " | ".join(reasons) if reasons else "",
            "severity":   severity,
            "change_24h": round(change_24h, 1),
            "rsi":        round(rsi_val, 1),
            "vol_ratio":  round(vol_ratio, 1),
            "ext_pct":    round(ext_pct, 1),
        }

    except Exception as e:
        log.warning(f"Exhaustion check error: {e}")
        return {"block_long": False, "short_opp": False,
                "reason": "", "severity": "NONE",
                "change_24h": 0, "rsi": 50, "vol_ratio": 1, "ext_pct": 0}


# ─── DYNAMIC LEVERAGE SUGGESTER ──────────────────────────────────────────────
def suggest_leverage(
    df: pd.DataFrame,
    signal_quality: float = 50.0,
    grade: str = "B",
    max_leverage: int = 10,
) -> dict:
    """
    Suggests safe leverage based on coin volatility (ATR%) and signal quality.

    Formula:
      base = clamp(round(1.0 / atr_pct * 8), 2, max_leverage)
      bonus: Grade A = +1, quality >= 65 = +1
      penalty: atr_pct > 5% = -2, atr_pct > 3% = -1

    Returns:
    {
      "suggested": int,   # recommended leverage
      "max_safe":  int,   # absolute ceiling for this coin
      "reason":    str,   # human-readable explanation
      "atr_pct":   float, # 14-period ATR as % of price
    }
    """
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        price  = float(close.iloc[-1])

        atr_s  = ta.atr(high, low, close, length=14)
        if atr_s is None or len(atr_s.dropna()) == 0:
            return {"suggested": 5, "max_safe": 7, "reason": "ATR unavailable — using conservative 5x", "atr_pct": 2.0}

        atr_val = float(atr_s.dropna().iloc[-1])
        atr_pct = (atr_val / price * 100) if price > 0 else 2.0

        # Base leverage — inversely proportional to volatility
        # atr_pct 1% → 8x, 2% → 4x, 3% → 2.7x, 5% → 1.6x
        base = round(8.0 / max(atr_pct, 0.5))
        base = max(2, min(base, max_leverage))

        # Adjustments
        adj    = 0
        reason_parts = [f"ATR {atr_pct:.1f}% → base {base}x"]

        if atr_pct > 5.0:
            adj -= 2
            reason_parts.append("very high volatility -2x")
        elif atr_pct > 3.0:
            adj -= 1
            reason_parts.append("high volatility -1x")

        if grade == "A":
            adj += 1
            reason_parts.append("Grade A +1x")

        if signal_quality >= 65:
            adj += 1
            reason_parts.append("high quality +1x")

        suggested = max(2, min(base + adj, max_leverage))
        max_safe  = max(2, min(base + 1, max_leverage))

        return {
            "suggested": suggested,
            "max_safe":  max_safe,
            "reason":    " | ".join(reason_parts),
            "atr_pct":   round(atr_pct, 2),
        }

    except Exception as e:
        log.warning(f"Leverage suggestion error: {e}")
        return {"suggested": 5, "max_safe": 7, "reason": "Error — using conservative 5x", "atr_pct": 2.0}


# ─── SUPERTREND ───────────────────────────────────────────────────────────────
def compute_supertrend(
    df: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> dict:
    """
    Computes Supertrend indicator.
    Standard settings: ATR period=10, multiplier=3.0

    Formula:
      Basic Upper Band = (high + low) / 2 + multiplier × ATR
      Basic Lower Band = (high + low) / 2 - multiplier × ATR
      Final bands adjust to never widen once price crosses them.
      Direction: 1 = bullish (price above), -1 = bearish (price below)

    Returns:
    {
      "value":       float,  # current Supertrend line level
      "direction":   int,    # 1=bullish, -1=bearish
      "bullish":     bool,   # price above Supertrend
      "bearish":     bool,   # price below Supertrend
      "just_flipped": bool,  # flipped on the last candle (strong signal)
      "dist_pct":    float,  # % distance from price to Supertrend line
    }
    """
    try:
        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        n     = len(close)

        if n < atr_period + 5:
            return _empty_supertrend(float(close[-1]))

        # ATR
        atr_s  = ta.atr(df["high"], df["low"], df["close"], length=atr_period)
        if atr_s is None or len(atr_s.dropna()) < 5:
            return _empty_supertrend(float(close[-1]))
        atr_v  = atr_s.bfill().values
        # Sanity check — ATR must be positive
        atr_v  = np.where(atr_v <= 0, np.nanmean(np.where(atr_v > 0, atr_v, np.nan)), atr_v)

        # Basic bands
        hl2       = (high + low) / 2.0
        basic_ub  = hl2 + multiplier * atr_v
        basic_lb  = hl2 - multiplier * atr_v

        # Final bands with memory
        final_ub  = np.zeros(n)
        final_lb  = np.zeros(n)
        direction = np.ones(n, dtype=int)  # 1=bullish, -1=bearish
        st_line   = np.zeros(n)

        final_ub[0] = basic_ub[0]
        final_lb[0] = basic_lb[0]

        for i in range(1, n):
            final_ub[i] = basic_ub[i] if (basic_ub[i] < final_ub[i-1] or close[i-1] > final_ub[i-1]) else final_ub[i-1]
            final_lb[i] = basic_lb[i] if (basic_lb[i] > final_lb[i-1] or close[i-1] < final_lb[i-1]) else final_lb[i-1]

            if direction[i-1] == -1 and close[i] > final_ub[i-1]:
                direction[i] = 1
            elif direction[i-1] == 1 and close[i] < final_lb[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

            st_line[i] = final_lb[i] if direction[i] == 1 else final_ub[i]

        cur_price    = float(close[-1])
        cur_st       = float(st_line[-1])
        cur_dir      = int(direction[-1])
        just_flipped = int(direction[-1]) != int(direction[-2]) if n >= 2 else False
        dist_pct     = (cur_price - cur_st) / cur_st * 100 if cur_st > 0 else 0.0

        return {
            "value":        round(cur_st, 6),
            "direction":    cur_dir,
            "bullish":      cur_dir == 1,
            "bearish":      cur_dir == -1,
            "just_flipped": just_flipped,
            "dist_pct":     round(dist_pct, 2),
        }

    except Exception as e:
        log.warning(f"Supertrend error: {e}")
        return _empty_supertrend(float(df["close"].iloc[-1]) if len(df) > 0 else 0)


def _empty_supertrend(price: float) -> dict:
    return {"value": price, "direction": 1, "bullish": True,
            "bearish": False, "just_flipped": False, "dist_pct": 0.0}

# ─── DOW THEORY PHASE DETECTOR ───────────────────────────────────────────────
def detect_dow_phase(df: pd.DataFrame, direction: str = "LONG") -> dict:
    """
    Identifies the current Dow Theory market phase.

    ACCUMULATION  — Smart money buying quietly. Flat price, rising OBV,
                    low volume, price near support. grade_delta = +3 (BEST entry)
    PARTICIPATION — Public joins in. Higher highs/lows, expanding volume.
                    grade_delta = +1 (good but move already started)
    DISTRIBUTION  — Smart money selling into retail. Price extended, volume
                    high but stalling. grade_delta = -2 (AVOID)
    MARKDOWN      — Confirmed downtrend. grade_delta = -2 for LONGs (SHORT opp)
    UNCLEAR       — Not enough evidence. grade_delta = 0
    """
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        price  = float(close.iloc[-1])

        acc_pts  = 0   # accumulation evidence
        part_pts = 0   # participation evidence
        dist_pts = 0   # distribution evidence
        signals  = []

        vol_ma = float(volume.rolling(20).mean().iloc[-1]) if len(volume) >= 20 else float(volume.mean())

        # ── 1. Price structure — Dow higher highs/lows ────────────────────────
        if len(high) >= 20:
            sh, sl = [], []
            for i in range(2, min(20, len(high) - 2)):
                if high.iloc[-i] > high.iloc[-i-1] and high.iloc[-i] > high.iloc[-i+1]:
                    sh.append(float(high.iloc[-i]))
                if low.iloc[-i] < low.iloc[-i-1] and low.iloc[-i] < low.iloc[-i+1]:
                    sl.append(float(low.iloc[-i]))

            if len(sh) >= 2 and len(sl) >= 2:
                if direction == "LONG":
                    if sh[0] > sh[1] and sl[0] > sl[1]:
                        part_pts += 2
                        signals.append("HH+HL confirmed — participation trend")
                    elif sh[0] <= sh[1] and sl[0] <= sl[1]:
                        acc_pts += 1
                        signals.append("No HH/HL yet — base still forming")
                else:
                    if sh[0] < sh[1] and sl[0] < sl[1]:
                        part_pts += 2
                        signals.append("LH+LL confirmed — participation downtrend")
                    elif sh[0] >= sh[1] and sl[0] >= sl[1]:
                        acc_pts += 1
                        signals.append("No LH/LL yet — topping structure")

        # ── 2. OBV vs price divergence ────────────────────────────────────────
        obv = ta.obv(close, volume)
        if obv is not None and len(obv) >= 15:
            ov = obv.fillna(0).values[-15:]
            pv = close.values[-15:]
            os = np.polyfit(range(15), ov, 1)[0]
            ps = np.polyfit(range(15), pv, 1)[0]

            if direction == "LONG":
                if os > 0 and ps <= 0:
                    acc_pts += 2
                    signals.append("OBV rising, price flat — smart money accumulating")
                elif os > 0 and ps > 0:
                    part_pts += 1
                    signals.append("OBV + price rising — public participation")
                elif os < 0 and ps > 0:
                    dist_pts += 2
                    signals.append("OBV falling, price rising — distribution")
            else:
                if os < 0 and ps >= 0:
                    acc_pts += 2
                    signals.append("OBV falling, price flat — smart money distributing")
                elif os < 0 and ps < 0:
                    part_pts += 1
                    signals.append("OBV + price falling — public participation short")
                elif os > 0 and ps < 0:
                    dist_pts += 2
                    signals.append("OBV rising, price falling — capitulation exhaustion")

        # ── 3. Volume pattern ─────────────────────────────────────────────────
        avg_recent = float(volume.tail(5).mean())
        if avg_recent < vol_ma * 0.7:
            acc_pts += 1
            signals.append("Volume contracting — quiet accumulation")
        elif avg_recent > vol_ma * 1.3:
            part_pts += 1
            signals.append("Volume expanding — public participation")

        # High volume but price stalling = distribution
        if avg_recent > vol_ma * 1.5:
            pr_range = (float(high.tail(5).max()) - float(low.tail(5).min())) / price * 100
            if pr_range < 2.0:
                dist_pts += 2
                signals.append("High volume, price stalling — distribution")

        # ── 4. RSI context ────────────────────────────────────────────────────
        rsi = ta.rsi(close, length=14)
        if rsi is not None and len(rsi.dropna()) >= 5:
            rv = float(rsi.dropna().iloc[-1])
            if direction == "LONG":
                if 42 < rv < 56:
                    acc_pts += 1
                    signals.append(f"RSI neutral ({rv:.0f}) — accumulation zone")
                elif 56 <= rv <= 70:
                    part_pts += 1
                    signals.append(f"RSI momentum zone ({rv:.0f}) — participation")
                elif rv > 70:
                    dist_pts += 1
                    signals.append(f"RSI overbought ({rv:.0f}) — distribution risk")
            else:
                if 44 < rv < 58:
                    acc_pts += 1
                    signals.append(f"RSI neutral ({rv:.0f}) — topping zone")
                elif 30 <= rv <= 44:
                    part_pts += 1
                    signals.append(f"RSI downtrend zone ({rv:.0f}) — participation")
                elif rv < 30:
                    dist_pts += 1
                    signals.append(f"RSI oversold ({rv:.0f}) — capitulation risk")

        # ── 5. EMA structure — trend maturity ────────────────────────────────
        ema9  = ta.ema(close, length=9)
        ema21 = ta.ema(close, length=21)
        ema50 = ta.ema(close, length=50)
        if ema9 is not None and ema21 is not None and ema50 is not None:
            e9  = float(ema9.dropna().iloc[-1])  if len(ema9.dropna())  > 0 else price
            e21 = float(ema21.dropna().iloc[-1]) if len(ema21.dropna()) > 0 else price
            e50 = float(ema50.dropna().iloc[-1]) if len(ema50.dropna()) > 0 else price

            if direction == "LONG":
                if price < e9 and price < e21 and price > e50:
                    acc_pts += 2
                    signals.append("Below EMA9/21 but above EMA50 — base forming")
                elif e9 > e21 > e50 and price > e9:
                    part_pts += 2
                    signals.append("Bullish EMA stack (9>21>50) — trend active")
                if price > e9 * 1.06:
                    dist_pts += 1
                    signals.append(f"Price {((price/e9-1)*100):.1f}% above EMA9 — extended")
            else:
                if price > e9 and price > e21 and price < e50:
                    acc_pts += 2
                    signals.append("Above EMA9/21 but below EMA50 — topping")
                elif e9 < e21 < e50 and price < e9:
                    part_pts += 2
                    signals.append("Bearish EMA stack (9<21<50) — downtrend active")
                if price < e9 * 0.94:
                    dist_pts += 1
                    signals.append("Price extended below EMA9 — oversold")

        # ── Determine phase ───────────────────────────────────────────────────
        if dist_pts >= 3:
            phase, conf, grade_delta = "DISTRIBUTION", "HIGH", -2
        elif dist_pts >= 2:
            phase, conf, grade_delta = "DISTRIBUTION", "MEDIUM", -2
        elif acc_pts >= 3 and acc_pts >= part_pts:
            phase, conf, grade_delta = "ACCUMULATION", "HIGH", +3
        elif acc_pts >= 2 and acc_pts >= part_pts:
            phase, conf, grade_delta = "ACCUMULATION", "MEDIUM", +3
        elif part_pts >= 3:
            phase, conf, grade_delta = "PARTICIPATION", "HIGH", +1
        elif part_pts >= 2:
            phase, conf, grade_delta = "PARTICIPATION", "MEDIUM", +1
        else:
            phase, conf, grade_delta = "UNCLEAR", "LOW", 0

        return {
            "phase":       phase,
            "confidence":  conf,
            "grade_delta": grade_delta,
            "signals":     signals,
            "acc_pts":     acc_pts,
            "part_pts":    part_pts,
            "dist_pts":    dist_pts,
        }

    except Exception as e:
        log.warning(f"Dow phase error: {e}")
        return {"phase": "UNCLEAR", "confidence": "LOW", "grade_delta": 0,
                "signals": [], "acc_pts": 0, "part_pts": 0, "dist_pts": 0}

def grade_signal(
    df_ltf: pd.DataFrame,
    df_htf: pd.DataFrame,
    direction: str,
    quality_score: float,
    rsi_val: float,
    funding_rate: float = 0.0,
) -> dict:
    """
    REWRITTEN — Clean grade assignment using the narrative output.

    Grade A — Signal is early: price structure not yet extended,
               volatility compressed, Supertrend just confirmed or aligning.
               Best entry. Catch before the main move.

    Grade B — Signal is confirmed: momentum running, Supertrend aligned,
               Dow phase supportive. Standard entry, mid-move.

    Grade C — Signal is late: overextended, high RSI, Supertrend long-confirmed
               with no fresh flip. Exit liquidity risk.

    In FULL pipeline mode (altseason): Supertrend 4H gate is hard — opposes = Grade C.
    In SIMPLIFIED mode: Supertrend contributes to score but doesn't hard-block.

    Returns same dict as before for bot.py compatibility.
    """
    grade_score = 0
    reasons     = []
    warnings    = []

    close  = df_ltf["close"]
    high   = df_ltf["high"]
    low    = df_ltf["low"]
    volume = df_ltf["volume"]
    price  = float(close.iloc[-1])

    try:
        # ── Entry timing: is the signal early or late? ────────────────────────

        # BB squeeze — volatility compressed = early (Grade A signal)
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty and len(bb) >= 10:
            bw_now  = float(bb.iloc[-1, 3]) if bb.shape[1] > 3 else 0
            bw_prev = float(bb.iloc[-6, 3]) if bb.shape[1] > 3 else 0
            if bw_now < bw_prev * 0.8:
                grade_score += 2
                reasons.append("BB squeeze — compression before move (early entry)")
            elif bw_now > bw_prev * 1.3:
                grade_score -= 1
                warnings.append("BB expanding — move already in progress (late entry)")

        # OBV divergence from price — smart money accumulating before price moves
        obv = ta.obv(close, volume)
        if obv is not None and len(obv) >= 10:
            ov = obv.fillna(0).values[-10:]
            pv = close.values[-10:]
            obv_slope   = np.polyfit(range(10), ov, 1)[0]
            price_slope = np.polyfit(range(10), pv, 1)[0]
            if direction == "LONG"  and obv_slope > 0 and price_slope <= 0:
                grade_score += 2
                reasons.append("OBV rising, price flat — accumulation (early)")
            elif direction == "SHORT" and obv_slope < 0 and price_slope >= 0:
                grade_score += 2
                reasons.append("OBV falling, price flat — distribution (early)")

        # ATR contracting — quiet market before explosion
        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is not None and len(atr_s.dropna()) >= 8:
            atr_now  = float(atr_s.dropna().iloc[-1])
            atr_prev = float(atr_s.dropna().iloc[-5])
            if atr_now < atr_prev * 0.85:
                grade_score += 1
                reasons.append("ATR contracting — quiet before move")
            elif atr_now > atr_prev * 1.4:
                grade_score -= 1
                warnings.append("ATR expanding — volatile, late entry risk")

        # RSI position — neutral zone = early, extreme = late
        if 44 < rsi_val < 58:
            grade_score += 1
            reasons.append(f"RSI neutral ({rsi_val:.1f}) — not chased")
        elif (direction == "LONG"  and rsi_val > 72) or              (direction == "SHORT" and rsi_val < 28):
            grade_score -= 2
            warnings.append(f"RSI extreme ({rsi_val:.1f}) — late entry, chasing move")

        # ── HTF confluence ────────────────────────────────────────────────────
        htf_rsi   = ta.rsi(df_htf["close"], length=14)
        htf_rsi_v = float(htf_rsi.dropna().iloc[-1]) if htf_rsi is not None and len(htf_rsi.dropna()) > 0 else 50.0
        htf_ema9  = ta.ema(df_htf["close"], length=9)
        htf_ema21 = ta.ema(df_htf["close"], length=21)
        if htf_ema9 is not None and htf_ema21 is not None:
            e9  = float(htf_ema9.dropna().iloc[-1])
            e21 = float(htf_ema21.dropna().iloc[-1])
            htf_bull = e9 > e21 and htf_rsi_v > 50
            htf_bear = e9 < e21 and htf_rsi_v < 50
            if   direction == "LONG"  and htf_bull: grade_score += 2; reasons.append("HTF aligned LONG")
            elif direction == "SHORT" and htf_bear:  grade_score += 2; reasons.append("HTF aligned SHORT")
            elif (direction == "LONG" and htf_bear) or (direction == "SHORT" and htf_bull):
                grade_score -= 1; warnings.append("HTF disagrees with direction")

    except Exception as e:
        log.warning(f"Grade signal error: {e}")

    # ── Supertrend gate ───────────────────────────────────────────────────────
    st_res = _empty_supertrend(price)
    try:
        st_res = compute_supertrend(df_ltf)
        st_htf = compute_supertrend(df_htf)

        if _SUPERTREND_GATE_ACTIVE:
            # FULL mode: 4H Supertrend opposing = hard Grade C
            if direction == "LONG"  and st_htf["bearish"]:
                warnings.append("GATE: ST 4H bearish — LONG blocked")
                grade_score -= 6
            elif direction == "SHORT" and st_htf["bullish"]:
                warnings.append("GATE: ST 4H bullish — SHORT blocked")
                grade_score -= 6
            else:
                if st_res["just_flipped"]:
                    grade_score += 2; reasons.append("Supertrend just flipped — fresh signal")
                else:
                    grade_score += 1; reasons.append(f"Supertrend aligned {direction}")
        else:
            # SIMPLIFIED mode: Supertrend contributes but doesn't hard-block
            if (direction == "LONG"  and st_res["bullish"]) or                (direction == "SHORT" and st_res["bearish"]):
                if st_res["just_flipped"]: grade_score += 2; reasons.append("Supertrend just flipped")
                else:                      grade_score += 1; reasons.append(f"Supertrend aligned")
            elif (direction == "LONG"  and st_res["bearish"] and st_htf["bearish"]) or                  (direction == "SHORT" and st_res["bullish"] and st_htf["bullish"]):
                grade_score -= 3
                warnings.append("\u26a0 Supertrend BEARISH both TFs — opposing signal direction (risky in simplified mode)")
            elif (direction == "LONG" and st_res["bearish"]) or \
                 (direction == "SHORT" and st_res["bullish"]):
                grade_score -= 1; warnings.append("Supertrend LTF opposing — caution")
    except Exception as e:
        log.warning("Supertrend grade error: " + str(e))

    # ── Dow Theory phase ──────────────────────────────────────────────────────
    dow = {"phase": "UNCLEAR", "confidence": "LOW", "grade_delta": 0, "signals": []}
    try:
        dow = detect_dow_phase(df_ltf, direction)
        grade_score += dow["grade_delta"]
        if dow["phase"] == "ACCUMULATION": reasons.append("Dow: ACCUMULATION — best entry zone")
        elif dow["phase"] == "DISTRIBUTION": warnings.append("Dow: DISTRIBUTION — smart money exiting")
        elif dow["phase"] == "PARTICIPATION": reasons.append("Dow: PARTICIPATION — move underway")
    except Exception as e:
        log.warning("Dow phase error: " + str(e))

    # ── VWAP ──────────────────────────────────────────────────────────────────
    vwap_res = {"vwap": 0, "vwap_dist_pct": 0, "bias": "AT", "strength": "WEAK", "signal": ""}
    try:
        vwap_window = 0 if len(df_ltf) <= 100 else 100
        vwap_res    = compute_vwap(df_ltf, window=vwap_window)
        vd          = vwap_grade_delta(vwap_res, direction)
        grade_score += vd
        if vwap_res["signal"]:
            (reasons if vd > 0 else warnings).append("VWAP: " + vwap_res["signal"])
    except Exception as e:
        log.warning("VWAP grade error: " + str(e))

    # ── Determine grade ───────────────────────────────────────────────────────
    all_reasons = reasons + warnings
    if   grade_score >= 4: grade = "A"
    elif grade_score >= 0: grade = "B"
    else:                  grade = "C"

    return {
        "grade":            grade,
        "grade_score":      grade_score,
        "reasons":          reasons,
        "warnings":         warnings,
        "all_reasons":      all_reasons,
        "dow_phase":        dow["phase"],
        "dow_confidence":   dow["confidence"],
        "dow_signals":      dow["signals"],
        "vwap":             vwap_res.get("vwap", 0),
        "vwap_dist_pct":    vwap_res.get("vwap_dist_pct", 0),
        "vwap_bias":        vwap_res.get("bias", "AT"),
        "vwap_strength":    vwap_res.get("strength", "WEAK"),
        "supertrend":       st_res.get("value", 0),
        "supertrend_dir":   st_res.get("direction", 1),
        "supertrend_flip":  st_res.get("just_flipped", False),
    }


def update_adaptive_thresholds(outcome: str):
    """
    Call this when a paper trade closes with WIN or LOSS.
    Adjusts CONFLUENCE_MIN and QUALITY_THRESHOLD_MEDIUM based on rolling win rate.
    """
    global CONFLUENCE_MIN, QUALITY_THRESHOLD_MEDIUM, _adaptive_state

    if outcome == "WIN":
        _adaptive_state["win_count"] += 1
    elif outcome == "LOSS":
        _adaptive_state["loss_count"] += 1
    else:
        return

    _adaptive_state["total"] += 1
    _adaptive_state["last_update"] = __import__("time").time()

    total = _adaptive_state["total"]
    if total < 10:  # need at least 10 trades before adapting
        return

    win_rate = _adaptive_state["win_count"] / total * 100

    # If win rate is poor — tighten thresholds (fewer but better signals)
    if win_rate < 35:
        CONFLUENCE_MIN = min(5, CONFLUENCE_MIN + 1)
        QUALITY_THRESHOLD_MEDIUM = min(60, QUALITY_THRESHOLD_MEDIUM + 5)
        log.info(f"Adaptive: win rate {win_rate:.0f}% low — tightened to C={CONFLUENCE_MIN} Q={QUALITY_THRESHOLD_MEDIUM}")

    # If win rate is strong — loosen slightly (more signals)
    elif win_rate > 65 and total >= 20:
        CONFLUENCE_MIN = max(3, CONFLUENCE_MIN - 1)
        QUALITY_THRESHOLD_MEDIUM = max(45, QUALITY_THRESHOLD_MEDIUM - 5)
        log.info(f"Adaptive: win rate {win_rate:.0f}% strong — loosened to C={CONFLUENCE_MIN} Q={QUALITY_THRESHOLD_MEDIUM}")

    # Reset rolling window every 30 trades
    if total >= 30:
        _adaptive_state = {
            "win_count":   0,
            "loss_count":  0,
            "total":       0,
            "last_update": _adaptive_state["last_update"],
        }
        log.info("Adaptive: rolling window reset after 30 trades")


# ─── VOLUME PROFILE ───────────────────────────────────────────────────────────
def compute_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
) -> dict:
    """
    Builds an OHLCV-approximated volume profile.

    Method:
    - Divide the full price range into N bins
    - For each candle, distribute its volume proportionally across the bins
      it touched (between low and high), weighted by close position
    - Find HVNs (high volume nodes), LVNs (low volume gaps), and POC

    Returns:
    {
      "poc":        float,   # Point of Control — highest volume price level
      "hvn_above":  float,   # nearest HVN above current price
      "hvn_below":  float,   # nearest HVN below current price
      "lvn_above":  float,   # nearest LVN above current price (fast move zone)
      "lvn_below":  float,   # nearest LVN below current price
      "hvns":       list,    # all HVN price levels
      "lvns":       list,    # all LVN price levels
      "in_lvn":     bool,    # True if current price is in a low volume zone
      "profile":    dict,    # full {price_level: volume} map
    }
    """
    try:
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values
        price  = float(close[-1])

        price_min = float(np.min(low))
        price_max = float(np.max(high))
        price_range = price_max - price_min

        if price_range == 0:
            return _empty_vp(price)

        # Build bin edges
        bin_size   = price_range / bins
        bin_edges  = [price_min + i * bin_size for i in range(bins + 1)]
        bin_centers= [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(bins)]
        bin_volumes = np.zeros(bins)

        # Distribute each candle's volume across bins it touched
        for i in range(len(close)):
            candle_low  = float(low[i])
            candle_high = float(high[i])
            candle_vol  = float(volume[i])
            candle_close= float(close[i])

            # Find which bins this candle spans
            lo_bin = max(0, int((candle_low  - price_min) / bin_size))
            hi_bin = min(bins - 1, int((candle_high - price_min) / bin_size))

            if lo_bin == hi_bin:
                bin_volumes[lo_bin] += candle_vol
            else:
                # Distribute proportionally — weight toward close price
                span = hi_bin - lo_bin + 1
                for b in range(lo_bin, hi_bin + 1):
                    bin_center = bin_centers[b]
                    # Closer to close = more volume weight
                    weight = 1.0 - abs(bin_center - candle_close) / (candle_high - candle_low + 1e-10)
                    weight = max(0.1, weight)
                    bin_volumes[b] += candle_vol * weight / span

        # Normalise
        total_vol = bin_volumes.sum()
        if total_vol == 0:
            return _empty_vp(price)

        bin_pct = bin_volumes / total_vol * 100

        # Find POC — bin with most volume
        poc_idx = int(np.argmax(bin_volumes))
        poc     = round(bin_centers[poc_idx], 6)

        # Find HVNs — bins with > 1.5x average volume
        avg_vol_pct = 100.0 / bins  # uniform distribution baseline
        hvn_threshold = avg_vol_pct * 1.5
        lvn_threshold = avg_vol_pct * 0.4

        hvns = [round(bin_centers[i], 6) for i in range(bins) if bin_pct[i] >= hvn_threshold]
        lvns = [round(bin_centers[i], 6) for i in range(bins) if bin_pct[i] <= lvn_threshold]

        # Nearest levels relative to current price
        hvn_above = min((h for h in hvns if h > price), default=None)
        hvn_below = max((h for h in hvns if h < price), default=None)
        lvn_above = min((l for l in lvns if l > price), default=None)
        lvn_below = max((l for l in lvns if l < price), default=None)

        # Is current price in a low volume zone?
        current_bin = min(bins - 1, int((price - price_min) / bin_size))
        in_lvn = bin_pct[current_bin] <= lvn_threshold

        # Build profile dict for display (top 10 levels only)
        profile = {round(bin_centers[i], 6): round(bin_pct[i], 2)
                   for i in sorted(range(bins), key=lambda x: bin_volumes[x], reverse=True)[:10]}

        return {
            "poc":       poc,
            "hvn_above": hvn_above,
            "hvn_below": hvn_below,
            "lvn_above": lvn_above,
            "lvn_below": lvn_below,
            "hvns":      hvns,
            "lvns":      lvns,
            "in_lvn":    in_lvn,
            "profile":   profile,
            "bin_size":  round(bin_size, 6),
        }

    except Exception as e:
        log.warning(f"Volume profile error: {e}")
        return _empty_vp(price if 'price' in dir() else 0)


def _empty_vp(price: float) -> dict:
    return {
        "poc": price, "hvn_above": None, "hvn_below": None,
        "lvn_above": None, "lvn_below": None,
        "hvns": [], "lvns": [], "in_lvn": False,
        "profile": {}, "bin_size": 0,
    }


def vp_adjusted_levels(
    direction: str,
    price: float,
    atr: float,
    vp: dict,
    sl_atr: float,
    tp1_atr: float,
    tp2_atr: float,
) -> tuple[float, float, float, dict]:
    """
    Adjusts ATR-based SL/TP levels to align with volume profile levels.

    Rules:
    - SL: snap to nearest HVN below (LONG) or above (SHORT) if within 0.5x ATR
    - TP1: snap to nearest HVN above (LONG) or below (SHORT) if within 1x ATR
    - TP2: snap to POC or next major HVN if within 2x ATR
    - If in LVN: widen TP targets slightly (fast move expected through LVN)

    Returns: (sl, tp1, tp2, vp_meta)
    """
    sl  = sl_atr
    tp1 = tp1_atr
    tp2 = tp2_atr
    adjustments = []

    try:
        if direction == "LONG":
            # SL — snap to HVN below if close enough (stronger support)
            hvn_b = vp.get("hvn_below")
            if hvn_b and abs(hvn_b - sl_atr) < atr * 0.5 and hvn_b < price:
                sl = round(hvn_b * 0.999, 6)  # just below the HVN
                adjustments.append(f"SL anchored to HVN ${sl:.4f}")

            # TP1 — snap to HVN above if close (natural resistance)
            hvn_a = vp.get("hvn_above")
            if hvn_a and hvn_a > price and abs(hvn_a - tp1_atr) < atr * 1.0:
                tp1 = round(hvn_a * 0.999, 6)  # just below the HVN
                adjustments.append(f"TP1 at HVN ${tp1:.4f}")

            # TP2 — POC or next major HVN
            poc = vp.get("poc")
            if poc and poc > tp1 and abs(poc - tp2_atr) < atr * 2.0:
                tp2 = round(poc, 6)
                adjustments.append(f"TP2 at POC ${tp2:.4f}")
            elif hvn_a and hvn_a > tp1:
                # Find next HVN above TP1
                hvns_above_tp1 = sorted([h for h in vp.get("hvns", []) if h > tp1])
                if len(hvns_above_tp1) >= 2:
                    tp2 = round(hvns_above_tp1[1] * 0.999, 6)
                    adjustments.append(f"TP2 at HVN ${tp2:.4f}")

        else:  # SHORT
            hvn_a = vp.get("hvn_above")
            if hvn_a and abs(hvn_a - sl_atr) < atr * 0.5 and hvn_a > price:
                sl = round(hvn_a * 1.001, 6)
                adjustments.append(f"SL anchored to HVN ${sl:.4f}")

            hvn_b = vp.get("hvn_below")
            if hvn_b and hvn_b < price and abs(hvn_b - tp1_atr) < atr * 1.0:
                tp1 = round(hvn_b * 1.001, 6)
                adjustments.append(f"TP1 at HVN ${tp1:.4f}")

            poc = vp.get("poc")
            if poc and poc < tp1 and abs(poc - tp2_atr) < atr * 2.0:
                tp2 = round(poc, 6)
                adjustments.append(f"TP2 at POC ${tp2:.4f}")
            elif hvn_b and hvn_b < tp1:
                hvns_below_tp1 = sorted([h for h in vp.get("hvns", []) if h < tp1], reverse=True)
                if len(hvns_below_tp1) >= 2:
                    tp2 = round(hvns_below_tp1[1] * 1.001, 6)
                    adjustments.append(f"TP2 at HVN ${tp2:.4f}")

        # LVN boost — fast move expected, widen TP slightly
        if vp.get("in_lvn") and not adjustments:
            tp2 = round(tp2 * 1.05 if direction == "LONG" else tp2 * 0.95, 6)
            adjustments.append("In LVN — TP2 widened (fast move zone)")

    except Exception as e:
        log.warning(f"VP level adjustment error: {e}")

    vp_meta = {
        "poc":         vp.get("poc"),
        "hvn_above":   vp.get("hvn_above"),
        "hvn_below":   vp.get("hvn_below"),
        "in_lvn":      vp.get("in_lvn", False),
        "adjustments": adjustments,
    }

    return sl, tp1, tp2, vp_meta


# ─── VWAP ────────────────────────────────────────────────────────────────────
def compute_vwap(
    df: pd.DataFrame,
    window: int = 0,
) -> dict:
    """
    Computes VWAP (Volume Weighted Average Price) from OHLCV data.

    Two modes:
    - window=0: Daily VWAP — resets each calendar day. Best for scalp (15m/1H).
    - window>0: Rolling VWAP over N candles. Best for swing (4H/1D).

    Method: typical_price = (high + low + close) / 3
            vwap = cumsum(typical_price × volume) / cumsum(volume)

    Returns:
    {
      "vwap":         float,   # current VWAP level
      "vwap_dist_pct": float,  # % price is above(+) or below(-) VWAP
      "bias":         str,     # "ABOVE" | "BELOW" | "AT"
      "strength":     str,     # "STRONG" | "MODERATE" | "WEAK"
      "signal":       str,     # human-readable context
      "grade_delta":  int,     # adjustment for grade_signal integration
                               # LONG below VWAP = +1, LONG above = -1
                               # SHORT above VWAP = +1, SHORT below = -1
    }
    """
    try:
        close  = df["close"]
        high   = df["high"]
        low    = df["low"]
        volume = df["volume"]
        price  = float(close.iloc[-1])

        typical = (high + low + close) / 3.0

        if window > 0:
            # Rolling VWAP — last N candles
            n = min(window, len(df))
            tp_v = typical.values[-n:]
            vol_v = volume.values[-n:]
        else:
            # Daily VWAP — group by date, use today's candles
            try:
                if hasattr(df.index, 'date'):
                    today = df.index[-1].date()
                    mask = pd.Series(df.index).apply(lambda x: x.date() == today).values
                    if mask.sum() >= 3:
                        tp_v  = typical.values[mask]
                        vol_v = volume.values[mask]
                    else:
                        # Fallback — last 24 candles
                        tp_v  = typical.values[-24:]
                        vol_v = volume.values[-24:]
                else:
                    tp_v  = typical.values[-24:]
                    vol_v = volume.values[-24:]
            except Exception:
                tp_v  = typical.values[-24:]
                vol_v = volume.values[-24:]

        # Compute VWAP
        cum_tp_vol = np.cumsum(tp_v * vol_v)
        cum_vol    = np.cumsum(vol_v)
        vwap_series = np.where(cum_vol > 0, cum_tp_vol / cum_vol, tp_v)
        vwap = float(vwap_series[-1])

        if vwap == 0:
            return _empty_vwap(price)

        dist_pct = (price - vwap) / vwap * 100

        # Bias
        if dist_pct > 0.5:
            bias = "ABOVE"
        elif dist_pct < -0.5:
            bias = "BELOW"
        else:
            bias = "AT"

        # Strength of deviation
        abs_dist = abs(dist_pct)
        if abs_dist >= 3.0:
            strength = "STRONG"
        elif abs_dist >= 1.5:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        # Context signal
        if bias == "ABOVE":
            signal = f"Price {dist_pct:.1f}% above VWAP ${vwap:.4f} — premium to fair value"
        elif bias == "BELOW":
            signal = f"Price {abs(dist_pct):.1f}% below VWAP ${vwap:.4f} — discount to fair value"
        else:
            signal = f"Price at VWAP ${vwap:.4f} — fair value"

        # Grade delta — VWAP context for entry quality
        # LONG entry below VWAP = buying at discount = good (+1)
        # LONG entry strongly above VWAP = buying at premium = bad (-1)
        # SHORT entry above VWAP = selling at premium = good (+1)
        # SHORT entry strongly below VWAP = selling at discount = bad (-1)
        grade_delta = 0
        return {
            "vwap":          round(vwap, 6),
            "vwap_dist_pct": round(dist_pct, 2),
            "bias":          bias,
            "strength":      strength,
            "signal":        signal,
            "grade_delta":   grade_delta,
        }

    except Exception as e:
        log.warning(f"VWAP error: {e}")
        return _empty_vwap(price if 'price' in dir() else 0)


def _empty_vwap(price: float) -> dict:
    return {
        "vwap": price, "vwap_dist_pct": 0.0,
        "bias": "AT", "strength": "WEAK",
        "signal": "", "grade_delta": 0,
    }


def vwap_grade_delta(vwap_result: dict, direction: str) -> int:
    """
    Returns grade_delta based on VWAP position relative to trade direction.

    LONG:
      Below VWAP (discount)         +1  — buying cheap, institutional support
      At VWAP                        0  — neutral
      Moderately above VWAP         -1  — paying slight premium
      Strongly above VWAP (>=3%)    -2  — chasing, expensive entry

    SHORT:
      Above VWAP (premium)          +1  — selling expensive, institutional resistance
      At VWAP                        0  — neutral
      Moderately below VWAP         -1  — shorting into discount
      Strongly below VWAP (>=3%)    -2  — chasing downside, oversold
    """
    bias     = vwap_result.get("bias", "AT")
    strength = vwap_result.get("strength", "WEAK")
    dist     = abs(vwap_result.get("vwap_dist_pct", 0))

    if direction == "LONG":
        if bias == "BELOW":
            return +1   # buying at discount
        elif bias == "AT":
            return 0
        elif bias == "ABOVE" and strength == "STRONG":
            return -2   # strongly overextended above VWAP
        elif bias == "ABOVE":
            return -1   # moderately above
    else:  # SHORT
        if bias == "ABOVE":
            return +1   # shorting at premium
        elif bias == "AT":
            return 0
        elif bias == "BELOW" and strength == "STRONG":
            return -2   # chasing downside
        elif bias == "BELOW":
            return -1   # moderately below
    return 0
