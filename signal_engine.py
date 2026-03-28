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
    Full Phase 1 signal quality computation.

    Returns a dict with:
      - quality_score: 0-100
      - direction: LONG | SHORT | NEUTRAL
      - confidence: HIGH | MEDIUM | LOW
      - regime: detected market regime
      - confluence_groups: how many groups agree
      - group_scores: individual group raw scores
      - signals: list of human-readable signal reasons
      - filters: dict of filter results (volatility, candle quality, TF alignment)
      - passed: bool — True if signal should fire
      - reject_reason: why it was rejected (if passed=False)
    """
    result = {
        "quality_score": 0,
        "direction": "NEUTRAL",
        "confidence": "LOW",
        "regime": {},
        "confluence_groups": 0,
        "group_scores": {},
        "signals": [],
        "filters": {},
        "passed": False,
        "reject_reason": "",
    }

    # ── 1. Volatility filter (fast fail) ──────────────────────────────────
    vol_check = volatility_check(df_ltf)
    result["filters"]["volatility"] = vol_check
    if not vol_check["pass"]:
        result["reject_reason"] = vol_check["reason"]
        return result

    # ── 2. Regime detection ───────────────────────────────────────────────
    regime = detect_regime(df_htf)
    result["regime"] = regime

    # ── 3. Score each group independently ────────────────────────────────
    trend_score,      trend_sigs      = score_trend_group(df_ltf, regime)
    momentum_score,   momentum_sigs   = score_momentum_group(df_ltf, regime)
    structure_score,  structure_sigs  = score_structure_group(df_ltf, regime)
    volume_score,     volume_sigs     = score_volume_group(df_ltf)
    oscillator_score, oscillator_sigs = score_oscillator_group(df_ltf)

    group_scores = {
        "trend":      trend_score,
        "momentum":   momentum_score,
        "structure":  structure_score,
        "volume":     volume_score,
        "oscillator": oscillator_score,
    }
    result["group_scores"] = group_scores

    all_signals = trend_sigs + momentum_sigs + structure_sigs + volume_sigs + oscillator_sigs
    result["signals"] = all_signals

    # ── 4. Determine direction by weighted vote ────────────────────────────
    weighted_sum = (
        trend_score      * WEIGHTS["trend"] +
        momentum_score   * WEIGHTS["momentum"] +
        structure_score  * WEIGHTS["structure"] +
        volume_score     * WEIGHTS["volume"] +
        oscillator_score * WEIGHTS["oscillator"]
    )
    max_weight = sum(WEIGHTS.values())  # 100
    normalized = weighted_sum / max_weight  # -1.0 to +1.0

    if normalized > 0.15:
        direction = "LONG"
    elif normalized < -0.15:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"

    result["direction"] = direction

    if direction == "NEUTRAL":
        result["reject_reason"] = "No directional bias — weighted score too weak"
        return result
      
    # ── Regime gate ───────────────────────────────────────────
    regime_name = regime.get("regime", "RANGING")

    if direction == "LONG" and regime_name == "TRENDING_DOWN":
        result["reject_reason"] = "LONG blocked — market is TRENDING_DOWN"
        result["direction"] = "NEUTRAL"
        return result

    if direction == "SHORT" and regime_name == "TRENDING_UP":
        result["reject_reason"] = "SHORT blocked — market is TRENDING_UP"
        result["direction"] = "NEUTRAL"
        return result

    if direction == "SHORT" and regime_name == "TRENDING_DOWN":
        if rsi_val < 70:
            result["reject_reason"] = f"SHORT blocked — RSI {rsi_val:.1f} not overbought (need > 70)"
            result["direction"] = "NEUTRAL"
            return result
        if funding_rate <= 0:
            result["reject_reason"] = f"SHORT blocked — funding rate {funding_rate:.4f} not positive"
            result["direction"] = "NEUTRAL"
            return result

    if direction == "LONG" and regime_name == "RANGING":
        result["reject_reason"] = "Swing LONG blocked in RANGING market — use scalp instead"
        result["direction"] = "NEUTRAL"
        return result

    if direction == "SHORT" and regime_name == "RANGING":
        result["reject_reason"] = "Swing SHORT blocked in RANGING market — use scalp instead"
        result["direction"] = "NEUTRAL"
        return result
  
    # ── 5. Confluence gate ────────────────────────────────────────────────
    threshold = 0.15  # a group must score above this to "agree"
    agreeing_groups = sum(
        1 for score in group_scores.values()
        if (direction == "LONG" and score > threshold) or
           (direction == "SHORT" and score < -threshold)
    )
    result["confluence_groups"] = agreeing_groups

    if agreeing_groups < CONFLUENCE_MIN:
        result["reject_reason"] = f"Confluence too low ({agreeing_groups}/{CONFLUENCE_MIN} groups agree)"
        return result

    # ── 6. Candle quality check ───────────────────────────────────────────
    cq = candle_quality_check(df_ltf, direction)
    result["filters"]["candle_quality"] = cq
    if not cq["pass"]:
        result["reject_reason"] = cq["reason"]
        return result

    # ── 7. Trend alignment filter ─────────────────────────────────────────
    # Get HTF scores for alignment check
    trend_htf, _     = score_trend_group(df_htf, regime)
    momentum_htf, _  = score_momentum_group(df_htf, regime)
    htf_combined     = (trend_htf * WEIGHTS["trend"] + momentum_htf * WEIGHTS["momentum"]) / (WEIGHTS["trend"] + WEIGHTS["momentum"])
    ltf_combined     = (trend_score * WEIGHTS["trend"] + momentum_score * WEIGHTS["momentum"]) / (WEIGHTS["trend"] + WEIGHTS["momentum"])

    alignment = trend_alignment_filter(htf_combined, ltf_combined)
    result["filters"]["tf_alignment"] = alignment
    if not alignment["pass"]:
        result["reject_reason"] = alignment["reason"]
        return result

    # ── 8. Compute final quality score (0-100) ────────────────────────────
    # Base: normalized absolute weighted score → 0-100
    base_score = abs(normalized) * 100

    # Confluence bonus: more agreeing groups = higher quality
    confluence_bonus = (agreeing_groups - CONFLUENCE_MIN) * 5  # +5 per extra group

    # Regime bonus: trending regime signals are higher quality
    regime_bonus = 5 if regime["regime"] != "RANGING" else -5

    # Alignment multiplier
    alignment_mult = alignment.get("multiplier", 1.0)

    quality = (base_score + confluence_bonus + regime_bonus) * alignment_mult
    quality = max(0, min(100, quality))

    result["quality_score"] = round(quality, 1)

    # ── 9. Confidence thresholds ──────────────────────────────────────────
    if quality >= QUALITY_THRESHOLD_HIGH:
        confidence = "HIGH"
    elif quality >= QUALITY_THRESHOLD_MEDIUM:
        confidence = "MEDIUM"
    else:
        result["reject_reason"] = f"Quality score too low ({quality:.1f} < {QUALITY_THRESHOLD_MEDIUM})"
        return result

    result["confidence"] = confidence
    result["passed"]     = True
    return result


# ─── DROP-IN REPLACEMENT for score_timeframe ─────────────────────────────────
def score_timeframe_v2(df: pd.DataFrame, regime: dict = None) -> tuple:
    """
    Drop-in replacement for the original score_timeframe().
    Returns same format: (score, signals, support, resistance, rsi_val, atr_val, price)
    But score is now quality-weighted and regime-aware.
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
    # Scale to legacy integer range (-20 to +20)
    score = int(weighted / 5)

    signals = trend_sigs + mom_sigs + struct_sigs + vol_sigs + osc_sigs

    return score, signals, support, resistance, rsi_val, atr_val, price


# ─── INTEGRATION HELPER ──────────────────────────────────────────────────────
def analyze_v2(symbol: str, fetch_ohlcv_fn, coin_labels: dict) -> dict:
    """
    Full Phase 1 analysis. Replaces the original analyze() function.
    Pass in your existing fetch_ohlcv function.
    """
    df_1h = fetch_ohlcv_fn(symbol, "1h", 200)
    df_4h = fetch_ohlcv_fn(symbol, "4h", 100)

    # Run the full quality engine
    # Get RSI for regime gate
    rsi_series = ta.rsi(df_1h["close"], length=14)
    rsi_current = float(rsi_series.dropna().iloc[-1]) if rsi_series is not None and len(rsi_series.dropna()) > 0 else 50.0

    quality_result = compute_signal_quality(df_1h, df_4h, symbol, funding_rate=0.0, rsi_val=rsi_current)

    # Also compute legacy scores for display compatibility
    regime = quality_result.get("regime") or detect_regime(df_4h)
    score_4h, signals_4h, support, resistance, rsi_4h, atr_4h, price = score_timeframe_v2(df_1h, regime)
    score_1d, signals_1d, _,       _,          rsi_1d, atr_1d, _     = score_timeframe_v2(df_4h, regime)

    direction  = quality_result.get("direction", "NEUTRAL")
    confidence = quality_result.get("confidence", "LOW")
    label      = coin_labels.get(symbol, symbol.split("/")[0])
    
    log.debug("analyze_v2 regime for %s: %s", symbol, regime)
    return {
        "symbol":    symbol,
        "label":     label,
        "direction": direction,
        "confidence": confidence,
        "score":     quality_result.get("quality_score", 0),
        "abs_score": quality_result.get("quality_score", 0),
        "score_4h":  score_4h,
        "score_1d":  score_1d,
        "tf_agree":  quality_result.get("filters", {}).get("tf_alignment", {}).get("pass", False),
        "signals":   quality_result.get("signals", [])[:10],
        "price":     price,
        "support":   support,
        "resistance": resistance,
        "rsi_4h":    rsi_4h,
        "rsi_1d":    rsi_1d,
        "atr":       atr_4h,
        "funding":   0,   # populated by caller
        "df_4h":     df_1h,
        "tf_labels": ("1H", "4H"),
        "regime":    regime,
        "quality":   quality_result,
        "signal_type": "NEUTRAL",   # placeholder — classify_signal() called in run_full_pipeline
        "type_conf":   confidence,
        "type_reason": quality_result.get("reject_reason", "") or
                       f"{quality_result.get('confluence_groups', 0)} groups confluent | Q:{quality_result.get('quality_score', 0)}",
    }
