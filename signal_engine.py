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
    "trend":      30,   # EMA alignment, BOS/CHOCH structural direction
    "momentum":   25,   # RSI, MACD, MBI, Rate of Change
    "structure":  30,   # SMC: BOS/CHOCH, OB retest, liquidity sweep + support/resistance
    "volume":     15,   # Volume confirmation, OBV, volume spikes
    "oscillator":  0,   # Removed — StochRSI/CCI/Williams%R were lagging RSI derivatives
}

# Minimum groups that must agree for a signal to fire (confluence gate)
CONFLUENCE_MIN = 4

# Minimum quality score to emit a signal
QUALITY_THRESHOLD_HIGH   = 65
QUALITY_THRESHOLD_MEDIUM = 50

# ATR multiplier thresholds
VOLATILITY_MAX_ATR_PCT = 8.0   # skip coin if ATR% > 8% (too erratic)
VOLATILITY_MIN_ATR_PCT = 0.3   # skip coin if ATR% < 0.3% (dead/no movement)

# Adaptive threshold state — updated by update_adaptive_thresholds() after each trade
_adaptive_state = {
    "win_count":   0,
    "loss_count":  0,
    "total":       0,
    "last_update": 0.0,
}


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
        adx_df = ta.adx(high, low, close, length=14)
        if adx_df is not None and not adx_df.empty:
            # Use named columns — positional iloc[0/1/2] breaks if pandas_ta changes column order
            adx_col = [c for c in adx_df.columns if c.startswith("ADX_")]
            dmp_col = [c for c in adx_df.columns if c.startswith("DMP_")]
            dmn_col = [c for c in adx_df.columns if c.startswith("DMN_")]
            adx_val = float(adx_df[adx_col[0]].iloc[-1]) if adx_col else 20.0
            dmp     = float(adx_df[dmp_col[0]].iloc[-1]) if dmp_col else 0.0
            dmn     = float(adx_df[dmn_col[0]].iloc[-1]) if dmn_col else 0.0
        else:
            adx_val, dmp, dmn = 20.0, 0.0, 0.0
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

    atr   = ta.atr(high, low, close, length=14)
    atr_v = 0.0
    if atr is not None and not atr.empty:
        _raw = float(atr.iloc[-1])
        atr_v = 0.0 if np.isnan(_raw) else _raw

    # Hard-fail if ATR is 0 or unavailable — can't size risk without it,
    # and NaN used to silently pass both threshold checks (NaN > 8.0 = False).
    if atr_v <= 0 or price <= 0:
        return {"pass": False, "atr_pct": 0.0, "reason": "ATR unavailable — insufficient data"}

    atr_pct = atr_v / price * 100

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

    # ── ZLEMA — primary trend (replaces EMA9/21, less lag) ──────────────────
    try:
        zl = compute_zlema(df, length=70, band_multiplier=1.2)
        total += 4
        if zl["just_flipped"] and zl["bullish"]:
            points += 4; sigs.append("ZLEMA flipped BULLISH — trend reversal ✦")
        elif zl["just_flipped"] and zl["bearish"]:
            points -= 4; sigs.append("ZLEMA flipped BEARISH — trend reversal ✦")
        elif zl["entry_long"]:
            points += 3; sigs.append("ZLEMA pullback entry LONG — price returning to ZLEMA in uptrend ✦")
        elif zl["entry_short"]:
            points -= 3; sigs.append("ZLEMA pullback entry SHORT — price returning to ZLEMA in downtrend ✦")
        elif zl["bullish"]:
            points += 2; sigs.append(f"ZLEMA trend BULLISH (price above band)")
        elif zl["bearish"]:
            points -= 2; sigs.append(f"ZLEMA trend BEARISH (price below band)")
    except Exception as _ze:
        log.warning(f"ZLEMA trend score error: {_ze}")
        # Fallback to EMA9/21
        ema9  = ta.ema(close, length=9)
        ema21 = ta.ema(close, length=21)
        if ema9 is not None and ema21 is not None and len(ema9.dropna()) >= 3:
            e9 = ema9.dropna().values; e21 = ema21.dropna().values
            if len(e9) >= 2 and len(e21) >= 2:
                total += 3
                if   e9[-1] > e21[-1] and e9[-2] <= e21[-2]: points += 3; sigs.append("EMA 9/21 bullish cross ✦")
                elif e9[-1] < e21[-1] and e9[-2] >= e21[-2]: points -= 3; sigs.append("EMA 9/21 bearish cross ✦")
                elif e9[-1] > e21[-1]: points += 1; sigs.append("EMA 9 > EMA 21")
                else: points -= 1; sigs.append("EMA 9 < EMA 21")

    # EMA 50 — medium trend
    ema50 = ta.ema(close, length=50)
    if ema50 is not None and len(ema50.dropna()) >= 2:
        e50 = float(ema50.dropna().iloc[-1])
        total += 2
        if price > e50:
            points += 2; sigs.append(f"Price above EMA50 (${e50:.4f})")
        else:
            points -= 2; sigs.append(f"Price below EMA50 (${e50:.4f})")

    # BOS / CHOCH — replaces MA50/200 golden/death cross.
    # MA cross is lagging by definition (50+200 bars of history averaged).
    # BOS/CHOCH reads structure directly from price — same logic SMC traders use.
    bos = detect_bos_choch(df)
    total += 2
    if bos["choch_bullish"] or bos["bos_bullish"]:
        points += 2
        # CHOCH is a stronger signal than BOS — append the right label
        _bos_tag = "CHOCH" if bos["choch_bullish"] else "BOS"
        sigs.append(f"{_bos_tag} bullish — structural trend UP ✦")
    elif bos["choch_bearish"] or bos["bos_bearish"]:
        points -= 2
        _bos_tag = "CHOCH" if bos["choch_bearish"] else "BOS"
        sigs.append(f"{_bos_tag} bearish — structural trend DOWN ✦")
    else:
        # No active break — use underlying structure as a mild directional bias
        if bos["structure"] == "BULLISH":
            points += 1; sigs.append("Structure: HH+HL (bullish bias)")
        elif bos["structure"] == "BEARISH":
            points -= 1; sigs.append("Structure: LH+LL (bearish bias)")

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
        # Use named columns — positional iloc[0/1/2] breaks if pandas_ta changes order
        _mc = [c for c in macd_df.columns if c.startswith("MACD_")]
        _mh = [c for c in macd_df.columns if c.startswith("MACDh_")]
        _ms = [c for c in macd_df.columns if c.startswith("MACDs_")]
        if _mc and _mh and _ms:
            macd_line = float(macd_df[_mc[0]].iloc[-1])
            sig_line  = float(macd_df[_ms[0]].iloc[-1])
            hist_now  = float(macd_df[_mh[0]].iloc[-1])
            hist_prev = float(macd_df[_mh[0]].iloc[-2])
            macd_prev = float(macd_df[_mc[0]].iloc[-2])
            sig_prev  = float(macd_df[_ms[0]].iloc[-2])
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


    # MBI — momentum acceleration (leading, confirms momentum building)
    try:
        mbi = compute_mbi(df, length=14, signal_length=9)
        total += 3
        if mbi["cross_above"] and mbi["bright_green"]:
            points += 3; sigs.append("MBI crossed above signal — momentum building ✦")
        elif mbi["cross_below"] and mbi["bright_red"]:
            points -= 3; sigs.append("MBI crossed below signal — momentum falling ✦")
        elif mbi["bright_green"]:
            points += 2; sigs.append("MBI bright green — strong bullish momentum")
        elif mbi["bright_red"]:
            points -= 2; sigs.append("MBI bright red — strong bearish momentum")
        elif mbi["bullish"]:
            points += 1; sigs.append("MBI positive")
        elif mbi["bearish"]:
            points -= 1; sigs.append("MBI negative")
    except Exception:
        pass

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

    # Support / Resistance — exclude current candle so breakout check can actually fire.
    # close can never exceed its own candle's high, so including the current candle
    # made `price > resistance_20` permanently False — breakout signal was dead.
    _hist_high    = high.iloc[-21:-1]   # last 20 completed bars, not current
    _hist_low     = low.iloc[-21:-1]
    support_20    = float(_hist_low.min())  if len(_hist_low)  > 0 else float(low.iloc[-1])
    resistance_20 = float(_hist_high.max()) if len(_hist_high) > 0 else float(high.iloc[-1])
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

    # SMC structure score — replaces single-candle patterns (hammer / engulfing etc).
    # Single-candle patterns generate false signals constantly because they have
    # no context: a hammer in the middle of a trend is noise.
    # smc_structure_score() adds BOS/CHOCH + OB retest + liquidity sweep — all
    # three require multi-bar confirmation and structural context to fire.
    smc_s, smc_sigs = smc_structure_score(df, regime)
    smc_pts = int(round(smc_s * 3))   # round() first — int(-2.7)=-2 but round(-2.7)=-3
    points += smc_pts
    total  += 3
    sigs.extend(smc_sigs)

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

    # ── VIDYA — adaptive volume-pressure trend (replaces OBV slope) ─────────
    try:
        vd = compute_vidya(df, length=10, momentum=20)
        total += 3
        if vd["just_flipped"] and vd["bullish"]:
            points += 3; sigs.append("VIDYA flipped bullish + volume delta ✦")
        elif vd["just_flipped"] and vd["bearish"]:
            points -= 3; sigs.append("VIDYA flipped bearish + volume delta ✦")
        elif vd["bullish"] and vd["delta_bullish"]:
            points += 2; sigs.append(f"VIDYA bullish + buy pressure (k={vd['k_value']:.2f})")
        elif vd["bearish"] and not vd["delta_bullish"]:
            points -= 2; sigs.append(f"VIDYA bearish + sell pressure (k={vd['k_value']:.2f})")
        elif vd["bullish"]:
            points += 1; sigs.append("VIDYA bullish (weak volume confirmation)")
        elif vd["bearish"]:
            points -= 1; sigs.append("VIDYA bearish (weak volume confirmation)")
    except Exception as _ve:
        log.warning(f"VIDYA score error: {_ve}")
        # Fallback to OBV
        try:
            obv = ta.obv(close, volume)
            if obv is not None:
                obv_clean = obv.fillna(0)
                if len(obv_clean) >= 10:
                    obv_slope = np.polyfit(range(10), obv_clean.values[-10:], 1)[0]
                    total += 2
                    if obv_slope > 0: points += 2; sigs.append("OBV rising (accumulation)")
                    else: points -= 2; sigs.append("OBV falling (distribution)")
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
    # score_oscillator_group (StochRSI, CCI, Williams %R) permanently removed —
    # all were RSI derivatives, redundant with RSI + MBI, and lagging.
    mom_s,  mom_sigs = score_momentum_group(df_ltf, regime)
    all_signals.extend(mom_sigs)
    mom_combined = mom_s
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
        "volume":    vol_s,
    }
    result["signals"] = all_signals

    # ── Layer 3: Narrative check ─────────────────────────────────────────────
    long_agrees  = votes.count("LONG")
    short_agrees = votes.count("SHORT")
    # True opposition = a group actively voting the opposite direction
    long_opposes  = votes.count("SHORT")  # SHORT votes when direction=LONG
    short_opposes = votes.count("LONG")   # LONG votes when direction=SHORT

    # Direction from majority — 2/3 agree OR 1 strong with no opposition
    if   long_agrees  >= 2: raw_dir = "LONG"
    elif short_agrees >= 2: raw_dir = "SHORT"
    elif long_agrees  == 1 and long_opposes == 0:
        raw_dir = "LONG"   # 1 agree, 2 neutral — weak but valid
    elif short_agrees == 1 and short_opposes == 0:
        raw_dir = "SHORT"  # 1 agree, 2 neutral — weak but valid
    else:
        result["reject_reason"] = f"No narrative consensus — votes: T={trend_vote} M={mom_vote} S={struct_vote}"
        return result

    # Block only on ACTIVE opposition — neutral groups are allowed
    if raw_dir == "LONG"  and long_opposes  > 0:
        result["reject_reason"] = f"Narrative conflict: {long_opposes} group(s) oppose LONG — {votes}"
        return result
    if raw_dir == "SHORT" and short_opposes > 0:
        result["reject_reason"] = f"Narrative conflict: {short_opposes} group(s) oppose SHORT — {votes}"
        return result

    # ── Regime gate — RANGING allowed but penalised, not hard blocked ──────────
    if raw_dir == "LONG"  and regime_name == "TRENDING_DOWN":
        result["reject_reason"] = "LONG blocked — regime TRENDING_DOWN"
        return result
    if raw_dir == "SHORT" and regime_name == "TRENDING_UP":
        result["reject_reason"] = "SHORT blocked — regime TRENDING_UP"
        return result
    # RANGING: allow if at least 2 groups agree (not 1), penalise quality
    if regime_name == "RANGING":
        if long_agrees < 2 and short_agrees < 2:
            result["reject_reason"] = "Blocked — RANGING regime with weak consensus"
            return result
        # else: allow but quality will be lower (no regime bonus)

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

    # ── Quality score ─────────────────────────────────────────────────────────
    # Previous formula used simple average — a signal with trend=0.6, mom=0.1,
    # struct=0.1 scored the same as trend=0.3, mom=0.3, struct=0.3. The first
    # is a weaker setup (two groups barely agreed) but rated identically.
    #
    # New: harmonic mean of AGREEING groups. Harmonic mean punishes imbalance
    # more than arithmetic mean — a weak group (0.05) drags the score down hard.
    # A signal only scores high if ALL agreeing groups are genuinely strong.
    agree_count = long_agrees if raw_dir == "LONG" else short_agrees

    agreeing_strengths = []
    for vote, strength in [
        (trend_vote,  abs(trend_combined)),
        (mom_vote,    abs(mom_combined)),
        (struct_vote, abs(struct_combined)),
    ]:
        if vote == raw_dir:
            agreeing_strengths.append(max(strength, 0.01))   # floor at 0.01 to avoid div/0

    if not agreeing_strengths:
        quality = 0.0
    else:
        n = len(agreeing_strengths)
        # Harmonic mean: n / Σ(1/xᵢ)  — punishes any weak group hard
        harmonic = n / sum(1.0 / s for s in agreeing_strengths)

        # Base: 0–70 from harmonic mean of agreeing groups
        base_score = min(70.0, harmonic * 180.0)

        # Unanimous: all 3 groups agree and are strong
        unanimous_bonus = 20 if n == 3 else (8 if n == 2 else 0)

        # Regime bonus: trending regime is more reliable
        regime_bonus = 5 if regime_name != "RANGING" else 0

        # HTF alignment: 4H confirms the same direction
        htf_align = 5 if (
            (raw_dir == "LONG"  and trend_htf > 0.15) or
            (raw_dir == "SHORT" and trend_htf < -0.15)
        ) else 0

        # SMC structure bonus: OB retest or BOS/CHOCH is a high-confidence signal
        # struct_str > 0.4 when smc_structure_score fired strongly
        smc_bonus = 5 if abs(struct_combined) > 0.4 else 0

        quality = min(100.0, base_score + unanimous_bonus + regime_bonus + htf_align + smc_bonus)

    quality = max(0.0, round(quality, 1))

    result["direction"]         = raw_dir
    result["quality_score"]     = quality
    result["confluence_groups"] = agree_count
    result["filters"]["tf_alignment"] = {
        "pass":       True,
        "multiplier": 1.0,
        "reason":     f"Narrative aligned {agree_count}/3 groups",
    }

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

    weighted = (
        trend_s  * WEIGHTS["trend"] +
        mom_s    * WEIGHTS["momentum"] +
        struct_s * WEIGHTS["structure"] +
        vol_s    * WEIGHTS["volume"]
    )
    score   = int(weighted / 4)
    signals = trend_sigs + mom_sigs + struct_sigs + vol_sigs
    return score, signals, support, resistance, rsi_val, atr_val, price


def analyze_v2(symbol: str, fetch_ohlcv_fn, coin_labels: dict) -> dict:
    """
    Full Phase 1 analysis — called by ai_validator.run_full_pipeline().

    FIXES:
      - "df_4h" key now correctly stores df_4h (was storing df_1h)
      - score_timeframe_v2 variable labels corrected (were inverted)
      - signal_type defaults to "MOMENTUM" (was "REVERSAL" causing silent discard)
      - Fetches df_15m for 15m entry timing
      - Exposes df_1h key explicitly for bot.py grade_signal fix
    """
    df_1h = fetch_ohlcv_fn(symbol, "1h", 200)
    df_4h = fetch_ohlcv_fn(symbol, "4h", 100)

    # Guard: if either primary df is missing, return minimal result
    if df_1h is None or len(df_1h) < 20:
        return {"symbol": symbol, "direction": "NEUTRAL", "confidence": "LOW",
                "score": 0, "abs_score": 0, "df_4h": df_4h, "df_1h": df_1h,
                "label": coin_labels.get(symbol, symbol.split("/")[0]),
                "signal_type": "MOMENTUM", "reject_reason": "Insufficient 1H data"}
    if df_4h is None or len(df_4h) < 20:
        return {"symbol": symbol, "direction": "NEUTRAL", "confidence": "LOW",
                "score": 0, "abs_score": 0, "df_4h": df_4h, "df_1h": df_1h,
                "label": coin_labels.get(symbol, symbol.split("/")[0]),
                "signal_type": "MOMENTUM", "reject_reason": "Insufficient 4H data"}

    # 15m data for entry timing — optional
    df_15m = None
    try:
        df_15m = fetch_ohlcv_fn(symbol, "15m", 100)
    except Exception:
        pass

    rsi_series  = ta.rsi(df_1h["close"], length=14)
    rsi_current = (float(rsi_series.dropna().iloc[-1])
                   if rsi_series is not None and len(rsi_series.dropna()) > 0 else 50.0)

    quality_result = compute_signal_quality(df_1h, df_4h, symbol, funding_rate=0.0, rsi_val=rsi_current)

    regime = quality_result.get("regime") or detect_regime(df_4h)

    # FIX: variable names were INVERTED — score_timeframe_v2(df_1h) produces 1H score
    score_1h, signals_1h, support, resistance, rsi_1h, atr_1h, price = score_timeframe_v2(df_1h, regime)
    score_4h, signals_4h, _,       _,          rsi_4h, atr_4h, _     = score_timeframe_v2(df_4h, regime)

    direction  = quality_result.get("direction", "NEUTRAL")
    confidence = quality_result.get("confidence", "LOW")
    label      = coin_labels.get(symbol, symbol.split("/")[0])

    # 15m entry context
    rsi_15m      = 50.0
    entry_timing = "NEUTRAL"
    if df_15m is not None and len(df_15m) >= 20:
        try:
            rsi_15m_s = ta.rsi(df_15m["close"], length=14)
            if rsi_15m_s is not None and len(rsi_15m_s.dropna()) > 0:
                rsi_15m = float(rsi_15m_s.dropna().iloc[-1])
                if rsi_15m > 70:
                    entry_timing = "OVERBOUGHT_15M"
                elif rsi_15m < 30:
                    entry_timing = "OVERSOLD_15M"
                else:
                    entry_timing = "NEUTRAL_15M"
        except Exception:
            pass

    return {
        "symbol":       symbol,
        "label":        label,
        "direction":    direction,
        "confidence":   confidence,
        "score":        quality_result.get("quality_score", 0),
        "abs_score":    quality_result.get("quality_score", 0),
        "score_1h":     score_1h,
        "score_4h":     score_4h,
        # Legacy keys kept for bot.py compatibility
        "score_4h_raw": score_1h,
        "score_1d":     score_4h,
        "tf_agree":     quality_result.get("filters", {}).get(
                            "tf_alignment", {}).get("pass", False),
        "signals":      quality_result.get("signals", [])[:10],
        "price":        price,
        "support":      support,
        "resistance":   resistance,
        "rsi_1h":       rsi_1h,
        "rsi_4h":       rsi_4h,        # FIX: now correctly labelled (was 1H stored as 4H)
        "rsi_15m":      rsi_15m,
        "entry_timing": entry_timing,
        "atr":          atr_4h,
        "funding":      0,
        "df_4h":        df_4h,         # FIX: was df_1h — corrupted all downstream grading
        "df_1h":        df_1h,         # NEW: expose 1H df for bot.py grade_signal
        "df_15m":       df_15m,        # NEW: 15m data for entry refinement
        "tf_labels":    ("1H", "4H"),
        "regime":       regime,
        "quality":      quality_result,
        "signal_type":  "MOMENTUM",    # FIX: was "REVERSAL" — silently discarded by get_actionable()
        "type_conf":    confidence,
        "type_reason":  (quality_result.get("reject_reason", "")
                         or f"{quality_result.get('confluence_groups', 0)} groups confluent"
                            f" | Q:{quality_result.get('quality_score', 0)}"),
    }


# ─── ZERO LAG EMA (ZLEMA) ────────────────────────────────────────────────────
def compute_zlema(df: pd.DataFrame, length: int = 70, band_multiplier: float = 1.2) -> dict:
    """
    AlgoAlpha Zero Lag Trend Signals — confirmed formula:
      lag   = floor((length-1)/2)
      zlema = EMA(price + (price - price[lag]), length)
      volatility = highest(ATR, length*3) * multiplier
      trend flips BULLISH when close > zlema + volatility
      trend flips BEARISH when close < zlema - volatility
      entry signal = close crosses back through zlema while in trend
    band_multiplier: 0.2=scalp, 1.2=swing
    """
    try:
        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        n     = len(close)
        if n < length + 10:
            return _empty_zlema(float(close[-1]))

        lag = int((length - 1) / 2)
        zl_price = np.array([close[i] + (close[i] - close[i-lag]) if i >= lag else close[i] for i in range(n)])

        alpha = 2.0 / (length + 1)
        zlema_arr = np.zeros(n)
        zlema_arr[0] = zl_price[0]
        for i in range(1, n):
            zlema_arr[i] = alpha * zl_price[i] + (1 - alpha) * zlema_arr[i-1]

        # ATR smoothed
        atr_arr = np.zeros(n)
        for i in range(1, n):
            atr_arr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
        atr_s = np.zeros(n)
        atr_s[0] = atr_arr[0]
        for i in range(1, n):
            atr_s[i] = alpha * atr_arr[i] + (1-alpha) * atr_s[i-1]

        # Volatility band = highest ATR over length*3
        lb = min(length*3, n)
        vol_arr = np.array([np.max(atr_s[max(0,i-lb):i+1]) * band_multiplier for i in range(n)])

        # Trend state machine
        trend = np.zeros(n, dtype=int)
        for i in range(1, n):
            prev = trend[i-1] if trend[i-1] != 0 else 1
            if   close[i] > zlema_arr[i] + vol_arr[i]: trend[i] =  1
            elif close[i] < zlema_arr[i] - vol_arr[i]: trend[i] = -1
            else: trend[i] = prev

        cur_z   = float(zlema_arr[-1])
        cur_vol = float(vol_arr[-1])
        cur_t   = int(trend[-1])
        prev_t  = int(trend[-2]) if n >= 2 else cur_t
        flip    = cur_t != prev_t
        e_long  = close[-1] > cur_z and close[-2] <= zlema_arr[-2] and cur_t == 1 and prev_t == 1
        e_short = close[-1] < cur_z and close[-2] >= zlema_arr[-2] and cur_t == -1 and prev_t == -1

        return {"zlema": round(cur_z,6), "trend": cur_t, "bullish": cur_t==1, "bearish": cur_t==-1,
                "just_flipped": flip, "entry_long": e_long, "entry_short": e_short,
                "upper_band": round(cur_z+cur_vol,6), "lower_band": round(cur_z-cur_vol,6),
                "volatility": round(cur_vol,6)}
    except Exception as e:
        log.warning(f"ZLEMA error: {e}")
        return _empty_zlema(float(df["close"].iloc[-1]))

def _empty_zlema(price: float) -> dict:
    """Neutral default — do NOT assume bullish on failure."""
    return {"zlema": price, "trend": 0, "bullish": False, "bearish": False,
            "just_flipped": False, "entry_long": False, "entry_short": False,
            "upper_band": price, "lower_band": price, "volatility": 0.0}


# ─── VOLUMATIC VIDYA ─────────────────────────────────────────────────────────
def compute_vidya(df: pd.DataFrame, length: int = 10, momentum: int = 20) -> dict:
    """
    Variable Index Dynamic Average with volume pressure delta.
    Confirmed Tushar Chande formula:
      alpha = 2/(length+1)
      k     = |CMO|/100  (CMO = Chande Momentum Oscillator)
      VIDYA = alpha*k*price + (1 - alpha*k)*VIDYA_prev
    Volume delta = cumulative buy-sell pressure over last 10 candles.
    """
    try:
        close  = df["close"].values
        high   = df["high"].values
        low    = df["low"].values
        volume = df["volume"].values
        n      = len(close)
        if n < momentum + 5:
            return _empty_vidya(float(close[-1]))

        gains  = np.maximum(np.diff(close), 0)
        losses = np.maximum(-np.diff(close), 0)
        cmo    = np.zeros(n)
        for i in range(momentum, n):
            g = np.sum(gains[i-momentum:i])
            l = np.sum(losses[i-momentum:i])
            cmo[i] = (g-l)/(g+l)*100 if (g+l) > 0 else 0

        alpha     = 2.0 / (length + 1)
        vidya_arr = np.zeros(n)
        vidya_arr[momentum] = close[momentum]
        for i in range(momentum+1, n):
            k = abs(cmo[i]) / 100.0
            vidya_arr[i] = alpha*k*close[i] + (1-alpha*k)*vidya_arr[i-1]

        cur_v  = float(vidya_arr[-1])
        cur_t  = 1 if close[-1] >= cur_v else -1
        prev_t = 1 if close[-2] >= vidya_arr[-2] else -1
        flip   = cur_t != prev_t

        # Volume delta — buy vs sell pressure over last 10 candles
        vol_delta = 0.0
        for i in range(max(1, n-10), n):
            d = 1 if close[i] >= close[i-1] else -1
            vol_delta += volume[i] * d

        return {"vidya": round(cur_v,6), "trend": cur_t, "bullish": cur_t==1, "bearish": cur_t==-1,
                "just_flipped": flip, "volume_delta": round(vol_delta,2),
                "delta_bullish": vol_delta > 0, "k_value": round(abs(cmo[-1])/100.0, 3)}
    except Exception as e:
        log.warning(f"VIDYA error: {e}")
        return _empty_vidya(float(df["close"].iloc[-1]))

def _empty_vidya(price: float) -> dict:
    """Neutral default — do NOT assume bullish on failure."""
    return {"vidya": price, "trend": 0, "bullish": False, "bearish": False,
            "just_flipped": False, "volume_delta": 0.0, "delta_bullish": False, "k_value": 0.0}


# ─── MOMENTUM BIAS INDEX (MBI) ────────────────────────────────────────────────
def compute_mbi(df: pd.DataFrame, length: int = 14, signal_length: int = 9) -> dict:
    """
    Momentum Bias Index — normalized momentum histogram with dynamic signal line.
    Signal fires when histogram crosses above/below its own EMA (signal line)
    AND histogram is on the correct side of zero (positive for BUY, negative for SELL).
    Bright green = strong building bullish momentum.
    Bright red   = strong building bearish momentum.
    """
    try:
        close = df["close"]
        high  = df["high"]
        low   = df["low"]
        n     = len(close)
        if n < length + signal_length + 5:
            return _empty_mbi()

        ema_v  = ta.ema(close, length=length)
        atr_v  = ta.atr(high, low, close, length=length)
        if ema_v is None or atr_v is None:
            return _empty_mbi()

        atr_clean = atr_v.bfill().replace(0, np.nan).ffill().bfill()  # pandas 2.x: method kwarg removed
        mom_norm  = ((close - ema_v) / atr_clean * 100).fillna(0)
        signal_v  = ta.ema(mom_norm, length=signal_length)
        if signal_v is None:
            return _empty_mbi()

        hist      = mom_norm - signal_v.bfill()
        cur_hist  = float(hist.iloc[-1])
        prev_hist = float(hist.iloc[-2]) if n >= 2 else cur_hist
        cur_mom   = float(mom_norm.iloc[-1])

        bright_green = cur_hist > 0 and cur_mom > 0
        bright_red   = cur_hist < 0 and cur_mom < 0

        return {"histogram": round(cur_hist,4), "signal": round(float(signal_v.iloc[-1]),4),
                "momentum": round(cur_mom,4), "bullish": bright_green, "bearish": bright_red,
                "cross_above": cur_hist > 0 and prev_hist <= 0,
                "cross_below": cur_hist < 0 and prev_hist >= 0,
                "bright_green": bright_green, "bright_red": bright_red}
    except Exception as e:
        log.warning(f"MBI error: {e}")
        return _empty_mbi()

def _empty_mbi() -> dict:
    return {"histogram": 0.0, "signal": 0.0, "momentum": 0.0,
            "bullish": False, "bearish": False, "cross_above": False,
            "cross_below": False, "bright_green": False, "bright_red": False}


# ═══════════════════════════════════════════════════════════════════════════════
# ─── SMC ENGINE — BOS/CHOCH · ORDER BLOCKS · LIQUIDITY POOLS ─────────────────
# ═══════════════════════════════════════════════════════════════════════════════

def _find_swing_points(
    df: pd.DataFrame,
    left: int     = 2,
    right: int    = 2,
    lookback: int = 60,
) -> tuple[list, list]:
    """
    Pivot-based swing detector.  A swing high at bar i requires high[i] to be
    strictly greater than the `left` bars before it AND the `right` bars after.
    Bars within `right` of the current bar are excluded — they haven't closed
    their right-side window yet and would generate false signals.

    Returns:
        (swing_highs, swing_lows) — each a list of (bar_index, price) tuples,
        ordered oldest → newest.
    """
    high  = df["high"].values
    low   = df["low"].values
    n     = len(high)
    start = max(left, n - lookback - right)

    swing_highs: list = []
    swing_lows:  list = []

    for i in range(start, n - right):
        h_l = high[i - left : i]
        h_r = high[i + 1   : i + right + 1]
        l_l = low[i  - left : i]
        l_r = low[i  + 1   : i + right + 1]

        if (len(h_l) == left and len(h_r) == right
                and high[i] > np.max(h_l) and high[i] > np.max(h_r)):
            swing_highs.append((i, float(high[i])))

        if (len(l_l) == left and len(l_r) == right
                and low[i] < np.min(l_l) and low[i] < np.min(l_r)):
            swing_lows.append((i, float(low[i])))

    return swing_highs, swing_lows


def detect_bos_choch(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Break of Structure (BOS) and Change of Character (CHOCH).

    Structure is determined from the last two confirmed swing highs and lows:
      HH + HL  → BULLISH   |   LH + LL  → BEARISH   |   else → NEUTRAL

    BOS:   current close breaks a swing level in the SAME direction as the
           existing structure (trend continuation — smart money adding).
    CHOCH: current close breaks a swing level AGAINST the existing structure
           (first sign of reversal — smart money flipping).

    CHOCH outranks BOS in the scoring engine because it signals the higher
    probability entry: at the START of a new move, not mid-trend.

    Returns:
    {
        "bos_bullish":    bool,   broke above prev swing high (continuation)
        "bos_bearish":    bool,   broke below prev swing low  (continuation)
        "choch_bullish":  bool,   bullish break in BEARISH structure (reversal)
        "choch_bearish":  bool,   bearish break in BULLISH structure (reversal)
        "structure":      str,    "BULLISH" | "BEARISH" | "NEUTRAL"
        "last_bos_level": float,  the level that was broken (0.0 = none)
        "swing_high_lvl": float,  most recent confirmed swing high
        "swing_low_lvl":  float,  most recent confirmed swing low
    }
    """
    EMPTY = {
        "bos_bullish": False, "bos_bearish": False,
        "choch_bullish": False, "choch_bearish": False,
        "structure": "NEUTRAL", "last_bos_level": 0.0,
        "swing_high_lvl": 0.0, "swing_low_lvl": 0.0,
    }
    try:
        if len(df) < lookback:
            return EMPTY

        sh, sl = _find_swing_points(df, left=2, right=2, lookback=lookback)
        if len(sh) < 2 or len(sl) < 2:
            return EMPTY

        n         = len(df)
        cur_close = float(df["close"].iloc[-1])

        # Exclude bars still within the right-side window of the pivot detector
        sh_confirmed = [(i, p) for i, p in sh if i < n - 2]
        sl_confirmed = [(i, p) for i, p in sl if i < n - 2]

        if not sh_confirmed or not sl_confirmed:
            return EMPTY

        # Most recent confirmed swing levels
        recent_sh = sh_confirmed[-1][1]
        recent_sl = sl_confirmed[-1][1]

        # ── Structure from last 2 swing highs + 2 swing lows ─────────────────
        structure = "NEUTRAL"
        if len(sh_confirmed) >= 2 and len(sl_confirmed) >= 2:
            hh = sh_confirmed[-1][1] > sh_confirmed[-2][1]
            hl = sl_confirmed[-1][1] > sl_confirmed[-2][1]
            lh = sh_confirmed[-1][1] < sh_confirmed[-2][1]
            ll = sl_confirmed[-1][1] < sl_confirmed[-2][1]
            if hh and hl:
                structure = "BULLISH"
            elif lh and ll:
                structure = "BEARISH"

        # ── Classify the break ───────────────────────────────────────────────
        bos_bullish = bos_bearish = choch_bullish = choch_bearish = False
        last_bos_level = 0.0

        if cur_close > recent_sh:
            last_bos_level = recent_sh
            if structure == "BEARISH":
                choch_bullish = True   # first bullish break in downtrend = reversal
            else:
                bos_bullish   = True   # continuation of uptrend / neutral

        elif cur_close < recent_sl:
            last_bos_level = recent_sl
            if structure == "BULLISH":
                choch_bearish = True   # first bearish break in uptrend = reversal
            else:
                bos_bearish   = True   # continuation of downtrend / neutral

        return {
            "bos_bullish":    bos_bullish,
            "bos_bearish":    bos_bearish,
            "choch_bullish":  choch_bullish,
            "choch_bearish":  choch_bearish,
            "structure":      structure,
            "last_bos_level": round(last_bos_level, 8),
            "swing_high_lvl": round(recent_sh, 8),
            "swing_low_lvl":  round(recent_sl, 8),
        }

    except Exception as e:
        log.warning(f"detect_bos_choch error: {e}")
        return EMPTY


def detect_order_blocks(df: pd.DataFrame, lookback: int = 40) -> dict:
    """
    Detects the most recent active Demand and Supply Order Blocks.

    Demand OB (long entry zone):
      The last BEARISH candle immediately before a bullish impulse move.
      Zone = that candle's full wick range (low → high).

    Supply OB (short entry zone):
      The last BULLISH candle immediately before a bearish impulse move.
      Zone = that candle's full wick range (low → high).

    Impulse threshold: single candle body > 1.0 × ATR14.
    Active:  price has NOT closed through the OB body since it formed.
             (demand OB: no close below ob_low | supply OB: no close above ob_high)
    At zone: current price is within the OB range ± 0.5% tolerance.

    Scanning direction: backward from current bar so we always find the
    MOST RECENT valid OB rather than the oldest one.

    Returns:
    {
        "demand_ob":          {"high", "low", "active"} | None,
        "supply_ob":          {"high", "low", "active"} | None,
        "price_at_demand_ob": bool,
        "price_at_supply_ob": bool,
    }
    """
    EMPTY = {"demand_ob": None, "supply_ob": None,
             "price_at_demand_ob": False, "price_at_supply_ob": False}
    try:
        n = len(df)
        if n < lookback + 5:
            return EMPTY

        close  = df["close"].values
        open_  = df["open"].values
        high   = df["high"].values
        low    = df["low"].values

        # ATR14 — impulse size threshold
        tr_arr = np.array([
            max(high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i]  - close[i - 1]))
            for i in range(1, n)
        ])
        atr14 = float(np.mean(tr_arr[-14:])) if len(tr_arr) >= 14 else float(np.mean(tr_arr))
        if atr14 <= 0:
            return EMPTY

        start     = max(3, n - lookback)
        demand_ob = None
        supply_ob = None

        # ── Scan backward — find most recent active OBs ───────────────────────
        for i in range(n - 2, start, -1):
            body = close[i] - open_[i]   # + = bullish, - = bearish

            # ── Demand OB: bullish impulse candle found ───────────────────────
            if demand_ob is None and body > atr14 * 1.0:
                # Walk back up to 4 bars to find the last bearish candle
                for j in range(i - 1, max(start, i - 5), -1):
                    if close[j] < open_[j]:            # bearish = OB candidate
                        ob_low  = float(low[j])
                        ob_high = float(high[j])
                        # Active: no close has gone below the OB low after formation
                        future = close[j + 1 : n]
                        if not np.any(future < ob_low):
                            demand_ob = {"high": round(ob_high, 8),
                                         "low":  round(ob_low,  8),
                                         "active": True}
                        break   # whether active or not, this is the OB candle — stop

            # ── Supply OB: bearish impulse candle found ───────────────────────
            if supply_ob is None and body < -atr14 * 1.0:
                for j in range(i - 1, max(start, i - 5), -1):
                    if close[j] > open_[j]:            # bullish = OB candidate
                        ob_low  = float(low[j])
                        ob_high = float(high[j])
                        # Active: no close has gone above the OB high after formation
                        future = close[j + 1 : n]
                        if not np.any(future > ob_high):
                            supply_ob = {"high": round(ob_high, 8),
                                         "low":  round(ob_low,  8),
                                         "active": True}
                        break

            if demand_ob and supply_ob:
                break

        price = float(close[-1])
        tol   = 0.005   # ±0.5% tolerance for "at zone"

        price_at_demand = (
            demand_ob is not None and demand_ob["active"]
            and demand_ob["low"] * (1 - tol) <= price <= demand_ob["high"] * (1 + tol)
        )
        price_at_supply = (
            supply_ob is not None and supply_ob["active"]
            and supply_ob["low"] * (1 - tol) <= price <= supply_ob["high"] * (1 + tol)
        )

        return {
            "demand_ob":          demand_ob,
            "supply_ob":          supply_ob,
            "price_at_demand_ob": price_at_demand,
            "price_at_supply_ob": price_at_supply,
        }

    except Exception as e:
        log.warning(f"detect_order_blocks error: {e}")
        return EMPTY


def detect_liquidity_pools(
    df: pd.DataFrame,
    lookback: int       = 60,
    tolerance_pct: float = 0.3,
) -> dict:
    """
    Identifies buy-side and sell-side liquidity pools.

    Buy-side  (BSL): 2+ swing highs clustered at the same price level.
                     Stop losses from short sellers rest just above these.
                     Price being drawn toward BSL = potential short entry target.
    Sell-side (SSL): 2+ swing lows clustered at the same price level.
                     Stop losses from long holders rest just below these.
                     Price being drawn toward SSL = potential long entry target.

    Sweep detection (last 4 bars):
      sweep_bullish — a wick pierced below SSL but price CLOSED back above it.
                      Interpretation: stops swept, reversal long setup confirmed.
      sweep_bearish — a wick pierced above BSL but price CLOSED back below it.
                      Interpretation: stops swept, reversal short setup confirmed.

    Returns:
    {
        "buy_side_liq":      bool,
        "sell_side_liq":     bool,
        "equal_highs_level": float,   price of the BSL cluster (0.0 if none)
        "equal_lows_level":  float,   price of the SSL cluster (0.0 if none)
        "sweep_bullish":     bool,    SSL swept → long reversal setup
        "sweep_bearish":     bool,    BSL swept → short reversal setup
    }
    """
    EMPTY = {
        "buy_side_liq": False, "sell_side_liq": False,
        "equal_highs_level": 0.0, "equal_lows_level": 0.0,
        "sweep_bullish": False, "sweep_bearish": False,
    }
    try:
        n = len(df)
        if n < 20:
            return EMPTY

        high  = df["high"].values
        low   = df["low"].values
        close = df["close"].values
        tol   = tolerance_pct / 100.0

        sh, sl = _find_swing_points(df, left=2, right=2, lookback=lookback)
        sh_prices = [p for _, p in sh]
        sl_prices = [p for _, p in sl]

        # ── Find equal-highs cluster (buy-side liquidity) ─────────────────────
        # Take the tightest cluster of recent highs to find the most relevant pool
        bsl_level = 0.0
        for ref in sorted(sh_prices, reverse=True):
            cluster = [p for p in sh_prices if ref > 0 and abs(p - ref) / ref < tol]
            if len(cluster) >= 2:
                bsl_level = float(np.mean(cluster))
                break

        # ── Find equal-lows cluster (sell-side liquidity) ─────────────────────
        ssl_level = 0.0
        for ref in sorted(sl_prices):
            cluster = [p for p in sl_prices if ref > 0 and abs(p - ref) / ref < tol]
            if len(cluster) >= 2:
                ssl_level = float(np.mean(cluster))
                break

        price = float(close[-1])

        # ── Sweep detection (last 4 candles' wicks vs pool level) ────────────
        sweep_bullish = False
        sweep_bearish = False

        if ssl_level > 0 and n >= 4:
            # Wick went below SSL in the last 4 bars but close is now above it
            if np.any(low[-4:] < ssl_level * (1 - tol)) and price > ssl_level:
                sweep_bullish = True

        if bsl_level > 0 and n >= 4:
            # Wick went above BSL in the last 4 bars but close is now below it
            if np.any(high[-4:] > bsl_level * (1 + tol)) and price < bsl_level:
                sweep_bearish = True

        return {
            "buy_side_liq":      bsl_level > 0,
            "sell_side_liq":     ssl_level > 0,
            "equal_highs_level": round(bsl_level, 8),
            "equal_lows_level":  round(ssl_level, 8),
            "sweep_bullish":     sweep_bullish,
            "sweep_bearish":     sweep_bearish,
        }

    except Exception as e:
        log.warning(f"detect_liquidity_pools error: {e}")
        return EMPTY


def smc_structure_score(df: pd.DataFrame, regime: dict) -> tuple[float, list[str]]:
    """
    SMC-based structure score — replaces single-candle pattern detection
    (hammer / shooting star / engulfing) which generated constant false signals.

    Combines three SMC detectors into one coherent structure vote:

      1. BOS / CHOCH  — structural confirmation or reversal (weight: 4 pts)
         CHOCH > BOS: a reversal at the start of a new move is higher precision
         than a mid-trend BOS signal.

      2. Order Block retest — is price retesting a demand/supply OB? (weight: 3 pts)
         The OB retest is the actual ENTRY trigger in SMC. Price may BOS and come
         back to retest the OB before continuing — that retest is the entry.

      3. Liquidity sweep — did price sweep stops before reversing? (weight: 3 pts)
         A sweep of sell-side liquidity followed by a close back above = classic
         SMC long setup. Buy-side sweep = classic short setup.

    Returns (raw_score -1.0 to +1.0, signals[])
    """
    points = 0
    total  = 0
    sigs   = []

    try:
        bos  = detect_bos_choch(df)
        ob   = detect_order_blocks(df)
        liq  = detect_liquidity_pools(df)
        price = float(df["close"].iloc[-1])

        # ── 1. BOS / CHOCH (4 pts max) ────────────────────────────────────────
        total += 4
        if bos["choch_bullish"]:
            points += 4
            sigs.append(
                f"CHOCH bullish — structure flipped above "
                f"${bos['last_bos_level']:.4f} ✦"
            )
        elif bos["bos_bullish"]:
            points += 3
            sigs.append(
                f"BOS bullish — broke above swing high "
                f"${bos['last_bos_level']:.4f} ✦"
            )
        elif bos["choch_bearish"]:
            points -= 4
            sigs.append(
                f"CHOCH bearish — structure flipped below "
                f"${bos['last_bos_level']:.4f} ✦"
            )
        elif bos["bos_bearish"]:
            points -= 3
            sigs.append(
                f"BOS bearish — broke below swing low "
                f"${bos['last_bos_level']:.4f} ✦"
            )
        else:
            # No active BOS/CHOCH — provide structure context as a mild bias
            if bos["structure"] == "BULLISH":
                points += 1
                sigs.append(
                    f"Structure BULLISH (HH+HL — "
                    f"high ${bos['swing_high_lvl']:.4f} | "
                    f"low ${bos['swing_low_lvl']:.4f})"
                )
            elif bos["structure"] == "BEARISH":
                points -= 1
                sigs.append(
                    f"Structure BEARISH (LH+LL — "
                    f"high ${bos['swing_high_lvl']:.4f} | "
                    f"low ${bos['swing_low_lvl']:.4f})"
                )

        # ── 2. Order Block retest (3 pts max) ─────────────────────────────────
        total += 3
        if ob["price_at_demand_ob"] and ob["demand_ob"]:
            points += 3
            sigs.append(
                f"Price retesting demand OB "
                f"${ob['demand_ob']['low']:.4f}–"
                f"${ob['demand_ob']['high']:.4f} — "
                f"entry zone ✦"
            )
        elif ob["price_at_supply_ob"] and ob["supply_ob"]:
            points -= 3
            sigs.append(
                f"Price retesting supply OB "
                f"${ob['supply_ob']['low']:.4f}–"
                f"${ob['supply_ob']['high']:.4f} — "
                f"short zone ✦"
            )
        elif ob["demand_ob"] and ob["demand_ob"]["active"]:
            dist = (price - ob["demand_ob"]["high"]) / price * 100
            sigs.append(
                f"Demand OB below: "
                f"${ob['demand_ob']['high']:.4f} ({dist:.1f}% away)"
            )
        elif ob["supply_ob"] and ob["supply_ob"]["active"]:
            dist = (ob["supply_ob"]["low"] - price) / price * 100
            sigs.append(
                f"Supply OB above: "
                f"${ob['supply_ob']['low']:.4f} ({dist:.1f}% away)"
            )

        # ── 3. Liquidity sweep (3 pts max) ────────────────────────────────────
        total += 3
        if liq["sweep_bullish"]:
            points += 3
            sigs.append(
                f"Sell-side liquidity swept at "
                f"${liq['equal_lows_level']:.4f} — "
                f"stop hunt complete, long setup ✦"
            )
        elif liq["sweep_bearish"]:
            points -= 3
            sigs.append(
                f"Buy-side liquidity swept at "
                f"${liq['equal_highs_level']:.4f} — "
                f"stop hunt complete, short setup ✦"
            )
        elif liq["buy_side_liq"] and liq["equal_highs_level"] > price:
            points -= 1
            sigs.append(
                f"Buy-side liquidity at "
                f"${liq['equal_highs_level']:.4f} — "
                f"stops above, potential magnet / target"
            )
        elif liq["sell_side_liq"] and liq["equal_lows_level"] < price and liq["equal_lows_level"] > 0:
            points += 1
            sigs.append(
                f"Sell-side liquidity at "
                f"${liq['equal_lows_level']:.4f} — "
                f"stops below, potential magnet / target"
            )

        raw = points / total if total > 0 else 0.0
        return max(-1.0, min(1.0, raw)), sigs

    except Exception as e:
        log.warning(f"smc_structure_score error: {e}")
        return 0.0, []


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

    # Funding thresholds — calibrated to Bitget per-8H decimal rates
    # Bitget normal: ~0.0001 (0.01%), elevated: 0.001 (0.1%), extreme: >0.005 (0.5%)
    FUNDING_VERY_NEGATIVE = -0.0008  # -0.08%/8H — shorts crowded / squeeze risk
    FUNDING_NEGATIVE      = -0.0002  # -0.02%/8H — mildly negative
    FUNDING_POSITIVE      = 0.0005   # +0.05%/8H — longs starting to crowd
    FUNDING_VERY_POSITIVE = 0.0015   # +0.15%/8H — overcrowded longs, flush risk HIGH

    # OI change (if available)
    oi_rising   = False
    oi_falling  = False
    oi_change_pct = 0.0
    if oi_now is not None and oi_prev is not None and oi_prev > 0:
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


# ─── MOMENTUM BIAS INDEX (duplicate removed) ────────────────────────────────

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
        # Clamp lookback to available bars — iloc[-24] assumed 1H candles and
        # raised IndexError / returned 0 silently when called with 15m data.
        change_24h   = 0.0
        lookback_24h = min(24, len(close) - 1)
        if lookback_24h > 0:
            price_24h_ago = float(close.iloc[-lookback_24h - 1])
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
        if funding_rate >= 0.0008:   # FIX: recalibrated from 0.03 (Bitget scale)
            short_signals += 1
            reasons.append(f"Funding {funding_rate*100:.3f}% — longs crowded")
        if funding_rate >= 0.0015:   # FIX: recalibrated from 0.05
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
    """Neutral default — do NOT assume bullish on failure."""
    return {"value": price, "direction": 0, "bullish": False,
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


# ─── ACCUMULATION SETUP DETECTOR ─────────────────────────────────────────────
def detect_accumulation_setup(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    symbol: str = "",
) -> dict:
    """
    Detects coins in early accumulation BEFORE they pump (pre-gainer scanner).

    Catches the setup that precedes the move rather than reacting after the
    coin is already up 15%+.

    Signals (need score >= 3):
      1. OBV rising while price flat/down (smart money absorbing) — weight 2
      2. Volume contracted below 65% of 20-period average
      3. Bollinger Band squeeze (volatility compressed)
      4. Price near 4H support (< 3% away)
      5. Tight 20-candle range (< 8% total range)
      6. ATR declining (volatility drying up)
      7. RSI 40-55 (accumulation zone — not extended)

    Returns:
    {
      "is_accumulating": bool,
      "score":           int,
      "confidence":      str,   # HIGH (5+) / MEDIUM (3-4) / LOW (<3)
      "signals":         list,
      "entry_zone":      float,
      "invalidation":    float,
    }
    """
    try:
        if df_1h is None or not isinstance(df_1h, pd.DataFrame) or len(df_1h) < 20:
            return {"is_accumulating": False, "score": 0, "confidence": "LOW",
                    "signals": [], "entry_zone": 0, "invalidation": 0}
        close  = df_1h["close"]
        high   = df_1h["high"]
        low    = df_1h["low"]
        volume = df_1h["volume"]
        price  = float(close.iloc[-1])

        score   = 0
        signals = []

        # 1. OBV rising, price flat — smart money absorbing (double weight)
        obv = ta.obv(close, volume)
        if obv is not None and len(obv) >= 20:
            ov = obv.fillna(0).values[-20:]
            pv = close.values[-20:]
            obv_slope   = np.polyfit(range(20), ov, 1)[0]
            price_slope = np.polyfit(range(20), pv, 1)[0]
            if obv_slope > 0 and price_slope <= price * 0.001:
                score += 2
                signals.append("OBV rising while price flat — smart money absorbing ✦")

        # 2. Volume contracting
        if len(volume) >= 20:
            vol_ma = float(volume.rolling(20).mean().iloc[-1])
            vol_5  = float(volume.tail(5).mean())
            if vol_5 < vol_ma * 0.65:
                score += 1
                signals.append(f"Volume at {vol_5/vol_ma:.0%} of average — quiet base forming")

        # 3. Bollinger Band squeeze
        bb = ta.bbands(close, length=20, std=2)
        if bb is not None and not bb.empty and len(bb) >= 10:
            try:
                bw_now  = float(bb.iloc[-1, 3])
                bw_prev = float(bb.iloc[-10, 3])
                if bw_now < bw_prev * 0.75:
                    score += 1
                    signals.append("BB squeeze — volatility compressed (coiled spring) ✦")
            except Exception:
                pass

        # 4. Price near 4H support
        if df_4h is not None and len(df_4h) >= 20:
            support_4h = float(df_4h["low"].tail(20).min())
            dist_to_support = abs(price - support_4h) / price * 100
            if dist_to_support < 3.0:
                score += 1
                signals.append(f"Near 4H support ${support_4h:.4f} ({dist_to_support:.1f}% away)")
        else:
            support_4h = float(low.tail(20).min())

        # 5. Tight range — base forming
        range_20 = (float(high.tail(20).max()) - float(low.tail(20).min())) / price * 100
        if range_20 < 8.0:
            score += 1
            signals.append(f"Tight 20-candle range {range_20:.1f}% — base forming")

        # 6. ATR declining
        atr_s = ta.atr(high, low, close, length=14)
        if atr_s is not None and len(atr_s.dropna()) >= 8:
            atr_now  = float(atr_s.dropna().iloc[-1])
            atr_prev = float(atr_s.dropna().iloc[-5])
            if atr_now < atr_prev * 0.80:
                score += 1
                signals.append("ATR declining — volatility drying up before expansion")

        # 7. RSI in accumulation zone
        rsi_s = ta.rsi(close, length=14)
        if rsi_s is not None and len(rsi_s.dropna()) > 0:
            rsi_v = float(rsi_s.dropna().iloc[-1])
            if 40 <= rsi_v <= 55:
                score += 1
                signals.append(f"RSI {rsi_v:.0f} — accumulation zone (not extended)")

        if score >= 5:
            confidence = "HIGH"
        elif score >= 3:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            "is_accumulating": score >= 3,
            "score":           score,
            "confidence":      confidence,
            "signals":         signals,
            "entry_zone":      round(price, 6),
            "invalidation":    round(support_4h * 0.985, 6),
        }

    except Exception as e:
        log.warning(f"Accumulation setup error for {symbol}: {e}")
        return {"is_accumulating": False, "score": 0, "confidence": "LOW",
                "signals": [], "entry_zone": 0, "invalidation": 0}




# ─── SHAKEOUT DETECTOR ────────────────────────────────────────────────────────
def detect_shakeout(
    df_15m: pd.DataFrame,
    df_1h:  pd.DataFrame,
    df_4h:  pd.DataFrame,
    symbol: str = "",
) -> dict:
    """
    Detects a shakeout (manipulated flush below support) that precedes a crimepump.

    A shakeout is NOT a real breakdown. Smart money pushes price below support to
    trigger retail stop-losses and buy the liquidity. The fingerprint is:

      REQUIRED (both must be true):
        R1. Price wicked below the 1H/4H support level (wick, not close)
        R2. Price CLOSED back ABOVE that support within 1-3 candles

      CONFIRMING (need 3+ of 6):
        C1. Volume on the flush candle is LOWER than prior 5 candles avg
            (no real panic selling — engineered flush on thin volume)
        C2. OBV does NOT confirm the breakdown
            (OBV flat or rising while price dips — absorption)
        C3. RSI on 15m was oversold during the wick AND is now recovering
            (V-shape RSI recovery)
        C4. The wick is >= 1.5x the candle body
            (long wick = rejection of the lower price, not acceptance)
        C5. 4H structure is still bullish: higher lows maintained
            (the macro trend hasn't broken — this is a micro flush)
        C6. Price is now within 1% of the 15m EMA9 from below
            (reclaimed the short-term mean — momentum resetting)

    Returns:
    {
      "is_shakeout":     bool,
      "confidence":      str,   # HIGH (5+) / MEDIUM (3-4) / LOW (<3)
      "score":           int,   # 0–8
      "signals":         list,  # human-readable evidence
      "invalidation":    float, # if price closes BELOW this, it's a real breakdown
      "pump_target":     float, # nearest resistance above — likely pump target
      "entry_zone":      float, # suggested entry (current price or slight pullback)
      "flush_low":       float, # the low of the shakeout wick
    }
    """
    EMPTY = {
        "is_shakeout": False, "confidence": "LOW", "score": 0,
        "signals": [], "invalidation": 0, "pump_target": 0,
        "entry_zone": 0, "flush_low": 0,
    }

    try:
        if df_15m is None or not isinstance(df_15m, pd.DataFrame) or len(df_15m) < 30:
            return EMPTY
        if df_1h is None or not isinstance(df_1h, pd.DataFrame) or len(df_1h) < 20:
            return EMPTY

        close_15m  = df_15m["close"].values
        high_15m   = df_15m["high"].values
        low_15m    = df_15m["low"].values
        open_15m   = df_15m["open"].values
        volume_15m = df_15m["volume"].values
        price      = float(close_15m[-1])

        score   = 0
        signals = []

        # ── Support level from 1H (last 20 candles) ──────────────────────────
        support_1h = float(df_1h["low"].tail(20).min())
        support_4h = float(df_4h["low"].tail(20).min()) if df_4h is not None and len(df_4h) >= 20 else support_1h
        # Use stronger (higher) support as the reference
        support    = max(support_1h, support_4h * 0.995)

        # ── REQUIRED CHECK R1: Did price wick below support recently? ─────────
        # Look back up to 6 candles for a wick below support
        flush_candle_idx = None
        flush_low        = 0.0

        for i in range(1, min(7, len(low_15m))):
            if low_15m[-i] < support:
                flush_low        = float(low_15m[-i])
                flush_candle_idx = len(low_15m) - i
                break

        if flush_candle_idx is None:
            return EMPTY  # no wick below support — not a shakeout

        # ── REQUIRED CHECK R2: Did price close BACK ABOVE support? ───────────
        # Candle that wicked must have closed back above, OR subsequent candle did
        wick_close     = float(close_15m[flush_candle_idx])
        current_close  = float(close_15m[-1])
        reclaimed      = wick_close > support or current_close > support

        if not reclaimed:
            # Price closed below support — could be real breakdown, not shakeout
            return EMPTY

        signals.append(f"Price wicked to ${flush_low:.4f} below support ${support:.4f} then reclaimed ✦")

        # How many candles since the flush?
        candles_since_flush = len(low_15m) - 1 - flush_candle_idx
        if candles_since_flush <= 3:
            score += 1
            signals.append(f"Fast reclaim in {candles_since_flush} candle(s) — rejection not acceptance")

        # ── CONFIRMING C1: Volume on flush was WEAK ───────────────────────────
        flush_vol   = float(volume_15m[flush_candle_idx])
        prior_5_avg = float(np.mean(volume_15m[max(0, flush_candle_idx-5):flush_candle_idx]))
        if prior_5_avg > 0 and flush_vol < prior_5_avg * 0.85:
            score += 2  # double weight — this is the most reliable shakeout signal
            signals.append(
                f"Flush volume {flush_vol/prior_5_avg:.0%} of prior avg — "
                f"no real panic (engineered) ✦"
            )
        elif prior_5_avg > 0 and flush_vol < prior_5_avg * 1.1:
            score += 1
            signals.append(f"Flush volume not elevated — weak selling pressure")

        # ── CONFIRMING C2: OBV doesn't confirm breakdown ──────────────────────
        try:
            obv    = ta.obv(df_15m["close"], df_15m["volume"])
            if obv is not None and len(obv) >= 10:
                obv_vals      = obv.fillna(0).values
                obv_at_flush  = obv_vals[flush_candle_idx]
                obv_before    = obv_vals[max(0, flush_candle_idx - 3)]
                obv_now       = obv_vals[-1]
                # OBV should NOT have dropped during the flush
                if obv_at_flush >= obv_before * 0.98:
                    score += 1
                    signals.append("OBV held during price flush — no real distribution ✦")
                if obv_now > obv_at_flush:
                    score += 1
                    signals.append("OBV recovering after flush — absorption confirmed")
        except Exception:
            pass

        # ── CONFIRMING C3: RSI V-shape recovery on 15m ───────────────────────
        try:
            rsi_15m_s = ta.rsi(df_15m["close"], length=14)
            if rsi_15m_s is not None and len(rsi_15m_s.dropna()) >= 5:
                rsi_vals = rsi_15m_s.fillna(50).values
                rsi_at_flush = float(rsi_vals[flush_candle_idx]) if flush_candle_idx < len(rsi_vals) else 50.0
                rsi_now      = float(rsi_vals[-1])
                if rsi_at_flush < 35 and rsi_now > rsi_at_flush + 8:
                    score += 2  # double weight — V-shape from oversold is very reliable
                    signals.append(
                        f"RSI V-shape: {rsi_at_flush:.0f} → {rsi_now:.0f} "
                        f"(oversold flush + recovery) ✦"
                    )
                elif rsi_at_flush < 45 and rsi_now > rsi_at_flush + 5:
                    score += 1
                    signals.append(f"RSI recovering from flush low ({rsi_at_flush:.0f} → {rsi_now:.0f})")
        except Exception:
            pass

        # ── CONFIRMING C4: Long wick vs body ratio ────────────────────────────
        flush_open  = float(open_15m[flush_candle_idx])
        flush_close = float(close_15m[flush_candle_idx])
        body        = abs(flush_close - flush_open)
        lower_wick  = min(flush_open, flush_close) - float(low_15m[flush_candle_idx])
        if body > 0 and lower_wick >= body * 1.5:
            score += 1
            signals.append(
                f"Long lower wick ({lower_wick/body:.1f}x body) — "
                f"strong rejection of lower price ✦"
            )
        elif body == 0 and lower_wick > 0:
            score += 1
            signals.append("Doji with long lower wick — indecision at support")

        # ── CONFIRMING C5: 4H macro structure still bullish ──────────────────
        if df_4h is not None and len(df_4h) >= 10:
            try:
                h4_highs = df_4h["high"].values[-10:]
                h4_lows  = df_4h["low"].values[-10:]
                # Check for higher lows in 4H — macro trend intact
                swing_lows_4h = []
                for i in range(1, len(h4_lows) - 1):
                    if h4_lows[i] < h4_lows[i-1] and h4_lows[i] < h4_lows[i+1]:
                        swing_lows_4h.append(h4_lows[i])
                if len(swing_lows_4h) >= 2 and swing_lows_4h[-1] > swing_lows_4h[-2]:
                    score += 1
                    signals.append("4H higher lows intact — macro trend not broken ✦")
                elif price > support_4h:
                    score += 1
                    signals.append("Price above 4H support — macro structure holding")
            except Exception:
                pass

        # ── CONFIRMING C6: Price near 15m EMA9 from below ────────────────────
        try:
            ema9   = ta.ema(df_15m["close"], length=9)
            if ema9 is not None and len(ema9.dropna()) > 0:
                ema9_v = float(ema9.dropna().iloc[-1])
                dist   = (price - ema9_v) / ema9_v * 100
                if -1.5 < dist < 1.5:
                    score += 1
                    signals.append(
                        f"Price at 15m EMA9 (${ema9_v:.4f}, {dist:+.1f}%) — "
                        f"mean reclaimed"
                    )
        except Exception:
            pass

        # ── Confidence + invalidation ─────────────────────────────────────────
        if score >= 6:
            confidence = "HIGH"
        elif score >= 4:
            confidence = "MEDIUM"
        elif score >= 3:
            confidence = "LOW"
        else:
            return EMPTY  # not enough evidence

        # Invalidation = if price closes back below flush low → real breakdown
        invalidation = round(flush_low * 0.995, 6)

        # Pump target = nearest resistance above current price (20-candle high on 1H)
        pump_target = round(float(df_1h["high"].tail(20).max()), 6)

        return {
            "is_shakeout":  score >= 3,
            "confidence":   confidence,
            "score":        score,
            "signals":      signals,
            "invalidation": invalidation,
            "pump_target":  pump_target,
            "entry_zone":   round(price, 6),
            "flush_low":    round(flush_low, 6),
        }

    except Exception as e:
        log.warning(f"Shakeout detector error for {symbol}: {e}")
        return EMPTY

# ─── 15M ENTRY QUALITY CHECK ─────────────────────────────────────────────────
def check_15m_entry_quality(df_15m: pd.DataFrame, direction: str,
                             demand_ob: dict = None, fvg_levels: list = None,
                             bos_data: dict = None) -> dict:
    """
    Checks 15m chart for optimal entry timing, now with SMC context.

    Verdict logic (revised):
      ENTER  — at least 1 enter signal AND no more than 1 wait signal
      WAIT   — 2+ wait signals, or 1 critical wait signal (RSI > 70 / < 30)
      CONDITIONAL — mixed signals: give structural level to wait for
      NEUTRAL — truly ambiguous, but ALWAYS provides structural context

    Pullback target priority (LONG):
      1. Demand OB top  2. FVG mid  3. 15m swing low  4. EMA9 fallback
    """
    try:
        if df_15m is None or not isinstance(df_15m, pd.DataFrame) or len(df_15m) < 20:
            return {"action": "NEUTRAL",
                    "reason": "15m data unavailable — base entry on 1H structure",
                    "rsi_15m": 50.0, "pullback_target": None, "wait_reason": ""}

        close = df_15m["close"]
        high  = df_15m["high"]
        low   = df_15m["low"]
        price = float(close.iloc[-1])

        # ── Indicators ───────────────────────────────────────────────────────
        rsi_s = ta.rsi(close, length=14)
        rsi_v = (float(rsi_s.dropna().iloc[-1])
                 if rsi_s is not None and len(rsi_s.dropna()) > 0 else 50.0)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_bearish_cross = macd_bullish_cross = False
        ml_now = sl_now = 0.0   # initialise so macd_bullish/bearish are always defined
        if macd_df is not None and not macd_df.empty and len(macd_df) >= 3:
            _mc = [c for c in macd_df.columns if c.startswith("MACD_")]
            _ms = [c for c in macd_df.columns if c.startswith("MACDs_")]
            if _mc and _ms:
                ml_now  = float(macd_df[_mc[0]].iloc[-1])
                sl_now  = float(macd_df[_ms[0]].iloc[-1])
                ml_prev = float(macd_df[_mc[0]].iloc[-2])
                sl_prev = float(macd_df[_ms[0]].iloc[-2])
                macd_bearish_cross = ml_now < sl_now and ml_prev >= sl_prev
                macd_bullish_cross = ml_now > sl_now and ml_prev <= sl_prev
        # Always defined — ml_now/sl_now default to 0.0 if MACD unavailable
        macd_bullish = ml_now > sl_now
        macd_bearish = ml_now < sl_now

        ema9   = ta.ema(close, length=9)
        ema9_v = (float(ema9.dropna().iloc[-1])
                  if ema9 is not None and len(ema9.dropna()) > 0 else price)
        ema21  = ta.ema(close, length=21)
        ema21_v = (float(ema21.dropna().iloc[-1])
                   if ema21 is not None and len(ema21.dropna()) > 0 else price)
        dist_ema9_pct = (price - ema9_v) / ema9_v * 100

        vol_ma   = float(df_15m["volume"].rolling(20).mean().iloc[-1])
        vol_now  = float(df_15m["volume"].iloc[-1])
        vol_weak = vol_now < vol_ma * 0.6
        vol_strong = vol_now > vol_ma * 1.4

        # ── SMC context ──────────────────────────────────────────────────────
        at_demand_ob = (demand_ob and demand_ob.get("active")
                        and demand_ob.get("low", 0) <= price <= demand_ob.get("high", 0) * 1.005)
        at_supply_ob = (demand_ob and demand_ob.get("active")
                        and demand_ob.get("low", 0) * 0.995 <= price <= demand_ob.get("high", 0))
        bos_bullish  = bos_data and (bos_data.get("bos_bullish") or bos_data.get("choch_bullish"))
        bos_bearish  = bos_data and (bos_data.get("bos_bearish") or bos_data.get("choch_bearish"))
        liq_sweep    = (bos_data and
                        (bos_data.get("sweep_bullish") or bos_data.get("sweep_bearish"))
                        if bos_data else False)

        # ── Score conditions ─────────────────────────────────────────────────
        wait_reasons  = []
        enter_reasons = []
        critical_wait = False   # single condition strong enough to force WAIT

        if direction == "LONG":
            # Critical waits (single condition enough)
            if rsi_v > 72:
                critical_wait = True
                wait_reasons.append(f"15m RSI extremely overbought ({rsi_v:.0f}) — high reversal risk")
            # Standard waits
            elif rsi_v > 65:
                wait_reasons.append(f"15m RSI overbought ({rsi_v:.0f}) — local top forming")
            if macd_bearish_cross:
                wait_reasons.append("15m MACD bearish cross — momentum turning down")
            if dist_ema9_pct > 4.0:
                wait_reasons.append(f"Price {dist_ema9_pct:.1f}% above 15m EMA9 — overextended")
            if vol_weak and not (at_demand_ob or macd_bullish_cross):
                wait_reasons.append("Weak volume on push — buyers not committed")
            # Enter signals
            if rsi_v < 68 and rsi_v > 30:   # healthy range: not chased (>30) and not overbought (<68)
                enter_reasons.append(f"15m RSI {rsi_v:.0f} — {'healthy, not chased' if rsi_v < 55 else 'elevated but not overbought'}")
            if macd_bullish_cross:
                enter_reasons.append("15m MACD bullish cross — momentum confirmed ✦")
            if macd_bullish and not macd_bullish_cross:
                enter_reasons.append("15m MACD bullish alignment")
            if abs(dist_ema9_pct) < 3.0:
                enter_reasons.append(f"Price near 15m EMA9 (within {abs(dist_ema9_pct):.1f}%) — clean zone")
            elif dist_ema9_pct > 0 and dist_ema9_pct < 5.0 and (bos_bullish or macd_bullish):
                # Extended but structure + momentum support continuation
                enter_reasons.append(f"Price {dist_ema9_pct:.1f}% above EMA9 — extended but trend supports continuation")
            if at_demand_ob:
                enter_reasons.append("Price retesting demand OB — structural entry ✦")
            if bos_bullish:
                enter_reasons.append("BOS/CHOCH bullish on structure ✦")
            if liq_sweep and bos_data.get("sweep_bullish"):
                enter_reasons.append("Sell-side liquidity swept — stop hunt complete ✦")
            if vol_strong and (macd_bullish or rsi_v > 45):
                enter_reasons.append("Strong volume confirming move")
        else:
            if rsi_v < 28:
                critical_wait = True
                wait_reasons.append(f"15m RSI extremely oversold ({rsi_v:.0f}) — bounce risk")
            elif rsi_v < 35:
                wait_reasons.append(f"15m RSI oversold ({rsi_v:.0f}) — counter-bounce likely")
            if macd_bullish_cross:
                wait_reasons.append("15m MACD bullish cross — counter-move forming")
            if dist_ema9_pct < -4.0:
                wait_reasons.append(f"Price {abs(dist_ema9_pct):.1f}% below 15m EMA9 — overextended")
            if vol_weak and not (at_supply_ob or macd_bearish_cross):
                wait_reasons.append("Weak volume on dump — sellers not committed")
            if rsi_v > 32 and rsi_v < 70:   # healthy short range: not oversold and not overbought
                enter_reasons.append(f"15m RSI {rsi_v:.0f} — {'healthy, not oversold' if rsi_v > 45 else 'slightly low but not extreme'}")
            if macd_bearish_cross:
                enter_reasons.append("15m MACD bearish cross — momentum confirmed ✦")
            if macd_bearish and not macd_bearish_cross:
                enter_reasons.append("15m MACD bearish alignment")
            if abs(dist_ema9_pct) < 3.0:
                enter_reasons.append(f"Price near 15m EMA9 (within {abs(dist_ema9_pct):.1f}%) — clean zone")
            elif dist_ema9_pct < 0 and dist_ema9_pct > -5.0 and (bos_bearish or macd_bearish):
                enter_reasons.append(f"Price {abs(dist_ema9_pct):.1f}% below EMA9 — extended but trend supports continuation")
            if at_supply_ob:
                enter_reasons.append("Price retesting supply OB — structural entry ✦")
            if bos_bearish:
                enter_reasons.append("BOS/CHOCH bearish on structure ✦")
            if liq_sweep and bos_data.get("sweep_bearish"):
                enter_reasons.append("Buy-side liquidity swept — stop hunt complete ✦")
            if vol_strong and (macd_bearish or rsi_v < 55):
                enter_reasons.append("Strong volume confirming dump")

        # ── Verdict ──────────────────────────────────────────────────────────
        n_wait  = len(wait_reasons)
        n_enter = len(enter_reasons)

        if critical_wait or n_wait >= 2:
            action = "WAIT"
            reason = " | ".join(wait_reasons)
        elif n_enter >= 1 and n_wait <= 1:
            # Enter even with 1 mild wait signal if there's SMC or MACD confirmation
            action = "ENTER"
            reason = " | ".join(enter_reasons)
        elif n_enter >= 1 and n_wait >= 1:
            # Mixed: give conditional verdict with the structural level
            action = "CONDITIONAL"
            reason = ("Mixed signals — " + enter_reasons[0] + " BUT " + wait_reasons[0]
                      + ". Wait for pullback to zone below.")
        else:
            # Truly neutral — still give structural context
            _str  = bos_data.get("structure", "NEUTRAL") if bos_data else "NEUTRAL"
            _ema9 = f"${ema9_v:.4f}"
            action = "NEUTRAL"
            reason = (f"No strong 15m signal — structure is {_str} | "
                      f"RSI: {rsi_v:.0f} | MACD: {'bullish' if macd_bullish else 'bearish' if macd_bearish else 'flat'} | "
                      f"EMA9: {_ema9}. Enter only if 1H/4H setup is clean.")

        # ── Pullback target ───────────────────────────────────────────────────
        pullback_target = None
        if action in ("WAIT", "CONDITIONAL"):
            if direction == "LONG":
                if demand_ob and demand_ob.get("active") and demand_ob.get("high", 0) < price:
                    pullback_target = round(demand_ob["high"], 6)
                elif fvg_levels:
                    bull_fvgs = sorted(
                        [f for f in fvg_levels if f.get("type") == "BULL" and f.get("mid", 0) < price],
                        key=lambda x: x["mid"], reverse=True)
                    if bull_fvgs:
                        pullback_target = round(bull_fvgs[0]["mid"], 6)
                if pullback_target is None:
                    recent_low = float(low.iloc[-10:].min())
                    if recent_low < price * 0.99:
                        pullback_target = round(recent_low * 1.002, 6)
                if pullback_target is None:
                    pullback_target = round(ema9_v, 6)
            else:
                if demand_ob and demand_ob.get("active") and demand_ob.get("low", 0) > price:
                    pullback_target = round(demand_ob["low"], 6)
                elif fvg_levels:
                    bear_fvgs = sorted(
                        [f for f in fvg_levels if f.get("type") == "BEAR" and f.get("mid", 0) > price],
                        key=lambda x: x["mid"])
                    if bear_fvgs:
                        pullback_target = round(bear_fvgs[0]["mid"], 6)
                if pullback_target is None:
                    recent_high = float(high.iloc[-10:].max())
                    if recent_high > price * 1.01:
                        pullback_target = round(recent_high * 0.998, 6)
                if pullback_target is None:
                    pullback_target = round(ema9_v, 6)

        return {
            "action":           action,
            "reason":           reason,
            "rsi_15m":          round(rsi_v, 1),
            "pullback_target":  pullback_target,
            "wait_reason":      " | ".join(wait_reasons) if wait_reasons else "",
        }

    except Exception as e:
        log.warning(f"15m entry check error: {e}")
        return {"action": "NEUTRAL", "reason": "15m check failed — base entry on 1H/4H structure",
                "rsi_15m": 50.0, "pullback_target": None, "wait_reason": ""}
    """
    Checks 15m chart for optimal entry timing.

    For LONG: WAIT if RSI > 65, MACD just crossed bearish, price extended above EMA9.
              ENTER if RSI < 45, MACD bullish cross, price at/near EMA9.

    For SHORT: mirrors above inverted.

    pullback_target priority (LONG):
      1. Demand OB top — best structural level, where buyers previously stepped in
      2. Nearest unfilled bullish FVG mid below price — imbalance fill zone
      3. Recent 15m swing low (last 10 bars) — structural support
      4. 15m EMA9 — fallback mean-reversion target

    Returns:
    {
      "action":          "ENTER" | "WAIT" | "NEUTRAL",
      "reason":          str,
      "rsi_15m":         float,
      "pullback_target": float | None,
      "wait_reason":     str,
    }
    """
    try:
        if df_15m is None or not isinstance(df_15m, pd.DataFrame) or len(df_15m) < 20:
            return {"action": "NEUTRAL", "reason": "15m data unavailable",
                    "rsi_15m": 50.0, "pullback_target": None, "wait_reason": ""}
        close  = df_15m["close"]
        high   = df_15m["high"]
        low    = df_15m["low"]
        price  = float(close.iloc[-1])

        rsi_s = ta.rsi(close, length=14)
        rsi_v = (float(rsi_s.dropna().iloc[-1])
                 if rsi_s is not None and len(rsi_s.dropna()) > 0 else 50.0)

        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        macd_bearish_cross = False
        macd_bullish_cross = False
        if macd_df is not None and not macd_df.empty and len(macd_df) >= 3:
            _mc = [c for c in macd_df.columns if c.startswith("MACD_")]
            _ms = [c for c in macd_df.columns if c.startswith("MACDs_")]
            if _mc and _ms:
                ml_now  = float(macd_df[_mc[0]].iloc[-1])
                sl_now  = float(macd_df[_ms[0]].iloc[-1])
                ml_prev = float(macd_df[_mc[0]].iloc[-2])
                sl_prev = float(macd_df[_ms[0]].iloc[-2])
                macd_bearish_cross = ml_now < sl_now and ml_prev >= sl_prev
                macd_bullish_cross = ml_now > sl_now and ml_prev <= sl_prev

        ema9   = ta.ema(close, length=9)
        ema9_v = (float(ema9.dropna().iloc[-1])
                  if ema9 is not None and len(ema9.dropna()) > 0 else price)
        dist_ema9_pct = (price - ema9_v) / ema9_v * 100

        vol_ma  = float(df_15m["volume"].rolling(20).mean().iloc[-1])
        vol_now = float(df_15m["volume"].iloc[-1])
        vol_weak = vol_now < vol_ma * 0.6

        wait_reasons  = []
        enter_reasons = []
        pullback_target = None

        if direction == "LONG":
            if rsi_v > 65:
                wait_reasons.append(f"15m RSI overbought ({rsi_v:.0f}) — local top forming")
            if macd_bearish_cross:
                wait_reasons.append("15m MACD bearish cross — momentum turning down")
            if dist_ema9_pct > 3.0:
                wait_reasons.append(f"Price {dist_ema9_pct:.1f}% above 15m EMA9 — overextended")
            if vol_weak and not macd_bullish_cross:
                wait_reasons.append("Low volume on push — weak buying pressure")
            if rsi_v < 45:
                enter_reasons.append(f"15m RSI {rsi_v:.0f} — not chased, good entry")
            if macd_bullish_cross:
                enter_reasons.append("15m MACD bullish cross — momentum confirming")
            if abs(dist_ema9_pct) < 1.0:
                enter_reasons.append("Price at 15m EMA9 — clean pullback entry")
        else:
            if rsi_v < 35:
                wait_reasons.append(f"15m RSI oversold ({rsi_v:.0f}) — local bottom forming")
            if macd_bullish_cross:
                wait_reasons.append("15m MACD bullish cross — counter-move likely")
            if dist_ema9_pct < -3.0:
                wait_reasons.append(f"Price {abs(dist_ema9_pct):.1f}% below 15m EMA9 — overextended")
            if vol_weak and not macd_bearish_cross:
                wait_reasons.append("Low volume on dump — weak selling pressure")
            if rsi_v > 55:
                enter_reasons.append(f"15m RSI {rsi_v:.0f} — not oversold, clean short entry")
            if macd_bearish_cross:
                enter_reasons.append("15m MACD bearish cross — momentum confirming")
            if abs(dist_ema9_pct) < 1.0:
                enter_reasons.append("Price at 15m EMA9 — clean pullback entry")

        if len(wait_reasons) >= 2:
            action = "WAIT"
            reason = " | ".join(wait_reasons)
        elif len(enter_reasons) >= 1 and not wait_reasons:
            action = "ENTER"
            reason = " | ".join(enter_reasons)
        else:
            action = "NEUTRAL"
            reason = "No strong 15m timing signal — use 1H/4H context"

        # ── Pullback target — use best available structural level ────────────
        # Previously only set inside RSI/EMA9 branches — MACD+volume WAIT gave
        # no target. Now always computed when action is WAIT, using structural
        # levels in priority order so the target is a real zone not just EMA9.
        if action == "WAIT":
            if direction == "LONG":
                # Priority 1: Demand OB top — actual buying zone
                if (demand_ob and demand_ob.get("active")
                        and demand_ob.get("high", 0) < price):
                    pullback_target = round(demand_ob["high"], 6)
                # Priority 2: Nearest bullish FVG mid below price
                elif fvg_levels:
                    bull_fvgs = sorted(
                        [f for f in fvg_levels
                         if f.get("type") == "BULL" and f.get("mid", 0) < price],
                        key=lambda x: x["mid"], reverse=True
                    )
                    if bull_fvgs:
                        pullback_target = round(bull_fvgs[0]["mid"], 6)
                # Priority 3: Recent 15m swing low (last 10 bars)
                if pullback_target is None:
                    recent_low = float(low.iloc[-10:].min())
                    if recent_low < price * 0.99:   # must be at least 1% below
                        pullback_target = round(recent_low * 1.002, 6)  # slight buffer above low
                # Priority 4: EMA9 fallback
                if pullback_target is None:
                    pullback_target = round(ema9_v, 6)
            else:
                # Priority 1: Supply OB bottom — actual selling zone
                if (demand_ob and demand_ob.get("active")
                        and demand_ob.get("low", 0) > price):
                    pullback_target = round(demand_ob["low"], 6)
                # Priority 2: Nearest bearish FVG mid above price
                elif fvg_levels:
                    bear_fvgs = sorted(
                        [f for f in fvg_levels
                         if f.get("type") == "BEAR" and f.get("mid", 0) > price],
                        key=lambda x: x["mid"]
                    )
                    if bear_fvgs:
                        pullback_target = round(bear_fvgs[0]["mid"], 6)
                # Priority 3: Recent 15m swing high (last 10 bars)
                if pullback_target is None:
                    recent_high = float(high.iloc[-10:].max())
                    if recent_high > price * 1.01:
                        pullback_target = round(recent_high * 0.998, 6)
                # Priority 4: EMA9 fallback
                if pullback_target is None:
                    pullback_target = round(ema9_v, 6)

        return {
            "action":           action,
            "reason":           reason,
            "rsi_15m":          round(rsi_v, 1),
            "pullback_target":  pullback_target,
            "wait_reason":      " | ".join(wait_reasons) if wait_reasons else "",
        }

    except Exception as e:
        log.warning(f"15m entry check error: {e}")
        return {"action": "NEUTRAL", "reason": "15m data unavailable",
                "rsi_15m": 50.0, "pullback_target": None, "wait_reason": ""}
