"""
Phyrobot — Phase 4: Advanced Risk Management
────────────────────────────────────────────────────────
Replaces the flat RISK_PCT position sizing with a full risk engine:

  1. Kelly Criterion sizing    — size based on rolling win rate, not fixed %
  2. Volatility-adjusted SL    — ATR-relative SL adapts to market conditions
  3. Correlation filter        — blocks correlated signals (no 5x BTC alts at once)
  4. Portfolio heat limiter    — caps total open risk exposure
  5. Session quality filter    — scores trading session (avoid dead hours)
  6. Dynamic leverage advisor  — suggests lower leverage in high-risk conditions
  7. Risk dashboard            — /risk command shows live exposure snapshot

All functions are DROP-IN — they wrap existing calc_position_size()
and calc_levels() without breaking anything.
"""

import os
import json
import logging
import time as _time
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
ACCOUNT_SIZE        = float(os.getenv("ACCOUNT_SIZE", "200"))
RISK_PCT            = float(os.getenv("RISK_PCT", "1.5"))
LEVERAGE            = int(os.getenv("LEVERAGE", "10"))

KELLY_LOOKBACK      = 20      # trades to compute Kelly fraction from
KELLY_FRACTION      = 0.25    # fractional Kelly (0.25 = quarter Kelly — conservative)
KELLY_MIN_PCT       = 0.5     # minimum risk % regardless of Kelly
KELLY_MAX_PCT       = 3.0     # maximum risk % regardless of Kelly

MAX_PORTFOLIO_HEAT  = 6.0     # max total open risk % of account (e.g. 6% = 4 trades at 1.5%)
MAX_CORRELATED      = 2       # max simultaneous signals in highly correlated group
CORRELATION_GROUPS  = {
    "BTC_FAMILY":  ["BTC", "ETH", "BNB"],
    "SOL_FAMILY":  ["SOL", "AVAX", "SUI", "APT"],
    "MEME_FAMILY": ["DOGE", "SHIB", "PEPE", "BONK", "WIF", "FLOKI"],
    "DEFI_FAMILY": ["LINK", "UNI", "AAVE", "CRV", "SNX"],
    "LAYER2":      ["ARB", "OP", "MATIC", "IMX", "STRK"],
}

# Session quality windows (UTC hours) — 0=worst, 1=best
SESSION_QUALITY = {
    # London open + NY overlap = best liquidity
    (13, 17): 1.0,   # NY open overlap with London
    (8,  13): 0.9,   # London session
    (17, 21): 0.85,  # NY afternoon
    (0,   8): 0.7,   # Asian session
    (21, 24): 0.6,   # NY close / dead zone
}
SESSION_DEAD_HOURS  = {0, 1, 2, 3, 4, 5}  # UTC hours — avoid signals entirely

# ATR multiplier ranges — tighter SL in low vol, wider in high vol
ATR_SL_RANGE   = (1.2, 2.5)   # min/max ATR multiplier for SL
ATR_TP1_RANGE  = (1.2, 2.5)   # matches SL for 1:1 risk/reward
ATR_TP2_RANGE  = (3.0, 5.0)   # TP2 always much further


# ─── 1. KELLY CRITERION POSITION SIZING ──────────────────────────────────────
def kelly_position_size(
    entry: float,
    sl: float,
    trade_history: list,
    account_size: float = ACCOUNT_SIZE,
    leverage: int = LEVERAGE,
) -> tuple[float, float, float]:
    """
    Compute position size using fractional Kelly Criterion.

    Kelly formula: f* = (W/L * win_rate - (1 - win_rate)) / (W/L)
    Where W = avg win %, L = avg loss %

    Returns: (risk_pct, position_usdt, contracts)
    """
    risk_pct = RISK_PCT  # fallback to default

    if len(trade_history) >= KELLY_LOOKBACK:
        recent = trade_history[-KELLY_LOOKBACK:]
        wins   = [t for t in recent if t.get("outcome") == "WIN"]
        losses = [t for t in recent if t.get("outcome") == "LOSS"]

        if len(wins) >= 3 and len(losses) >= 1:
            win_rate = len(wins) / len(recent)
            avg_win  = abs(sum(t.get("pnl_pct", 0) for t in wins) / len(wins))
            avg_loss = abs(sum(t.get("pnl_pct", 0) for t in losses) / len(losses))

            if avg_loss > 0 and avg_win > 0:
                wl_ratio = avg_win / avg_loss
                kelly_f  = (wl_ratio * win_rate - (1 - win_rate)) / wl_ratio
                # Apply fractional Kelly and clamp to safe range
                kelly_risk = kelly_f * KELLY_FRACTION * 100
                risk_pct   = max(KELLY_MIN_PCT, min(KELLY_MAX_PCT, kelly_risk))
                log.info(
                    f"Kelly: win_rate={win_rate:.1%} W/L={wl_ratio:.2f} "
                    f"raw_kelly={kelly_f:.3f} → risk={risk_pct:.2f}%"
                )

    sl_distance = abs(entry - sl) / entry
    if sl_distance == 0:
        return risk_pct, 0.0, 0.0

    risk_amount   = account_size * (risk_pct / 100)
    position_usdt = round(risk_amount / sl_distance, 2)
    contracts     = round(position_usdt * leverage / entry, 4)

    return round(risk_pct, 2), position_usdt, contracts


# ─── 2. VOLATILITY-ADJUSTED LEVELS ───────────────────────────────────────────
def calc_levels_v2(
    direction: str,
    price: float,
    atr: float,
    atr_pct: float = None,
) -> tuple[float, float, float, dict]:
    """
    Volatility-adaptive SL/TP levels.

    Low volatility  (ATR < 0.5%) → tighter SL (1.2x ATR), closer TP2 (3x)
    Medium vol      (0.5-2%)     → standard SL (1.5x ATR), TP2 (4x)
    High volatility (ATR > 2%)   → wider SL (2.0x ATR), TP2 (5x) to let it breathe

    Returns: (sl, tp1, tp2, meta_dict)
    """
    if atr_pct is None:
        atr_pct = (atr / price * 100) if price > 0 else 1.0

    # Determine volatility regime
    if atr_pct < 0.5:
        sl_mult  = 1.2
        tp1_mult = 1.2
        tp2_mult = 3.0
        vol_regime = "LOW"
    elif atr_pct < 2.0:
        sl_mult  = 1.5
        tp1_mult = 1.5
        tp2_mult = 4.0
        vol_regime = "MEDIUM"
    elif atr_pct < 4.0:
        sl_mult  = 2.0
        tp1_mult = 2.0
        tp2_mult = 4.5
        vol_regime = "HIGH"
    else:
        sl_mult  = 2.5
        tp1_mult = 2.5
        tp2_mult = 5.0
        vol_regime = "EXTREME"

    if direction == "LONG":
        sl  = round(price - atr * sl_mult,  6)
        tp1 = round(price + atr * tp1_mult, 6)
        tp2 = round(price + atr * tp2_mult, 6)
    else:
        sl  = round(price + atr * sl_mult,  6)
        tp1 = round(price - atr * tp1_mult, 6)
        tp2 = round(price - atr * tp2_mult, 6)

    meta = {
        "vol_regime": vol_regime,
        "atr_pct":    round(atr_pct, 3),
        "sl_mult":    sl_mult,
        "tp2_mult":   tp2_mult,
        "rr_ratio":   round(tp2_mult / sl_mult, 1),
    }

    return sl, tp1, tp2, meta


# ─── 3. CORRELATION FILTER ────────────────────────────────────────────────────
def check_correlation(
    new_symbol: str,
    active_signals: dict,
    max_correlated: int = MAX_CORRELATED,
) -> dict:
    """
    Checks if adding this signal would create too many correlated positions.
    Returns {"pass": bool, "reason": str, "group": str}
    """
    new_label = new_symbol.split("/")[0].upper()

    # Find which family the new coin belongs to
    new_group = None
    for group, members in CORRELATION_GROUPS.items():
        if new_label in members:
            new_group = group
            break

    if new_group is None:
        return {"pass": True, "reason": "No correlation group", "group": "NONE"}

    # Count how many active signals are in the same group
    same_group_count = 0
    same_group_coins = []
    for sym in active_signals:
        label = sym.split("/")[0].upper()
        if label in CORRELATION_GROUPS.get(new_group, []):
            same_group_count += 1
            same_group_coins.append(label)

    if same_group_count >= max_correlated:
        return {
            "pass":   False,
            "reason": f"Too many {new_group} positions open: {', '.join(same_group_coins)}",
            "group":  new_group,
        }

    return {
        "pass":   True,
        "reason": f"{new_group}: {same_group_count}/{max_correlated} slots used",
        "group":  new_group,
    }


# ─── 4. PORTFOLIO HEAT MONITOR ───────────────────────────────────────────────
def get_portfolio_heat(
    active_signals: dict,
    account_size: float = ACCOUNT_SIZE,
) -> dict:
    """
    Computes total current risk exposure across all open positions.
    "Heat" = sum of (SL distance % × leverage) for each open trade.

    Returns:
    {
      "heat_pct": float,        # total % of account at risk
      "can_open": bool,         # whether new trade is allowed
      "slots_used": int,
      "max_heat": float,
      "breakdown": list
    }
    """
    breakdown   = []
    total_heat  = 0.0

    for symbol, sig in active_signals.items():
        entry = sig.get("entry", 0)
        sl    = sig.get("sl", 0)
        if entry == 0:
            continue
        sl_dist_pct = abs(entry - sl) / entry * 100
        # Risk per trade = SL distance × leverage, expressed as % of account
        trade_heat  = sl_dist_pct * LEVERAGE * (RISK_PCT / 100)
        total_heat += trade_heat
        breakdown.append({
            "symbol":    symbol.split("/")[0],
            "sl_dist":   round(sl_dist_pct, 2),
            "heat_pct":  round(trade_heat, 2),
            "direction": sig.get("direction", "?"),
        })

    can_open = total_heat < MAX_PORTFOLIO_HEAT

    return {
        "heat_pct":   round(total_heat, 2),
        "can_open":   can_open,
        "slots_used": len(active_signals),
        "max_heat":   MAX_PORTFOLIO_HEAT,
        "breakdown":  breakdown,
        "reason":     "" if can_open else f"Portfolio heat {total_heat:.1f}% ≥ max {MAX_PORTFOLIO_HEAT}%",
    }


# ─── 5. SESSION QUALITY FILTER ───────────────────────────────────────────────
def get_session_quality() -> dict:
    """
    Returns a quality score (0.0-1.0) for the current trading session.
    Based on typical crypto liquidity patterns (UTC).

    Low quality sessions have wider spreads, more false breakouts,
    and lower volume — signals are less reliable.
    """
    now_utc = datetime.now(timezone.utc)
    hour    = now_utc.hour

    # Hard block during dead hours
    if hour in SESSION_DEAD_HOURS:
        return {
            "quality":    0.0,
            "session":    "DEAD",
            "hour_utc":   hour,
            "pass":       False,
            "reason":     f"Dead zone hour {hour}:00 UTC — low liquidity",
        }

    quality  = 0.7  # default
    session  = "ASIAN"

    for (start, end), q in SESSION_QUALITY.items():
        if start <= hour < end:
            quality = q
            if start >= 13:
                session = "NY_OVERLAP"
            elif start >= 8:
                session = "LONDON"
            elif start >= 17:
                session = "NY"
            break

    return {
        "quality":  quality,
        "session":  session,
        "hour_utc": hour,
        "pass":     quality >= 0.7,
        "reason":   f"{session} session ({hour}:00 UTC) — quality {quality:.0%}",
    }


# ─── 6. DYNAMIC LEVERAGE ADVISOR ─────────────────────────────────────────────
def suggest_leverage(
    atr_pct: float,
    quality_score: float,
    portfolio_heat: float,
    session_quality: float,
    base_leverage: int = LEVERAGE,
) -> dict:
    """
    Suggests a leverage adjustment based on current conditions.
    Never suggests going ABOVE base_leverage — only reduces it.

    Risk factors that reduce leverage:
    - High ATR% (volatile market)
    - Low signal quality
    - High portfolio heat
    - Poor session quality
    """
    suggested = base_leverage
    reasons   = []

    # High volatility → reduce leverage
    if atr_pct > 4.0:
        suggested = min(suggested, 5)
        reasons.append(f"ATR {atr_pct:.1f}% is extreme")
    elif atr_pct > 2.5:
        suggested = min(suggested, 7)
        reasons.append(f"ATR {atr_pct:.1f}% is high")

    # Low quality signal → reduce leverage
    if quality_score < 50:
        suggested = min(suggested, 7)
        reasons.append(f"Signal quality {quality_score:.0f}/100 is marginal")
    elif quality_score < 40:
        suggested = min(suggested, 5)
        reasons.append(f"Signal quality {quality_score:.0f}/100 is low")

    # Portfolio heat building → reduce leverage
    if portfolio_heat > MAX_PORTFOLIO_HEAT * 0.7:
        suggested = min(suggested, 7)
        reasons.append(f"Portfolio heat {portfolio_heat:.1f}% is elevated")

    # Poor session → reduce leverage
    if session_quality < 0.75:
        suggested = min(suggested, 7)
        reasons.append(f"Low liquidity session")

    return {
        "suggested":  suggested,
        "base":       base_leverage,
        "reduced":    suggested < base_leverage,
        "reasons":    reasons,
        "warning":    len(reasons) >= 2,
    }


# ─── 7. FULL RISK GATE ────────────────────────────────────────────────────────
def risk_gate(
    symbol: str,
    direction: str,
    price: float,
    atr: float,
    quality_score: float,
    active_signals: dict,
    trade_history: list,
    enforce_session: bool = True,
) -> dict:
    """
    Master risk gate — runs ALL checks and returns a unified decision.
    Call this BEFORE sending any signal to Telegram.

    Returns:
    {
      "approved": bool,
      "reject_reason": str (if not approved),
      "sl": float, "tp1": float, "tp2": float,
      "risk_pct": float,
      "position_usdt": float,
      "contracts": float,
      "suggested_leverage": int,
      "checks": dict  (full breakdown of each check)
    }
    """
    atr_pct = (atr / price * 100) if price > 0 else 1.0

    # ── Run all checks ────────────────────────────────────────────────────
    session  = get_session_quality()
    heat     = get_portfolio_heat(active_signals)
    corr     = check_correlation(symbol, active_signals)
    lev_adv  = suggest_leverage(atr_pct, quality_score, heat["heat_pct"], session["quality"])

    # ── Compute levels first (needed for sizing) ──────────────────────────
    sl, tp1, tp2, levels_meta = calc_levels_v2(direction, price, atr, atr_pct)

    # ── Kelly position sizing ─────────────────────────────────────────────
    risk_pct, position_usdt, contracts = kelly_position_size(
        price, sl, trade_history
    )

    checks = {
        "session":     session,
        "heat":        heat,
        "correlation": corr,
        "leverage":    lev_adv,
        "levels":      levels_meta,
    }

    # ── Gate decisions (hard blocks) ──────────────────────────────────────
    if enforce_session and not session["pass"]:
        return {
            "approved": False,
            "reject_reason": session["reason"],
            "checks": checks,
            **_empty_levels(sl, tp1, tp2, risk_pct, position_usdt, contracts, lev_adv),
        }

    if not heat["can_open"]:
        return {
            "approved": False,
            "reject_reason": heat["reason"],
            "checks": checks,
            **_empty_levels(sl, tp1, tp2, risk_pct, position_usdt, contracts, lev_adv),
        }

    if not corr["pass"]:
        return {
            "approved": False,
            "reject_reason": corr["reason"],
            "checks": checks,
            **_empty_levels(sl, tp1, tp2, risk_pct, position_usdt, contracts, lev_adv),
        }

    # ── Approved ──────────────────────────────────────────────────────────
    return {
        "approved":           True,
        "reject_reason":      "",
        "sl":                 sl,
        "tp1":                tp1,
        "tp2":                tp2,
        "risk_pct":           risk_pct,
        "position_usdt":      position_usdt,
        "contracts":          contracts,
        "suggested_leverage": lev_adv["suggested"],
        "atr_pct":            atr_pct,
        "vol_regime":         levels_meta["vol_regime"],
        "rr_ratio":           levels_meta["rr_ratio"],
        "checks":             checks,
    }


def _empty_levels(sl, tp1, tp2, risk_pct, pos, contracts, lev_adv):
    return {
        "sl": sl, "tp1": tp1, "tp2": tp2,
        "risk_pct": risk_pct,
        "position_usdt": pos,
        "contracts": contracts,
        "suggested_leverage": lev_adv["suggested"],
        "atr_pct": 0, "vol_regime": "UNKNOWN", "rr_ratio": 0,
    }


# ─── TELEGRAM /risk COMMAND ───────────────────────────────────────────────────
async def cmd_risk(update, ctx, active_signals: dict, trade_history: list):
    """
    /risk — Live portfolio risk dashboard.
    Shows current heat, open positions, session quality, Kelly sizing.
    """
    heat    = get_portfolio_heat(active_signals)
    session = get_session_quality()
    kelly_r, _, _ = kelly_position_size(100, 98.5, trade_history)  # dummy entry/sl for rate only

    heat_bar = _heat_bar(heat["heat_pct"], heat["max_heat"])
    heat_emoji = "🟢" if heat["heat_pct"] < 3 else "🟡" if heat["heat_pct"] < 5 else "🔴"
    sess_emoji = "🟢" if session["quality"] >= 0.85 else "🟡" if session["quality"] >= 0.7 else "🔴"

    msg  = "📊 *Risk Dashboard*\n\n"
    msg += f"*Portfolio Heat:*\n"
    msg += f"  {heat_emoji} `{heat_bar}` {heat['heat_pct']:.1f}% / {heat['max_heat']}%\n"
    msg += f"  Open positions: `{heat['slots_used']}`\n\n"

    if heat["breakdown"]:
        msg += "*Open Risk Breakdown:*\n"
        for b in heat["breakdown"]:
            d_emoji = "🟢" if b["direction"] == "LONG" else "🔴"
            msg += f"  {d_emoji} {b['symbol']}: SL `{b['sl_dist']}%` away | heat `{b['heat_pct']}%`\n"
        msg += "\n"

    msg += f"*Session:*\n"
    msg += f"  {sess_emoji} {session['session']} | {session['hour_utc']}:00 UTC | Quality `{session['quality']:.0%}`\n\n"

    msg += f"*Kelly Sizing:*\n"
    msg += f"  Current risk/trade: `{kelly_r:.2f}%` (default: {RISK_PCT}%)\n"
    msg += f"  Based on last {KELLY_LOOKBACK} trades\n\n"

    if not heat["can_open"]:
        msg += "⛔ *NEW TRADES BLOCKED* — Portfolio heat too high\n"
    elif not session["pass"]:
        msg += "⛔ *NEW TRADES BLOCKED* — Dead zone session\n"
    else:
        msg += "✅ *Ready to trade*\n"

    await update.message.reply_text(msg, parse_mode="Markdown")


def _heat_bar(current: float, maximum: float, width: int = 10) -> str:
    filled = int((current / maximum) * width) if maximum > 0 else 0
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)


# ─── RISK BLOCK IN format_signal ADDON ───────────────────────────────────────
def format_risk_block(risk_result: dict) -> str:
    """
    Appends a risk management block to the Telegram signal message.
    Shows vol regime, R:R ratio, Kelly sizing, and leverage advice.
    """
    if not risk_result or not risk_result.get("approved"):
        return ""

    vol       = risk_result.get("vol_regime", "MEDIUM")
    rr        = risk_result.get("rr_ratio", 0)
    risk_pct  = risk_result.get("risk_pct", RISK_PCT)
    sug_lev   = risk_result.get("suggested_leverage", LEVERAGE)
    lev_adv   = risk_result.get("checks", {}).get("leverage", {})
    corr      = risk_result.get("checks", {}).get("correlation", {})
    session   = risk_result.get("checks", {}).get("session", {})

    vol_emoji = {"LOW": "😴", "MEDIUM": "⚖️", "HIGH": "⚠️", "EXTREME": "🚨"}.get(vol, "⚖️")
    sess_emoji = "🟢" if session.get("quality", 1) >= 0.85 else "🟡"

    msg  = "\n⚙️ *Risk Management*\n"
    msg += f"{vol_emoji} Volatility: `{vol}` | R:R Ratio: `1:{rr}`\n"
    msg += f"💰 Kelly Size: `{risk_pct:.2f}%` of account\n"
    msg += f"{sess_emoji} Session: `{session.get('session', '?')}` ({session.get('hour_utc', '?')}:00 UTC)\n"

    if sug_lev < LEVERAGE:
        reasons_str = " | ".join(lev_adv.get("reasons", []))
        msg += f"⚠️ Suggested leverage: `{sug_lev}x` (not {LEVERAGE}x) — {reasons_str}\n"

    if corr.get("group") not in (None, "NONE"):
        msg += f"🔗 Correlation group: `{corr.get('group', '?')}` — {corr.get('reason', '')}\n"

    return msg
