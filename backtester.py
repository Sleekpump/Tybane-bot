"""
Phyrobot — Phase 3: Backtesting Framework
────────────────────────────────────────────────────────
Replays historical OHLCV data through the Phase 1 scoring engine,
simulates trade execution, and produces a full performance report.

Features:
  1.  Walk-forward backtesting — no lookahead bias
  2.  Per-coin win rate, avg PnL, Sharpe, max drawdown
  3.  Per-regime performance (trending vs ranging)
  4.  Per-signal-type performance (MOMENTUM vs REVERSAL)
  5.  Auto-threshold tuner — finds optimal confluence/quality settings
  6.  Equity curve + drawdown tracking
  7.  Telegram /backtest command integration
  8.  JSON results saved to backtest_results.json

Usage:
  python3 backtester.py --coins BTC ETH SOL --days 90
  python3 backtester.py --all --days 60 --tune
  Or via Telegram: /backtest 30   (runs on top 10 coins, last 30 days)
"""

import os, json, logging, argparse, time
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional
import pandas as pd
import pandas_ta as ta
import numpy as np
import ccxt
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ─── BACKTEST CONFIG ──────────────────────────────────────────────────────────
LEVERAGE       = int(os.getenv("LEVERAGE", "10"))
ACCOUNT_SIZE   = float(os.getenv("ACCOUNT_SIZE", "200"))
RISK_PCT       = float(os.getenv("RISK_PCT", "1.5"))
ATR_SL_MULT    = 1.5   # SL = entry ± ATR * 1.5
ATR_TP1_MULT   = 1.5   # TP1 = entry ± ATR * 1.5
ATR_TP2_MULT   = 4.0   # TP2 = entry ± ATR * 4.0
MAX_HOLD_BARS  = 72    # max candles to hold before expiry (72 x 1H = 3 days)
RESULTS_FILE   = "backtest_results.json"
SLIPPAGE_PCT   = 0.05  # 0.05% slippage on entry/exit
TAKER_FEE_PCT  = 0.06  # 0.06% taker fee per side (Bitget)

# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────
@dataclass
class BacktestTrade:
    symbol:       str
    direction:    str
    entry_bar:    int
    entry_price:  float
    sl:           float
    tp1:          float
    tp2:          float
    quality:      float
    confidence:   str
    regime:       str
    signal_type:  str    # MOMENTUM | REVERSAL | WEAK
    confluence:   int

    exit_bar:     int    = 0
    exit_price:   float  = 0.0
    outcome:      str    = "OPEN"   # WIN | LOSS | BREAKEVEN | EXPIRED
    pnl_pct:      float  = 0.0      # % move (pre-leverage)
    pnl_usdt:     float  = 0.0
    tp1_hit:      bool   = False
    bars_held:    int    = 0

@dataclass
class CoinStats:
    symbol:      str
    total:       int   = 0
    wins:        int   = 0
    losses:      int   = 0
    expired:     int   = 0
    total_pnl:   float = 0.0
    peak_equity: float = 0.0
    max_dd:      float = 0.0
    trades:      list  = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        closed = self.wins + self.losses
        return round(self.wins / closed * 100, 1) if closed > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return round(self.total_pnl / self.total, 2) if self.total > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        gross_profit = sum(t.pnl_usdt for t in self.trades if t.pnl_usdt > 0)
        gross_loss   = abs(sum(t.pnl_usdt for t in self.trades if t.pnl_usdt < 0))
        return round(gross_profit / gross_loss, 2) if gross_loss > 0 else float("inf")


# ─── EXCHANGE SETUP ───────────────────────────────────────────────────────────
exchange = ccxt.bitget({"options": {"defaultType": "swap"}})

def fetch_historical(symbol: str, timeframe: str = "1h", days: int = 90) -> pd.DataFrame:
    """
    Fetch historical OHLCV data going back `days` days.
    Handles pagination for longer lookbacks.
    """
    since_ms = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_data = []
    limit    = 500
    while True:
        try:
            raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=limit)
            if not raw:
                break
            all_data.extend(raw)
            if len(raw) < limit:
                break
            since_ms = raw[-1][0] + 1
            time.sleep(0.2)
        except Exception as e:
            log.error(f"Fetch error {symbol} {timeframe}: {e}")
            break

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


# ─── WALK-FORWARD SIGNAL GENERATOR ───────────────────────────────────────────
def generate_signals_walkforward(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    warmup_bars: int = 210,
    step: int = 4,         # check for new signal every 4 bars (every 4 hours)
) -> list[BacktestTrade]:
    """
    Walk-forward signal generation — at each bar, only uses data UP TO that bar.
    No lookahead. Returns a list of BacktestTrade objects (unfilled).
    """
    # Lazy import to avoid circular dependency
    from signal_engine import compute_signal_quality, detect_regime, score_timeframe_v2

    trades    = []
    n         = len(df_1h)
    active    = {}   # symbol→trade currently open (one at a time per symbol)

    for i in range(warmup_bars, n, step):
        # Slice data up to current bar — NO FUTURE DATA
        slice_1h = df_1h.iloc[:i+1].copy()
        # Map 4H bars up to same timestamp
        current_ts = df_1h.iloc[i]["timestamp"]
        slice_4h   = df_4h[df_4h["timestamp"] <= current_ts].copy()

        if len(slice_1h) < warmup_bars or len(slice_4h) < 50:
            continue

        # Skip if already in an active trade
        if "open" in active:
            continue

        try:
            quality_result = compute_signal_quality(slice_1h, slice_4h)
        except Exception as e:
            log.debug(f"Signal error at bar {i}: {e}")
            continue

        if not quality_result.get("passed"):
            continue

        direction = quality_result["direction"]
        if direction == "NEUTRAL":
            continue

        price   = float(slice_1h["close"].iloc[-1])
        atr_s   = ta.atr(slice_1h["high"], slice_1h["low"], slice_1h["close"], length=14)
        atr_val = float(atr_s.dropna().iloc[-1]) if atr_s is not None and len(atr_s.dropna()) > 0 else price * 0.01

        # Levels
        if direction == "LONG":
            sl  = price - atr_val * ATR_SL_MULT
            tp1 = price + atr_val * ATR_TP1_MULT
            tp2 = price + atr_val * ATR_TP2_MULT
        else:
            sl  = price + atr_val * ATR_SL_MULT
            tp1 = price - atr_val * ATR_TP1_MULT
            tp2 = price - atr_val * ATR_TP2_MULT

        # Entry with slippage
        slip     = price * (SLIPPAGE_PCT / 100)
        entry    = price + slip if direction == "LONG" else price - slip

        regime   = quality_result.get("regime", {})
        # Simple signal type classification
        gs       = quality_result.get("group_scores", {})
        struct_s = gs.get("structure", 0)
        mom_s    = gs.get("momentum", 0)
        if direction == "LONG":
            is_momentum = struct_s > 0.3 and mom_s > 0.2
        else:
            is_momentum = struct_s < -0.3 and mom_s < -0.2
        signal_type = "MOMENTUM" if is_momentum else "REVERSAL"

        trade = BacktestTrade(
            symbol      = "SIM",
            direction   = direction,
            entry_bar   = i,
            entry_price = entry,
            sl          = sl,
            tp1         = tp1,
            tp2         = tp2,
            quality     = quality_result.get("quality_score", 0),
            confidence  = quality_result.get("confidence", "MEDIUM"),
            regime      = regime.get("regime", "UNKNOWN"),
            signal_type = signal_type,
            confluence  = quality_result.get("confluence_groups", 0),
        )
        active["open"] = trade
        trades.append(trade)

    return trades


# ─── TRADE SIMULATOR ──────────────────────────────────────────────────────────
def simulate_trades(
    trades: list[BacktestTrade],
    df_1h:  pd.DataFrame,
) -> list[BacktestTrade]:
    """
    For each pending trade, walk forward through subsequent bars
    and determine outcome: WIN (TP2), LOSS (SL), or EXPIRED.
    Accounts for slippage and fees.
    """
    risk_per_trade = ACCOUNT_SIZE * (RISK_PCT / 100)

    for trade in trades:
        start = trade.entry_bar + 1
        end   = min(start + MAX_HOLD_BARS, len(df_1h))

        for j in range(start, end):
            bar_high = float(df_1h["high"].iloc[j])
            bar_low  = float(df_1h["low"].iloc[j])

            if trade.direction == "LONG":
                # Check SL first (worst case within bar)
                if bar_low <= trade.sl:
                    trade.exit_price = trade.sl
                    trade.outcome    = "LOSS"
                    trade.exit_bar   = j
                    break
                # TP1 hit
                if not trade.tp1_hit and bar_high >= trade.tp1:
                    trade.tp1_hit = True
                # TP2 hit
                if bar_high >= trade.tp2:
                    trade.exit_price = trade.tp2
                    trade.outcome    = "WIN"
                    trade.exit_bar   = j
                    break
            else:
                # SHORT
                if bar_high >= trade.sl:
                    trade.exit_price = trade.sl
                    trade.outcome    = "LOSS"
                    trade.exit_bar   = j
                    break
                if not trade.tp1_hit and bar_low <= trade.tp1:
                    trade.tp1_hit = True
                if bar_low <= trade.tp2:
                    trade.exit_price = trade.tp2
                    trade.outcome    = "WIN"
                    trade.exit_bar   = j
                    break
        else:
            # Max hold expired
            trade.exit_price = float(df_1h["close"].iloc[min(end - 1, len(df_1h) - 1)])
            trade.outcome    = "EXPIRED"
            trade.exit_bar   = end - 1

        trade.bars_held = trade.exit_bar - trade.entry_bar

        # PnL calculation
        entry = trade.entry_price
        exit_ = trade.exit_price
        fee   = (entry + exit_) * (TAKER_FEE_PCT / 100)

        if trade.direction == "LONG":
            move_pct    = (exit_ - entry) / entry * 100
        else:
            move_pct    = (entry - exit_) / entry * 100

        leveraged_pct = move_pct * LEVERAGE
        fee_pct       = fee / entry * 100 * LEVERAGE
        net_pct       = leveraged_pct - fee_pct

        trade.pnl_pct   = round(net_pct, 2)
        trade.pnl_usdt  = round(risk_per_trade * net_pct / 100, 2)

        # Reclassify expired trades
        if trade.outcome == "EXPIRED":
            if trade.pnl_usdt > 0:
                trade.outcome = "BREAKEVEN"

    return trades


# ─── EQUITY CURVE & DRAWDOWN ──────────────────────────────────────────────────
def compute_equity_curve(trades: list[BacktestTrade]) -> dict:
    """
    Compute running equity, peak equity, max drawdown, and Sharpe ratio.
    """
    equity  = ACCOUNT_SIZE
    peak    = ACCOUNT_SIZE
    max_dd  = 0.0
    curve   = [ACCOUNT_SIZE]
    returns = []

    sorted_trades = sorted(trades, key=lambda t: t.exit_bar)
    for t in sorted_trades:
        equity += t.pnl_usdt
        curve.append(equity)
        returns.append(t.pnl_usdt / ACCOUNT_SIZE)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    # Sharpe (annualized, assuming 6 trades/day avg)
    if len(returns) > 1:
        avg_r    = np.mean(returns)
        std_r    = np.std(returns)
        sharpe   = round((avg_r / std_r) * np.sqrt(len(returns)), 2) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    return {
        "final_equity":  round(equity, 2),
        "peak_equity":   round(peak, 2),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        sharpe,
        "total_return":  round((equity - ACCOUNT_SIZE) / ACCOUNT_SIZE * 100, 2),
        "curve":         [round(c, 2) for c in curve[-50:]],  # last 50 points
    }


# ─── PERFORMANCE BREAKDOWN ────────────────────────────────────────────────────
def compute_stats(trades: list[BacktestTrade], label: str = "Overall") -> dict:
    if not trades:
        return {"label": label, "total": 0}

    closed  = [t for t in trades if t.outcome in ("WIN","LOSS","BREAKEVEN","EXPIRED")]
    wins    = [t for t in closed if t.outcome == "WIN"]
    losses  = [t for t in closed if t.outcome == "LOSS"]
    total   = len(closed)
    wr      = round(len(wins) / total * 100, 1) if total > 0 else 0
    avg_pnl = round(sum(t.pnl_usdt for t in closed) / total, 2) if total > 0 else 0
    gross_p = sum(t.pnl_usdt for t in closed if t.pnl_usdt > 0)
    gross_l = abs(sum(t.pnl_usdt for t in closed if t.pnl_usdt < 0))
    pf      = round(gross_p / gross_l, 2) if gross_l > 0 else float("inf")
    avg_bars = round(sum(t.bars_held for t in closed) / total, 1) if total > 0 else 0

    return {
        "label":          label,
        "total":          total,
        "wins":           len(wins),
        "losses":         len(losses),
        "expired":        len([t for t in closed if t.outcome in ("EXPIRED","BREAKEVEN")]),
        "win_rate":       wr,
        "avg_pnl_usdt":   avg_pnl,
        "total_pnl_usdt": round(sum(t.pnl_usdt for t in closed), 2),
        "profit_factor":  pf,
        "avg_bars_held":  avg_bars,
    }


# ─── AUTO THRESHOLD TUNER ─────────────────────────────────────────────────────
def tune_thresholds(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
) -> dict:
    """
    Grid search over confluence and quality thresholds.
    Finds the combination that maximizes win rate on historical data.
    Warning: only use for tuning — validate on out-of-sample data.
    """
    from signal_engine import CONFLUENCE_MIN, QUALITY_THRESHOLD_MEDIUM
    import signal_engine as se

    results = []
    original_conf    = se.CONFLUENCE_MIN
    original_quality = se.QUALITY_THRESHOLD_MEDIUM

    for conf in [2, 3, 4]:
        for quality in [35, 45, 55, 65]:
            se.CONFLUENCE_MIN          = conf
            se.QUALITY_THRESHOLD_MEDIUM = quality

            try:
                trades   = generate_signals_walkforward(df_1h, df_4h)
                trades   = simulate_trades(trades, df_1h)
                stats    = compute_stats(trades)
                eq       = compute_equity_curve(trades)
                results.append({
                    "confluence_min":    conf,
                    "quality_threshold": quality,
                    "total_trades":      stats.get("total", 0),
                    "win_rate":          stats.get("win_rate", 0),
                    "total_pnl":         stats.get("total_pnl_usdt", 0),
                    "profit_factor":     stats.get("profit_factor", 0),
                    "max_drawdown":      eq.get("max_drawdown", 100),
                    "sharpe":            eq.get("sharpe", 0),
                })
            except Exception as e:
                log.error(f"Tuning error conf={conf} quality={quality}: {e}")

    # Restore originals
    se.CONFLUENCE_MIN          = original_conf
    se.QUALITY_THRESHOLD_MEDIUM = original_quality

    if not results:
        return {}

    # Score each config: balance win rate, trade count, drawdown, PF
    def config_score(r):
        if r["total_trades"] < 5:
            return 0
        return (
            r["win_rate"] * 0.4 +
            min(r["profit_factor"], 5) * 10 +
            r["sharpe"] * 10 -
            r["max_drawdown"] * 0.5
        )

    results.sort(key=config_score, reverse=True)
    best = results[0]
    log.info(f"Best config: confluence={best['confluence_min']} quality={best['quality_threshold']} | WR={best['win_rate']}% | PF={best['profit_factor']}")
    return {"best": best, "all": results[:6]}


# ─── MAIN BACKTEST RUNNER ─────────────────────────────────────────────────────
def run_backtest(
    symbols: list[str],
    days:    int  = 60,
    tune:    bool = False,
) -> dict:
    """
    Run full backtest across a list of symbols.
    Returns aggregated results dict.
    """
    all_trades       = []
    per_coin_stats   = {}
    regime_trades    = {"TRENDING_UP": [], "TRENDING_DOWN": [], "RANGING": []}
    type_trades      = {"MOMENTUM": [], "REVERSAL": []}
    tuning_results   = {}
    start_time       = time.time()

    log.info(f"Starting backtest: {len(symbols)} coins | {days} days")

    for i, symbol in enumerate(symbols):
        log.info(f"[{i+1}/{len(symbols)}] Backtesting {symbol}...")
        try:
            df_1h = fetch_historical(symbol, "1h", days + 10)  # +10 for warmup
            df_4h = fetch_historical(symbol, "4h", days + 10)

            if len(df_1h) < 220 or len(df_4h) < 50:
                log.warning(f"Not enough data for {symbol}")
                continue

            # Generate signals
            trades = generate_signals_walkforward(df_1h, df_4h)
            if not trades:
                log.info(f"No signals for {symbol}")
                continue

            # Simulate exits
            trades = simulate_trades(trades, df_1h)

            # Tag symbol
            for t in trades:
                t.symbol = symbol

            all_trades.extend(trades)

            # Per-coin stats
            stats = compute_stats(trades, symbol)
            eq    = compute_equity_curve(trades)
            stats.update(eq)
            per_coin_stats[symbol] = stats

            # Regime breakdown
            for t in trades:
                regime_trades.get(t.regime, []).append(t)

            # Signal type breakdown
            for t in trades:
                type_trades.get(t.signal_type, []).append(t)

            # Tune on first coin only (representative)
            if tune and i == 0:
                log.info("Running threshold tuner...")
                tuning_results = tune_thresholds(df_1h, df_4h)

            time.sleep(0.3)

        except Exception as e:
            log.error(f"Backtest error {symbol}: {e}")
            continue

    # Overall stats
    overall       = compute_stats(all_trades, "Overall")
    equity_curve  = compute_equity_curve(all_trades)
    regime_stats  = {r: compute_stats(t, r) for r, t in regime_trades.items()}
    type_stats    = {st: compute_stats(t, st) for st, t in type_trades.items()}

    elapsed = round(time.time() - start_time, 1)

    results = {
        "meta": {
            "coins":      len(symbols),
            "days":       days,
            "total_trades": len(all_trades),
            "run_time_s": elapsed,
            "timestamp":  datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        },
        "overall":    overall,
        "equity":     equity_curve,
        "per_coin":   per_coin_stats,
        "by_regime":  regime_stats,
        "by_type":    type_stats,
        "tuning":     tuning_results,
    }

    # Save
    try:
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        log.error(f"Save error: {e}")

    return results


# ─── TELEGRAM REPORT FORMATTER ────────────────────────────────────────────────
def format_backtest_report(results: dict) -> list[str]:
    """
    Format backtest results into Telegram-ready messages.
    Returns list of message strings (split to avoid 4096 char limit).
    """
    overall  = results.get("overall", {})
    equity   = results.get("equity", {})
    meta     = results.get("meta", {})
    by_type  = results.get("by_type", {})
    by_regime = results.get("by_regime", {})
    per_coin = results.get("per_coin", {})
    tuning   = results.get("tuning", {})

    wr       = overall.get("win_rate", 0)
    wr_emoji = "🟢" if wr >= 60 else "🟡" if wr >= 50 else "🔴"

    # ── Message 1: Summary ────────────────────────────────────────────────
    msg1  = "📊 *Backtest Report*\n"
    msg1 += f"_{meta.get('coins', 0)} coins | {meta.get('days', 0)} days | {meta.get('timestamp', '')}_ \n\n"

    msg1 += "*Overall Performance:*\n"
    msg1 += f"  Total Trades: `{overall.get('total', 0)}`\n"
    msg1 += f"  {wr_emoji} Win Rate: `{wr}%`\n"
    msg1 += f"  Profit Factor: `{overall.get('profit_factor', 0)}`\n"
    msg1 += f"  Total PnL: `${overall.get('total_pnl_usdt', 0):+.2f}`\n"
    msg1 += f"  Total Return: `{equity.get('total_return', 0):+.1f}%`\n"
    msg1 += f"  Max Drawdown: `{equity.get('max_drawdown', 0):.1f}%`\n"
    msg1 += f"  Sharpe Ratio: `{equity.get('sharpe', 0)}`\n"
    msg1 += f"  Avg Hold Time: `{overall.get('avg_bars_held', 0)} bars`\n\n"

    # Signal type breakdown
    msg1 += "*By Signal Type:*\n"
    for stype, stats in by_type.items():
        if stats.get("total", 0) > 0:
            emoji = "⚡" if stype == "MOMENTUM" else "🔄"
            msg1 += f"  {emoji} {stype}: `{stats['total']}` trades | WR `{stats.get('win_rate', 0)}%` | PF `{stats.get('profit_factor', 0)}`\n"

    msg1 += "\n*By Market Regime:*\n"
    regime_emojis = {"TRENDING_UP": "📈", "TRENDING_DOWN": "📉", "RANGING": "↔️"}
    for regime, stats in by_regime.items():
        if stats.get("total", 0) > 0:
            re = regime_emojis.get(regime, "⚪")
            msg1 += f"  {re} {regime}: `{stats['total']}` trades | WR `{stats.get('win_rate', 0)}%`\n"

    # ── Message 2: Per-coin breakdown ────────────────────────────────────
    msg2  = "📊 *Per-Coin Performance:*\n\n"
    sorted_coins = sorted(
        per_coin.items(),
        key=lambda x: x[1].get("win_rate", 0),
        reverse=True
    )
    for symbol, stats in sorted_coins[:15]:
        label = symbol.split("/")[0]
        wr_c  = stats.get("win_rate", 0)
        total = stats.get("total", 0)
        pnl   = stats.get("total_pnl_usdt", 0)
        pf    = stats.get("profit_factor", 0)
        dd    = stats.get("max_drawdown", 0)
        ce    = "🟢" if wr_c >= 60 else "🟡" if wr_c >= 50 else "🔴"
        if total > 0:
            msg2 += f"{ce} *{label}*: WR `{wr_c}%` | `{total}` trades | PnL `${pnl:+.2f}` | PF `{pf}` | DD `{dd:.1f}%`\n"

    # ── Message 3: Tuning recommendations ────────────────────────────────
    msg3 = ""
    if tuning and tuning.get("best"):
        best = tuning["best"]
        msg3  = "🔧 *Auto-Tuner Recommendations:*\n\n"
        msg3 += f"Optimal settings found:\n"
        msg3 += f"  `CONFLUENCE_MIN = {best.get('confluence_min', 3)}`\n"
        msg3 += f"  `QUALITY_THRESHOLD_MEDIUM = {best.get('quality_threshold', 45)}`\n\n"
        msg3 += f"Expected improvement:\n"
        msg3 += f"  Win Rate: `{best.get('win_rate', 0)}%`\n"
        msg3 += f"  Profit Factor: `{best.get('profit_factor', 0)}`\n"
        msg3 += f"  Max Drawdown: `{best.get('max_drawdown', 0):.1f}%`\n\n"
        msg3 += "_Apply by editing signal_engine.py constants_\n"

        msg3 += "\n*All configs tested:*\n"
        for cfg in tuning.get("all", [])[:6]:
            msg3 += (
                f"  C={cfg['confluence_min']} Q={cfg['quality_threshold']}: "
                f"WR `{cfg['win_rate']}%` | "
                f"PF `{cfg['profit_factor']}` | "
                f"DD `{cfg['max_drawdown']:.1f}%`\n"
            )

    messages = [m for m in [msg1, msg2, msg3] if m.strip()]
    return messages


# ─── TELEGRAM COMMAND HANDLER ─────────────────────────────────────────────────
async def cmd_backtest(update, ctx, exchange_ref, coin_labels: dict):
    """
    /backtest [days] [--tune]
    Examples:
      /backtest        → 30 days, top 10 coins
      /backtest 60     → 60 days, top 10 coins
      /backtest 30 tune → 30 days + auto-tuner
    """
    args  = ctx.args if ctx.args else []
    days  = 30
    tune  = False

    for arg in args:
        if arg.isdigit():
            days = min(int(arg), 90)
        if arg.lower() == "tune":
            tune = True

    # Use top 10 coins for speed (full 40 would take ~20 mins)
      try:
             import json
             with open("paper_trades.json") as f:
                 trades = json.load(f)
             recent_symbols = list(dict.fromkeys([t["symbol"] for t in trades[-20:]]))[:10]
             symbols = recent_symbols if recent_symbols else list(coin_labels.keys())[:10]
             log.info("Backtest using paper trade coins: " + str([s.split("/")[0] for s in symbols]))
         except Exception:
             symbols = list(coin_labels.keys())[:10]

    await update.message.reply_text(
        f"🔬 *Starting backtest...*\n\n"
        f"Coins: {len(symbols)}\n"
        f"Period: {days} days\n"
        f"Auto-tune: {'Yes' if tune else 'No'}\n\n"
        f"_This takes 3-8 minutes. Results incoming..._",
        parse_mode="Markdown"
    )

    try:
        import asyncio
        loop    = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None,
            lambda: run_backtest(symbols, days, tune)
        )
        messages = format_backtest_report(results)
        for msg in messages:
            if msg.strip():
                await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"Backtest error: {str(e)}")


# ─── CLI RUNNER ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phyrobot Backtester")
    parser.add_argument("--coins", nargs="+", default=["BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT"])
    parser.add_argument("--days",  type=int,   default=60)
    parser.add_argument("--tune",  action="store_true")
    args = parser.parse_args()

    results  = run_backtest(args.coins, args.days, args.tune)
    messages = format_backtest_report(results)
    print("\n" + "="*60)
    for msg in messages:
        # Strip Markdown for CLI output
        clean = msg.replace("*","").replace("`","").replace("_","")
        print(clean)
        print("-"*60)
