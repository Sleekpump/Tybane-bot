"""
Phyrobot — Phase 2: AI Signal Validation Layer
────────────────────────────────────────────────────────
Uses Groq (LLaMA 3 70B) as a second-opinion analyst that:

  1. Reviews the full signal context (regime, group scores, price action)
  2. Identifies hidden risks the scoring engine can't see
  3. Returns a structured verdict: APPROVE / DOWNGRADE / REJECT
  4. Adjusts confidence level and quality score based on AI verdict
  5. Adds a human-readable AI rationale to the Telegram message

The AI does NOT replace the scoring engine — it acts as a senior analyst
reviewing a junior analyst's work. The scoring engine sets the floor,
the AI raises or lowers it.

Architecture:
  Phase 1 (signal_engine.py) → quality_result dict
  Phase 2 (ai_validator.py)  → enriched quality_result dict
  bot.py                     → sends final signal

Groq model: llama3-70b-8192 (fast, cheap, strong reasoning)
Fallback: if Groq unavailable, Phase 1 result passes through unchanged
"""

import os
import json
import logging
import time
import hashlib
from typing import Optional

log = logging.getLogger(__name__)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
AI_VALIDATION_ENABLED  = True    # toggle off to bypass AI (falls back to Phase 1)
AI_TIMEOUT_SECONDS     = 8       # max wait for Groq response
AI_MIN_QUALITY_TO_CALL = 70      # raised 65→70: cuts ~30% of calls, Grade C signals don't need AI
AI_VETO_THRESHOLD      = 30      # AI confidence below this = REJECT
AI_BOOST_THRESHOLD     = 70      # AI confidence above this = confidence upgrade possible
AI_CACHE_TTL           = 1800    # raised: cache for 30 min instead of 15 (saves ~30% tokens)

# ─── GEMINI FALLBACK CONFIG ───────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# gemini-1.5-flash-latest was removed from v1beta — replaced with gemini-2.0-flash-lite.
# gemini-2.0-flash-lite: free tier, 1500 RPD / 15 RPM, available on v1beta.
# Override via GEMINI_MODEL env var if needed.
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
try:
    import google.generativeai as genai
    _GEMINI_IMPORT_OK = True
except ImportError:
    _GEMINI_IMPORT_OK = False
    log.warning("google-generativeai not installed — Gemini disabled. Run: pip install google-generativeai --break-system-packages")
GEMINI_ENABLED = bool(GEMINI_API_KEY) and _GEMINI_IMPORT_OK

# ─── GEMINI RATE LIMITER ──────────────────────────────────────────────────────
# Free tier limits: 15 RPM, 1500 RPD, 1M TPM.
# Without this, every Groq overflow (up to 34 coins/cycle) hits Gemini at once
# blowing the per-minute limit immediately and burning the daily quota by midday.
_gemini_last_call:     float = 0.0
_gemini_call_lock             = None
_gemini_blocked_until: float = 0.0   # set when 429 fires — parse retry_after from error
_gemini_daily_count:   int   = 0
_gemini_daily_reset:   float = 0.0
GEMINI_MIN_INTERVAL    = 5.0          # 5s between calls = 12 RPM (under 15 RPM limit)
GEMINI_MAX_PER_CYCLE   = 3            # max Gemini calls per scan cycle
GEMINI_MAX_PER_DAY     = 1400         # conservative — real limit is 1500
_gemini_cycle_count:   int   = 0
_gemini_cycle_reset:   float = 0.0

def _gemini_rate_ok() -> bool:
    """Returns True if a Gemini call is allowed right now."""
    global _gemini_last_call, _gemini_call_lock, _gemini_blocked_until
    global _gemini_daily_count, _gemini_daily_reset
    global _gemini_cycle_count, _gemini_cycle_reset
    import threading
    if _gemini_call_lock is None:
        _gemini_call_lock = threading.Lock()
    with _gemini_call_lock:
        now = time.time()
        # Hard block from 429 — respect retry_after from the API response
        if now < _gemini_blocked_until:
            wait = round(_gemini_blocked_until - now, 1)
            log.debug(f"Gemini blocked for {wait}s (429 backoff)")
            return False
        # Daily cap
        if now - _gemini_daily_reset > 86400:
            _gemini_daily_count = 0
            _gemini_daily_reset = now
        if _gemini_daily_count >= GEMINI_MAX_PER_DAY:
            log.warning("Gemini daily cap reached — using Phase 1 only for remainder of day")
            return False
        # Per-cycle cap (reset every 5 min)
        if now - _gemini_cycle_reset > 300:
            _gemini_cycle_count = 0
            _gemini_cycle_reset = now
        if _gemini_cycle_count >= GEMINI_MAX_PER_CYCLE:
            return False
        # Per-call interval
        if now - _gemini_last_call < GEMINI_MIN_INTERVAL:
            return False
        return True

def _gemini_record_call():
    global _gemini_last_call, _gemini_daily_count, _gemini_cycle_count
    _gemini_last_call    = time.time()
    _gemini_daily_count += 1
    _gemini_cycle_count += 1

def _gemini_set_backoff(retry_after_seconds: float = 30.0):
    """Called when Gemini returns 429 — block calls until retry window passes."""
    global _gemini_blocked_until
    _gemini_blocked_until = time.time() + retry_after_seconds + 2.0  # +2s buffer
    log.warning(f"Gemini 429 — backing off for {retry_after_seconds:.0f}s")


# Groq free tier: ~30 RPM for llama-3.3-70b-versatile.
# Every non-NEUTRAL coin in a 40-coin scan fires within seconds = instant 429.
# Solution: enforce minimum gap between calls + cap per scan cycle.
_groq_last_call: float = 0.0          # timestamp of last successful Groq call
_groq_call_lock = None                 # threading.Lock — created lazily
GROQ_MIN_INTERVAL = 3.0               # min seconds between Groq calls (~20 RPM max)
GROQ_MAX_PER_CYCLE = 6                # max Groq calls per scan cycle
_groq_cycle_count: int = 0            # calls made this cycle
_groq_cycle_reset: float = 0.0        # timestamp when cycle counter was reset

def _reset_groq_cycle():
    """Call at the start of each scan cycle to reset the per-cycle counter."""
    global _groq_cycle_count, _groq_cycle_reset
    _groq_cycle_count = 0
    _groq_cycle_reset = time.time()

def _groq_rate_ok() -> bool:
    """
    Returns True if a Groq call is allowed right now.
    Enforces both per-call interval and per-cycle cap.
    """
    global _groq_last_call, _groq_call_lock, _groq_cycle_count, _groq_cycle_reset
    import threading
    if _groq_call_lock is None:
        _groq_call_lock = threading.Lock()
    with _groq_call_lock:
        now = time.time()
        # Reset cycle counter every 5 minutes (one scan cycle)
        if now - _groq_cycle_reset > 300:
            _groq_cycle_count = 0
            _groq_cycle_reset = now
        # Check cap
        if _groq_cycle_count >= GROQ_MAX_PER_CYCLE:
            return False
        # Check interval
        if now - _groq_last_call < GROQ_MIN_INTERVAL:
            return False
        return True

def _groq_record_call():
    """Record that a Groq call was made."""
    global _groq_last_call, _groq_cycle_count
    _groq_last_call  = time.time()
    _groq_cycle_count += 1


_ai_cache: dict = {}

def _cleanup_ai_cache():
    """Remove stale cache entries. Call once per scan cycle."""
    now = time.time()
    stale = [k for k, v in _ai_cache.items() if now - v.get("ts", 0) > AI_CACHE_TTL]
    for k in stale:
        del _ai_cache[k]
    if stale:
        log.debug(f"AI cache: evicted {len(stale)} stale entries")

def _cache_key(symbol: str, direction: str, quality_score: float) -> str:
    raw = f"{symbol}_{direction}_{int(quality_score)}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]

def _get_cached(key: str) -> Optional[dict]:
    _cleanup_ai_cache()   # evict stale entries on each call
    entry = _ai_cache.get(key)
    if entry and time.time() - entry["ts"] < AI_CACHE_TTL:
        return entry["result"]
    return None

def _set_cached(key: str, result: dict):
    _ai_cache[key] = {"result": result, "ts": time.time()}


# ─── CONTEXT BUILDER ──────────────────────────────────────────────────────────
def build_ai_context(
    symbol: str,
    quality_result: dict,
    price: float,
    funding_rate: float,
    news_sentiment: str = "NEUTRAL",
    news_points: list = None,
) -> str:
    """
    Build a compact, information-dense prompt context for the AI.
    Keeps token count low while giving the AI everything it needs.
    """
    direction     = quality_result.get("direction", "NEUTRAL")
    quality       = quality_result.get("quality_score", 0)
    confidence    = quality_result.get("confidence", "LOW")
    regime        = quality_result.get("regime", {})
    group_scores  = quality_result.get("group_scores", {})
    signals       = quality_result.get("signals", [])
    confluence    = quality_result.get("confluence_groups", 0)
    filters       = quality_result.get("filters", {})

    regime_str   = regime.get("regime", "UNKNOWN")
    adx_val      = regime.get("adx", 0)
    slope        = regime.get("slope", 0)
    atr_pct      = filters.get("volatility", {}).get("atr_pct", 0)
    tf_aligned   = filters.get("tf_alignment", {}).get("pass", False)

    # Format group scores as clean percentages
    gs_str = " | ".join([
        f"{k.upper()[:4]}:{'+' if v >= 0 else ''}{v:.2f}"
        for k, v in group_scores.items()
    ])

    # Top signals (max 6)
    sigs_str = "\n".join([f"  - {s}" for s in signals[:6]])

    # News
    news_str = f"Market news sentiment: {news_sentiment}"
    if news_points:
        news_str += "\n" + "\n".join([f"  - {p}" for p in news_points[:3]])

    prompt = f"""You are a professional crypto futures trader and risk analyst reviewing a trading signal generated by a quantitative algorithm.

SIGNAL CONTEXT:
  Symbol: {symbol}
  Direction: {direction}
  Entry Price: ${price:.4f}
  Algorithm Quality Score: {quality}/100 ({confidence})
  Confluence: {confluence}/5 indicator groups agree
  Timeframe Aligned: {"YES" if tf_aligned else "NO"}

MARKET REGIME:
  Regime: {regime_str}
  ADX Strength: {adx_val}
  EMA50 Slope: {slope:+.3f}%
  ATR Volatility: {atr_pct:.2f}% of price

INDICATOR GROUP SCORES (-1.0 bearish to +1.0 bullish):
  {gs_str}

KEY SIGNALS DETECTED:
{sigs_str}

FUNDING RATE: {funding_rate*100:.4f}% (positive = longs paying shorts)

{news_str}

YOUR TASK:
Analyze this signal as a senior trader would. Consider:
1. Does the regime support this signal type?
2. Are there any conflicting signals or hidden risks?
3. Is the funding rate a concern (extreme positive = crowded long, extreme negative = crowded short)?
4. Does the confluence and quality score justify the trade?
5. Is there any reason this signal could be a false positive?

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "verdict": "APPROVE" | "DOWNGRADE" | "REJECT",
  "ai_confidence": <integer 0-100>,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "rationale": "<one concise sentence explaining your verdict>",
  "key_risk": "<one sentence — the biggest risk with this trade>",
  "suggested_action": "<one actionable sentence for the trader>"
}}"""

    return prompt


# ─── AI VALIDATOR ─────────────────────────────────────────────────────────────
def validate_signal_with_ai(
    groq_client,
    symbol: str,
    quality_result: dict,
    price: float,
    funding_rate: float = 0.0,
    news_sentiment: str = "NEUTRAL",
    news_points: list = None,
) -> dict:
    """
    Core AI validation function. Calls Groq and returns enriched result.

    Returns:
    {
      "verdict": "APPROVE" | "DOWNGRADE" | "REJECT" | "BYPASS",
      "ai_confidence": int 0-100,
      "risk_level": "LOW" | "MEDIUM" | "HIGH",
      "rationale": str,
      "key_risk": str,
      "suggested_action": str,
      "final_confidence": "HIGH" | "MEDIUM" | "LOW",
      "final_quality": float,
      "ai_used": bool,
    }
    """
    default = {
        "verdict": "BYPASS",
        "ai_confidence": 50,
        "risk_level": "MEDIUM",
        "rationale": "AI validation bypassed",
        "key_risk": "Unknown",
        "suggested_action": "Follow scoring engine signal",
        "final_confidence": quality_result.get("confidence") or "MEDIUM",
        "final_quality": quality_result.get("quality_score") or quality_result.get("abs_score", 0),
        "ai_used": False,
    }

    if not AI_VALIDATION_ENABLED or groq_client is None:
        return default

    quality = quality_result.get("quality_score") or quality_result.get("abs_score", 0)
    if quality < AI_MIN_QUALITY_TO_CALL:
        default["rationale"] = f"Quality too low for AI review ({quality}/100)"
        return default

    # Rate limit check — prevents 429 cascade during scan cycles
    if not _groq_rate_ok():
        log.debug(f"Groq rate limit gate — skipping AI for {symbol} (cycle cap or interval)")
        exhausted = dict(default)
        exhausted["groq_exhausted"] = True
        exhausted["rationale"] = "Groq rate gate — Gemini fallback triggered"
        return exhausted

    # Cache check
    cache_key = _cache_key(symbol, quality_result.get("direction", ""), quality)
    cached = _get_cached(cache_key)
    if cached:
        log.info(f"AI cache hit for {symbol}")
        return cached

    try:
        prompt = build_ai_context(
            symbol, quality_result, price, funding_rate,
            news_sentiment, news_points or []
        )

        _groq_record_call()   # record before call so burst is tracked even on failure
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=400,
            temperature=0.1,        # low temperature = consistent, analytical responses
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a disciplined crypto futures risk analyst. "
                        "You respond ONLY with valid JSON. No markdown, no explanation outside JSON. "
                        "Be conservative — when in doubt, DOWNGRADE not APPROVE."
                    )
                },
                {"role": "user", "content": prompt}
            ],
        )

        raw = response.choices[0].message.content.strip()

        # Strip any accidental markdown fences
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        ai_data = json.loads(raw)

        # Validate required fields
        verdict          = ai_data.get("verdict", "BYPASS")
        ai_confidence    = int(ai_data.get("ai_confidence", 50))
        risk_level       = ai_data.get("risk_level", "MEDIUM")
        rationale        = ai_data.get("rationale", "")
        key_risk         = ai_data.get("key_risk", "")
        suggested_action = ai_data.get("suggested_action", "")

        if verdict not in ("APPROVE", "DOWNGRADE", "REJECT"):
            verdict = "BYPASS"

        # ── Compute final confidence & quality adjustment ─────────────────
        original_quality     = quality_result.get("quality_score", 0)
        original_confidence  = quality_result.get("confidence", "MEDIUM")

        if verdict == "APPROVE" and ai_confidence >= AI_BOOST_THRESHOLD:
            # AI strongly approves — boost quality slightly, maintain or upgrade confidence
            final_quality = min(100, original_quality * 1.15)
            if original_confidence == "MEDIUM" and ai_confidence >= 80:
                final_confidence = "HIGH"
            else:
                final_confidence = original_confidence

        elif verdict == "DOWNGRADE":
            # AI sees risks — reduce quality, downgrade confidence
            penalty = (100 - ai_confidence) / 100  # higher penalty for lower AI confidence
            final_quality = original_quality * (1 - penalty * 0.4)
            final_confidence = "MEDIUM" if original_confidence == "HIGH" else "LOW"

        elif verdict == "REJECT" or ai_confidence < AI_VETO_THRESHOLD:
            # AI veto — quality collapses
            final_quality    = original_quality * 0.3
            final_confidence = "LOW"
            verdict          = "REJECT"

        else:
            # BYPASS or neutral
            final_quality    = original_quality
            final_confidence = original_confidence

        result = {
            "verdict":          verdict,
            "ai_confidence":    ai_confidence,
            "risk_level":       risk_level,
            "rationale":        rationale,
            "key_risk":         key_risk,
            "suggested_action": suggested_action,
            "final_confidence": final_confidence,
            "final_quality":    round(final_quality, 1),
            "ai_used":          True,
        }

        _set_cached(cache_key, result)
        log.info(
            f"AI verdict for {symbol}: {verdict} | "
            f"AI conf: {ai_confidence} | "
            f"Risk: {risk_level} | "
            f"Final quality: {final_quality:.1f}"
        )
        return result

    except json.JSONDecodeError as e:
        log.error(f"AI JSON parse error for {symbol}: {e}")
        return default
    except Exception as e:
        err_str = str(e)
        # Flag 429 rate limit errors so async wrapper knows to try Gemini
        if "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower():
            log.warning(f"Groq rate limit hit for {symbol} — flagging for Gemini fallback")
            exhausted = dict(default)
            exhausted["groq_exhausted"] = True
            exhausted["rationale"] = "Groq rate limit reached — Gemini fallback triggered"
            return exhausted
        log.error(f"AI validation error for {symbol}: {e}")
        return default




# ─── GEMINI FALLBACK ──────────────────────────────────────────────────────────
def validate_signal_with_gemini(
    symbol: str,
    quality_result: dict,
    price: float,
    funding_rate: float = 0.0,
    news_sentiment: str = "NEUTRAL",
    news_points: list = None,
) -> dict:
    """
    Gemini Flash-Lite fallback validator — called when Groq is exhausted (429).
    Uses google-generativeai SDK. Free tier: 1000 RPD, 15 RPM.
    """
    default = {
        "verdict": "BYPASS",
        "ai_confidence": 50,
        "risk_level": "MEDIUM",
        "rationale": "Gemini fallback — AI validation bypassed",
        "key_risk": "Unknown",
        "suggested_action": "Follow scoring engine signal",
        "final_confidence": quality_result.get("confidence") or "MEDIUM",
        "final_quality": quality_result.get("quality_score") or quality_result.get("abs_score", 0),
        "ai_used": False,
    }

    if not GEMINI_ENABLED:
        return default

    # Gemini eligibility — Grade A or score>=80 only
    _grade = quality_result.get("grade", "B")
    _score = quality_result.get("quality_score") or quality_result.get("abs_score", 0)
    if _grade != "A" and _score < 80:
        default["rationale"] = "Gemini reserved for Grade A / score>=80 — using Phase 1"
        return default

    # Gemini rate gate
    if not _gemini_rate_ok():
        default["rationale"] = "Gemini rate gate — using Phase 1"
        return default

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(
            GEMINI_MODEL,
            system_instruction=(
                "You are a disciplined crypto futures risk analyst. "
                "You respond ONLY with valid JSON. No markdown, no explanation outside JSON. "
                "Be conservative — when in doubt, DOWNGRADE not APPROVE."
            )
        )

        prompt = build_ai_context(
            symbol, quality_result, price, funding_rate,
            news_sentiment, news_points or []
        )

        try:
            from google.generativeai import types as _gtypes
            gen_cfg = _gtypes.GenerationConfig(temperature=0.1, max_output_tokens=400)
        except Exception:
            gen_cfg = {"temperature": 0.1, "max_output_tokens": 400}

        _gemini_record_call()
        response = model.generate_content(prompt, generation_config=gen_cfg)

        raw = response.text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        ai_data = json.loads(raw)

        verdict          = ai_data.get("verdict", "BYPASS")
        ai_confidence    = int(ai_data.get("ai_confidence", 50))
        risk_level       = ai_data.get("risk_level", "MEDIUM")
        rationale        = ai_data.get("rationale", "")
        key_risk         = ai_data.get("key_risk", "")
        suggested_action = ai_data.get("suggested_action", "")

        if verdict not in ("APPROVE", "DOWNGRADE", "REJECT"):
            verdict = "BYPASS"

        original_quality    = quality_result.get("quality_score", 0)
        original_confidence = quality_result.get("confidence", "MEDIUM")

        if verdict == "APPROVE" and ai_confidence >= AI_BOOST_THRESHOLD:
            final_quality = min(100, original_quality * 1.15)
            final_confidence = "HIGH" if original_confidence == "MEDIUM" and ai_confidence >= 80 else original_confidence
        elif verdict == "DOWNGRADE":
            penalty = (100 - ai_confidence) / 100
            final_quality = original_quality * (1 - penalty * 0.4)
            final_confidence = "MEDIUM" if original_confidence == "HIGH" else "LOW"
        elif verdict == "REJECT" or ai_confidence < AI_VETO_THRESHOLD:
            final_quality = original_quality * 0.3
            final_confidence = "LOW"
            verdict = "REJECT"
        else:
            final_quality = original_quality
            final_confidence = original_confidence

        result = {
            "verdict":          verdict,
            "ai_confidence":    ai_confidence,
            "risk_level":       risk_level,
            "rationale":        f"[Gemini] {rationale}",
            "key_risk":         key_risk,
            "suggested_action": suggested_action,
            "final_confidence": final_confidence,
            "final_quality":    round(final_quality, 1),
            "ai_used":          True,
        }

        log.info(f"Gemini verdict for {symbol}: {verdict} | conf:{ai_confidence} | quality:{final_quality:.1f}")
        return result

    except json.JSONDecodeError as e:
        log.error(f"Gemini JSON parse error for {symbol}: {e}")
        return default
    except Exception as e:
        err_str = str(e)
        # Parse retry_after from Gemini 429 response and set backoff
        # Error contains "Please retry in X.XXXs" — extract and honour it
        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
            import re as _re
            _retry_match = _re.search(r"retry in (\d+(?:\.\d+)?)", err_str, _re.IGNORECASE)
            _retry_secs  = float(_retry_match.group(1)) if _retry_match else 60.0
            _gemini_set_backoff(_retry_secs)
        log.error(f"Gemini validation error for {symbol}: {err_str[:200]}")
        return default

# ─── ASYNC WRAPPER WITH FALLBACK CHAIN ───────────────────────────────────────
async def validate_signal_async(
    groq_client,
    symbol: str,
    quality_result: dict,
    price: float,
    funding_rate: float = 0.0,
    news_sentiment: str = "NEUTRAL",
    news_points: list = None,
) -> dict:
    """
    Async wrapper with fallback chain:
      1. Groq (llama-3.3-70b-versatile) — primary
      2. Gemini Flash-Lite               — fallback on Groq 429
      3. Phase 1 score only (BYPASS)     — fallback if both fail
    """
    import asyncio

    # Cache check first — avoid any API call if cached
    quality = quality_result.get("quality_score") or quality_result.get("abs_score", 0)
    cache_key = _cache_key(symbol, quality_result.get("direction", ""), quality)
    cached = _get_cached(cache_key)
    if cached:
        log.info(f"AI cache hit for {symbol}")
        return cached

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()

    # ── 1. Try Groq ───────────────────────────────────────────────────────────
    try:
        result = await loop.run_in_executor(
            None,
            lambda: validate_signal_with_ai(
                groq_client, symbol, quality_result, price,
                funding_rate, news_sentiment, news_points
            )
        )
        # If Groq returned a real verdict (not BYPASS due to rate limit), cache and return
        if result.get("verdict") != "BYPASS" or not result.get("groq_exhausted"):
            _set_cached(cache_key, result)
            return result
    except Exception as e:
        log.warning(f"Groq failed for {symbol}: {e} — trying Gemini fallback")

    # ── 2. Groq exhausted — try Gemini ───────────────────────────────────────
    if GEMINI_ENABLED:
        try:
            log.info(f"Using Gemini fallback for {symbol}")
            result = await loop.run_in_executor(
                None,
                lambda: validate_signal_with_gemini(
                    symbol, quality_result, price,
                    funding_rate, news_sentiment, news_points
                )
            )
            if result.get("ai_used"):
                _set_cached(cache_key, result)
                return result
        except Exception as e:
            log.warning(f"Gemini fallback also failed for {symbol}: {e}")

    # ── 3. Both failed — return Phase 1 bypass ───────────────────────────────
    log.warning(f"All AI providers exhausted for {symbol} — using Phase 1 score only")
    return {
        "verdict": "BYPASS",
        "ai_confidence": 50,
        "risk_level": "MEDIUM",
        "rationale": "All AI providers unavailable — Phase 1 score only",
        "key_risk": "No AI validation available",
        "suggested_action": "Use Phase 1 signal with caution",
        "final_confidence": quality_result.get("confidence") or "MEDIUM",
        "final_quality": quality,
        "ai_used": False,
    }


# ─── SIGNAL ENRICHMENT ────────────────────────────────────────────────────────
def enrich_signal_with_ai(analysis_result: dict, ai_result: dict) -> dict:
    """
    Merge the Phase 1 analysis result with the Phase 2 AI verdict.
    Returns an enriched result dict ready for format_signal().
    """
    enriched = dict(analysis_result)

    verdict       = ai_result.get("verdict", "BYPASS")
    final_conf    = ai_result.get("final_confidence", analysis_result.get("confidence", "MEDIUM"))
    final_quality = ai_result.get("final_quality", analysis_result.get("quality", {}).get("quality_score", 0))
    ai_used       = ai_result.get("ai_used", False)

    # If AI rejected the signal, mark as NEUTRAL so it won't fire
    if verdict == "REJECT":
        enriched["direction"]  = "NEUTRAL"
        enriched["confidence"] = "LOW"
        enriched["ai_reject_reason"] = ai_result.get("rationale", "AI rejected signal")
        enriched["ai_result"]  = ai_result
        return enriched

    # Apply AI adjustments
    enriched["confidence"] = final_conf
    enriched["abs_score"]  = final_quality
    enriched["score"]      = final_quality if analysis_result.get("direction") == "LONG" else -final_quality
    enriched["ai_result"]  = ai_result

    # Add AI signals to the signal list
    if ai_used:
        ai_signals = []
        if ai_result.get("rationale"):
            ai_signals.append(f"🤖 AI: {ai_result['rationale']}")
        if ai_result.get("key_risk"):
            ai_signals.append(f"⚠ Risk: {ai_result['key_risk']}")
        enriched["signals"] = ai_signals + enriched.get("signals", [])

    return enriched


# ─── TELEGRAM MESSAGE FORMATTER ADDON ─────────────────────────────────────────
def format_ai_block(ai_result: dict) -> str:
    """
    Returns a formatted Telegram message block for the AI verdict.
    Append this to the existing format_signal() output.
    """
    if not ai_result or not ai_result.get("ai_used"):
        return ""

    verdict    = ai_result.get("verdict", "BYPASS")
    ai_conf    = ai_result.get("ai_confidence", 0)
    risk       = ai_result.get("risk_level", "MEDIUM")
    rationale  = ai_result.get("rationale", "")
    key_risk   = ai_result.get("key_risk", "")
    suggestion = ai_result.get("suggested_action", "")

    verdict_emoji = {
        "APPROVE":   "✅",
        "DOWNGRADE": "⚠",
        "REJECT":    "🚫",
        "BYPASS":    "⏭",
    }.get(verdict, "🤖")

    risk_emoji = {
        "LOW":    "🟢",
        "MEDIUM": "🟡",
        "HIGH":   "🔴",
    }.get(risk, "🟡")

    msg  = "\n🤖 *AI Analyst Verdict*\n"
    msg += f"{verdict_emoji} *{verdict}* | AI Confidence: `{ai_conf}%` | {risk_emoji} Risk: {risk}\n"
    if rationale:
        msg += f"📋 {rationale}\n"
    if key_risk:
        msg += f"⚠ Key Risk: {key_risk}\n"
    if suggestion:
        msg += f"💡 {suggestion}\n"

    return msg


# ─── FULL PIPELINE RUNNER ─────────────────────────────────────────────────────
async def run_full_pipeline(
    symbol: str,
    fetch_ohlcv_fn,
    coin_labels: dict,
    groq_client,
    exchange,
    news_context: dict = None,
) -> dict:
    """
    Complete Phase 1 + Phase 2 pipeline in one call.
    Replaces the original analyze() function entirely.

    Usage in bot.py:
        from ai_validator import run_full_pipeline
        r = await run_full_pipeline(symbol, fetch_ohlcv, COIN_LABELS, ai_client, exchange, news_context)
    """
    from signal_engine import analyze_v2

    # ── Phase 1: Scoring engine ───────────────────────────────────────────
    r = analyze_v2(symbol, fetch_ohlcv_fn, coin_labels)

    # ── Funding rate ──────────────────────────────────────────────────────
    try:
        funding     = exchange.fetch_funding_rate(symbol)
        r["funding"] = float(funding["fundingRate"]) if funding else 0
    except Exception:
        r["funding"] = 0

    # ── Skip AI if signal already NEUTRAL after Phase 1 ──────────────────
    if r["direction"] == "NEUTRAL":
        r["ai_result"] = {"verdict": "BYPASS", "ai_used": False}
        return r

    # ── Phase 2: AI validation ────────────────────────────────────────────
    news_sentiment = (news_context or {}).get("sentiment", "NEUTRAL")
    news_points    = (news_context or {}).get("key_points", [])

    quality_data = r.get("quality") or {}
    quality_data["quality_score"] = r.get("abs_score", 0)
    quality_data["direction"] = r.get("direction", "NEUTRAL")
    quality_data["confidence"] = r.get("confidence", "LOW")

    ai_result = await validate_signal_async(
        groq_client, symbol,
        quality_data,
        r["price"], r["funding"],
        news_sentiment, news_points,
    )

    # ── Merge results ─────────────────────────────────────────────────────
    r = enrich_signal_with_ai(r, ai_result)

    # signal_type classification happens in bot.py after this returns
    # (avoids circular import — bot imports ai_validator, ai_validator cannot import bot)
    return r
