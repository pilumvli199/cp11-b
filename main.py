"""
🚀 ETH OPTIONS BOT - DELTA EXCHANGE INDIA v8.0 PRO
====================================================
v8.0 FIXES FROM v7.0:
- FIXED: Strikes ±10 → ±3 (noise कमी, only meaningful strikes)
- FIXED: Phase 1 OI threshold 5% → 12% (less false signals)
- FIXED: Phase comparison 5min → 15min ago (more reliable)
- FIXED: Absolute OI minimum check in Phase detection
- FIXED: Phase detection ATM ±1 strikes पण check करतो
- FIXED: PCR delta (change rate) tracking added
- FIXED: ATM shift between snapshots handled
- FIXED: Volume midnight reset false spike handled
- FIXED: Expiry rollover graceful handling
- FIXED: DeepSeek JSON robust parsing
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from statistics import mode as stat_mode
import json
import logging
import os
import hmac
import hashlib
import time as time_module
import re

# ============================================================
#  CONFIGURATION
# ============================================================
DELTA_API_KEY      = os.getenv("DELTA_API_KEY",      "YOUR_API_KEY")
DELTA_API_SECRET   = os.getenv("DELTA_API_SECRET",   "YOUR_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY",   "YOUR_DEEPSEEK_KEY")

DELTA_BASE_URL = "https://api.india.delta.exchange"

UNDERLYING       = "ETH"
# ✅ FIX 1: ±10 → ±3 (only meaningful strikes, noise कमी)
ATM_STRIKE_FETCH = 3
ATM_STRIKE_AI    = 3
STRIKE_INTERVAL  = 20

SNAPSHOT_INTERVAL = 5 * 60
ANALYSIS_INTERVAL = 30 * 60

CACHE_5MIN_SIZE  = 72
CACHE_30MIN_SIZE = 12
CANDLE_COUNT     = 24

# Signal thresholds
MIN_OI_CHANGE    = 10.0
STRONG_OI_CHANGE = 20.0
MIN_VOLUME_CHG   = 15.0
PCR_BULL         = 1.3
PCR_BEAR         = 0.7
MIN_CONFIDENCE   = 7

# ✅ FIX 2: Phase thresholds updated
PHASE1_OI_BUILD_PCT   = 12.0   # 5% → 12% (less noise)
PHASE1_VOL_MAX_PCT    = 15.0   # 10% → 15% (more tolerance)
PHASE1_MIN_ABS_OI     = 100    # ✅ FIX 3: Minimum absolute OI contracts
PHASE1_COMPARE_SNAPS  = 3      # ✅ FIX 4: 1 snap(5min) → 3 snaps(15min)

PHASE2_VOL_SPIKE_PCT  = 20.0
PHASE2_OI_MIN_PCT     = 5.0    # 3% → 5%
PHASE2_MIN_ABS_OI     = 150    # Absolute OI minimum for Phase 2

PHASE3_PRICE_MOVE_PCT = 0.4
PHASE3_CONFIRM_ALL    = True

# Standalone alert thresholds
OI_ALERT_PCT     = 15.0
VOL_SPIKE_PCT    = 25.0
PCR_ALERT_PCT    = 12.0
ATM_PROX_USD     = 50

# ✅ FIX 5: Minimum absolute OI for any analysis
MIN_ABS_OI_THRESHOLD  = 50     # Ignore strikes with OI < 50 contracts
MIN_ABS_VOL_THRESHOLD = 10

# Strike weights
ATM_WEIGHT      = 3.0
NEAR_ATM_WEIGHT = 2.0
FAR_WEIGHT      = 1.0

# ✅ FIX 6: Volume midnight reset window (ignore spikes 23:50-00:10 UTC)
MIDNIGHT_RESET_BUFFER_MIN = 10

# API settings
MAX_RETRIES      = 3
API_DELAY        = 0.35
DEEPSEEK_TIMEOUT = 45

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
#  DATA STRUCTURES
# ============================================================

@dataclass
class OISnapshot:
    strike:    float
    ce_oi:     float
    pe_oi:     float
    ce_volume: float
    pe_volume: float
    ce_ltp:    float
    pe_ltp:    float
    pcr:       float
    timestamp: datetime


@dataclass
class MarketSnapshot:
    timestamp:     datetime
    spot_price:    float
    atm_strike:    float
    expiry:        str
    strikes_oi:    Dict[float, OISnapshot]
    overall_pcr:   float
    total_ce_oi:   float
    total_pe_oi:   float
    total_ce_vol:  float
    total_pe_vol:  float
    # ✅ FIX 7: PCR delta tracking
    pcr_5min_ago:  float = 0.0
    pcr_delta:     float = 0.0  # PCR change rate per 5 min


@dataclass
class PhaseSignal:
    phase:            int
    dominant_side:    str
    direction:        str
    oi_change_pct:    float
    oi_abs_value:     float      # ✅ NEW: Absolute OI value
    vol_change_pct:   float
    price_change_pct: float
    atm_strike:       float
    spot_price:       float
    confidence:       float
    pcr_delta:        float      # ✅ NEW: PCR change rate
    message:          str


@dataclass
class PriceActionInsight:
    price_change_5m:   float
    price_change_15m:  float
    price_change_30m:  float
    price_momentum:    str
    vol_rolling_avg:   float
    vol_spike_ratio:   float
    oi_vol_corr:       float
    price_oi_corr:     float
    support_levels:    List[float]
    resistance_levels: List[float]
    trend_strength:    float
    triple_confirmed:  bool
    pcr_delta:         float     # ✅ NEW: PCR change rate
    pcr_acceleration:  str       # ✅ NEW: RISING_FAST / RISING / FLAT / FALLING / FALLING_FAST


@dataclass
class StrikeAnalysis:
    strike:        float
    is_atm:        bool
    distance_atm:  float
    weight:        float
    ce_oi:         float
    pe_oi:         float
    ce_volume:     float
    pe_volume:     float
    ce_ltp:        float
    pe_ltp:        float
    ce_oi_15:      float
    pe_oi_15:      float
    ce_vol_15:     float
    pe_vol_15:     float
    pcr_ch_15:     float
    ce_oi_30:      float
    pe_oi_30:      float
    ce_vol_30:     float
    pe_vol_30:     float
    pcr_ch_30:     float
    ce_oi_60:      float
    pe_oi_60:      float
    pcr:           float
    # ✅ NEW: Strike-wise PCR delta
    pcr_delta:     float
    ce_action:     str
    pe_action:     str
    tf15_signal:   str
    tf30_signal:   str
    mtf_confirmed: bool
    vol_confirms:  bool
    vol_strength:  str
    is_support:    bool
    is_resistance: bool
    bull_strength: float
    bear_strength: float
    recommendation: str
    confidence:    float


@dataclass
class SupportResistance:
    support_strike:     float
    support_put_oi:     float
    resistance_strike:  float
    resistance_call_oi: float
    near_support:       bool
    near_resistance:    bool


# ============================================================
#  DUAL CACHE
# ============================================================

class DualCache:
    def __init__(self):
        self._c5   = deque(maxlen=CACHE_5MIN_SIZE)
        self._c30  = deque(maxlen=CACHE_30MIN_SIZE)
        self._lock = asyncio.Lock()

    async def add_5min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c5.append(snap)
        logger.info(f"📦 5-min cache: {len(self._c5)}/{CACHE_5MIN_SIZE} | PCR:{snap.overall_pcr:.2f} Δ:{snap.pcr_delta:+.3f}")

    async def add_30min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c30.append(snap)
        logger.info(f"📦 30-min cache: {len(self._c30)}/{CACHE_30MIN_SIZE}")

    async def get_5min_ago(self, n: int) -> Optional[MarketSnapshot]:
        async with self._lock:
            idx = len(self._c5) - 1 - n
            return self._c5[idx] if idx >= 0 else None

    async def get_30min_ago(self, n: int) -> Optional[MarketSnapshot]:
        async with self._lock:
            idx = len(self._c30) - 1 - n
            return self._c30[idx] if idx >= 0 else None

    async def get_recent_snapshots(self, n: int) -> List[MarketSnapshot]:
        async with self._lock:
            lst = list(self._c5)
            return lst[-n:] if len(lst) >= n else lst

    async def latest(self) -> Optional[MarketSnapshot]:
        async with self._lock:
            return self._c5[-1] if self._c5 else None

    def sizes(self) -> Tuple[int, int]:
        return len(self._c5), len(self._c30)

    def has_data(self) -> bool:
        return len(self._c5) >= 3


# ============================================================
#  DELTA CLIENT
# ============================================================

class DeltaClient:

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = None
        # ✅ FIX: Track current expiry for rollover detection
        self._current_expiry: str = ""

    async def init(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))

    async def close(self):
        if self.session:
            await self.session.close()

    def _auth_headers(self, method: str, path: str, query: str = "", body: str = "") -> Dict:
        ts  = str(int(time_module.time()))
        qs  = f"?{query}" if query else ""
        msg = method.upper() + ts + path + qs + body
        sig = hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()
        return {"api-key": self.api_key, "timestamp": ts, "signature": sig, "Content-Type": "application/json"}

    async def _get(self, path: str, params: Dict = None, auth: bool = True) -> Optional[Dict]:
        url  = DELTA_BASE_URL + path
        qs   = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        hdrs = self._auth_headers("GET", path, qs) if auth else {}

        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(url, params=params, headers=hdrs) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429:
                        await asyncio.sleep((attempt + 1) * 3)
                        continue
                    logger.warning(f"⚠️ GET {path} → {r.status}")
                    return None
            except aiohttp.ClientConnectorError:
                logger.error(f"❌ Network error ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Request error: {e}")
                await asyncio.sleep(1)
        return None

    async def get_eth_spot(self) -> float:
        data = await self._get("/v2/tickers/ETHUSD")
        if data and data.get("result"):
            r = data["result"]
            price = r.get("spot_price") or r.get("mark_price") or r.get("close") or 0
            return float(price)
        return 0.0

    async def get_daily_eth_options(self) -> Tuple[List[Dict], str]:
        data = await self._get("/v2/products", params={
            "contract_types": "call_options,put_options",
            "states":         "live"
        })
        if not data or not data.get("result"):
            return [], ""

        products = data["result"]
        eth_opts = [
            p for p in products
            if (p.get("underlying_asset_symbol") == UNDERLYING
                or UNDERLYING in p.get("symbol", "").upper())
               and p.get("contract_type") in ("call_options", "put_options")
        ]
        if not eth_opts:
            return [], ""

        now_utc    = datetime.now(timezone.utc)
        by_expiry: Dict[str, List] = {}
        for p in eth_opts:
            raw = p.get("settlement_time") or p.get("expiry_time") or ""
            if not raw:
                continue
            try:
                dt  = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt <= now_utc:
                    continue
                key = dt.date().isoformat()
                by_expiry.setdefault(key, []).append(p)
            except Exception:
                continue

        if not by_expiry:
            return [], ""

        nearest = sorted(by_expiry.keys())[0]

        # ✅ FIX: Expiry rollover detection
        if self._current_expiry and nearest != self._current_expiry:
            logger.warning(f"🔄 EXPIRY ROLLOVER: {self._current_expiry} → {nearest}")
        self._current_expiry = nearest

        return by_expiry[nearest], nearest

    async def get_option_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        data = await self._get("/v2/tickers", params={"contract_types": "call_options,put_options"})
        if not data or not data.get("result"):
            return {}
        sym_set = set(symbols)
        return {t["symbol"]: t for t in data["result"] if t.get("symbol") in sym_set}

    async def get_candles(self, symbol: str, resolution: str, count: int) -> pd.DataFrame:
        end_ts  = int(time_module.time())
        res_str = resolution.lower().strip()
        if res_str.endswith("m"):
            res_min = int(res_str[:-1])
        elif res_str.endswith("h"):
            res_min = int(res_str[:-1]) * 60
        elif res_str.endswith("d"):
            res_min = int(res_str[:-1]) * 1440
        else:
            res_min = int(res_str)
        res_sec  = res_min * 60
        start_ts = end_ts - (count * res_sec) - res_sec * 3

        data = await self._get("/v2/history/candles", params={
            "symbol": symbol, "resolution": resolution,
            "start": start_ts, "end": end_ts
        })
        if not data or not data.get("result"):
            return pd.DataFrame()

        # ✅ FIX: list OR dict handle
        result = data["result"]
        raw    = result if isinstance(result, list) else result.get("candles", result)
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            try:
                if isinstance(c, list):
                    ts = datetime.fromtimestamp(int(c[0]), tz=timezone.utc)
                    o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5] if len(c) > 5 else 0)
                else:
                    ts = datetime.fromtimestamp(int(c.get("time", c.get("timestamp", 0))), tz=timezone.utc)
                    o, h, l, cl, v = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c.get("volume", 0))
                rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": cl, "volume": v})
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df.tail(count)

    async def fetch_snapshot(self, cache: 'DualCache') -> Optional[MarketSnapshot]:
        spot = await self.get_eth_spot()
        if spot <= 0:
            return None
        logger.info(f"💰 ETH: ${spot:,.2f}")

        await asyncio.sleep(API_DELAY)
        options, expiry = await self.get_daily_eth_options()
        if not options:
            return None

        strikes_all = sorted({float(p["strike_price"]) for p in options if p.get("strike_price")})
        if not strikes_all:
            return None

        if len(strikes_all) >= 2:
            diffs = [strikes_all[i+1] - strikes_all[i] for i in range(min(8, len(strikes_all)-1))]
            try:
                strike_step = stat_mode(diffs)
            except Exception:
                strike_step = min(diffs)
        else:
            strike_step = STRIKE_INTERVAL

        atm     = min(strikes_all, key=lambda s: abs(s - spot))
        atm_idx = strikes_all.index(atm)
        # ✅ FIX 1: ±3 strikes only
        lo      = max(0, atm_idx - ATM_STRIKE_FETCH)
        hi      = min(len(strikes_all), atm_idx + ATM_STRIKE_FETCH + 1)
        selected = set(strikes_all[lo:hi])

        calls = {float(p["strike_price"]): p for p in options
                 if p["contract_type"] == "call_options" and float(p["strike_price"]) in selected}
        puts  = {float(p["strike_price"]): p for p in options
                 if p["contract_type"] == "put_options"  and float(p["strike_price"]) in selected}

        await asyncio.sleep(API_DELAY)
        all_syms = [p["symbol"] for p in options if float(p.get("strike_price", 0)) in selected]
        tickers  = await self.get_option_tickers(all_syms)

        strikes_oi: Dict[float, OISnapshot] = {}
        t_ce_oi = t_pe_oi = t_ce_vol = t_pe_vol = 0.0

        for strike in sorted(selected):
            cp = calls.get(strike)
            pp = puts.get(strike)
            if not cp or not pp:
                continue

            ct = tickers.get(cp["symbol"], {})
            pt = tickers.get(pp["symbol"], {})

            ce_oi  = float(ct.get("oi", ct.get("open_interest", 0)) or 0)
            pe_oi  = float(pt.get("oi", pt.get("open_interest", 0)) or 0)
            ce_vol = float(ct.get("volume", 0) or 0)
            pe_vol = float(pt.get("volume", 0) or 0)
            ce_ltp = float(ct.get("close", ct.get("mark_price", 0)) or 0)
            pe_ltp = float(pt.get("close", pt.get("mark_price", 0)) or 0)
            pcr    = (pe_oi / ce_oi) if ce_oi > 0 else 0.0

            t_ce_oi  += ce_oi;  t_pe_oi  += pe_oi
            t_ce_vol += ce_vol; t_pe_vol += pe_vol

            strikes_oi[strike] = OISnapshot(
                strike=strike, ce_oi=ce_oi, pe_oi=pe_oi,
                ce_volume=ce_vol, pe_volume=pe_vol,
                ce_ltp=ce_ltp, pe_ltp=pe_ltp, pcr=pcr,
                timestamp=datetime.now(timezone.utc)
            )

        if not strikes_oi:
            return None

        overall_pcr = (t_pe_oi / t_ce_oi) if t_ce_oi > 0 else 0.0

        # ✅ FIX: PCR delta calculation
        prev = await cache.get_5min_ago(0)
        pcr_5min_ago = prev.overall_pcr if prev else overall_pcr
        pcr_delta    = overall_pcr - pcr_5min_ago

        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            spot_price=spot, atm_strike=atm, expiry=expiry,
            strikes_oi=strikes_oi, overall_pcr=overall_pcr,
            total_ce_oi=t_ce_oi, total_pe_oi=t_pe_oi,
            total_ce_vol=t_ce_vol, total_pe_vol=t_pe_vol,
            pcr_5min_ago=pcr_5min_ago, pcr_delta=pcr_delta
        )


# ============================================================
#  MIDNIGHT RESET GUARD
# ============================================================

def is_midnight_reset_window() -> bool:
    """
    ✅ FIX: Volume resets at midnight UTC
    Ignore volume spikes in ±10 min window around 00:00 UTC
    """
    now = datetime.now(timezone.utc)
    minutes_from_midnight = now.hour * 60 + now.minute
    return (minutes_from_midnight < MIDNIGHT_RESET_BUFFER_MIN or
            minutes_from_midnight > (24 * 60 - MIDNIGHT_RESET_BUFFER_MIN))


# ============================================================
#  PANDAS / NUMPY PRE-CALCULATOR
# ============================================================

class PriceActionCalculator:

    @staticmethod
    def calculate(snapshots: List[MarketSnapshot], candles_15m: pd.DataFrame) -> PriceActionInsight:
        if len(snapshots) < 3:
            return PriceActionCalculator._empty()

        prices    = np.array([s.spot_price for s in snapshots])
        curr_price = prices[-1]

        def pct_change(ago_idx: int) -> float:
            if len(prices) > ago_idx:
                prev = prices[-(ago_idx + 1)]
                return ((curr_price - prev) / prev * 100) if prev > 0 else 0.0
            return 0.0

        p5m  = pct_change(1)
        p15m = pct_change(3)
        p30m = pct_change(6)
        price_mom = "BULLISH" if p5m > 0.3 else "BEARISH" if p5m < -0.3 else "NEUTRAL"

        vols = np.array([s.total_ce_vol + s.total_pe_vol for s in snapshots])
        vol_rolling = float(np.mean(vols[:-1])) if len(vols) > 1 else float(vols[-1])
        curr_vol    = float(vols[-1])
        vol_spike   = (curr_vol / vol_rolling) if vol_rolling > 0 else 1.0

        ce_ois = np.array([s.total_ce_oi for s in snapshots])
        pe_ois = np.array([s.total_pe_oi for s in snapshots])
        oi_total = ce_ois + pe_ois

        if len(oi_total) > 2 and np.std(oi_total) > 0 and np.std(vols) > 0:
            oi_vol_corr = float(np.corrcoef(oi_total, vols)[0, 1])
        else:
            oi_vol_corr = 0.0

        if len(prices) > 2 and np.std(prices) > 0 and np.std(oi_total) > 0:
            price_oi_corr = float(np.corrcoef(prices, oi_total)[0, 1])
        else:
            price_oi_corr = 0.0

        # ✅ FIX: PCR delta tracking
        pcr_deltas = np.array([s.pcr_delta for s in snapshots])
        avg_pcr_delta = float(np.mean(pcr_deltas[-3:])) if len(pcr_deltas) >= 3 else 0.0

        if avg_pcr_delta > 0.05:      pcr_accel = "RISING_FAST"
        elif avg_pcr_delta > 0.01:    pcr_accel = "RISING"
        elif avg_pcr_delta < -0.05:   pcr_accel = "FALLING_FAST"
        elif avg_pcr_delta < -0.01:   pcr_accel = "FALLING"
        else:                          pcr_accel = "FLAT"

        # S/R from candles
        support_levels    = []
        resistance_levels = []
        if not candles_15m.empty and len(candles_15m) >= 5:
            df    = candles_15m.tail(20)
            lows  = df["low"].values
            highs = df["high"].values
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    support_levels.append(float(lows[i]))
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_levels.append(float(highs[i]))
            support_levels    = sorted(support_levels,    key=lambda x: abs(x - curr_price))[:3]
            resistance_levels = sorted(resistance_levels, key=lambda x: abs(x - curr_price))[:3]

        # Trend strength
        ts = 0.0
        if abs(p5m) >= 0.5:   ts += 3.0
        elif abs(p5m) >= 0.3:  ts += 1.5
        if vol_spike >= 1.5:   ts += 3.0
        elif vol_spike >= 1.2:  ts += 1.5
        oi_ch = ((oi_total[-1] - oi_total[0]) / oi_total[0] * 100) if oi_total[0] > 0 else 0
        if abs(oi_ch) >= 10:   ts += 4.0
        elif abs(oi_ch) >= 5:  ts += 2.0
        trend_strength = min(10.0, ts)

        # Triple confirmation
        price_bull  = p5m > 0.3
        price_bear  = p5m < -0.3
        oi_bull     = pe_ois[-1] > pe_ois[0] if len(pe_ois) > 1 else False
        oi_bear     = ce_ois[-1] > ce_ois[0] if len(ce_ois) > 1 else False
        vol_confirm = vol_spike >= 1.2
        triple_bull = price_bull and oi_bull and vol_confirm
        triple_bear = price_bear and oi_bear and vol_confirm

        return PriceActionInsight(
            price_change_5m=round(p5m, 3),
            price_change_15m=round(p15m, 3),
            price_change_30m=round(p30m, 3),
            price_momentum=price_mom,
            vol_rolling_avg=round(vol_rolling, 0),
            vol_spike_ratio=round(vol_spike, 2),
            oi_vol_corr=round(oi_vol_corr, 2),
            price_oi_corr=round(price_oi_corr, 2),
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            trend_strength=round(trend_strength, 1),
            triple_confirmed=(triple_bull or triple_bear),
            pcr_delta=round(avg_pcr_delta, 4),
            pcr_acceleration=pcr_accel
        )

    @staticmethod
    def _empty() -> PriceActionInsight:
        return PriceActionInsight(
            price_change_5m=0, price_change_15m=0, price_change_30m=0,
            price_momentum="NEUTRAL", vol_rolling_avg=0, vol_spike_ratio=1.0,
            oi_vol_corr=0, price_oi_corr=0, support_levels=[], resistance_levels=[],
            trend_strength=0, triple_confirmed=False, pcr_delta=0, pcr_acceleration="FLAT"
        )


# ============================================================
#  PHASE DETECTOR v8.0
# ============================================================

class PhaseDetector:

    COOLDOWN_PHASE1 = 20 * 60   # 15min → 20min (less spam)
    COOLDOWN_PHASE2 = 10 * 60
    COOLDOWN_PHASE3 =  5 * 60

    def __init__(self):
        self._last: Dict[str, float] = {}
        self._phase1_fired_at: float = 0
        self._phase2_fired_at: float = 0
        self._phase1_side: str = ""

    def _can(self, key: str, cooldown: int) -> bool:
        return (time_module.time() - self._last.get(key, 0)) >= cooldown

    def _mark(self, key: str):
        self._last[key] = time_module.time()

    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        return ((curr - prev) / prev * 100) if prev > 0 else 0.0

    async def detect(self, curr: MarketSnapshot,
                     prev_15m: Optional[MarketSnapshot],   # ✅ FIX: 15min ago (was 5min)
                     pa: PriceActionInsight) -> List[PhaseSignal]:
        signals = []

        # ✅ FIX: Midnight volume reset guard
        if is_midnight_reset_window():
            logger.info("🌙 Midnight reset window — skipping phase detection")
            return signals

        if not prev_15m:
            return signals

        # ✅ FIX: ATM ±1 strikes check (not just ATM)
        strikes_to_check = [curr.atm_strike]
        all_strikes = sorted(curr.strikes_oi.keys())
        atm_idx = all_strikes.index(curr.atm_strike) if curr.atm_strike in all_strikes else -1
        if atm_idx > 0:
            strikes_to_check.append(all_strikes[atm_idx - 1])  # ATM-1
        if atm_idx >= 0 and atm_idx < len(all_strikes) - 1:
            strikes_to_check.append(all_strikes[atm_idx + 1])  # ATM+1

        # Aggregate OI across ATM ±1
        ce_oi_curr = pe_oi_curr = ce_oi_prev = pe_oi_prev = 0.0
        ce_vol_curr = pe_vol_curr = ce_vol_prev = pe_vol_prev = 0.0

        for strike in strikes_to_check:
            atm_c = curr.strikes_oi.get(strike)
            atm_p = prev_15m.strikes_oi.get(strike)
            if not atm_c or not atm_p:
                continue
            # ✅ FIX: ATM gets 3x weight, ±1 gets 1x
            w = 3.0 if strike == curr.atm_strike else 1.0
            ce_oi_curr  += atm_c.ce_oi     * w
            pe_oi_curr  += atm_c.pe_oi     * w
            ce_oi_prev  += atm_p.ce_oi     * w
            pe_oi_prev  += atm_p.pe_oi     * w
            ce_vol_curr += atm_c.ce_volume * w
            pe_vol_curr += atm_c.pe_volume * w
            ce_vol_prev += atm_p.ce_volume * w
            pe_vol_prev += atm_p.pe_volume * w

        if ce_oi_curr == 0 and pe_oi_curr == 0:
            return signals

        ce_oi_ch  = self._pct(ce_oi_curr,  ce_oi_prev)
        pe_oi_ch  = self._pct(pe_oi_curr,  pe_oi_prev)
        ce_vol_ch = self._pct(ce_vol_curr, ce_vol_prev)
        pe_vol_ch = self._pct(pe_vol_curr, pe_vol_prev)

        # ✅ FIX: ATM shift detection
        # ✅ FIX: अगर ATM change झाला तर prev data match होणार नाही
        if curr.atm_strike != prev_15m.atm_strike:
            logger.info(f"⚠️ ATM shifted: {prev_15m.atm_strike} → {curr.atm_strike} — phase detection skipped")
            return signals

        call_building = ce_oi_ch >= PHASE1_OI_BUILD_PCT
        put_building  = pe_oi_ch >= PHASE1_OI_BUILD_PCT
        dominant_side = "PUT" if put_building and pe_oi_ch >= ce_oi_ch else ("CALL" if call_building else "")

        if not dominant_side:
            return signals

        oi_ch     = pe_oi_ch   if dominant_side == "PUT"  else ce_oi_ch
        vol_ch    = pe_vol_ch  if dominant_side == "PUT"  else ce_vol_ch
        abs_oi    = pe_oi_curr if dominant_side == "PUT"  else ce_oi_curr
        direction = "BULLISH"  if dominant_side == "PUT"  else "BEARISH"

        # ✅ FIX: Absolute OI check
        if abs_oi < PHASE1_MIN_ABS_OI:
            logger.info(f"⚠️ Phase 1 skipped — OI too low: {abs_oi:.0f} < {PHASE1_MIN_ABS_OI}")
            return signals

        now = time_module.time()

        # ── PHASE 1 ──
        if (oi_ch >= PHASE1_OI_BUILD_PCT
                and abs(vol_ch) < PHASE1_VOL_MAX_PCT
                and self._can("PHASE1", self.COOLDOWN_PHASE1)):

            self._phase1_fired_at = now
            self._phase1_side     = dominant_side
            self._mark("PHASE1")

            # PCR delta confirmation
            pcr_confirms = (
                (direction == "BULLISH" and pa.pcr_delta >= 0) or
                (direction == "BEARISH" and pa.pcr_delta <= 0)
            )

            signals.append(PhaseSignal(
                phase=1, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, oi_abs_value=abs_oi,
                vol_change_pct=vol_ch, price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, oi_ch / 3),
                pcr_delta=pa.pcr_delta,
                message=(
                    f"⚡ <b>PHASE 1 — SMART MONEY POSITIONING</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b>\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"{'PUT' if dominant_side == 'PUT' else 'CALL'} OI: <b>{oi_ch:+.1f}%</b> ({abs_oi:.0f} contracts)\n"
                    f"Volume: {vol_ch:+.1f}% (still low — smart money only)\n"
                    f"PCR Δ: {pa.pcr_delta:+.3f} ({pa.pcr_acceleration})\n"
                    f"PCR confirms: {'✅' if pcr_confirms else '❌'}\n\n"
                    f"Signal: {direction} | Compared vs 15min ago\n"
                    f"⚠️ Wait for Phase 2 (Volume spike)\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
            ))

        # ── PHASE 2 ──
        phase1_recent = (now - self._phase1_fired_at) < (25 * 60)
        # ✅ FIX: Volume midnight guard
        vol_spike_valid = (pa.vol_spike_ratio >= (1 + PHASE2_VOL_SPIKE_PCT / 100)
                           and not is_midnight_reset_window())

        if (vol_spike_valid
                and oi_ch >= PHASE2_OI_MIN_PCT
                and abs_oi >= PHASE2_MIN_ABS_OI
                and phase1_recent
                and self._phase1_side == dominant_side
                and self._can("PHASE2", self.COOLDOWN_PHASE2)):

            self._phase2_fired_at = now
            self._mark("PHASE2")

            signals.append(PhaseSignal(
                phase=2, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, oi_abs_value=abs_oi,
                vol_change_pct=vol_ch, price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, pa.vol_spike_ratio * 3),
                pcr_delta=pa.pcr_delta,
                message=(
                    f"🔥 <b>PHASE 2 — VOLUME SPIKE! MOVE IMMINENT</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b>\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"Volume: <b>{pa.vol_spike_ratio:.1f}x</b> above avg 🔥\n"
                    f"OI: {oi_ch:+.1f}% ({abs_oi:.0f} contracts)\n"
                    f"PCR Δ: {pa.pcr_delta:+.3f} ({pa.pcr_acceleration})\n\n"
                    f"Signal: <b>{direction}</b>\n"
                    f"🎯 {'BUY CALL' if direction == 'BULLISH' else 'BUY PUT'} near ${curr.atm_strike:,.0f}\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
            ))

        # ── PHASE 3 ──
        phase2_recent = (now - self._phase2_fired_at) < (15 * 60)
        price_confirms = (
            (direction == "BULLISH" and pa.price_change_5m >= PHASE3_PRICE_MOVE_PCT) or
            (direction == "BEARISH" and pa.price_change_5m <= -PHASE3_PRICE_MOVE_PCT)
        )

        if (phase2_recent and price_confirms and pa.triple_confirmed
                and self._can("PHASE3", self.COOLDOWN_PHASE3)):

            self._mark("PHASE3")
            rec = "BUY_CALL" if direction == "BULLISH" else "BUY_PUT"

            signals.append(PhaseSignal(
                phase=3, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, oi_abs_value=abs_oi,
                vol_change_pct=vol_ch, price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, 7 + pa.trend_strength / 3),
                pcr_delta=pa.pcr_delta,
                message=(
                    f"✅ <b>PHASE 3 — CONFIRMED! EXECUTE NOW!</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b> ({pa.price_change_5m:+.2f}% / 5min)\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"🎯 Signal: <b>{rec}</b>\n"
                    f"💯 Triple: OI ✅ + Volume ✅ + Price ✅\n\n"
                    f"OI Change: {oi_ch:+.1f}% ({abs_oi:.0f} contracts)\n"
                    f"Vol Spike: {pa.vol_spike_ratio:.1f}x avg\n"
                    f"Price: {pa.price_change_5m:+.2f}%\n"
                    f"PCR Δ: {pa.pcr_delta:+.3f} ({pa.pcr_acceleration})\n"
                    f"Trend: {pa.trend_strength:.1f}/10\n\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
                    f"🤖 Sending to AI..."
                )
            ))

        return signals


# ============================================================
#  MTF ANALYZER v8.0
# ============================================================

class MTFAnalyzer:

    def __init__(self, cache: DualCache):
        self.cache = cache

    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        return ((curr - prev) / prev * 100) if prev > 0 else 0.0

    @staticmethod
    def _action(oi_ch: float) -> str:
        if oi_ch >= 10:  return "BUILDING"
        if oi_ch <= -10: return "UNWINDING"
        return "NEUTRAL"

    @staticmethod
    def _tf_signal(ce_ch: float, pe_ch: float, ce_vol: float, pe_vol: float) -> str:
        if pe_ch >= MIN_OI_CHANGE and pe_vol >= MIN_VOLUME_CHG:  return "BULLISH"
        if ce_ch >= MIN_OI_CHANGE and ce_vol >= MIN_VOLUME_CHG:  return "BEARISH"
        if pe_ch <= -MIN_OI_CHANGE:                               return "BEARISH"
        if ce_ch <= -MIN_OI_CHANGE:                               return "BULLISH"
        return "NEUTRAL"

    @staticmethod
    def _vol_confirm(oi_ch: float, vol_ch: float) -> Tuple[bool, str]:
        if oi_ch > 10 and vol_ch > MIN_VOLUME_CHG: return True,  "STRONG"
        if oi_ch > 5  and vol_ch > 10:             return True,  "MODERATE"
        if abs(oi_ch) < 5 and abs(vol_ch) < 5:    return True,  "WEAK"
        return False, "WEAK"

    def _signal_strength(self, ce30: float, pe30: float,
                         ce_vol30: float, pe_vol30: float,
                         weight: float, mtf: bool) -> Tuple[float, float]:
        bull = bear = 0.0
        boost = 1.5 if mtf else 0.8

        if   pe30 >= STRONG_OI_CHANGE: bull = 9.0
        elif pe30 >= MIN_OI_CHANGE:    bull = 7.0
        elif pe30 >= 5:                bull = 4.0

        if   ce30 >= STRONG_OI_CHANGE: bear = 9.0
        elif ce30 >= MIN_OI_CHANGE:    bear = 7.0
        elif ce30 >= 5:                bear = 4.0

        if pe30 <= -STRONG_OI_CHANGE:  bear = max(bear, 8.0)
        elif pe30 <= -MIN_OI_CHANGE:   bear = max(bear, 6.0)
        if ce30 <= -STRONG_OI_CHANGE:  bull = max(bull, 8.0)
        elif ce30 <= -MIN_OI_CHANGE:   bull = max(bull, 6.0)

        return min(10.0, bull * weight * boost), min(10.0, bear * weight * boost)

    async def analyze(self, current: MarketSnapshot) -> Dict:
        s_15m = await self.cache.get_5min_ago(3)
        s_30m = await self.cache.get_5min_ago(6)
        s_60m = await self.cache.get_5min_ago(12)

        if not s_15m:
            return {"available": False, "reason": "⏳ Building cache (need ≥ 15 min)..."}

        analyses: List[StrikeAnalysis] = []

        for strike in sorted(current.strikes_oi.keys()):
            curr = current.strikes_oi[strike]

            # ✅ FIX: Absolute OI threshold — ignore low OI strikes
            if curr.ce_oi < MIN_ABS_OI_THRESHOLD and curr.pe_oi < MIN_ABS_OI_THRESHOLD:
                logger.debug(f"⚠️ Skipping strike ${strike} — OI too low")
                continue

            p15  = s_15m.strikes_oi.get(strike)
            p30  = s_30m.strikes_oi.get(strike) if s_30m else None
            p60  = s_60m.strikes_oi.get(strike) if s_60m else None

            ce15 = self._pct(curr.ce_oi,     p15.ce_oi     if p15 else 0)
            pe15 = self._pct(curr.pe_oi,     p15.pe_oi     if p15 else 0)
            cv15 = self._pct(curr.ce_volume, p15.ce_volume  if p15 else 0)
            pv15 = self._pct(curr.pe_volume, p15.pe_volume  if p15 else 0)
            pcr15= self._pct(curr.pcr,       p15.pcr        if p15 else 0)

            ce30 = self._pct(curr.ce_oi,     p30.ce_oi     if p30 else 0)
            pe30 = self._pct(curr.pe_oi,     p30.pe_oi     if p30 else 0)
            cv30 = self._pct(curr.ce_volume, p30.ce_volume  if p30 else 0)
            pv30 = self._pct(curr.pe_volume, p30.pe_volume  if p30 else 0)
            pcr30= self._pct(curr.pcr,       p30.pcr        if p30 else 0)

            ce60 = self._pct(curr.ce_oi, p60.ce_oi if p60 else 0)
            pe60 = self._pct(curr.pe_oi, p60.pe_oi if p60 else 0)

            # ✅ FIX: Strike-wise PCR delta
            pcr_delta = curr.pcr - (p15.pcr if p15 else curr.pcr)

            is_atm = (strike == current.atm_strike)
            dist   = abs(strike - current.atm_strike)
            if is_atm:       weight = ATM_WEIGHT
            elif dist <= 40: weight = NEAR_ATM_WEIGHT
            else:            weight = FAR_WEIGHT

            tf15 = self._tf_signal(ce15, pe15, cv15, pv15)
            tf30 = self._tf_signal(ce30, pe30, cv30, pv30)
            mtf  = (tf15 == tf30 and tf15 != "NEUTRAL")

            avg_oi  = (ce30 + pe30) / 2
            avg_vol = (cv30 + pv30) / 2
            vc, vs  = self._vol_confirm(avg_oi, avg_vol)

            bull, bear = self._signal_strength(ce30, pe30, cv30, pv30, weight, mtf)
            if mtf:
                bull = min(10.0, bull * 1.2)
                bear = min(10.0, bear * 1.2)

            if bull >= 7 and bull > bear:   rec, conf = "STRONG_CALL", bull
            elif bear >= 7 and bear > bull: rec, conf = "STRONG_PUT",  bear
            else:                            rec, conf = "WAIT", max(bull, bear)

            analyses.append(StrikeAnalysis(
                strike=strike, is_atm=is_atm, distance_atm=dist, weight=weight,
                ce_oi=curr.ce_oi, pe_oi=curr.pe_oi,
                ce_volume=curr.ce_volume, pe_volume=curr.pe_volume,
                ce_ltp=curr.ce_ltp, pe_ltp=curr.pe_ltp,
                ce_oi_15=ce15, pe_oi_15=pe15, ce_vol_15=cv15, pe_vol_15=pv15, pcr_ch_15=pcr15,
                ce_oi_30=ce30, pe_oi_30=pe30, ce_vol_30=cv30, pe_vol_30=pv30, pcr_ch_30=pcr30,
                ce_oi_60=ce60, pe_oi_60=pe60,
                pcr=curr.pcr, pcr_delta=pcr_delta,
                ce_action=self._action(ce15), pe_action=self._action(pe15),
                tf15_signal=tf15, tf30_signal=tf30, mtf_confirmed=mtf,
                vol_confirms=vc, vol_strength=vs,
                is_support=False, is_resistance=False,
                bull_strength=bull, bear_strength=bear,
                recommendation=rec, confidence=conf
            ))

        sr = self._find_sr(current, analyses)
        for sa in analyses:
            sa.is_support    = (sa.strike == sr.support_strike)
            sa.is_resistance = (sa.strike == sr.resistance_strike)

        prev_pcr  = s_30m.overall_pcr if s_30m else current.overall_pcr
        pcr_trend = "BULLISH" if current.overall_pcr > prev_pcr else "BEARISH"
        pcr_ch    = self._pct(current.overall_pcr, prev_pcr)

        total_bull = sum(sa.bull_strength for sa in analyses)
        total_bear = sum(sa.bear_strength for sa in analyses)
        if   total_bull > total_bear and total_bull >= 15: overall = "BULLISH"
        elif total_bear > total_bull and total_bear >= 15: overall = "BEARISH"
        else:                                               overall = "NEUTRAL"

        has_strong = any(sa.mtf_confirmed and sa.confidence >= MIN_CONFIDENCE for sa in analyses)

        return {
            "available":       True,
            "strike_analyses": analyses,
            "sr":              sr,
            "overall":         overall,
            "total_bull":      total_bull,
            "total_bear":      total_bear,
            "overall_pcr":     current.overall_pcr,
            "pcr_trend":       pcr_trend,
            "pcr_ch_pct":      pcr_ch,
            "pcr_delta":       current.pcr_delta,
            "has_15m":         s_15m is not None,
            "has_30m":         s_30m is not None,
            "has_strong":      has_strong
        }

    def _find_sr(self, current: MarketSnapshot, analyses: List[StrikeAnalysis]) -> SupportResistance:
        max_pe = max(analyses, key=lambda x: x.pe_oi, default=None)
        max_ce = max(analyses, key=lambda x: x.ce_oi, default=None)
        sup    = max_pe.strike if max_pe else current.atm_strike
        res    = max_ce.strike if max_ce else current.atm_strike
        return SupportResistance(
            support_strike=sup, support_put_oi=max_pe.pe_oi if max_pe else 0,
            resistance_strike=res, resistance_call_oi=max_ce.ce_oi if max_ce else 0,
            near_support=abs(current.spot_price - sup) <= ATM_PROX_USD,
            near_resistance=abs(current.spot_price - res) <= ATM_PROX_USD
        )


# ============================================================
#  STANDALONE ALERT CHECKER
# ============================================================

class AlertChecker:
    COOLDOWN = 30 * 60

    def __init__(self, cache: DualCache, alerter):
        self.cache   = cache
        self.alerter = alerter
        self._last: Dict[str, float] = {}

    def _can(self, key: str) -> bool:
        return (time_module.time() - self._last.get(key, 0)) >= self.COOLDOWN

    def _mark(self, key: str):
        self._last[key] = time_module.time()

    async def check_all(self, curr: MarketSnapshot):
        prev = await self.cache.get_5min_ago(6)
        if not prev:
            return
        await self._oi_change(curr, prev)
        # ✅ FIX: Volume spike alert skips midnight window
        if not is_midnight_reset_window():
            await self._vol_spike(curr, prev)
        await self._pcr_change(curr, prev)
        await self._atm_proximity(curr)

    async def _oi_change(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("OI"):
            return
        # ✅ FIX: ATM shift check
        if curr.atm_strike != prev.atm_strike:
            return
        atm_c = curr.strikes_oi.get(curr.atm_strike)
        atm_p = prev.strikes_oi.get(curr.atm_strike)
        if not atm_c or not atm_p or atm_p.ce_oi == 0 or atm_p.pe_oi == 0:
            return
        # ✅ FIX: Absolute OI check
        if atm_c.ce_oi < MIN_ABS_OI_THRESHOLD and atm_c.pe_oi < MIN_ABS_OI_THRESHOLD:
            return
        ce_ch = (atm_c.ce_oi - atm_p.ce_oi) / atm_p.ce_oi * 100
        pe_ch = (atm_c.pe_oi - atm_p.pe_oi) / atm_p.pe_oi * 100
        if abs(ce_ch) < OI_ALERT_PCT and abs(pe_ch) < OI_ALERT_PCT:
            return
        txt = (
            f"⚠️ <b>OI CHANGE ALERT (30-min)</b>\n\n"
            f"ETH: ${curr.spot_price:,.2f}\n"
            f"ATM: ${curr.atm_strike:,.0f}\n\n"
            f"CALL OI: {ce_ch:+.1f}%  ({atm_c.ce_oi:.0f} contracts)  {'🔴 BUILDING' if ce_ch > 0 else '🟢 UNWINDING'}\n"
            f"PUT  OI: {pe_ch:+.1f}%  ({atm_c.pe_oi:.0f} contracts)  {'🟢 BUILDING' if pe_ch > 0 else '🔴 UNWINDING'}\n\n"
            f"Overall PCR: {curr.overall_pcr:.2f}  Δ: {curr.pcr_delta:+.3f}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("OI")

    async def _vol_spike(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("VOL"):
            return
        if curr.atm_strike != prev.atm_strike:
            return
        atm_c = curr.strikes_oi.get(curr.atm_strike)
        atm_p = prev.strikes_oi.get(curr.atm_strike)
        if not atm_c or not atm_p or atm_p.ce_volume == 0 or atm_p.pe_volume == 0:
            return
        ce_v = (atm_c.ce_volume - atm_p.ce_volume) / atm_p.ce_volume * 100
        pe_v = (atm_c.pe_volume - atm_p.pe_volume) / atm_p.pe_volume * 100
        if max(ce_v, pe_v) < VOL_SPIKE_PCT:
            return
        dom  = "CALL" if ce_v >= pe_v else "PUT"
        bias = "🔴 BEARISH" if dom == "CALL" else "🟢 BULLISH"
        txt  = (
            f"🔥 <b>VOLUME SPIKE ALERT</b>\n\n"
            f"ETH: ${curr.spot_price:,.2f}\n"
            f"ATM: ${curr.atm_strike:,.0f}\n\n"
            f"CALL Vol: {ce_v:+.1f}%\n"
            f"PUT  Vol: {pe_v:+.1f}%\n\n"
            f"Dominant: {dom}  →  {bias}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("VOL")

    async def _pcr_change(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("PCR") or prev.overall_pcr <= 0:
            return
        pcr_ch = (curr.overall_pcr - prev.overall_pcr) / prev.overall_pcr * 100
        if abs(pcr_ch) < PCR_ALERT_PCT:
            return
        trend  = "📈 BULLS GAINING" if pcr_ch > 0 else "📉 BEARS GAINING"
        interp = ("Strong PUT base → Bullish" if curr.overall_pcr > PCR_BULL
                  else "Strong CALL base → Bearish" if curr.overall_pcr < PCR_BEAR
                  else "Neutral zone")
        txt = (
            f"📊 <b>PCR CHANGE ALERT</b>\n\n"
            f"ETH: ${curr.spot_price:,.2f}\n\n"
            f"PCR: {prev.overall_pcr:.2f} → <b>{curr.overall_pcr:.2f}</b>  ({pcr_ch:+.1f}%)\n"
            f"PCR Δ: {curr.pcr_delta:+.3f}/5min\n"
            f"Trend: {trend}\nBias: {interp}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("PCR")

    async def _atm_proximity(self, curr: MarketSnapshot):
        if not self._can("PROX"):
            return
        valid = {k: v for k, v in curr.strikes_oi.items()
                 if v.pe_oi >= MIN_ABS_OI_THRESHOLD or v.ce_oi >= MIN_ABS_OI_THRESHOLD}
        if not valid:
            return
        max_pe = max(valid.items(), key=lambda x: x[1].pe_oi, default=None)
        max_ce = max(valid.items(), key=lambda x: x[1].ce_oi, default=None)
        spot   = curr.spot_price
        for level, item, kind, emoji in [
            ("SUPPORT",    max_pe, "PUT",  "🟢"),
            ("RESISTANCE", max_ce, "CALL", "🔴")
        ]:
            if not item:
                continue
            strike, oi_snap = item
            dist   = abs(spot - strike)
            if dist > ATM_PROX_USD:
                continue
            oi_val = oi_snap.pe_oi if kind == "PUT" else oi_snap.ce_oi
            note   = ("PUT writers defending → Watch for bounce"
                      if kind == "PUT" else "CALL writers defending → Watch for rejection")
            txt = (
                f"{emoji} <b>PRICE NEAR {level} ALERT</b>\n\n"
                f"ETH: ${spot:,.2f}\n"
                f"{level}: ${strike:,.0f}  (OI: {oi_val:,.0f} contracts)\n"
                f"Dist: ${dist:.2f}\n\n{note}\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            await self.alerter.send_raw(txt)
            self._mark("PROX")
            break


# ============================================================
#  CANDLESTICK PATTERN DETECTOR
# ============================================================

class PatternDetector:

    @staticmethod
    def detect(df: pd.DataFrame) -> List[Dict]:
        patterns = []
        if df.empty or len(df) < 2:
            return patterns
        for i in range(1, len(df)):
            cur, prv = df.iloc[i], df.iloc[i-1]
            body_c = abs(cur.close - cur.open)
            body_p = abs(prv.close - prv.open)
            rng    = cur.high - cur.low
            if rng == 0:
                continue
            if (cur.close > cur.open and prv.close < prv.open
                    and cur.open <= prv.close and cur.close >= prv.open
                    and body_c > body_p * 1.2):
                patterns.append({"time": cur.name, "pattern": "BULLISH_ENGULFING", "type": "BULLISH", "strength": 8})
            elif (cur.close < cur.open and prv.close > prv.open
                    and cur.open >= prv.close and cur.close <= prv.open
                    and body_c > body_p * 1.2):
                patterns.append({"time": cur.name, "pattern": "BEARISH_ENGULFING", "type": "BEARISH", "strength": 8})
            else:
                lo_wick = min(cur.open, cur.close) - cur.low
                hi_wick = cur.high - max(cur.open, cur.close)
                if lo_wick > body_c * 2 and hi_wick < body_c * 0.3 and body_c < rng * 0.35:
                    patterns.append({"time": cur.name, "pattern": "HAMMER", "type": "BULLISH", "strength": 7})
                elif hi_wick > body_c * 2 and lo_wick < body_c * 0.3 and body_c < rng * 0.35:
                    patterns.append({"time": cur.name, "pattern": "SHOOTING_STAR", "type": "BEARISH", "strength": 7})
                elif body_c < rng * 0.1:
                    patterns.append({"time": cur.name, "pattern": "DOJI", "type": "NEUTRAL", "strength": 4})
        return patterns[-5:]

    @staticmethod
    def support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty or len(df) < 5:
            return 0.0, 0.0
        last = df.tail(20)
        return float(last.low.min()), float(last.high.max())


# ============================================================
#  PROMPT BUILDER v8.0 (PCR delta added)
# ============================================================

class PromptBuilder:

    @staticmethod
    def _filter_strikes(analyses: List[StrikeAnalysis]) -> List[StrikeAnalysis]:
        # ±3 strikes च्या आत आहेत ते सगळे + significant OI असलेले
        return [sa for sa in analyses
                if (sa.ce_oi >= MIN_ABS_OI_THRESHOLD or sa.pe_oi >= MIN_ABS_OI_THRESHOLD)]

    @staticmethod
    def _candle_table(df: pd.DataFrame, label: str) -> str:
        if df.empty:
            return f"{label}: no data\n"
        out = f"\n{label} CANDLES (TIME|OPEN|HIGH|LOW|CLOSE|VOL|DIR):\n"
        for ts, row in df.tail(CANDLE_COUNT).iterrows():
            t = ts.strftime("%H:%M")
            d = "↑" if row.close > row.open else "↓" if row.close < row.open else "→"
            out += f"{t}|{row.open:.0f}|{row.high:.0f}|{row.low:.0f}|{row.close:.0f}|{row.volume:.0f}|{d}\n"
        return out

    @staticmethod
    def build(spot: float, atm: float, expiry: str, oi: Dict,
              c15: pd.DataFrame, c30: pd.DataFrame,
              patterns: List[Dict], p_sup: float, p_res: float,
              pa: PriceActionInsight,
              phase_signal: Optional[PhaseSignal] = None) -> str:

        now = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sr  = oi["sr"]
        pcr = oi["overall_pcr"]
        filtered = PromptBuilder._filter_strikes(oi["strike_analyses"])

        p  = "You are an expert ETH options trader on Delta Exchange India.\n"
        p += "Analyze OI, Volume, PCR delta, Price Action, and Candlesticks for a precise trade signal.\n\n"

        p += f"=== MARKET SNAPSHOT | {now} | Expiry: {expiry} ===\n"
        p += f"ETH Spot: ${spot:,.2f}\n"
        p += f"ATM Strike: ${atm:,.0f}\n"
        p += f"Overall PCR: {pcr:.2f} ({oi['pcr_trend']})  Δ30m: {oi['pcr_ch_pct']:+.1f}%\n"
        p += f"PCR Delta (rate): {oi['pcr_delta']:+.4f}/5min  ← Speed of PCR change\n"
        p += f"OI Support: ${sr.support_strike:,.0f} | OI Resistance: ${sr.resistance_strike:,.0f}\n"
        if sr.near_support:    p += "⚡ PRICE NEAR OI-SUPPORT!\n"
        if sr.near_resistance: p += "⚡ PRICE NEAR OI-RESISTANCE!\n"

        p += f"\n=== PRICE ACTION ===\n"
        p += f"Price: 5m={pa.price_change_5m:+.2f}%  15m={pa.price_change_15m:+.2f}%  30m={pa.price_change_30m:+.2f}%\n"
        p += f"Momentum: {pa.price_momentum}\n"
        p += f"Vol Spike: {pa.vol_spike_ratio:.2f}x rolling avg\n"
        p += f"OI-Vol Corr: {pa.oi_vol_corr:.2f}  Price-OI Corr: {pa.price_oi_corr:.2f}\n"
        p += f"Trend Strength: {pa.trend_strength:.1f}/10\n"
        p += f"Triple Confirm: {' ✅ YES' if pa.triple_confirmed else ' ❌ NO'}\n"
        p += f"PCR Acceleration: {pa.pcr_acceleration}  Δ={pa.pcr_delta:+.4f}\n"
        if pa.support_levels:
            p += f"Price Support: {', '.join(f'${s:.0f}' for s in pa.support_levels)}\n"
        if pa.resistance_levels:
            p += f"Price Resistance: {', '.join(f'${r:.0f}' for r in pa.resistance_levels)}\n"

        if phase_signal:
            p += f"\n=== PHASE {phase_signal.phase} TRIGGERED ===\n"
            p += f"Side: {phase_signal.dominant_side} | Direction: {phase_signal.direction}\n"
            p += f"OI: {phase_signal.oi_change_pct:+.1f}% ({phase_signal.oi_abs_value:.0f} contracts)\n"
            p += f"Vol: {phase_signal.vol_change_pct:+.1f}% | Price: {phase_signal.price_change_pct:+.2f}%\n"
            p += f"PCR Δ: {phase_signal.pcr_delta:+.4f}\n"

        p += f"\n=== OI MULTI-TIMEFRAME ({len(filtered)} strikes, min OI={MIN_ABS_OI_THRESHOLD}) ===\n"
        p += "STRIKE | CE_OI(abs) CE%15m CE%30m | PE_OI(abs) PE%15m PE%30m | PCR PCRdelta | TF15 TF30 MTF | Bull Bear\n"
        for sa in filtered:
            tag = "⭐ATM" if sa.is_atm else ("🟢SUP" if sa.is_support else ("🔴RES" if sa.is_resistance else "    "))
            mtf = "✅" if sa.mtf_confirmed else "❌"
            p += (f"${sa.strike:,.0f}{tag} | "
                  f"CE:{sa.ce_oi:.0f} {sa.ce_oi_15:+.0f}% {sa.ce_oi_30:+.0f}% | "
                  f"PE:{sa.pe_oi:.0f} {sa.pe_oi_15:+.0f}% {sa.pe_oi_30:+.0f}% | "
                  f"PCR:{sa.pcr:.2f} Δ{sa.pcr_delta:+.3f} | "
                  f"{sa.tf15_signal[:3]} {sa.tf30_signal[:3]} {mtf} | "
                  f"B:{sa.bull_strength:.0f} Br:{sa.bear_strength:.0f}\n")

        p += PromptBuilder._candle_table(c15, "15-MIN")
        p += PromptBuilder._candle_table(c30, "30-MIN")

        if patterns:
            p += "\n=== CANDLESTICK PATTERNS ===\n"
            for pat in patterns:
                p += f"{pat['time'].strftime('%H:%M')} | {pat['pattern']} | {pat['type']} | Str:{pat['strength']}/10\n"

        if p_sup or p_res:
            p += f"\nCandle S/R: Support=${p_sup:.0f}  Resistance=${p_res:.0f}\n"

        p += f"""
=== TRADING RULES ===
PCR DELTA RULES:
• PCR Δ > +0.05/5min = RISING_FAST = Strong bullish bias
• PCR Δ < -0.05/5min = FALLING_FAST = Strong bearish bias
• Use PCR acceleration to confirm signal direction

OI RULES:
• CALL OI↑ (large abs) + Vol↑ = Resistance = BEARISH → BUY_PUT
• PUT OI↑ (large abs) + Vol↑ = Support = BULLISH → BUY_CALL
• OI↑ but Vol flat = TRAP

ENTRY RULES:
• MTF confirmed + Vol confirms + Price confirms = EXECUTE
• Min confidence 7/10 for BUY signal

RESPOND ONLY VALID JSON:
{{
  "signal": "BUY_CALL"|"BUY_PUT"|"WAIT",
  "primary_strike": {atm},
  "confidence": 0-10,
  "stop_loss_strike": 0,
  "target_strike": 0,
  "mtf": {{"tf15": "", "tf30": "", "confirmed": true|false}},
  "price_action": {{"momentum": "", "triple_confirmed": true|false, "confirms_signal": true|false}},
  "candle_pattern": {{"latest_pattern": "", "type": "", "confirms_signal": true|false, "near_sr": true|false}},
  "atm": {{"ce_action": "", "pe_action": "", "vol_confirms": true|false, "strength": ""}},
  "pcr": {{"value": {pcr:.2f}, "trend": "{oi['pcr_trend']}", "delta": {oi['pcr_delta']:.4f}, "acceleration": "", "supports": true|false}},
  "volume": {{"ok": true|false, "spike_ratio": {pa.vol_spike_ratio:.2f}, "trap_warning": ""}},
  "entry": {{"now": true|false, "reason": "", "wait_for": ""}},
  "rr": {{"sl_pts": 0, "tgt_pts": 0, "ratio": 0}},
  "levels": {{"support": {sr.support_strike}, "resistance": {sr.resistance_strike}}}
}}"""
        return p


# ============================================================
#  DEEPSEEK CLIENT v8.0 (Robust JSON parsing)
# ============================================================

class DeepSeekClient:
    URL   = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self, key: str):
        self.key = key

    async def analyze(self, prompt: str) -> Optional[Dict]:
        hdrs    = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": self.MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.2, "max_tokens": 1500}
        try:
            timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(self.URL, headers=hdrs, json=payload) as r:
                    if r.status != 200:
                        logger.error(f"❌ DeepSeek {r.status}: {await r.text()}")
                        return None
                    data    = await r.json()
                    content = data["choices"][0]["message"]["content"].strip()

                    # ✅ FIX: Robust JSON extraction
                    content = re.sub(r"```json|```", "", content).strip()
                    # Find JSON object in response
                    match = re.search(r"\{[\s\S]*\}", content)
                    if match:
                        content = match.group(0)
                    return json.loads(content)
        except asyncio.TimeoutError:
            logger.error(f"❌ DeepSeek timeout")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ DeepSeek JSON error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ DeepSeek error: {e}")
            return None


# ============================================================
#  TELEGRAM ALERTER
# ============================================================

class TelegramAlerter:

    def __init__(self, token: str, chat_id: str):
        self.token   = token
        self.chat_id = chat_id
        self.session = None

    async def _sess(self):
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        if self.session:
            await self.session.close()

    async def send_raw(self, text: str):
        await self._sess()
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            async with self.session.post(url, json={
                "chat_id": self.chat_id, "text": text, "parse_mode": "HTML"
            }) as r:
                if r.status == 200:
                    logger.info("✅ Telegram sent")
                else:
                    logger.error(f"❌ Telegram {r.status}")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

    async def send_signal(self, sig: Dict, snap: MarketSnapshot, oi: Dict, pa: PriceActionInsight):
        mtf  = sig.get("mtf",           {})
        atma = sig.get("atm",           {})
        pcra = sig.get("pcr",           {})
        vol  = sig.get("volume",        {})
        ent  = sig.get("entry",         {})
        rr   = sig.get("rr",            {})
        prce = sig.get("price_action",  {})
        cndl = sig.get("candle_pattern", {})

        signal_type = sig.get("signal", "WAIT")
        option_type = "CE" if "CALL" in signal_type else "PE" if "PUT" in signal_type else ""

        msg = (
            f"🚨 <b>ETH OPTIONS SIGNAL v8.0 PRO 🇮🇳</b>\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%d-%b %H:%M UTC')}\n\n"
            f"💰 ETH: <b>${snap.spot_price:,.2f}</b>\n"
            f"📊 Signal: <b>{signal_type}</b>\n"
            f"⭐ Confidence: <b>{sig.get('confidence', 0)}/10</b>\n"
            f"📅 Expiry: {snap.expiry}\n\n"
            f"💼 <b>TRADE SETUP:</b>\n"
            f"Entry: ${sig.get('primary_strike', 0):,.0f} {option_type}\n"
            f"SL:    ${sig.get('stop_loss_strike', 0):,.0f}\n"
            f"Tgt:   ${sig.get('target_strike', 0):,.0f}\n"
            f"RR:    {rr.get('ratio', 'N/A')}\n\n"
            f"🔗 <b>MTF:</b> {mtf.get('tf15','N/A')} / {mtf.get('tf30','N/A')} "
            f"{'✅ HIGH CONF' if mtf.get('confirmed') else '❌ Single TF'}\n\n"
            f"📈 <b>PRICE ACTION:</b>\n"
            f"5m:{pa.price_change_5m:+.2f}%  15m:{pa.price_change_15m:+.2f}%\n"
            f"Momentum: {pa.price_momentum}\n"
            f"Triple: {'✅' if pa.triple_confirmed else '❌'}\n\n"
            f"📊 <b>PCR:</b> {snap.overall_pcr:.2f}  Δ:{snap.pcr_delta:+.3f}/5m ({pa.pcr_acceleration})\n"
            f"PCR supports: {'✅' if pcra.get('supports') else '❌'}\n\n"
            f"🕯️ <b>CANDLE:</b> {cndl.get('latest_pattern','None')} "
            f"{'✅' if cndl.get('confirms_signal') else '❌'}\n\n"
            f"📊 <b>OI:</b> CE {atma.get('ce_action','N/A')} | PE {atma.get('pe_action','N/A')}\n"
            f"Vol: {'✅' if atma.get('vol_confirms') else '❌ TRAP?'}\n\n"
            f"⏰ Enter: {'✅ NOW' if ent.get('now') else '⏳ WAIT'}\n"
            f"{ent.get('reason','')}\n\n"
            f"🤖 DeepSeek V3 + MTF v8.0 PRO\n"
            f"🇮🇳 Delta Exchange India"
        )
        if vol.get("trap_warning"):
            msg += f"\n⚠️ {vol['trap_warning']}"
        await self.send_raw(msg)


# ============================================================
#  MAIN BOT v8.0
# ============================================================

class ETHOptionsBot:

    def __init__(self):
        self.delta   = DeltaClient(DELTA_API_KEY, DELTA_API_SECRET)
        self.cache   = DualCache()
        self.mtf     = MTFAnalyzer(self.cache)
        self.ai      = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.checker = AlertChecker(self.cache, self.alerter)
        self.phase   = PhaseDetector()
        self._cycle  = 0

    async def run(self):
        logger.info("="*60)
        logger.info("🚀 ETH OPTIONS BOT v8.0 PRO – DELTA EXCHANGE INDIA 🇮🇳")
        logger.info("="*60)
        logger.info(f"Strikes: ATM ±{ATM_STRIKE_FETCH} | Phase1 OI: {PHASE1_OI_BUILD_PCT}% | Compare: {PHASE1_COMPARE_SNAPS*5}min ago")
        logger.info(f"Min Abs OI: {MIN_ABS_OI_THRESHOLD} | Midnight guard: ±{MIDNIGHT_RESET_BUFFER_MIN}min")
        logger.info("="*60)

        await self.delta.init()
        try:
            while True:
                try:
                    await self._cycle_run()
                except aiohttp.ClientConnectorError:
                    logger.error("❌ Network error — retry 60s")
                    await asyncio.sleep(60)
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(60)
                s5, s30 = self.cache.sizes()
                logger.info(f"⏰ Next 5min | Cache: 5m={s5} 30m={s30}\n")
                await asyncio.sleep(SNAPSHOT_INTERVAL)
        except KeyboardInterrupt:
            logger.info("🛑 Stopped")
        finally:
            await self.delta.close()
            await self.alerter.close()

    async def _cycle_run(self):
        self._cycle += 1
        is_analysis = (self._cycle % 6 == 0)

        logger.info(f"\n{'='*60}")
        logger.info(f"{'🚀 FULL ANALYSIS' if is_analysis else '📦 SNAPSHOT'} #{self._cycle}")
        logger.info(f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        if is_midnight_reset_window():
            logger.info("🌙 Midnight reset window active")
        logger.info("="*60)

        snap = await self.delta.fetch_snapshot(self.cache)
        if not snap:
            logger.warning("⚠️ Snapshot failed")
            return

        await self.cache.add_5min(snap)
        if is_analysis:
            await self.cache.add_30min(snap)

        await self.checker.check_all(snap)

        recent_snaps = await self.cache.get_recent_snapshots(12)
        c15 = await self.delta.get_candles("ETHUSD", "15m", CANDLE_COUNT)
        pa  = PriceActionCalculator.calculate(recent_snaps, c15)

        logger.info(f"📊 5m={pa.price_change_5m:+.2f}% | Vol:{pa.vol_spike_ratio:.2f}x | PCR Δ:{pa.pcr_delta:+.3f}({pa.pcr_acceleration}) | Triple:{pa.triple_confirmed}")

        # ✅ FIX: Phase uses 15min ago (3 snaps) instead of 5min ago
        prev_15m      = await self.cache.get_5min_ago(PHASE1_COMPARE_SNAPS)
        phase_signals = await self.phase.detect(snap, prev_15m, pa)

        for ps in phase_signals:
            logger.info(f"🚨 Phase {ps.phase}: {ps.direction} | OI:{ps.oi_change_pct:+.1f}%({ps.oi_abs_value:.0f})")
            await self.alerter.send_raw(ps.message)
            if ps.phase == 3:
                await self._trigger_ai_analysis(snap, pa, ps)

        if is_analysis:
            await self._full_analysis(snap, pa)

    async def _trigger_ai_analysis(self, snap: MarketSnapshot,
                                   pa: PriceActionInsight,
                                   phase_signal: PhaseSignal):
        logger.info("🤖 Phase 3 → AI call...")
        c15 = await self.delta.get_candles("ETHUSD", "15m", CANDLE_COUNT)
        await asyncio.sleep(API_DELAY)
        c30 = await self.delta.get_candles("ETHUSD", "30m", CANDLE_COUNT)
        patterns     = PatternDetector.detect(c15)    if not c15.empty else []
        p_sup, p_res = PatternDetector.support_resistance(c15) if not c15.empty else (0.0, 0.0)

        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            oi = {
                "available": True, "strike_analyses": [],
                "sr": SupportResistance(snap.atm_strike, 0, snap.atm_strike, 0, False, False),
                "overall": phase_signal.direction, "total_bull": 0, "total_bear": 0,
                "overall_pcr": snap.overall_pcr, "pcr_trend": "N/A",
                "pcr_ch_pct": 0, "pcr_delta": snap.pcr_delta,
                "has_15m": False, "has_30m": False, "has_strong": True
            }

        prompt = PromptBuilder.build(
            spot=snap.spot_price, atm=snap.atm_strike, expiry=snap.expiry,
            oi=oi, c15=c15, c30=c30, patterns=patterns,
            p_sup=p_sup, p_res=p_res, pa=pa, phase_signal=phase_signal
        )
        ai_sig = await self.ai.analyze(prompt)
        if not ai_sig:
            return
        if ai_sig.get("confidence", 0) >= MIN_CONFIDENCE and ai_sig.get("signal") != "WAIT":
            await self.alerter.send_signal(ai_sig, snap, oi, pa)

    async def _full_analysis(self, snap: MarketSnapshot, pa: PriceActionInsight):
        logger.info("🔍 Full MTF analysis...")
        oi = await self.mtf.analyze(snap)

        if not oi["available"]:
            logger.info(f"⏳ {oi['reason']}")
            return
        if not oi["has_strong"]:
            logger.info("📊 No strong MTF signal")
            return

        c15 = await self.delta.get_candles("ETHUSD", "15m", CANDLE_COUNT)
        await asyncio.sleep(API_DELAY)
        c30 = await self.delta.get_candles("ETHUSD", "30m", CANDLE_COUNT)
        patterns     = PatternDetector.detect(c15)    if not c15.empty else []
        p_sup, p_res = PatternDetector.support_resistance(c15) if not c15.empty else (0.0, 0.0)

        prompt = PromptBuilder.build(
            spot=snap.spot_price, atm=snap.atm_strike, expiry=snap.expiry,
            oi=oi, c15=c15, c30=c30, patterns=patterns,
            p_sup=p_sup, p_res=p_res, pa=pa, phase_signal=None
        )
        ai_sig = await self.ai.analyze(prompt)

        if not ai_sig:
            logger.warning("⚠️ AI timeout — fallback")
            atm_sa = next((sa for sa in oi["strike_analyses"] if sa.is_atm), None)
            fb, fc = ("BUY_CALL" if atm_sa and atm_sa.bull_strength > atm_sa.bear_strength else "BUY_PUT",
                      min(10, max(atm_sa.bull_strength, atm_sa.bear_strength)) if atm_sa and atm_sa.mtf_confirmed else 3)
            ai_sig = {
                "signal": fb if atm_sa and atm_sa.mtf_confirmed else "WAIT",
                "confidence": fc, "primary_strike": snap.atm_strike,
                "mtf": {"tf15": "N/A", "tf30": "N/A", "confirmed": False},
                "entry": {"now": False, "reason": "AI timeout"},
                "price_action": {"momentum": pa.price_momentum, "triple_confirmed": pa.triple_confirmed, "confirms_signal": False},
                "candle_pattern": {"latest_pattern": "N/A", "confirms_signal": False, "near_sr": False},
                "volume": {"ok": False, "spike_ratio": pa.vol_spike_ratio, "trap_warning": ""},
                "pcr": {"value": snap.overall_pcr, "delta": snap.pcr_delta, "acceleration": pa.pcr_acceleration, "supports": False},
                "rr": {}, "atm": {}, "levels": {}
            }

        conf   = ai_sig.get("confidence", 0)
        signal = ai_sig.get("signal", "WAIT")
        logger.info(f"🎯 {signal} | Conf: {conf}/10")

        if conf >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa)
        else:
            logger.info(f"⏳ Conf {conf} < {MIN_CONFIDENCE} — skip")


# ============================================================
#  HTTP HEALTH
# ============================================================

bot_instance: Optional[ETHOptionsBot] = None

async def health(request):
    s5, s30 = bot_instance.cache.sizes() if bot_instance else (0, 0)
    midnight = "🌙 MIDNIGHT GUARD ACTIVE" if is_midnight_reset_window() else ""
    return aiohttp.web.Response(
        text=f"✅ ETH Bot v8.0 ALIVE | {datetime.now(timezone.utc).strftime('%H:%M UTC')} | Cache:5m={s5}/{CACHE_5MIN_SIZE} 30m={s30}/{CACHE_30MIN_SIZE} {midnight}",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )

async def start_bot(app):
    global bot_instance
    bot_instance = ETHOptionsBot()
    app["task"] = asyncio.create_task(bot_instance.run())


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from aiohttp import web

    app = web.Application()
    app.router.add_get("/",       health)
    app.router.add_get("/health", health)
    app.on_startup.append(start_bot)

    port = int(os.getenv("PORT", 8000))

    print(f"""
╔══════════════════════════════════════════════════════════╗
║   🚀 ETH OPTIONS BOT v8.0 PRO – DELTA EXCHANGE INDIA 🇮🇳║
╚══════════════════════════════════════════════════════════╝

✅ v8.0 FIXES:
  1. Strikes: ±10 → ±3 (noise कमी)
  2. Phase1 OI: 5% → 12% (less false signals)
  3. Phase compare: 5min → 15min ago (more reliable)
  4. Absolute OI minimum check (fake % change block)
  5. Phase detection: ATM ±1 weighted aggregate
  6. PCR delta (change rate) tracking added
  7. ATM shift between snapshots handled
  8. Volume midnight reset guard (±10min)
  9. Expiry rollover detection
  10. DeepSeek robust JSON parsing (regex extraction)

🚨 PHASE DETECTION:
  Phase 1: OI ≥ {PHASE1_OI_BUILD_PCT}% + Abs OI ≥ {PHASE1_MIN_ABS_OI} + Vol < {PHASE1_VOL_MAX_PCT}% (vs 15min ago)
  Phase 2: Vol spike {PHASE2_VOL_SPIKE_PCT}%+ + OI building → Move imminent
  Phase 3: Price {PHASE3_PRICE_MOVE_PCT}%+ + Triple confirmed → EXECUTE

Starting on port {port}...
""")

    web.run_app(app, host="0.0.0.0", port=port)
