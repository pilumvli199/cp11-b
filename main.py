"""
🚀 ETH OPTIONS BOT - DELTA EXCHANGE INDIA v7.0 PRO
====================================================
Platform  : Delta Exchange India (api.india.delta.exchange)
Asset     : ETH Daily Options
Updated   : Feb 2026

✅ v7.0 CHANGES FROM v6.0:
- 5-min polling (was 15-min) — faster signals
- PHASE 1 ALERT: OI building quietly (Smart Money accumulation)
- PHASE 2 ALERT: Volume spike (Move imminent)
- PHASE 3 ALERT: Price confirms + AI signal
- Pandas/Numpy pre-calculation before AI call
- Smart AI trigger: Only call DeepSeek when Phase 1+2 detected
- FIXED: DeepSeek model → deepseek-chat (V3, was wrongly set to R1)
- FIXED: Strike interval mode calculation (statistics.mode)
- FIXED: Koyeb sleep (Cache-Control header)
- IMPROVED: DeepSeek prompt with candlestick + S/R + price action logic
- Price correlation with OI + Volume (Triple confirmation)
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

# ============================================================
#  CONFIGURATION
# ============================================================
DELTA_API_KEY      = os.getenv("DELTA_API_KEY",      "YOUR_API_KEY")
DELTA_API_SECRET   = os.getenv("DELTA_API_SECRET",   "YOUR_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY",   "YOUR_DEEPSEEK_KEY")

# API
DELTA_BASE_URL = "https://api.india.delta.exchange"

# Asset
UNDERLYING       = "ETH"
ATM_STRIKE_FETCH = 10
ATM_STRIKE_AI    = 5
STRIKE_INTERVAL  = 20

# ✅ FIX: 5-min polling (was 15-min)
SNAPSHOT_INTERVAL = 5 * 60     # 5 min  → always collect snapshot
ANALYSIS_INTERVAL = 30 * 60    # 30 min → full analysis

# Cache sizes
CACHE_5MIN_SIZE  = 72   # 72 × 5-min  = 6 hr
CACHE_30MIN_SIZE = 12   # 12 × 30-min = 6 hr

# Candles
CANDLE_COUNT = 24

# Signal thresholds
MIN_OI_CHANGE    = 10.0
STRONG_OI_CHANGE = 20.0
MIN_VOLUME_CHG   = 15.0
PCR_BULL         = 1.3
PCR_BEAR         = 0.7
MIN_CONFIDENCE   = 7

# ── PHASE DETECTION THRESHOLDS ──
# Phase 1: OI accumulation (Smart Money)
PHASE1_OI_BUILD_PCT   = 5.0    # OI ≥ 5% increase (quiet accumulation)
PHASE1_VOL_MAX_PCT    = 10.0   # BUT Volume still low (< 10% change)

# Phase 2: Volume spike (Move imminent)
PHASE2_VOL_SPIKE_PCT  = 20.0   # Volume ≥ 20% above rolling avg
PHASE2_OI_MIN_PCT     = 3.0    # OI already building ≥ 3%

# Phase 3: Price confirmation
PHASE3_PRICE_MOVE_PCT = 0.4    # Price ≥ 0.4% move in 5 min
PHASE3_CONFIRM_ALL    = True   # Require OI + Vol + Price all confirm

# Standalone alert thresholds
OI_ALERT_PCT     = 15.0
VOL_SPIKE_PCT    = 25.0
PCR_ALERT_PCT    = 12.0
ATM_PROX_USD     = 50

# Strike weights
ATM_WEIGHT      = 3.0
NEAR_ATM_WEIGHT = 2.0
FAR_WEIGHT      = 1.0

# API settings
MAX_RETRIES      = 3
API_DELAY        = 0.35
DEEPSEEK_TIMEOUT = 45

# AI filtering
MIN_OI_THRESHOLD  = 50
MIN_VOL_THRESHOLD = 10

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


@dataclass
class PhaseSignal:
    """Early warning phase detection result"""
    phase:          int      # 1, 2, or 3
    dominant_side:  str      # CALL or PUT
    direction:      str      # BULLISH or BEARISH
    oi_change_pct:  float
    vol_change_pct: float
    price_change_pct: float
    atm_strike:     float
    spot_price:     float
    confidence:     float
    message:        str


@dataclass
class PriceActionInsight:
    """Pandas/Numpy pre-calculated insights"""
    price_change_5m:   float
    price_change_15m:  float
    price_change_30m:  float
    price_momentum:    str      # BULLISH / BEARISH / NEUTRAL
    vol_rolling_avg:   float
    vol_spike_ratio:   float    # current vol / rolling avg
    oi_vol_corr:       float    # -1 to 1
    price_oi_corr:     float
    support_levels:    List[float]
    resistance_levels: List[float]
    trend_strength:    float    # 0-10
    triple_confirmed:  bool     # OI + Vol + Price all agree


@dataclass
class StrikeAnalysis:
    strike:           float
    is_atm:           bool
    distance_atm:     float
    weight:           float
    ce_oi:            float
    pe_oi:            float
    ce_volume:        float
    pe_volume:        float
    ce_ltp:           float
    pe_ltp:           float
    ce_oi_15:         float
    pe_oi_15:         float
    ce_vol_15:        float
    pe_vol_15:        float
    pcr_ch_15:        float
    ce_oi_30:         float
    pe_oi_30:         float
    ce_vol_30:        float
    pe_vol_30:        float
    pcr_ch_30:        float
    ce_oi_60:         float
    pe_oi_60:         float
    pcr:              float
    ce_action:        str
    pe_action:        str
    tf15_signal:      str
    tf30_signal:      str
    mtf_confirmed:    bool
    vol_confirms:     bool
    vol_strength:     str
    is_support:       bool
    is_resistance:    bool
    bull_strength:    float
    bear_strength:    float
    recommendation:   str
    confidence:       float


@dataclass
class SupportResistance:
    support_strike:     float
    support_put_oi:     float
    resistance_strike:  float
    resistance_call_oi: float
    near_support:       bool
    near_resistance:    bool


# ============================================================
#  DUAL CACHE (5-min + 30-min)
# ============================================================

class DualCache:
    """5-min cache (72 snaps = 6hr) + 30-min cache (12 snaps = 6hr)"""

    def __init__(self):
        self._c5  = deque(maxlen=CACHE_5MIN_SIZE)
        self._c30 = deque(maxlen=CACHE_30MIN_SIZE)
        self._lock = asyncio.Lock()

    async def add_5min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c5.append(snap)
        logger.info(f"📦 5-min cache: {len(self._c5)}/{CACHE_5MIN_SIZE} | PCR:{snap.overall_pcr:.2f}")

    async def add_30min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c30.append(snap)
        logger.info(f"📦 30-min cache: {len(self._c30)}/{CACHE_30MIN_SIZE}")

    async def get_5min_ago(self, n: int) -> Optional[MarketSnapshot]:
        """n × 5-min ago"""
        async with self._lock:
            idx = len(self._c5) - 1 - n
            return self._c5[idx] if idx >= 0 else None

    async def get_30min_ago(self, n: int) -> Optional[MarketSnapshot]:
        async with self._lock:
            idx = len(self._c30) - 1 - n
            return self._c30[idx] if idx >= 0 else None

    async def get_recent_snapshots(self, n: int) -> List[MarketSnapshot]:
        """Last n snapshots (for Pandas/Numpy calculations)"""
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
#  DELTA EXCHANGE INDIA CLIENT
# ============================================================

class DeltaClient:

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = None

    async def init(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )

    async def close(self):
        if self.session:
            await self.session.close()

    def _auth_headers(self, method: str, path: str, query: str = "", body: str = "") -> Dict:
        ts  = str(int(time_module.time()))
        qs  = f"?{query}" if query else ""
        msg = method.upper() + ts + path + qs + body
        sig = hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()
        return {
            "api-key":       self.api_key,
            "timestamp":     ts,
            "signature":     sig,
            "Content-Type":  "application/json"
        }

    async def _get(self, path: str, params: Dict = None, auth: bool = True) -> Optional[Dict]:
        url = DELTA_BASE_URL + path
        qs  = "&".join(f"{k}={v}" for k, v in (params or {}).items())
        hdrs = self._auth_headers("GET", path, qs) if auth else {}

        for attempt in range(MAX_RETRIES):
            try:
                async with self.session.get(url, params=params, headers=hdrs) as r:
                    if r.status == 200:
                        return await r.json()
                    if r.status == 429:
                        await asyncio.sleep((attempt + 1) * 3)
                        continue
                    txt = await r.text()
                    logger.warning(f"⚠️  GET {path} → {r.status}: {txt[:120]}")
                    return None
            except aiohttp.ClientConnectorError:
                logger.error(f"❌ Network error ({attempt+1}/{MAX_RETRIES}) — retrying...")
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"❌ Request error ({attempt+1}/{MAX_RETRIES}): {e}")
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
            logger.error("❌ No products from Delta Exchange India")
            return [], ""

        products  = data["result"]
        eth_opts  = [
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
        return by_expiry[nearest], nearest

    async def get_option_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        data = await self._get("/v2/tickers", params={
            "contract_types": "call_options,put_options"
        })
        if not data or not data.get("result"):
            return {}
        sym_set = set(symbols)
        return {t["symbol"]: t for t in data["result"] if t.get("symbol") in sym_set}

    async def get_candles(self, symbol: str, resolution: str, count: int) -> pd.DataFrame:
        end_ts   = int(time_module.time())
        res_sec  = int(resolution) * 60
        start_ts = end_ts - (count * res_sec) - res_sec * 3

        data = await self._get("/v2/history/candles", params={
            "symbol":     symbol,
            "resolution": resolution,
            "start":      start_ts,
            "end":        end_ts
        })

        if not data or not data.get("result"):
            return pd.DataFrame()

        raw  = data["result"].get("candles", data["result"])
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

    async def fetch_snapshot(self) -> Optional[MarketSnapshot]:
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

        # ✅ FIX: Use statistics.mode for correct strike interval detection
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

            ce_oi  = float(ct.get("oi",     ct.get("open_interest", 0)) or 0)
            pe_oi  = float(pt.get("oi",     pt.get("open_interest", 0)) or 0)
            ce_vol = float(ct.get("volume", 0) or 0)
            pe_vol = float(pt.get("volume", 0) or 0)
            ce_ltp = float(ct.get("close",  ct.get("mark_price", 0)) or 0)
            pe_ltp = float(pt.get("close",  pt.get("mark_price", 0)) or 0)
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
        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            spot_price=spot, atm_strike=atm, expiry=expiry,
            strikes_oi=strikes_oi, overall_pcr=overall_pcr,
            total_ce_oi=t_ce_oi, total_pe_oi=t_pe_oi,
            total_ce_vol=t_ce_vol, total_pe_vol=t_pe_vol
        )


# ============================================================
#  PANDAS / NUMPY PRE-CALCULATOR
# ============================================================

class PriceActionCalculator:
    """
    Pre-calculates all insights using Pandas + Numpy
    BEFORE calling DeepSeek — so AI gets clean, processed data
    not raw numbers.
    """

    @staticmethod
    def calculate(snapshots: List[MarketSnapshot], candles_15m: pd.DataFrame) -> PriceActionInsight:
        if len(snapshots) < 3:
            return PriceActionCalculator._empty()

        # ── Price changes ──
        prices    = np.array([s.spot_price for s in snapshots])
        curr_price = prices[-1]

        def pct_change(ago_idx: int) -> float:
            if len(prices) > ago_idx:
                prev = prices[-(ago_idx + 1)]
                return ((curr_price - prev) / prev * 100) if prev > 0 else 0.0
            return 0.0

        p5m  = pct_change(1)    # 1 snap ago = 5 min
        p15m = pct_change(3)    # 3 snaps ago = 15 min
        p30m = pct_change(6)    # 6 snaps ago = 30 min

        price_mom = "BULLISH" if p5m > 0.3 else "BEARISH" if p5m < -0.3 else "NEUTRAL"

        # ── Volume rolling average ──
        vols = np.array([s.total_ce_vol + s.total_pe_vol for s in snapshots])
        vol_rolling = float(np.mean(vols[:-1])) if len(vols) > 1 else float(vols[-1])
        curr_vol    = float(vols[-1])
        vol_spike   = (curr_vol / vol_rolling) if vol_rolling > 0 else 1.0

        # ── OI series ──
        ce_ois = np.array([s.total_ce_oi for s in snapshots])
        pe_ois = np.array([s.total_pe_oi for s in snapshots])
        price_series = prices

        # ── Correlations (Numpy) ──
        # OI vs Volume correlation
        oi_total = ce_ois + pe_ois
        if len(oi_total) > 2 and np.std(oi_total) > 0 and np.std(vols) > 0:
            oi_vol_corr = float(np.corrcoef(oi_total, vols)[0, 1])
        else:
            oi_vol_corr = 0.0

        # Price vs OI correlation
        if len(price_series) > 2 and np.std(price_series) > 0 and np.std(oi_total) > 0:
            price_oi_corr = float(np.corrcoef(price_series, oi_total)[0, 1])
        else:
            price_oi_corr = 0.0

        # ── S/R from candles ──
        support_levels    = []
        resistance_levels = []
        if not candles_15m.empty and len(candles_15m) >= 5:
            df  = candles_15m.tail(20)
            # Swing lows = support
            lows  = df["low"].values
            highs = df["high"].values
            for i in range(1, len(lows) - 1):
                if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                    support_levels.append(float(lows[i]))
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    resistance_levels.append(float(highs[i]))
            # Keep top 3 nearest to current price
            support_levels    = sorted(support_levels,    key=lambda x: abs(x - curr_price))[:3]
            resistance_levels = sorted(resistance_levels, key=lambda x: abs(x - curr_price))[:3]

        # ── Trend strength (0-10) ──
        # Based on: price momentum + OI buildup + volume confirmation
        ts = 0.0
        if abs(p5m) >= 0.5:  ts += 3.0
        elif abs(p5m) >= 0.3: ts += 1.5
        if vol_spike >= 1.5:  ts += 3.0
        elif vol_spike >= 1.2: ts += 1.5
        oi_ch = ((oi_total[-1] - oi_total[0]) / oi_total[0] * 100) if oi_total[0] > 0 else 0
        if abs(oi_ch) >= 10:  ts += 4.0
        elif abs(oi_ch) >= 5: ts += 2.0
        trend_strength = min(10.0, ts)

        # ── Triple confirmation ──
        # Price + OI + Volume all pointing same direction
        price_bull  = p5m > 0.3
        price_bear  = p5m < -0.3
        oi_bull     = pe_ois[-1] > pe_ois[0] if len(pe_ois) > 1 else False  # PUT OI building = bull
        oi_bear     = ce_ois[-1] > ce_ois[0] if len(ce_ois) > 1 else False  # CALL OI building = bear
        vol_confirm = vol_spike >= 1.2

        triple_bull = price_bull and oi_bull and vol_confirm
        triple_bear = price_bear and oi_bear and vol_confirm
        triple_confirmed = triple_bull or triple_bear

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
            triple_confirmed=triple_confirmed
        )

    @staticmethod
    def _empty() -> PriceActionInsight:
        return PriceActionInsight(
            price_change_5m=0, price_change_15m=0, price_change_30m=0,
            price_momentum="NEUTRAL", vol_rolling_avg=0, vol_spike_ratio=1.0,
            oi_vol_corr=0, price_oi_corr=0, support_levels=[], resistance_levels=[],
            trend_strength=0, triple_confirmed=False
        )


# ============================================================
#  PHASE DETECTOR (EARLY WARNING SYSTEM) 🚨
# ============================================================

class PhaseDetector:
    """
    Phase 1: OI building quietly (Smart Money accumulation)
    Phase 2: Volume spike (Move is imminent)
    Phase 3: Price confirms the move

    Logic:
    ──────
    Phase 1 triggers when:
      - ATM CE or PE OI increased ≥ 5% vs 5-min ago
      - BUT volume still low (< 10% change)  ← Smart Money quietly positioning
      
    Phase 2 triggers when:
      - Volume spike ≥ 20% above rolling average
      - OI already building (Phase 1 condition met)  ← Retail follows
      
    Phase 3 triggers when:
      - Price moves ≥ 0.4% in 5 min
      - OI + Volume already confirmed  ← Execute!
    """

    COOLDOWN_PHASE1 = 15 * 60   # 15 min cooldown
    COOLDOWN_PHASE2 = 10 * 60   # 10 min cooldown
    COOLDOWN_PHASE3 =  5 * 60   #  5 min cooldown

    def __init__(self):
        self._last: Dict[str, float] = {}
        # Track if Phase 1+2 fired recently for Phase 3 to confirm
        self._phase1_fired_at: float = 0
        self._phase2_fired_at: float = 0
        self._phase1_side: str = ""   # CALL or PUT

    def _can(self, key: str, cooldown: int) -> bool:
        return (time_module.time() - self._last.get(key, 0)) >= cooldown

    def _mark(self, key: str):
        self._last[key] = time_module.time()

    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        return ((curr - prev) / prev * 100) if prev > 0 else 0.0

    async def detect(self, curr: MarketSnapshot, prev_5m: Optional[MarketSnapshot],
                     pa: PriceActionInsight) -> List[PhaseSignal]:
        signals = []
        if not prev_5m:
            return signals

        atm_c = curr.strikes_oi.get(curr.atm_strike)
        atm_p = prev_5m.strikes_oi.get(curr.atm_strike)
        if not atm_c or not atm_p:
            return signals

        # ── Per-strike OI + Volume changes ──
        ce_oi_ch  = self._pct(atm_c.ce_oi,     atm_p.ce_oi)
        pe_oi_ch  = self._pct(atm_c.pe_oi,     atm_p.pe_oi)
        ce_vol_ch = self._pct(atm_c.ce_volume,  atm_p.ce_volume)
        pe_vol_ch = self._pct(atm_c.pe_volume,  atm_p.pe_volume)

        # Which side is building?
        call_building = ce_oi_ch >= PHASE1_OI_BUILD_PCT
        put_building  = pe_oi_ch >= PHASE1_OI_BUILD_PCT
        dominant_side = "PUT" if put_building and pe_oi_ch >= ce_oi_ch else ("CALL" if call_building else "")

        if not dominant_side:
            return signals

        oi_ch  = pe_oi_ch  if dominant_side == "PUT"  else ce_oi_ch
        vol_ch = pe_vol_ch if dominant_side == "PUT"  else ce_vol_ch
        direction = "BULLISH" if dominant_side == "PUT" else "BEARISH"
        # (PUT OI building = writers protecting support = BULLISH)
        # (CALL OI building = writers protecting resistance = BEARISH)

        now = time_module.time()

        # ── PHASE 1: OI building, Volume still quiet ──
        if (oi_ch >= PHASE1_OI_BUILD_PCT
                and abs(vol_ch) < PHASE1_VOL_MAX_PCT
                and self._can("PHASE1", self.COOLDOWN_PHASE1)):

            self._phase1_fired_at = now
            self._phase1_side     = dominant_side
            self._mark("PHASE1")

            signals.append(PhaseSignal(
                phase=1, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, vol_change_pct=vol_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, oi_ch / 2),
                message=(
                    f"⚡ <b>PHASE 1 — SMART MONEY POSITIONING</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b>\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"{'PUT' if dominant_side == 'PUT' else 'CALL'} OI: <b>{oi_ch:+.1f}%</b> (building quietly)\n"
                    f"Volume: {vol_ch:+.1f}% (still low — smart money only)\n\n"
                    f"Signal: {direction}\n"
                    f"⚠️ Volume spike not yet — WAIT for Phase 2\n"
                    f"👁️ Keep an eye — move may come in 10-20 min!\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
            ))

        # ── PHASE 2: Volume spike with OI already building ──
        phase1_recent = (now - self._phase1_fired_at) < (25 * 60)  # Phase 1 within 25 min
        if (pa.vol_spike_ratio >= (1 + PHASE2_VOL_SPIKE_PCT / 100)
                and oi_ch >= PHASE2_OI_MIN_PCT
                and phase1_recent
                and self._phase1_side == dominant_side
                and self._can("PHASE2", self.COOLDOWN_PHASE2)):

            self._phase2_fired_at = now
            self._mark("PHASE2")

            signals.append(PhaseSignal(
                phase=2, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, vol_change_pct=vol_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, pa.vol_spike_ratio * 3),
                message=(
                    f"🔥 <b>PHASE 2 — VOLUME SPIKE! MOVE IMMINENT</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b>\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"Volume: <b>{pa.vol_spike_ratio:.1f}x</b> above average 🔥\n"
                    f"OI: {oi_ch:+.1f}% (already building since Phase 1)\n\n"
                    f"Signal: <b>{direction}</b>\n"
                    f"⚡ Finger on trigger! Wait for Price confirmation!\n"
                    f"🎯 {'BUY CALL' if direction == 'BULLISH' else 'BUY PUT'} near ${curr.atm_strike:,.0f}\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
                )
            ))

        # ── PHASE 3: Price confirms ──
        phase2_recent = (now - self._phase2_fired_at) < (15 * 60)
        price_confirms = (
            (direction == "BULLISH" and pa.price_change_5m >= PHASE3_PRICE_MOVE_PCT) or
            (direction == "BEARISH" and pa.price_change_5m <= -PHASE3_PRICE_MOVE_PCT)
        )

        if (phase2_recent
                and price_confirms
                and pa.triple_confirmed
                and self._can("PHASE3", self.COOLDOWN_PHASE3)):

            self._mark("PHASE3")
            rec = "BUY_CALL" if direction == "BULLISH" else "BUY_PUT"

            signals.append(PhaseSignal(
                phase=3, dominant_side=dominant_side, direction=direction,
                oi_change_pct=oi_ch, vol_change_pct=vol_ch,
                price_change_pct=pa.price_change_5m,
                atm_strike=curr.atm_strike, spot_price=curr.spot_price,
                confidence=min(10, 7 + pa.trend_strength / 3),
                message=(
                    f"✅ <b>PHASE 3 — PRICE CONFIRMED! EXECUTE NOW!</b>\n\n"
                    f"ETH: <b>${curr.spot_price:,.2f}</b> ({pa.price_change_5m:+.2f}% / 5min)\n"
                    f"ATM: ${curr.atm_strike:,.0f}\n\n"
                    f"🎯 Signal: <b>{rec}</b>\n"
                    f"💯 Triple Confirmed: OI ✅ + Volume ✅ + Price ✅\n\n"
                    f"OI Change:    {oi_ch:+.1f}%\n"
                    f"Vol Spike:    {pa.vol_spike_ratio:.1f}x avg\n"
                    f"Price Move:   {pa.price_change_5m:+.2f}%\n"
                    f"Trend Strength: {pa.trend_strength:.1f}/10\n\n"
                    f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}\n"
                    f"🤖 Sending to AI for final confirmation..."
                )
            ))

        return signals


# ============================================================
#  MTF ANALYZER (unchanged logic, minor updates)
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
        if pe_ch >= MIN_OI_CHANGE and pe_vol >= MIN_VOLUME_CHG:   return "BULLISH"
        if ce_ch >= MIN_OI_CHANGE and ce_vol >= MIN_VOLUME_CHG:   return "BEARISH"
        if pe_ch <= -MIN_OI_CHANGE:                                return "BEARISH"
        if ce_ch <= -MIN_OI_CHANGE:                                return "BULLISH"
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
        # Use 5-min cache: 3 snaps=15min, 6 snaps=30min, 12 snaps=60min
        s_15m = await self.cache.get_5min_ago(3)
        s_30m = await self.cache.get_5min_ago(6)
        s_60m = await self.cache.get_5min_ago(12)

        if not s_15m:
            return {"available": False, "reason": "⏳ Building cache (need ≥ 15 min)..."}

        analyses: List[StrikeAnalysis] = []

        for strike in sorted(current.strikes_oi.keys()):
            curr = current.strikes_oi[strike]
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

            is_atm = (strike == current.atm_strike)
            dist   = abs(strike - current.atm_strike)
            step   = max(20, min(100, int(dist / max(1, ATM_STRIKE_FETCH - 1))))
            if is_atm:             weight = ATM_WEIGHT
            elif dist <= step * 2: weight = NEAR_ATM_WEIGHT
            else:                  weight = FAR_WEIGHT

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

            if bull >= 7 and bull > bear:
                rec, conf = "STRONG_CALL", bull
            elif bear >= 7 and bear > bull:
                rec, conf = "STRONG_PUT",  bear
            else:
                rec, conf = "WAIT", max(bull, bear)

            analyses.append(StrikeAnalysis(
                strike=strike, is_atm=is_atm, distance_atm=dist, weight=weight,
                ce_oi=curr.ce_oi, pe_oi=curr.pe_oi,
                ce_volume=curr.ce_volume, pe_volume=curr.pe_volume,
                ce_ltp=curr.ce_ltp, pe_ltp=curr.pe_ltp,
                ce_oi_15=ce15, pe_oi_15=pe15, ce_vol_15=cv15, pe_vol_15=pv15, pcr_ch_15=pcr15,
                ce_oi_30=ce30, pe_oi_30=pe30, ce_vol_30=cv30, pe_vol_30=pv30, pcr_ch_30=pcr30,
                ce_oi_60=ce60, pe_oi_60=pe60,
                pcr=curr.pcr, ce_action=self._action(ce15), pe_action=self._action(pe15),
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
            support_strike=sup,    support_put_oi=max_pe.pe_oi if max_pe else 0,
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
        prev = await self.cache.get_5min_ago(6)  # 30-min ago
        if not prev:
            return
        await self._oi_change(curr, prev)
        await self._vol_spike(curr, prev)
        await self._pcr_change(curr, prev)
        await self._atm_proximity(curr)

    async def _oi_change(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("OI"):
            return
        atm_c = curr.strikes_oi.get(curr.atm_strike)
        atm_p = prev.strikes_oi.get(curr.atm_strike)
        if not atm_c or not atm_p or atm_p.ce_oi == 0 or atm_p.pe_oi == 0:
            return
        ce_ch = (atm_c.ce_oi - atm_p.ce_oi) / atm_p.ce_oi * 100
        pe_ch = (atm_c.pe_oi - atm_p.pe_oi) / atm_p.pe_oi * 100
        if abs(ce_ch) < OI_ALERT_PCT and abs(pe_ch) < OI_ALERT_PCT:
            return
        txt = (
            f"⚠️ <b>OI CHANGE ALERT (30-min)</b>\n\n"
            f"ETH: ${curr.spot_price:,.2f}\n"
            f"ATM: ${curr.atm_strike:,.0f}\n\n"
            f"CALL OI: {ce_ch:+.1f}%  {'🔴 BUILDING' if ce_ch > 0 else '🟢 UNWINDING'}\n"
            f"PUT  OI: {pe_ch:+.1f}%  {'🟢 BUILDING' if pe_ch > 0 else '🔴 UNWINDING'}\n\n"
            f"Overall PCR: {curr.overall_pcr:.2f}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("OI")

    async def _vol_spike(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("VOL"):
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
            f"Trend: {trend}\nBias: {interp}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("PCR")

    async def _atm_proximity(self, curr: MarketSnapshot):
        if not self._can("PROX"):
            return
        max_pe = max(curr.strikes_oi.items(), key=lambda x: x[1].pe_oi, default=None)
        max_ce = max(curr.strikes_oi.items(), key=lambda x: x[1].ce_oi, default=None)
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
                f"{level}: ${strike:,.0f}  (OI: {oi_val:,.0f})\n"
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
                    patterns.append({"time": cur.name, "pattern": "HAMMER",        "type": "BULLISH", "strength": 7})
                elif hi_wick > body_c * 2 and lo_wick < body_c * 0.3 and body_c < rng * 0.35:
                    patterns.append({"time": cur.name, "pattern": "SHOOTING_STAR", "type": "BEARISH", "strength": 7})
                elif body_c < rng * 0.1:
                    patterns.append({"time": cur.name, "pattern": "DOJI",          "type": "NEUTRAL", "strength": 4})
        return patterns[-5:]

    @staticmethod
    def support_resistance(df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty or len(df) < 5:
            return 0.0, 0.0
        last = df.tail(20)
        return float(last.low.min()), float(last.high.max())


# ============================================================
#  PROMPT BUILDER v7.0 (IMPROVED WITH PRICE ACTION + S/R + CANDLES)
# ============================================================

class PromptBuilder:

    @staticmethod
    def _filter_strikes(analyses: List[StrikeAnalysis], atm: float, sr) -> List[StrikeAnalysis]:
        filtered = []
        for sa in analyses:
            if sa.distance_atm <= (5 * STRIKE_INTERVAL):
                filtered.append(sa)
            elif sa.is_support or sa.is_resistance:
                filtered.append(sa)
            elif (sa.ce_oi >= MIN_OI_THRESHOLD or sa.pe_oi >= MIN_OI_THRESHOLD or
                  sa.ce_volume >= MIN_VOL_THRESHOLD or sa.pe_volume >= MIN_VOL_THRESHOLD):
                filtered.append(sa)
        return filtered

    @staticmethod
    def _candle_table(df: pd.DataFrame, label: str) -> str:
        """Full price candle table — easier for AI to read"""
        if df.empty:
            return f"{label}: no data\n"
        out  = f"\n{label} CANDLES (TIME|OPEN|HIGH|LOW|CLOSE|VOL|DIR):\n"
        for ts, row in df.tail(CANDLE_COUNT).iterrows():
            t = ts.strftime("%H:%M")
            d = "↑" if row.close > row.open else "↓" if row.close < row.open else "→"
            out += f"{t}|{row.open:.0f}|{row.high:.0f}|{row.low:.0f}|{row.close:.0f}|{row.volume:.0f}|{d}\n"
        return out

    @staticmethod
    def build(spot: float, atm: float, expiry: str,
              oi: Dict,
              c15: pd.DataFrame, c30: pd.DataFrame,
              patterns: List[Dict],
              p_sup: float, p_res: float,
              pa: PriceActionInsight,
              phase_signal: Optional[PhaseSignal] = None) -> str:

        now  = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sr   = oi["sr"]
        pcr  = oi["overall_pcr"]

        all_analyses      = oi["strike_analyses"]
        filtered_analyses = PromptBuilder._filter_strikes(all_analyses, atm, sr)

        # ── SYSTEM ROLE ──
        p  = "You are an expert ETH options trader on Delta Exchange India.\n"
        p += "Analyze OI, Volume, Price Action, and Candlesticks to give a precise trade signal.\n\n"

        # ── MARKET SNAPSHOT ──
        p += f"=== MARKET SNAPSHOT | {now} | Expiry: {expiry} ===\n"
        p += f"ETH Spot: ${spot:,.2f}\n"
        p += f"ATM Strike: ${atm:,.0f}\n"
        p += f"Overall PCR: {pcr:.2f} ({oi['pcr_trend']})  Δ30m: {oi['pcr_ch_pct']:+.1f}%\n"
        p += f"OI Support: ${sr.support_strike:,.0f} | OI Resistance: ${sr.resistance_strike:,.0f}\n"
        if sr.near_support:    p += "⚡ PRICE NEAR OI-SUPPORT!\n"
        if sr.near_resistance: p += "⚡ PRICE NEAR OI-RESISTANCE!\n"

        # ── PRICE ACTION (PRE-CALCULATED BY PANDAS/NUMPY) ──
        p += f"\n=== PRICE ACTION (Pandas/Numpy Pre-Calculated) ===\n"
        p += f"Price Change:  5min={pa.price_change_5m:+.2f}%  15min={pa.price_change_15m:+.2f}%  30min={pa.price_change_30m:+.2f}%\n"
        p += f"Momentum:      {pa.price_momentum}\n"
        p += f"Volume Spike:  {pa.vol_spike_ratio:.2f}x above rolling avg\n"
        p += f"OI-Vol Corr:   {pa.oi_vol_corr:.2f}  (>0.5 = OI and volume moving together)\n"
        p += f"Price-OI Corr: {pa.price_oi_corr:.2f}  (>0.5 = price confirmed by OI)\n"
        p += f"Trend Strength:{pa.trend_strength:.1f}/10\n"
        p += f"Triple Confirm:{' ✅ YES — OI + Volume + Price all agree!' if pa.triple_confirmed else ' ❌ NOT YET'}\n"
        if pa.support_levels:
            p += f"Price Support:  {', '.join(f'${s:.0f}' for s in pa.support_levels)}\n"
        if pa.resistance_levels:
            p += f"Price Resistance: {', '.join(f'${r:.0f}' for r in pa.resistance_levels)}\n"

        # ── PHASE SIGNAL (Early Warning Context) ──
        if phase_signal:
            p += f"\n=== EARLY WARNING PHASE {phase_signal.phase} TRIGGERED ===\n"
            p += f"Side: {phase_signal.dominant_side} OI building\n"
            p += f"Direction: {phase_signal.direction}\n"
            p += f"OI Change: {phase_signal.oi_change_pct:+.1f}%\n"
            p += f"Vol Change: {phase_signal.vol_change_pct:+.1f}%\n"
            p += f"Price Change: {phase_signal.price_change_pct:+.2f}%\n"

        # ── OI MULTI-TIMEFRAME TABLE ──
        p += f"\n=== OI MULTI-TIMEFRAME ({len(filtered_analyses)} key strikes) ===\n"
        p += "Format: STRIKE | CE_OI_15% CE_OI_30% | PE_OI_15% PE_OI_30% | TF15 TF30 | MTF | Conf\n"
        for sa in filtered_analyses:
            tag = "⭐ATM" if sa.is_atm else ("🟢SUP" if sa.is_support else ("🔴RES" if sa.is_resistance else "    "))
            mtf = "✅MTF" if sa.mtf_confirmed else "❌   "
            p += (f"${sa.strike:,.0f} {tag} W{sa.weight:.0f}x | "
                  f"CE: {sa.ce_oi_15:+.0f}%/15m {sa.ce_oi_30:+.0f}%/30m ({sa.ce_action}) | "
                  f"PE: {sa.pe_oi_15:+.0f}%/15m {sa.pe_oi_30:+.0f}%/30m ({sa.pe_action}) | "
                  f"TF15:{sa.tf15_signal[:3]} TF30:{sa.tf30_signal[:3]} {mtf} | "
                  f"Bull:{sa.bull_strength:.0f} Bear:{sa.bear_strength:.0f} Conf:{sa.confidence:.0f}\n")

        # ── CANDLESTICK DATA (FULL PRICE — easier for AI) ──
        p += PromptBuilder._candle_table(c15, "15-MIN")
        p += PromptBuilder._candle_table(c30, "30-MIN")

        # ── CANDLESTICK PATTERNS ──
        if patterns:
            p += "\n=== CANDLESTICK PATTERNS DETECTED ===\n"
            for pat in patterns:
                p += f"{pat['time'].strftime('%H:%M')} | {pat['pattern']} | {pat['type']} | Strength:{pat['strength']}/10\n"

        # ── PRICE S/R FROM CANDLES ──
        if p_sup or p_res:
            p += f"\nCandle S/R: Support=${p_sup:.2f}  Resistance=${p_res:.2f}\n"

        # ── TRADING RULES (IMPROVED PROMPT v7.0) ──
        p += f"""
=== TRADING RULES ===
CANDLESTICK RULES:
• Bullish Engulfing near OI-Support + PUT OI building = STRONG BUY_CALL
• Bearish Engulfing near OI-Resistance + CALL OI building = STRONG BUY_PUT
• Hammer at support = BUY_CALL signal
• Shooting Star at resistance = BUY_PUT signal
• Doji alone = WAIT (no clear direction)
• Multiple candles same direction = Trend continuation

OI + VOLUME RULES:
• CALL OI↑ + Vol↑ = Resistance building = BEARISH → BUY_PUT
• PUT  OI↑ + Vol↑ = Support building   = BULLISH → BUY_CALL
• OI↑ but Vol flat = TRAP — ignore signal
• MTF ✅ = both 15min + 30min agree = HIGH CONFIDENCE

PRICE ACTION RULES:
• Price + OI + Volume all agree = TRIPLE CONFIRMED = highest quality signal
• Price moving INTO support/resistance = reversal likely
• Price momentum must confirm direction — opposite = WAIT
• Price-OI Corr > 0.5 = strong confirmation

PCR RULES:
• PCR > 1.3 = bullish bias (more PUT writers)
• PCR < 0.7 = bearish bias (more CALL writers)
• PCR moving in signal direction = confirmation

ENTRY / EXIT RULES:
• Enter ONLY when: MTF confirmed + Volume confirms + Price confirms
• Stop Loss: Previous candle high/low (outside the setup candle)
• Target: Minimum 1:1.5 Risk:Reward | Next S/R level
• Do NOT enter if: Volume mismatch (TRAP) OR price going against signal
• Minimum confidence 7/10 to give BUY signal

RESPOND ONLY VALID JSON (no extra text):
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
  "pcr": {{"value": {pcr:.2f}, "trend": "{oi['pcr_trend']}", "note": "", "supports": true|false}},
  "volume": {{"ok": true|false, "spike_ratio": {pa.vol_spike_ratio:.2f}, "trap_warning": ""}},
  "entry": {{"now": true|false, "reason": "", "wait_for": ""}},
  "rr": {{"sl_pts": 0, "tgt_pts": 0, "ratio": 0}},
  "levels": {{"support": {sr.support_strike}, "resistance": {sr.resistance_strike}, "candle_sup": {p_sup:.0f}, "candle_res": {p_res:.0f}}}
}}"""
        return p


# ============================================================
#  DEEPSEEK CLIENT v7.0
# ============================================================

class DeepSeekClient:

    URL   = "https://api.deepseek.com/v1/chat/completions"
    # ✅ FIX: deepseek-chat = DeepSeek V3 (was wrongly "deepseek-reasoner" = R1)
    MODEL = "deepseek-chat"

    def __init__(self, key: str):
        self.key = key

    async def analyze(self, prompt: str) -> Optional[Dict]:
        hdrs    = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {
            "model":       self.MODEL,
            "messages":    [{"role": "user", "content": prompt}],
            "temperature": 0.2,      # Lower temp = more consistent JSON
            "max_tokens":  1500
        }
        try:
            timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(self.URL, headers=hdrs, json=payload) as r:
                    if r.status != 200:
                        logger.error(f"❌ DeepSeek {r.status}: {await r.text()}")
                        return None
                    data    = await r.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    for fence in ("```json", "```"):
                        content = content.replace(fence, "")
                    return json.loads(content.strip())
        except asyncio.TimeoutError:
            logger.error(f"❌ DeepSeek timeout (>{DEEPSEEK_TIMEOUT}s)")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"❌ DeepSeek JSON parse error: {e}")
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
                    logger.info("✅ Telegram alert sent")
                else:
                    logger.error(f"❌ Telegram {r.status}: {await r.text()}")
        except Exception as e:
            logger.error(f"❌ Telegram error: {e}")

    async def send_signal(self, sig: Dict, snap: MarketSnapshot, oi: Dict, pa: PriceActionInsight):
        """Full AI trade signal alert"""
        mtf  = sig.get("mtf",          {})
        atma = sig.get("atm",          {})
        pcra = sig.get("pcr",          {})
        vol  = sig.get("volume",       {})
        ent  = sig.get("entry",        {})
        rr   = sig.get("rr",           {})
        prce = sig.get("price_action", {})
        cndl = sig.get("candle_pattern",{})

        signal_type = sig.get("signal", "WAIT")
        option_type = "CE" if "CALL" in signal_type else "PE" if "PUT" in signal_type else ""

        msg = (
            f"🚨 <b>ETH OPTIONS SIGNAL v7.0 PRO 🇮🇳</b>\n"
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
            f"🔗 <b>MTF CONFIRMATION:</b>\n"
            f"TF-15min: {mtf.get('tf15', 'N/A')}\n"
            f"TF-30min: {mtf.get('tf30', 'N/A')}\n"
            f"Confirmed: {'✅ YES – HIGH CONFIDENCE' if mtf.get('confirmed') else '❌ Single TF only'}\n\n"
            f"📈 <b>PRICE ACTION:</b>\n"
            f"5min: {pa.price_change_5m:+.2f}%  15min: {pa.price_change_15m:+.2f}%\n"
            f"Momentum: {pa.price_momentum}\n"
            f"Triple Confirmed: {'✅ OI + Vol + Price' if pa.triple_confirmed else '❌'}\n"
            f"Price confirms signal: {'✅' if prce.get('confirms_signal') else '❌'}\n\n"
            f"🕯️ <b>CANDLESTICK:</b>\n"
            f"Pattern: {cndl.get('latest_pattern', 'None')}\n"
            f"Near S/R: {'✅' if cndl.get('near_sr') else '❌'}\n"
            f"Confirms: {'✅' if cndl.get('confirms_signal') else '❌'}\n\n"
            f"📊 <b>OI ANALYSIS:</b>\n"
            f"CE Writers: {atma.get('ce_action', 'N/A')}\n"
            f"PE Writers: {atma.get('pe_action', 'N/A')}\n"
            f"Volume: {'✅ Confirms' if atma.get('vol_confirms') else '❌ MISMATCH – TRAP?'}\n\n"
            f"📈 PCR: {pcra.get('value', 'N/A')} ({pcra.get('trend', 'N/A')}) "
            f"{'✅' if pcra.get('supports') else '❌'}\n\n"
            f"⏰ Enter Now: {'✅ YES' if ent.get('now') else '⏳ WAIT'}\n"
            f"{ent.get('reason', '')}\n\n"
            f"🤖 DeepSeek V3 + MTF v7.0 PRO\n"
            f"🇮🇳 Delta Exchange India"
        )
        if vol.get("trap_warning"):
            msg += f"\n⚠️ {vol['trap_warning']}"

        await self.send_raw(msg)


# ============================================================
#  MAIN BOT v7.0
# ============================================================

class ETHOptionsBot:
    """
    v7.0 Cycle logic:
      Every 5 min  → fetch snapshot → Phase detection → Standalone alerts
      Phase 1/2/3  → Smart AI trigger (only when move is building)
      Every 30 min → Full MTF analysis (regardless of phase)
    """

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
        logger.info("\n" + "="*60)
        logger.info("🚀 ETH OPTIONS BOT v7.0 PRO – DELTA EXCHANGE INDIA 🇮🇳")
        logger.info("="*60)
        logger.info(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"📦 Cache: 5-min×{CACHE_5MIN_SIZE} + 30-min×{CACHE_30MIN_SIZE}")
        logger.info(f"📊 Polling: every {SNAPSHOT_INTERVAL//60}min | Analysis: every {ANALYSIS_INTERVAL//60}min")
        logger.info(f"🚨 Phase Detection: P1(OI buildup) → P2(Vol spike) → P3(Price confirm)")
        logger.info(f"🤖 DeepSeek V3 (deepseek-chat) | Min confidence: {MIN_CONFIDENCE}/10")
        logger.info("="*60 + "\n")

        await self.delta.init()

        try:
            while True:
                try:
                    await self._cycle_run()
                except aiohttp.ClientConnectorError:
                    logger.error("❌ Network error — retrying in 60s")
                    await asyncio.sleep(60)
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(60)
                s5, s30 = self.cache.sizes()
                logger.info(f"⏰ Next in {SNAPSHOT_INTERVAL//60}min | Cache: 5m={s5} 30m={s30}\n")
                await asyncio.sleep(SNAPSHOT_INTERVAL)
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped")
        finally:
            await self.delta.close()
            await self.alerter.close()

    async def _cycle_run(self):
        self._cycle += 1
        is_analysis = (self._cycle % 6 == 0)  # Every 6th = 30 min

        logger.info(f"\n{'='*60}")
        logger.info(f"{'🚀 FULL ANALYSIS' if is_analysis else '📦 SNAPSHOT'} CYCLE #{self._cycle}")
        logger.info(f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        logger.info("="*60)

        # 1. Fetch snapshot
        snap = await self.delta.fetch_snapshot()
        if not snap:
            logger.warning("⚠️ Snapshot failed — skipping")
            return

        # 2. Add to 5-min cache
        await self.cache.add_5min(snap)

        if is_analysis:
            await self.cache.add_30min(snap)

        # 3. Standalone alerts (every cycle)
        await self.checker.check_all(snap)

        # 4. Get recent snapshots for Pandas/Numpy calculation
        recent_snaps = await self.cache.get_recent_snapshots(12)

        # 5. Fetch 15-min candles for Price Action calc
        c15 = await self.delta.get_candles("ETHUSD", "15", CANDLE_COUNT)

        # 6. Pre-calculate all insights with Pandas + Numpy
        pa = PriceActionCalculator.calculate(recent_snaps, c15)
        logger.info(f"📊 Price: 5m={pa.price_change_5m:+.2f}% | Vol spike: {pa.vol_spike_ratio:.2f}x | Triple: {pa.triple_confirmed}")

        # 7. Phase detection (Early Warning)
        prev_5m       = await self.cache.get_5min_ago(1)
        phase_signals = await self.phase.detect(snap, prev_5m, pa)

        for ps in phase_signals:
            logger.info(f"🚨 Phase {ps.phase} detected: {ps.direction}")
            await self.alerter.send_raw(ps.message)

            # Phase 3 = Price confirmed → trigger AI immediately
            if ps.phase == 3:
                await self._trigger_ai_analysis(snap, pa, ps)

        # 8. Full 30-min MTF analysis
        if is_analysis:
            await self._full_analysis(snap, pa)

    async def _trigger_ai_analysis(self, snap: MarketSnapshot,
                                   pa: PriceActionInsight,
                                   phase_signal: PhaseSignal):
        """Called when Phase 3 fires — immediate AI call"""
        logger.info("🤖 Phase 3 triggered — calling DeepSeek immediately...")

        c15 = await self.delta.get_candles("ETHUSD", "15", CANDLE_COUNT)
        await asyncio.sleep(API_DELAY)
        c30 = await self.delta.get_candles("ETHUSD", "30", CANDLE_COUNT)

        patterns     = PatternDetector.detect(c15)    if not c15.empty else []
        p_sup, p_res = PatternDetector.support_resistance(c15) if not c15.empty else (0.0, 0.0)

        # Quick MTF for OI data
        oi = await self.mtf.analyze(snap)
        if not oi["available"]:
            # Build minimal OI dict for prompt
            oi = {
                "available": True, "strike_analyses": [],
                "sr": SupportResistance(snap.atm_strike, 0, snap.atm_strike, 0, False, False),
                "overall": phase_signal.direction,
                "total_bull": 0, "total_bear": 0,
                "overall_pcr": snap.overall_pcr,
                "pcr_trend": "N/A", "pcr_ch_pct": 0,
                "has_15m": False, "has_30m": False, "has_strong": True
            }

        prompt = PromptBuilder.build(
            spot=snap.spot_price, atm=snap.atm_strike, expiry=snap.expiry,
            oi=oi, c15=c15, c30=c30, patterns=patterns,
            p_sup=p_sup, p_res=p_res, pa=pa, phase_signal=phase_signal
        )

        ai_sig = await self.ai.analyze(prompt)
        if not ai_sig:
            logger.warning("⚠️ DeepSeek timeout on Phase 3 trigger")
            return

        conf   = ai_sig.get("confidence", 0)
        signal = ai_sig.get("signal", "WAIT")
        logger.info(f"🎯 Phase 3 AI: {signal} | Conf: {conf}/10")

        if conf >= MIN_CONFIDENCE and signal != "WAIT":
            await self.alerter.send_signal(ai_sig, snap, oi, pa)

    async def _full_analysis(self, snap: MarketSnapshot, pa: PriceActionInsight):
        """Full 30-min MTF analysis"""
        logger.info("\n🔍 Running full MTF analysis...")
        oi = await self.mtf.analyze(snap)

        if not oi["available"]:
            logger.info(f"⏳ {oi['reason']}")
            return

        if not oi["has_strong"]:
            logger.info("📊 No strong MTF signal — skipping AI")
            return

        logger.info("📈 Fetching candles for full analysis...")
        c15 = await self.delta.get_candles("ETHUSD", "15", CANDLE_COUNT)
        await asyncio.sleep(API_DELAY)
        c30 = await self.delta.get_candles("ETHUSD", "30", CANDLE_COUNT)

        patterns     = PatternDetector.detect(c15)    if not c15.empty else []
        p_sup, p_res = PatternDetector.support_resistance(c15) if not c15.empty else (0.0, 0.0)

        prompt = PromptBuilder.build(
            spot=snap.spot_price, atm=snap.atm_strike, expiry=snap.expiry,
            oi=oi, c15=c15, c30=c30, patterns=patterns,
            p_sup=p_sup, p_res=p_res, pa=pa, phase_signal=None
        )

        logger.info(f"🤖 Sending to DeepSeek V3...")
        ai_sig = await self.ai.analyze(prompt)

        if not ai_sig:
            logger.warning("⚠️ DeepSeek timeout — fallback")
            atm_sa = next((sa for sa in oi["strike_analyses"] if sa.is_atm), None)
            if atm_sa and atm_sa.mtf_confirmed:
                fb = "BUY_CALL" if atm_sa.bull_strength > atm_sa.bear_strength else "BUY_PUT"
                fc = min(10, max(atm_sa.bull_strength, atm_sa.bear_strength))
            else:
                fb, fc = "WAIT", 3
            ai_sig = {
                "signal": fb, "confidence": fc,
                "primary_strike": snap.atm_strike,
                "mtf":   {"tf15": "N/A", "tf30": "N/A", "confirmed": False},
                "entry": {"now": False, "reason": "AI timeout – fallback"},
                "price_action": {"momentum": pa.price_momentum, "triple_confirmed": pa.triple_confirmed, "confirms_signal": False},
                "candle_pattern": {"latest_pattern": "N/A", "confirms_signal": False, "near_sr": False},
                "volume": {"ok": False, "spike_ratio": pa.vol_spike_ratio, "trap_warning": ""},
                "rr": {}, "atm": {}, "pcr": {}, "levels": {}
            }

        conf   = ai_sig.get("confidence", 0)
        signal = ai_sig.get("signal", "WAIT")
        logger.info(f"🎯 Signal: {signal} | Conf: {conf}/10")

        if conf >= MIN_CONFIDENCE:
            await self.alerter.send_signal(ai_sig, snap, oi, pa)
        else:
            logger.info(f"⏳ Conf {conf}/10 < {MIN_CONFIDENCE} — no alert")


# ============================================================
#  HTTP (Koyeb keep-alive) ✅ FIX
# ============================================================

bot_instance: Optional[ETHOptionsBot] = None

async def health(request):
    s5, s30 = bot_instance.cache.sizes() if bot_instance else (0, 0)
    return aiohttp.web.Response(
        text=f"✅ ETH Bot v7.0 ALIVE | {datetime.now(timezone.utc).strftime('%H:%M UTC')} | Cache: 5m={s5}/{CACHE_5MIN_SIZE} 30m={s30}/{CACHE_30MIN_SIZE}",
        # ✅ FIX: Cache-Control prevents Koyeb from caching the response
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
║   🚀 ETH OPTIONS BOT v7.0 PRO – DELTA EXCHANGE INDIA 🇮🇳║
╚══════════════════════════════════════════════════════════╝

✅ v7.0 CHANGES:
  • 5-min polling (was 15-min)
  • Phase 1: OI building quietly (Smart Money alert)
  • Phase 2: Volume spike (Move imminent alert)
  • Phase 3: Price confirms → AI triggered immediately
  • Pandas/Numpy pre-calculation before AI call
  • Smart AI trigger: DeepSeek only called when needed
  • FIXED: DeepSeek model = deepseek-chat (V3, was R1)
  • FIXED: Strike interval = statistics.mode (correct)
  • FIXED: Koyeb Cache-Control header
  • IMPROVED: DeepSeek prompt with candlestick + S/R + price action

🚨 PHASE DETECTION:
  Phase 1: OI ≥ {PHASE1_OI_BUILD_PCT}% + Vol < {PHASE1_VOL_MAX_PCT}% → Smart Money positioning
  Phase 2: Vol spike {PHASE2_VOL_SPIKE_PCT}%+ + OI building → Move imminent
  Phase 3: Price {PHASE3_PRICE_MOVE_PCT}%+ + Triple confirmed → EXECUTE

🔑 ENV VARS: DELTA_API_KEY, DELTA_API_SECRET,
              TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
              DEEPSEEK_API_KEY

Starting on port {port}...
""")

    web.run_app(app, host="0.0.0.0", port=port)
