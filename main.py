"""
🚀 ETH OPTIONS BOT - DELTA EXCHANGE GLOBAL v6.0
=================================================
Platform  : Delta Exchange Global (api.delta.exchange)
Asset     : ETH Daily Options
Updated   : Feb 2026

✅ v6.0 FEATURES:
- Delta Exchange Global API (api_key + secret auth)
- ETH Daily Expiry auto-detection
- Snapshot every 15 min (stores to 15-min cache)
- Full Analysis every 30 min (stores to 30-min cache + DeepSeek)
- DUAL CACHE: 15-min × 24 snaps + 30-min × 12 snaps = 6hr each
- MULTI-TIMEFRAME: 15-min + 30-min OI/Volume/PCR confirmation
- MTF Confirmed signal = Both TFs agree → HIGH CONFIDENCE
- 24 Candles per TF (ultra-short format: TIME|O|H|L|C|↑)
- 4 Standalone Alerts: OI Change / Volume Spike / PCR Change / ATM Proximity
- TRAP detection: OI up but Volume flat = ignore
- DeepSeek V3 (45s timeout) + fallback logic
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import logging
import os
import hmac
import hashlib
import time as time_module

# ============================================================
#  CONFIGURATION
# ============================================================
DELTA_API_KEY    = os.getenv("DELTA_API_KEY",    "YOUR_API_KEY")
DELTA_API_SECRET = os.getenv("DELTA_API_SECRET", "YOUR_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "YOUR_CHAT_ID")
DEEPSEEK_API_KEY   = os.getenv("DEEPSEEK_API_KEY",   "YOUR_DEEPSEEK_KEY")

# API
DELTA_BASE_URL = "https://api.delta.exchange"

# Asset
UNDERLYING       = "ETH"
ATM_STRIKE_COUNT = 5     # ±5 strikes from ATM (11 total)
STRIKE_INTERVAL  = 50    # Default $50; auto-detected at runtime

# Timing
SNAPSHOT_INTERVAL = 15 * 60   # 15 min  → always collect snapshot
ANALYSIS_INTERVAL = 30 * 60   # 30 min  → also run full analysis (every 2nd snapshot)

# Cache sizes  (6 hours each)
CACHE_15MIN_SIZE = 24   # 24 × 15-min = 6 hr
CACHE_30MIN_SIZE = 12   # 12 × 30-min = 6 hr

# Candles
CANDLE_COUNT = 24   # Last 24 candles per timeframe

# Signal thresholds
MIN_OI_CHANGE    = 10.0   # % for normal signal
STRONG_OI_CHANGE = 20.0   # % for strong signal
MIN_VOLUME_CHG   = 15.0   # % volume change for confirmation
PCR_BULL         = 1.3    # PCR > 1.3 → bullish bias
PCR_BEAR         = 0.7    # PCR < 0.7 → bearish bias
MIN_CONFIDENCE   = 7      # Minimum confidence to send trade alert

# Standalone alert thresholds
OI_ALERT_PCT     = 15.0   # % OI change (30-min) → standalone alert
VOL_SPIKE_PCT    = 25.0   # % volume spike → standalone alert
PCR_ALERT_PCT    = 12.0   # % PCR change  → standalone alert
ATM_PROX_USD     = 50     # $50 near high-OI strike → proximity alert

# Strike weights
ATM_WEIGHT      = 3.0
NEAR_ATM_WEIGHT = 2.0
FAR_WEIGHT      = 1.0

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
    # 15-min TF changes
    ce_oi_15:         float
    pe_oi_15:         float
    ce_vol_15:        float
    pe_vol_15:        float
    pcr_ch_15:        float
    # 30-min TF changes
    ce_oi_30:         float
    pe_oi_30:         float
    ce_vol_30:        float
    pe_vol_30:        float
    pcr_ch_30:        float
    # 60-min changes (trend context)
    ce_oi_60:         float
    pe_oi_60:         float
    # Analysis
    pcr:              float
    ce_action:        str   # BUILDING / UNWINDING / NEUTRAL
    pe_action:        str
    tf15_signal:      str   # BULLISH / BEARISH / NEUTRAL
    tf30_signal:      str
    mtf_confirmed:    bool  # Both TFs agree
    vol_confirms:     bool
    vol_strength:     str   # STRONG / MODERATE / WEAK
    is_support:       bool
    is_resistance:    bool
    bull_strength:    float
    bear_strength:    float
    recommendation:   str   # STRONG_CALL / STRONG_PUT / WAIT
    confidence:       float


@dataclass
class SupportResistance:
    support_strike:    float
    support_put_oi:    float
    resistance_strike: float
    resistance_call_oi:float
    near_support:      bool
    near_resistance:   bool


# ============================================================
#  DUAL CACHE
# ============================================================

class DualCache:
    """15-min cache (24 snaps) + 30-min cache (12 snaps) = 6 hr each"""

    def __init__(self):
        self._c15 = deque(maxlen=CACHE_15MIN_SIZE)
        self._c30 = deque(maxlen=CACHE_30MIN_SIZE)
        self._lock = asyncio.Lock()

    async def add_15min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c15.append(snap)
        logger.info(f"📦 15-min cache: {len(self._c15)}/{CACHE_15MIN_SIZE} | PCR:{snap.overall_pcr:.2f}")

    async def add_30min(self, snap: MarketSnapshot):
        async with self._lock:
            self._c30.append(snap)
        logger.info(f"📦 30-min cache: {len(self._c30)}/{CACHE_30MIN_SIZE}")

    async def get_15min_ago(self, n: int) -> Optional[MarketSnapshot]:
        """n × 15-min ago (n=1 → 15 min, n=2 → 30 min, n=4 → 60 min)"""
        async with self._lock:
            idx = len(self._c15) - 1 - n
            return self._c15[idx] if idx >= 0 else None

    async def get_30min_ago(self, n: int) -> Optional[MarketSnapshot]:
        """n × 30-min ago"""
        async with self._lock:
            idx = len(self._c30) - 1 - n
            return self._c30[idx] if idx >= 0 else None

    async def latest(self) -> Optional[MarketSnapshot]:
        async with self._lock:
            return self._c15[-1] if self._c15 else None

    def sizes(self) -> Tuple[int, int]:
        return len(self._c15), len(self._c30)

    def has_data(self) -> bool:
        return len(self._c15) >= 2


# ============================================================
#  DELTA EXCHANGE CLIENT
# ============================================================

class DeltaClient:
    """Delta Exchange Global v2 API client with HMAC authentication"""

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
        """HMAC-SHA256 signature for Delta Exchange"""
        ts  = str(int(time_module.time()))
        qs  = f"?{query}" if query else ""
        msg = method.upper() + ts + path + qs + body
        sig = hmac.new(
            self.api_secret.encode(),
            msg.encode(),
            hashlib.sha256
        ).hexdigest()
        return {
            "api-key":   self.api_key,
            "timestamp": ts,
            "signature": sig,
            "Content-Type": "application/json"
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
            except Exception as e:
                logger.error(f"❌ Request error ({attempt+1}/{MAX_RETRIES}): {e}")
                await asyncio.sleep(1)
        return None

    # ── ETH Spot ──────────────────────────────────────────────
    async def get_eth_spot(self) -> float:
        """ETH spot price via ETHUSD perp ticker"""
        data = await self._get("/v2/tickers/ETHUSD")
        if data and data.get("result"):
            r = data["result"]
            price = r.get("spot_price") or r.get("mark_price") or r.get("close") or 0
            return float(price)
        return 0.0

    # ── Daily ETH Options ─────────────────────────────────────
    async def get_daily_eth_options(self) -> Tuple[List[Dict], str]:
        """
        Returns (list_of_option_products, expiry_date_str)
        Finds the nearest daily expiry for ETH options.
        """
        data = await self._get("/v2/products", params={
            "contract_types": "call_options,put_options",
            "states":         "live"
        })
        if not data or not data.get("result"):
            logger.error("❌ No products from Delta Exchange")
            return [], ""

        products = data["result"]

        # Filter ETH
        eth_opts = [
            p for p in products
            if (p.get("underlying_asset_symbol") == UNDERLYING
                or UNDERLYING in p.get("symbol", "").upper())
               and p.get("contract_type") in ("call_options", "put_options")
        ]

        if not eth_opts:
            logger.error("❌ No ETH options found in products list")
            return [], ""

        now_utc = datetime.now(timezone.utc)

        # Group by expiry date
        by_expiry: Dict[str, List] = {}
        for p in eth_opts:
            raw = p.get("settlement_time") or p.get("expiry_time") or ""
            if not raw:
                continue
            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt <= now_utc:
                    continue          # Already expired
                key = dt.date().isoformat()
                by_expiry.setdefault(key, []).append(p)
            except Exception:
                continue

        if not by_expiry:
            logger.error("❌ No future ETH option expiries found")
            return [], ""

        nearest = sorted(by_expiry.keys())[0]
        logger.info(f"📅 Daily expiry: {nearest} | {len(by_expiry[nearest])} contracts")
        return by_expiry[nearest], nearest

    # ── Option Tickers ────────────────────────────────────────
    async def get_option_tickers(self, symbols: List[str]) -> Dict[str, Dict]:
        """Bulk OI + Volume for option symbols"""
        data = await self._get("/v2/tickers", params={
            "contract_types": "call_options,put_options"
        })
        if not data or not data.get("result"):
            return {}
        sym_set = set(symbols)
        return {t["symbol"]: t for t in data["result"] if t.get("symbol") in sym_set}

    # ── Candles ───────────────────────────────────────────────
    async def get_candles(self, symbol: str, resolution: str, count: int) -> pd.DataFrame:
        """
        OHLCV candles.
        symbol     : "ETHUSD"
        resolution : "15" or "30" (minutes)
        count      : number of candles to return
        """
        end_ts   = int(time_module.time())
        res_sec  = int(resolution) * 60
        start_ts = end_ts - (count * res_sec) - res_sec * 3  # small buffer

        data = await self._get("/v2/history/candles", params={
            "symbol":     symbol,
            "resolution": resolution,
            "start":      start_ts,
            "end":        end_ts
        })

        if not data or not data.get("result"):
            logger.warning(f"⚠️  No candles for {symbol} {resolution}m")
            return pd.DataFrame()

        raw = data["result"].get("candles", data["result"])
        if not raw:
            return pd.DataFrame()

        rows = []
        for c in raw:
            try:
                if isinstance(c, list):
                    ts = datetime.fromtimestamp(int(c[0]), tz=timezone.utc)
                    o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5] if len(c) > 5 else 0)
                else:  # dict
                    ts = datetime.fromtimestamp(int(c.get("time", c.get("timestamp", 0))), tz=timezone.utc)
                    o, h, l, cl, v = float(c["open"]), float(c["high"]), float(c["low"]), float(c["close"]), float(c.get("volume", 0))
                rows.append({"timestamp": ts, "open": o, "high": h, "low": l, "close": cl, "volume": v})
            except Exception:
                continue

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return df.tail(count)

    # ── Full Market Snapshot ──────────────────────────────────
    async def fetch_snapshot(self) -> Optional[MarketSnapshot]:
        """Fetch ETH spot + all ATM±5 strikes OI/Volume"""

        # 1. Spot price
        spot = await self.get_eth_spot()
        if spot <= 0:
            logger.error("❌ Could not get ETH spot price")
            return None
        logger.info(f"💰 ETH: ${spot:,.2f}")

        await asyncio.sleep(API_DELAY)

        # 2. Daily options
        options, expiry = await self.get_daily_eth_options()
        if not options:
            return None

        # 3. Detect strike interval + ATM
        strikes_all = sorted({float(p["strike_price"]) for p in options if p.get("strike_price")})
        if not strikes_all:
            logger.error("❌ No strikes in option products")
            return None

        if len(strikes_all) >= 2:
            diffs = [strikes_all[i+1] - strikes_all[i] for i in range(min(8, len(strikes_all)-1))]
            strike_step = min(set(diffs), key=diffs.count)
        else:
            strike_step = STRIKE_INTERVAL

        atm = min(strikes_all, key=lambda s: abs(s - spot))
        atm_idx = strikes_all.index(atm)
        lo, hi = max(0, atm_idx - ATM_STRIKE_COUNT), min(len(strikes_all), atm_idx + ATM_STRIKE_COUNT + 1)
        selected = set(strikes_all[lo:hi])

        logger.info(f"📊 ATM:${atm:,.0f} | Step:${strike_step:.0f} | Strikes:{len(selected)}")

        # 4. Separate calls/puts
        calls = {float(p["strike_price"]): p for p in options
                 if p["contract_type"] == "call_options" and float(p["strike_price"]) in selected}
        puts  = {float(p["strike_price"]): p for p in options
                 if p["contract_type"] == "put_options"  and float(p["strike_price"]) in selected}

        await asyncio.sleep(API_DELAY)

        # 5. Tickers (OI + Volume)
        all_syms = [p["symbol"] for p in options if float(p.get("strike_price", 0)) in selected]
        tickers  = await self.get_option_tickers(all_syms)

        # 6. Build strikes_oi
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
            logger.error("❌ No valid strike data (need call + put)")
            return None

        overall_pcr = (t_pe_oi / t_ce_oi) if t_ce_oi > 0 else 0.0
        logger.info(f"✅ {len(strikes_oi)} strikes | PCR:{overall_pcr:.2f} | Expiry:{expiry}")

        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            spot_price=spot, atm_strike=atm, expiry=expiry,
            strikes_oi=strikes_oi, overall_pcr=overall_pcr,
            total_ce_oi=t_ce_oi, total_pe_oi=t_pe_oi,
            total_ce_vol=t_ce_vol, total_pe_vol=t_pe_vol
        )


# ============================================================
#  MULTI-TIMEFRAME OI ANALYZER
# ============================================================

class MTFAnalyzer:
    """OI + Volume analysis across 15-min + 30-min timeframes"""

    def __init__(self, cache: DualCache):
        self.cache = cache

    # ── Helpers ───────────────────────────────────────────────
    @staticmethod
    def _pct(curr: float, prev: float) -> float:
        return ((curr - prev) / prev * 100) if prev > 0 else 0.0

    @staticmethod
    def _action(oi_ch: float) -> str:
        if oi_ch >= 10:   return "BUILDING"
        if oi_ch <= -10:  return "UNWINDING"
        return "NEUTRAL"

    @staticmethod
    def _tf_signal(ce_ch: float, pe_ch: float, ce_vol: float, pe_vol: float) -> str:
        """Signal for ONE timeframe"""
        # PUT OI up + volume up → writers building support → BULLISH
        if pe_ch >= MIN_OI_CHANGE and pe_vol >= MIN_VOLUME_CHG:
            return "BULLISH"
        # CALL OI up + volume up → writers building resistance → BEARISH
        if ce_ch >= MIN_OI_CHANGE and ce_vol >= MIN_VOLUME_CHG:
            return "BEARISH"
        # PUT OI unwinding → BEARISH
        if pe_ch <= -MIN_OI_CHANGE:
            return "BEARISH"
        # CALL OI unwinding → BULLISH
        if ce_ch <= -MIN_OI_CHANGE:
            return "BULLISH"
        return "NEUTRAL"

    @staticmethod
    def _vol_confirm(oi_ch: float, vol_ch: float) -> Tuple[bool, str]:
        if oi_ch > 10 and vol_ch > MIN_VOLUME_CHG:  return True,  "STRONG"
        if oi_ch > 5  and vol_ch > 10:              return True,  "MODERATE"
        if abs(oi_ch) < 5 and abs(vol_ch) < 5:      return True,  "WEAK"
        return False, "WEAK"   # Mismatch = possible TRAP

    def _signal_strength(self, ce30: float, pe30: float,
                         ce_vol30: float, pe_vol30: float,
                         weight: float, mtf: bool) -> Tuple[float, float]:
        bull = bear = 0.0
        boost = 1.5 if mtf else 0.8

        # PUT OI building → BULLISH
        if   pe30 >= STRONG_OI_CHANGE: bull = 9.0
        elif pe30 >= MIN_OI_CHANGE:    bull = 7.0
        elif pe30 >= 5:                bull = 4.0

        # CALL OI building → BEARISH
        if   ce30 >= STRONG_OI_CHANGE: bear = 9.0
        elif ce30 >= MIN_OI_CHANGE:    bear = 7.0
        elif ce30 >= 5:                bear = 4.0

        # Unwinding (reversal signals)
        if pe30 <= -STRONG_OI_CHANGE:  bear = max(bear, 8.0)
        elif pe30 <= -MIN_OI_CHANGE:   bear = max(bear, 6.0)
        if ce30 <= -STRONG_OI_CHANGE:  bull = max(bull, 8.0)
        elif ce30 <= -MIN_OI_CHANGE:   bull = max(bull, 6.0)

        return min(10.0, bull * weight * boost), min(10.0, bear * weight * boost)

    # ── Main Analysis ─────────────────────────────────────────
    async def analyze(self, current: MarketSnapshot) -> Dict:
        # Historical snapshots (via 15-min cache)
        s_15m = await self.cache.get_15min_ago(1)   # 15 min ago
        s_30m = await self.cache.get_15min_ago(2)   # 30 min ago
        s_60m = await self.cache.get_15min_ago(4)   # 60 min ago

        if not s_15m:
            return {"available": False, "reason": "⏳ Building 15-min cache (need ≥ 15 min)..."}

        analyses: List[StrikeAnalysis] = []

        for strike in sorted(current.strikes_oi.keys()):
            curr = current.strikes_oi[strike]
            p15  = s_15m.strikes_oi.get(strike)
            p30  = s_30m.strikes_oi.get(strike) if s_30m else None
            p60  = s_60m.strikes_oi.get(strike) if s_60m else None

            # ── Compute changes per TF ──
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

            # ── Weight ──
            is_atm   = (strike == current.atm_strike)
            dist     = abs(strike - current.atm_strike)
            step     = max(50, min(100, int(dist / max(1, ATM_STRIKE_COUNT - 1))))
            if is_atm:               weight = ATM_WEIGHT
            elif dist <= step * 2:   weight = NEAR_ATM_WEIGHT
            else:                    weight = FAR_WEIGHT

            # ── TF signals ──
            tf15 = self._tf_signal(ce15, pe15, cv15, pv15)
            tf30 = self._tf_signal(ce30, pe30, cv30, pv30)
            mtf  = (tf15 == tf30 and tf15 != "NEUTRAL")

            # ── Volume confirmation (using 30-min as primary) ──
            avg_oi  = (ce30 + pe30) / 2
            avg_vol = (cv30 + pv30) / 2
            vc, vs  = self._vol_confirm(avg_oi, avg_vol)

            # ── Signal strength ──
            bull, bear = self._signal_strength(ce30, pe30, cv30, pv30, weight, mtf)

            # Extra boost if 15-min also confirms
            if mtf:
                bull = min(10.0, bull * 1.2)
                bear = min(10.0, bear * 1.2)

            # ── Recommendation ──
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

        # ── Support / Resistance ──
        sr = self._find_sr(current, analyses)
        for sa in analyses:
            sa.is_support    = (sa.strike == sr.support_strike)
            sa.is_resistance = (sa.strike == sr.resistance_strike)

        # ── Overall PCR trend (vs 30-min ago) ──
        prev_pcr    = s_30m.overall_pcr if s_30m else current.overall_pcr
        pcr_trend   = "BULLISH" if current.overall_pcr > prev_pcr else "BEARISH"
        pcr_ch      = self._pct(current.overall_pcr, prev_pcr)

        # ── Aggregate signal ──
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
        sup = max_pe.strike if max_pe else current.atm_strike
        res = max_ce.strike if max_ce else current.atm_strike
        return SupportResistance(
            support_strike=sup,    support_put_oi=max_pe.pe_oi if max_pe else 0,
            resistance_strike=res, resistance_call_oi=max_ce.ce_oi if max_ce else 0,
            near_support=abs(current.spot_price - sup) <= ATM_PROX_USD,
            near_resistance=abs(current.spot_price - res) <= ATM_PROX_USD
        )


# ============================================================
#  STANDALONE ALERT CHECKER  (4 alert types)
# ============================================================

class AlertChecker:
    """
    Runs every 15-min snapshot.
    Sends immediate alerts without waiting for full AI analysis.
    30-min cooldown per alert type to prevent spam.
    """

    COOLDOWN = 30 * 60   # 30 minutes

    def __init__(self, cache: DualCache, alerter):
        self.cache   = cache
        self.alerter = alerter
        self._last: Dict[str, float] = {}

    def _can(self, key: str) -> bool:
        return (time_module.time() - self._last.get(key, 0)) >= self.COOLDOWN

    def _mark(self, key: str):
        self._last[key] = time_module.time()

    async def check_all(self, curr: MarketSnapshot):
        prev = await self.cache.get_15min_ago(2)   # 30-min ago
        if not prev:
            return
        await self._oi_change(curr, prev)
        await self._vol_spike(curr, prev)
        await self._pcr_change(curr, prev)
        await self._atm_proximity(curr)

    # ── 1. OI Change ──────────────────────────────────────────
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

    # ── 2. Volume Spike ───────────────────────────────────────
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
        dom   = "CALL" if ce_v >= pe_v else "PUT"
        bias  = "🔴 BEARISH" if dom == "CALL" else "🟢 BULLISH"
        txt   = (
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

    # ── 3. PCR Change ────────────────────────────────────────
    async def _pcr_change(self, curr: MarketSnapshot, prev: MarketSnapshot):
        if not self._can("PCR") or prev.overall_pcr <= 0:
            return
        pcr_ch = (curr.overall_pcr - prev.overall_pcr) / prev.overall_pcr * 100
        if abs(pcr_ch) < PCR_ALERT_PCT:
            return
        trend = "📈 BULLS GAINING (more PUT writers)" if pcr_ch > 0 else "📉 BEARS GAINING (more CALL writers)"
        interp = ("Strong PUT base → Bullish bias" if curr.overall_pcr > PCR_BULL
                  else "Strong CALL base → Bearish bias" if curr.overall_pcr < PCR_BEAR
                  else "Neutral zone")
        txt = (
            f"📊 <b>PCR CHANGE ALERT</b>\n\n"
            f"ETH: ${curr.spot_price:,.2f}\n\n"
            f"PCR: {prev.overall_pcr:.2f} → <b>{curr.overall_pcr:.2f}</b>  ({pcr_ch:+.1f}%)\n"
            f"Trend: {trend}\n"
            f"Bias: {interp}\n"
            f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        await self.alerter.send_raw(txt)
        self._mark("PCR")

    # ── 4. ATM Proximity ─────────────────────────────────────
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
            dist = abs(spot - strike)
            if dist > ATM_PROX_USD:
                continue
            oi_val = oi_snap.pe_oi if kind == "PUT" else oi_snap.ce_oi
            note   = ("PUT writers defending → Watch for bounce"
                      if kind == "PUT" else "CALL writers defending → Watch for rejection")
            txt = (
                f"{emoji} <b>PRICE NEAR {level} ALERT</b>\n\n"
                f"ETH:   ${spot:,.2f}\n"
                f"{level}: ${strike:,.0f}  (OI: {oi_val:,.0f})\n"
                f"Dist:  ${dist:.2f}\n\n"
                f"{note}\n"
                f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
            )
            await self.alerter.send_raw(txt)
            self._mark("PROX")
            break   # Only one proximity alert per cycle


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
            # Bullish Engulfing
            if (cur.close > cur.open and prv.close < prv.open
                    and cur.open <= prv.close and cur.close >= prv.open
                    and body_c > body_p * 1.2):
                patterns.append({"time": cur.name, "pattern": "BULLISH_ENGULFING", "type": "BULLISH", "strength": 8})
            # Bearish Engulfing
            elif (cur.close < cur.open and prv.close > prv.open
                    and cur.open >= prv.close and cur.close <= prv.open
                    and body_c > body_p * 1.2):
                patterns.append({"time": cur.name, "pattern": "BEARISH_ENGULFING", "type": "BEARISH", "strength": 8})
            else:
                lo_wick = min(cur.open, cur.close) - cur.low
                hi_wick = cur.high - max(cur.open, cur.close)
                if lo_wick > body_c * 2 and hi_wick < body_c * 0.3 and body_c < rng * 0.35:
                    patterns.append({"time": cur.name, "pattern": "HAMMER",        "type": "BULLISH", "strength": 6})
                elif hi_wick > body_c * 2 and lo_wick < body_c * 0.3 and body_c < rng * 0.35:
                    patterns.append({"time": cur.name, "pattern": "SHOOTING_STAR", "type": "BEARISH", "strength": 6})
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
#  PROMPT BUILDER
# ============================================================

class PromptBuilder:

    @staticmethod
    def _short_candles(df: pd.DataFrame, label: str, base: int) -> str:
        """
        Ultra-compact candle format:
        15MIN (BASE+3000):
        14:30|245|268|238|261|↑
        """
        if df.empty:
            return f"{label}: no data\n"
        out = f"{label} (BASE+{base}, format:TIME|O|H|L|C|DIR):\n"
        for ts, row in df.tail(CANDLE_COUNT).iterrows():
            t  = ts.strftime("%H:%M")
            o  = int(row.open)  - base
            h  = int(row.high)  - base
            l  = int(row.low)   - base
            c  = int(row.close) - base
            d  = "↑" if row.close > row.open else "↓" if row.close < row.open else "→"
            out += f"{t}|{o}|{h}|{l}|{c}|{d}\n"
        return out

    @staticmethod
    def build(spot: float, atm: float, expiry: str,
              oi: Dict,
              c15: pd.DataFrame, c30: pd.DataFrame,
              patterns: List[Dict],
              p_sup: float, p_res: float) -> str:

        now   = datetime.now(timezone.utc).strftime("%H:%M UTC")
        sr    = oi["sr"]
        pcr   = oi["overall_pcr"]
        base  = int(spot / 1000) * 1000   # e.g. 3000 for ETH at $3245

        # ── Header ──
        p  = f"ETH OPTIONS | {now} | Expiry: {expiry}\n"
        p += f"ETH:${spot:,.2f} ATM:${atm:,.0f} PCR:{pcr:.2f}({oi['pcr_trend']}) Δ30m:{oi['pcr_ch_pct']:+.1f}%\n"
        p += f"OI-Sup:${sr.support_strike:,.0f} OI-Res:${sr.resistance_strike:,.0f}\n"
        if sr.near_support:    p += "⚡ NEAR OI-SUPPORT!\n"
        if sr.near_resistance: p += "⚡ NEAR OI-RESISTANCE!\n"
        p += "\n"

        # ── Strike table ──
        p += "OI+VOL MULTI-TIMEFRAME (CE=Call, PE=Put):\n"
        for sa in oi["strike_analyses"]:
            tag = "⭐ATM" if sa.is_atm else ("🟢SUP" if sa.is_support else "🔴RES" if sa.is_resistance else "")
            mtf = "✅MTF" if sa.mtf_confirmed else "❌"
            p += (f"\n${sa.strike:,.0f} {tag} W:{sa.weight:.0f}x {mtf}\n"
                  f"CE: OI{sa.ce_oi_15:+.0f}%/15m {sa.ce_oi_30:+.0f}%/30m "
                  f"Vol{sa.ce_vol_15:+.0f}%/15m {sa.ce_vol_30:+.0f}%/30m ({sa.ce_action})\n"
                  f"PE: OI{sa.pe_oi_15:+.0f}%/15m {sa.pe_oi_30:+.0f}%/30m "
                  f"Vol{sa.pe_vol_15:+.0f}%/15m {sa.pe_vol_30:+.0f}%/30m ({sa.pe_action})\n"
                  f"PCR:{sa.pcr:.2f} TF15:{sa.tf15_signal} TF30:{sa.tf30_signal} "
                  f"Vol:{sa.vol_strength} Bull:{sa.bull_strength:.0f} Bear:{sa.bear_strength:.0f} Conf:{sa.confidence:.0f}\n")

        # ── Candles (short format) ──
        p += f"\n{PromptBuilder._short_candles(c15, '15MIN', base)}"
        p += f"\n{PromptBuilder._short_candles(c30, '30MIN', base)}"

        # ── Patterns ──
        if patterns:
            p += "\nPATTERNS:\n"
            for pat in patterns:
                p += f"{pat['time'].strftime('%H:%M')}|{pat['pattern']}|{pat['type']}|{pat['strength']}/10\n"

        # ── Price S/R ──
        if p_sup or p_res:
            p += f"\nPrice S/R: Sup ${p_sup:.2f} Res ${p_res:.2f}\n"

        # ── Instructions ──
        p += f"""
RULES:
• MTF = both TF15+TF30 agree → HIGH confidence
• OI↑+Vol↑ = real move | OI↑+Vol↓ = TRAP ignore
• CALL OI↑+Vol↑ = resistance building = BEARISH → BUY_PUT
• PUT  OI↑+Vol↑ = support building   = BULLISH → BUY_CALL
• PCR > 1.3 = bullish bias | PCR < 0.7 = bearish bias
• ATM strike (3x weight) is primary signal

RESPOND ONLY VALID JSON:
{{
  "signal": "BUY_CALL"|"BUY_PUT"|"WAIT",
  "primary_strike": {atm},
  "confidence": 0-10,
  "stop_loss_strike": 0,
  "target_strike": 0,
  "mtf": {{"tf15": "", "tf30": "", "confirmed": true}},
  "atm": {{"ce_action": "", "pe_action": "", "vol_confirms": true, "strength": ""}},
  "pcr": {{"value": {pcr:.2f}, "trend": "{oi['pcr_trend']}", "note": "", "supports": true}},
  "volume": {{"ok": true, "trap_warning": "", "quality": ""}},
  "entry": {{"now": true, "reason": "", "wait_for": ""}},
  "rr": {{"sl_pts": 0, "tgt_pts": 0, "ratio": 0}},
  "levels": {{"support": {sr.support_strike}, "resistance": {sr.resistance_strike}, "pos": ""}}
}}"""
        return p


# ============================================================
#  DEEPSEEK CLIENT
# ============================================================

class DeepSeekClient:

    URL   = "https://api.deepseek.com/v1/chat/completions"
    MODEL = "deepseek-chat"

    def __init__(self, key: str):
        self.key = key

    async def analyze(self, prompt: str) -> Optional[Dict]:
        hdrs    = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        payload = {"model": self.MODEL, "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.3, "max_tokens": 1200}
        try:
            timeout = aiohttp.ClientTimeout(total=DEEPSEEK_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(self.URL, headers=hdrs, json=payload) as r:
                    if r.status != 200:
                        logger.error(f"❌ DeepSeek {r.status}")
                        return None
                    data    = await r.json()
                    content = data["choices"][0]["message"]["content"].strip()
                    # Strip markdown fences
                    for fence in ("```json", "```"):
                        content = content.replace(fence, "")
                    return json.loads(content.strip())
        except asyncio.TimeoutError:
            logger.error(f"❌ DeepSeek timeout (>{DEEPSEEK_TIMEOUT}s)")
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

    async def send_signal(self, sig: Dict, snap: MarketSnapshot, oi: Dict):
        """Full AI trade signal alert"""
        atm_info = next((sa for sa in oi["strike_analyses"] if sa.is_atm), None)

        mtf  = sig.get("mtf",    {})
        atma = sig.get("atm",    {})
        pcra = sig.get("pcr",    {})
        vol  = sig.get("volume", {})
        ent  = sig.get("entry",  {})
        rr   = sig.get("rr",     {})

        signal_type = sig.get("signal", "WAIT")
        option_type = "CE" if "CALL" in signal_type else "PE" if "PUT" in signal_type else ""

        msg = (
            f"🚨 <b>ETH OPTIONS SIGNAL v6.0</b>\n"
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
            f"📊 <b>ATM ANALYSIS:</b>\n"
            f"CE Writers: {atma.get('ce_action', 'N/A')}\n"
            f"PE Writers: {atma.get('pe_action', 'N/A')}\n"
            f"Volume: {'✅ Confirms' if atma.get('vol_confirms') else '❌ MISMATCH – POSSIBLE TRAP'}\n"
            f"Quality: {atma.get('strength', 'N/A')}\n\n"
            f"📈 <b>PCR:</b> {pcra.get('value', 'N/A')} ({pcra.get('trend', 'N/A')})\n"
            f"Supports signal: {'✅' if pcra.get('supports') else '❌'}\n"
            f"{pcra.get('note', '')}\n\n"
            f"⚡ Volume: {'✅ OK' if vol.get('ok') else '❌ CAUTION'}\n"
        )
        if vol.get("trap_warning"):
            msg += f"⚠️ {vol['trap_warning']}\n"
        msg += (
            f"\n⏰ Enter Now: {'✅ YES' if ent.get('now') else '⏳ WAIT'}\n"
            f"{ent.get('reason', '')}\n\n"
            f"🤖 DeepSeek V3 + MTF v6.0"
        )

        await self.send_raw(msg)


# ============================================================
#  MAIN BOT
# ============================================================

class ETHOptionsBot:
    """
    Cycle logic:
      Every 15 min  → fetch snapshot → cache_15min → standalone alerts
      Every 30 min  → also → cache_30min → full MTF analysis → DeepSeek → trade alert
    """

    def __init__(self):
        self.delta   = DeltaClient(DELTA_API_KEY, DELTA_API_SECRET)
        self.cache   = DualCache()
        self.mtf     = MTFAnalyzer(self.cache)
        self.ai      = DeepSeekClient(DEEPSEEK_API_KEY)
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.checker = AlertChecker(self.cache, self.alerter)
        self._cycle  = 0

    # ── Main loop ────────────────────────────────────────────
    async def run(self):
        logger.info("\n" + "="*60)
        logger.info("🚀 ETH OPTIONS BOT v6.0 – DELTA EXCHANGE GLOBAL")
        logger.info("="*60)
        logger.info(f"⏰ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"📦 Cache: 15-min×{CACHE_15MIN_SIZE} + 30-min×{CACHE_30MIN_SIZE} (6hr each)")
        logger.info(f"📊 Snapshot: every {SNAPSHOT_INTERVAL//60}min | Analysis: every {ANALYSIS_INTERVAL//60}min")
        logger.info(f"🔗 Multi-TF: 15-min + 30-min confirmation")
        logger.info(f"🤖 DeepSeek V3 ({DEEPSEEK_TIMEOUT}s) | Min confidence: {MIN_CONFIDENCE}/10")
        logger.info("="*60 + "\n")

        await self.delta.init()

        try:
            while True:
                try:
                    await self._cycle_run()
                except Exception as e:
                    logger.error(f"❌ Cycle error: {e}")
                    logger.exception("Traceback:")
                    await asyncio.sleep(60)
                s15, s30 = self.cache.sizes()
                logger.info(f"⏰ Next snapshot in {SNAPSHOT_INTERVAL//60} min | Cache: 15m={s15} 30m={s30}\n")
                await asyncio.sleep(SNAPSHOT_INTERVAL)
        except KeyboardInterrupt:
            logger.info("🛑 Bot stopped")
        finally:
            await self.delta.close()
            await self.alerter.close()

    async def _cycle_run(self):
        self._cycle += 1
        is_analysis = (self._cycle % 2 == 0)   # Every 2nd → 30-min cycle

        logger.info(f"\n{'='*60}")
        logger.info(f"{'🚀 ANALYSIS' if is_analysis else '📦 SNAPSHOT'} CYCLE #{self._cycle}")
        logger.info(f"⏰ {datetime.now(timezone.utc).strftime('%H:%M UTC')}")
        logger.info("="*60)

        # 1. Fetch snapshot
        snap = await self.delta.fetch_snapshot()
        if not snap:
            logger.warning("⚠️  Snapshot fetch failed – skipping")
            return

        # 2. Always: 15-min cache + standalone alerts
        await self.cache.add_15min(snap)
        await self.checker.check_all(snap)

        # 3. Every 30-min: full analysis
        if is_analysis:
            await self.cache.add_30min(snap)
            await self._full_analysis(snap)

    # ── Full Analysis ────────────────────────────────────────
    async def _full_analysis(self, snap: MarketSnapshot):
        logger.info("\n🔍 Running multi-timeframe OI analysis...")
        oi = await self.mtf.analyze(snap)

        if not oi["available"]:
            logger.info(f"⏳ {oi['reason']}")
            return

        self._log_analysis(snap, oi)

        if not oi["has_strong"]:
            logger.info("📊 No MTF-confirmed signal ≥ 7 confidence – skipping AI")
            return

        logger.info("🚨 Strong MTF signal found!")

        # Fetch candles
        logger.info("📈 Fetching 15-min + 30-min candles...")
        c15 = await self.delta.get_candles("ETHUSD", "15", CANDLE_COUNT)
        await asyncio.sleep(API_DELAY)
        c30 = await self.delta.get_candles("ETHUSD", "30", CANDLE_COUNT)

        # Patterns + S/R from 15-min
        patterns       = PatternDetector.detect(c15) if not c15.empty else []
        p_sup, p_res   = PatternDetector.support_resistance(c15) if not c15.empty else (0.0, 0.0)

        # Build prompt
        prompt = PromptBuilder.build(
            spot=snap.spot_price, atm=snap.atm_strike, expiry=snap.expiry,
            oi=oi, c15=c15, c30=c30,
            patterns=patterns, p_sup=p_sup, p_res=p_res
        )

        # DeepSeek
        logger.info(f"🤖 Sending to DeepSeek (timeout:{DEEPSEEK_TIMEOUT}s)...")
        ai_sig = await self.ai.analyze(prompt)

        if not ai_sig:
            logger.warning("⚠️  DeepSeek timeout – using fallback")
            atm_sa = next((sa for sa in oi["strike_analyses"] if sa.is_atm), None)
            if atm_sa and atm_sa.mtf_confirmed:
                fb = "BUY_CALL" if atm_sa.bull_strength > atm_sa.bear_strength else "BUY_PUT"
                fc = min(10, max(atm_sa.bull_strength, atm_sa.bear_strength))
            else:
                fb, fc = "WAIT", 3
            ai_sig = {
                "signal": fb, "confidence": fc,
                "primary_strike": snap.atm_strike,
                "mtf": {"tf15": "N/A", "tf30": "N/A", "confirmed": False},
                "entry": {"now": False, "reason": "AI timeout – fallback logic"}
            }

        conf   = ai_sig.get("confidence", 0)
        signal = ai_sig.get("signal", "WAIT")
        logger.info(f"🎯 Signal: {signal} | Confidence: {conf}/10")

        if conf >= MIN_CONFIDENCE:
            logger.info("📤 Sending trade alert to Telegram...")
            await self.alerter.send_signal(ai_sig, snap, oi)
        else:
            logger.info(f"⏳ Confidence {conf}/10 < {MIN_CONFIDENCE} – no trade alert")

    # ── Debug Logger ─────────────────────────────────────────
    def _log_analysis(self, snap: MarketSnapshot, oi: Dict):
        logger.info("\n" + "="*60)
        logger.info("📊 PRE-AI ANALYSIS DATA")
        logger.info("="*60)
        logger.info(f"ETH:${snap.spot_price:,.2f} | ATM:${snap.atm_strike:,.0f} | Expiry:{snap.expiry}")
        logger.info(f"PCR:{snap.overall_pcr:.2f} ({oi['pcr_trend']}) | Δ30m:{oi['pcr_ch_pct']:+.1f}%")
        logger.info(f"Overall: {oi['overall']} | Bull:{oi['total_bull']:.1f} Bear:{oi['total_bear']:.1f}")
        sr = oi["sr"]
        logger.info(f"Sup:${sr.support_strike:,.0f}(PUT OI:{sr.support_put_oi:,.0f}) "
                    f"Res:${sr.resistance_strike:,.0f}(CE OI:{sr.resistance_call_oi:,.0f})")
        logger.info("-"*60)
        for sa in oi["strike_analyses"]:
            atm_m = " ⭐ATM" if sa.is_atm else ""
            mtf_m = " ✅MTF" if sa.mtf_confirmed else ""
            logger.info(
                f"${sa.strike:,.0f}{atm_m}{mtf_m} | "
                f"CE OI:{sa.ce_oi_15:+.0f}%/15m {sa.ce_oi_30:+.0f}%/30m | "
                f"PE OI:{sa.pe_oi_15:+.0f}%/15m {sa.pe_oi_30:+.0f}%/30m | "
                f"TF15:{sa.tf15_signal[:3]} TF30:{sa.tf30_signal[:3]} | "
                f"Conf:{sa.confidence:.1f}"
            )
        logger.info("="*60)


# ============================================================
#  HTTP WRAPPER (Koyeb / Railway deployment)
# ============================================================

async def health(request):
    s15, s30 = bot_instance.cache.sizes() if bot_instance else (0, 0)
    return aiohttp.web.Response(
        text=f"✅ ETH Options Bot v6.0 | Cache: 15m={s15}/{CACHE_15MIN_SIZE} 30m={s30}/{CACHE_30MIN_SIZE}"
    )

bot_instance: Optional[ETHOptionsBot] = None

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
║   🚀 ETH OPTIONS BOT v6.0 – DELTA EXCHANGE GLOBAL      ║
╚══════════════════════════════════════════════════════════╝

📋 CONFIG:
  Asset      : ETH Daily Options (global.delta.exchange)
  Snapshot   : Every 15 min  → 15-min cache (24 × 15min = 6hr)
  Analysis   : Every 30 min  → 30-min cache (12 × 30min = 6hr)
  Candles    : 15-min × 24 + 30-min × 24 (short format)
  Strikes    : ATM ± {ATM_STRIKE_COUNT} (11 total)

✅ FEATURES:
  • Multi-TF: 15-min + 30-min OI/Volume/PCR comparison
  • MTF confirmed = both TFs agree → HIGH confidence
  • TRAP filter: OI↑ but Vol flat → ignore
  • 4 standalone alerts (no AI needed):
      1. OI Change   ≥{OI_ALERT_PCT:.0f}% (30-min)
      2. Volume Spike≥{VOL_SPIKE_PCT:.0f}% 
      3. PCR Change  ≥{PCR_ALERT_PCT:.0f}%
      4. Price near high-OI strike (±${ATM_PROX_USD})
  • DeepSeek V3 ({DEEPSEEK_TIMEOUT}s timeout) + fallback
  • Min confidence: {MIN_CONFIDENCE}/10 to send trade alert

🔑 ENV VARS NEEDED:
  DELTA_API_KEY, DELTA_API_SECRET
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
  DEEPSEEK_API_KEY

Starting on port {port}...
""")

    web.run_app(app, host="0.0.0.0", port=port)
