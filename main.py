#!/usr/bin/env python3
"""
main.py - Full patched Advanced 720-Candles Crypto Trading Bot
- 4H + 1H 720 candles analysis
- Redis tracking with JSON serialization
- Candle cache TTL = 10 days (864000 seconds)
- HTTP concurrency control (semaphore)
- AI confirmation via OpenAI (no unsupported kwargs)
- Telegram alerts (optional)
- Robust logging & error handling
"""
import os
import sys
import json
import time
import math
import traceback
import asyncio
import aiohttp
import redis
from datetime import datetime
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

# ---------------- Load config ----------------
load_dotenv()

# Symbols (default list; override via env SYMBOLS="BTCUSDT,ETHUSDT,...")
SYMBOLS = os.getenv("SYMBOLS",
                    "BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,AAVEUSDT,TRXUSDT,DOGEUSDT,BNBUSDT,ADAUSDT,LTCUSDT,LINKUSDT") \
    .split(",")

POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", "1800")))  # seconds between scans
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", "85.0"))  # final conf %
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# Redis config: prefer REDIS_URL, fallback to host/port/user/pass
REDIS_URL = os.getenv("REDIS_URL")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_USER = os.getenv("REDIS_USER", os.getenv("REDIS_USERNAME", None))
REDIS_PASS = os.getenv("REDIS_PASS", os.getenv("REDIS_PASSWORD", None))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# Candle cache TTL: 10 days in seconds
CANDLE_CACHE_TTL = 10 * 24 * 3600  # 864000

# HTTP concurrency
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "6"))
HTTP_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Binance endpoints
CANDLE_4H_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h&limit=720"
CANDLE_1H_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"
TICKER_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol={symbol}"

MIN_CANDLES_REQUIRED = 720
MIN_RR_RATIO = float(os.getenv("MIN_RR_RATIO", "2.0"))
TREND_REVERSAL_THRESHOLD = float(os.getenv("TREND_REVERSAL_THRESHOLD", "0.02"))  # 2%
TARGET_PROGRESS_LEVELS = [0.25, 0.50, 0.75, 0.90]

# Initialize OpenAI client (if key present)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# ---------------- Redis init ----------------
redis_client = None
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, db=REDIS_DB, decode_responses=True)
    else:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
                                   username=REDIS_USER, password=REDIS_PASS, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

# ---------------- Utils ----------------
def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"

def now_iso():
    return datetime.utcnow().isoformat()

# ---------------- Candle fetch & cache ----------------
async def fetch_json(session, url, retries=3, timeout_seconds=40):
    for i in range(retries):
        try:
            async with HTTP_SEMAPHORE:
                async with session.get(url, timeout=timeout_seconds) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        text = await resp.text()
                        print(f"HTTP {resp.status} for {url} -> {text[:200]}")
        except Exception as e:
            print(f"Fetch error for {url} (attempt {i+1}): {e}")
            if i < retries - 1:
                await asyncio.sleep(1 + i)
    return None

async def fetch_candles_with_cache(session, symbol, timeframe):
    """
    Fetch candles for given symbol/timeframe with Redis caching.
    Cache TTL is set to CANDLE_CACHE_TTL (10 days).
    """
    cache_key = f"candles:{symbol}:{timeframe}"
    try:
        if redis_client:
            cached = redis_client.get(cache_key)
            if cached:
                try:
                    return json.loads(cached)
                except Exception:
                    # if corrupted, delete and fetch fresh
                    try:
                        redis_client.delete(cache_key)
                    except Exception:
                        pass

        url = CANDLE_1H_URL.format(symbol=symbol) if timeframe == "1h" else CANDLE_4H_URL.format(symbol=symbol)
        data = await fetch_json(session, url)
        if data and redis_client:
            try:
                redis_client.set(cache_key, json.dumps(data), ex=CANDLE_CACHE_TTL)
            except Exception as e:
                print(f"Redis set error for {cache_key}: {e}")
        return data
    except Exception as e:
        print(f"fetch_candles_with_cache error: {e}")
        return None

# ---------------- Candle normalization ----------------
def normalize_klines(raw_klines):
    out = []
    for row in raw_klines or []:
        try:
            if len(row) >= 6:
                ts = int(row[0])
                o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); v = float(row[5])
                body = abs(c - o)
                rng = h - l if h - l != 0 else 0.0
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l
                out.append({
                    "open": o, "high": h, "low": l, "close": c, "volume": v, "ts": ts,
                    "body_size": body, "total_range": rng, "upper_wick": upper_wick, "lower_wick": lower_wick,
                    "is_bullish": c > o,
                    "body_ratio": (body / rng) if rng > 0 else 0,
                    "upper_wick_ratio": (upper_wick / rng) if rng > 0 else 0,
                    "lower_wick_ratio": (lower_wick / rng) if rng > 0 else 0
                })
            else:
                if len(row) >= 4:
                    o = float(row[0]); h = float(row[1]); l = float(row[2]); c = float(row[3])
                    out.append({"open": o, "high": h, "low": l, "close": c, "volume": 0.0, "ts": None,
                                "body_size": abs(c-o), "total_range": h-l if h-l!=0 else 0.0,
                                "upper_wick": h-max(o,c), "lower_wick": min(o,c)-l,
                                "is_bullish": c>o, "body_ratio": 0, "upper_wick_ratio": 0, "lower_wick_ratio": 0})
                else:
                    continue
        except Exception:
            continue
    return out

# ---------------- EMA & helpers ----------------
def ema(values, period):
    if not values or len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema_vals.append(sma)
    prev = sma
    for i in range(period, len(values)):
        v = values[i]
        prev = v * k + prev * (1 - k)
        ema_vals.append(prev)
    if len(ema_vals) < len(values):
        pad = [None] * (len(values) - len(ema_vals))
        ema_vals = pad + ema_vals
    return ema_vals

def calculate_emas_9_20(closes):
    try:
        e9 = ema(closes, 9)
        e20 = ema(closes, 20)
        return {"ema_9": e9[-1] if e9 else None, "ema_20": e20[-1] if e20 else None, "e9_series": e9, "e20_series": e20}
    except Exception:
        return {"ema_9": None, "ema_20": None, "e9_series": [], "e20_series": []}

def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    try:
        length = min(len(closes), lookback)
        pts = closes[-length:] + highs[-length:] + lows[-length:]
    except Exception:
        return []
    levels = []
    for p in pts:
        if p is None or p == 0:
            continue
        found = False
        for lv in levels:
            try:
                if abs((lv["price"] - p) / p) < binsize:
                    lv["count"] += 1
                    lv["price"] = (lv["price"] * (lv["count"] - 1) + p) / lv["count"]
                    found = True
                    break
            except Exception:
                continue
        if not found:
            levels.append({"price": p, "count": 1})
    levels.sort(key=lambda x: -x["count"])
    return [lv["price"] for lv in levels][:5]

# ---------------- Pattern detection ----------------
def detect_single_candle_patterns(c):
    patterns = []
    if c['total_range'] <= 0:
        return patterns
    if c['body_ratio'] < 0.1:
        if c['upper_wick_ratio'] > 0.4 and c['lower_wick_ratio'] > 0.4:
            patterns.append("Long_Legged_Doji")
        elif c['upper_wick_ratio'] < 0.1 and c['lower_wick_ratio'] < 0.1:
            patterns.append("Four_Price_Doji")
        elif c['upper_wick_ratio'] > c['lower_wick_ratio'] * 2:
            patterns.append("Gravestone_Doji")
        elif c['lower_wick_ratio'] > c['upper_wick_ratio'] * 2:
            patterns.append("Dragonfly_Doji")
        else:
            patterns.append("Doji")
    elif (c['body_ratio'] < 0.3 and c['lower_wick_ratio'] > 0.5 and c['upper_wick_ratio'] < 0.1):
        patterns.append("Hammer" if c['is_bullish'] else "Hanging_Man")
    elif (c['body_ratio'] < 0.3 and c['upper_wick_ratio'] > 0.5 and c['lower_wick_ratio'] < 0.1):
        patterns.append("Inverted_Hammer" if c['is_bullish'] else "Shooting_Star")
    elif (c['body_ratio'] > 0.95):
        patterns.append("Bullish_Marubozu" if c['is_bullish'] else "Bearish_Marubozu")
    elif (0.1 < c['body_ratio'] < 0.3 and c['upper_wick_ratio'] > 0.2 and c['lower_wick_ratio'] > 0.2):
        patterns.append("Spinning_Top")
    return patterns

def detect_two_candle_patterns(c1, c2):
    patterns = []
    if (c1['is_bullish'] != c2['is_bullish'] and
        c2['body_size'] > c1['body_size'] * 1.1 and
        ((c2['is_bullish'] and c2['close'] > c1['high'] and c2['open'] < c1['low']) or
         (not c2['is_bullish'] and c2['close'] < c1['low'] and c2['open'] > c1['high']))):
        patterns.append("Bullish_Engulfing" if c2['is_bullish'] else "Bearish_Engulfing")
    elif (c1['is_bullish'] != c2['is_bullish'] and
          c2['body_size'] < c1['body_size'] * 0.8 and
          c2['high'] < c1['high'] and c2['low'] > c1['low']):
        patterns.append("Bullish_Harami" if c2['is_bullish'] else "Bearish_Harami")
    elif c1['is_bullish'] != c2['is_bullish']:
        if (not c1['is_bullish'] and c2['is_bullish'] and
            c2['open'] < c1['low'] and c2['close'] > (c1['open'] + c1['close']) / 2):
            patterns.append("Piercing_Line")
        elif (c1['is_bullish'] and not c2['is_bullish'] and
              c2['open'] > c1['high'] and c2['close'] < (c1['open'] + c1['close']) / 2):
            patterns.append("Dark_Cloud_Cover")
    elif c1['high'] > 0 and c2['high'] > 0 and abs(c1['high'] - c2['high']) / c1['high'] < 0.002:
        patterns.append("Tweezer_Top")
    elif c1['low'] > 0 and c2['low'] > 0 and abs(c1['low'] - c2['low']) / c1['low'] < 0.002:
        patterns.append("Tweezer_Bottom")
    return patterns

def detect_three_candle_patterns(c1, c2, c3):
    patterns = []
    if (c2['body_ratio'] < 0.3):
        if (not c1['is_bullish'] and c3['is_bullish'] and
            c3['close'] > (c1['open'] + c1['close']) / 2):
            patterns.append("Morning_Star")
        elif (c1['is_bullish'] and not c3['is_bullish'] and
              c3['close'] < (c1['open'] + c1['close']) / 2):
            patterns.append("Evening_Star")
    if (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'] and
        c2['close'] > c1['close'] and c3['close'] > c2['close'] and
        all(c['body_ratio'] > 0.6 for c in [c1, c2, c3])):
        patterns.append("Three_White_Soldiers")
    elif (not c1['is_bullish'] and not c2['is_bullish'] and not c3['is_bullish'] and
          c2['close'] < c1['close'] and c3['close'] < c2['close'] and
          all(c['body_ratio'] > 0.6 for c in [c1, c2, c3])):
        patterns.append("Three_Black_Crows")
    if (c2['body_ratio'] < 0.1 and
        c1['is_bullish'] != c3['is_bullish'] and
        ((c2['high'] < min(c1['low'], c3['low'])) or
         (c2['low'] > max(c1['high'], c3['high'])))):
        patterns.append("Abandoned_Baby_Bullish" if c3['is_bullish'] else "Abandoned_Baby_Bearish")
    return patterns

def detect_five_candle_patterns(c1, c2, c3, c4, c5):
    patterns = []
    if (c1['is_bullish'] and c5['is_bullish'] and
        c5['close'] > c1['close'] and c5['high'] > c1['high']):
        if all(not c['is_bullish'] for c in [c2, c3, c4]):
            patterns.append("Rising_Three_Methods")
    elif (not c1['is_bullish'] and not c5['is_bullish'] and
          c5['close'] < c1['close'] and c5['low'] < c1['low']):
        if all(c['is_bullish'] for c in [c2, c3, c4]):
            patterns.append("Falling_Three_Methods")
    return patterns

def detect_all_candlestick_patterns(candles):
    if len(candles) < 5:
        return []
    patterns = []
    c1, c2, c3, c4, c5 = candles[-5:]
    patterns.extend(detect_single_candle_patterns(c5))
    patterns.extend(detect_two_candle_patterns(c4, c5))
    patterns.extend(detect_three_candle_patterns(c3, c4, c5))
    patterns.extend(detect_five_candle_patterns(c1, c2, c3, c4, c5))
    return list(set(patterns))

# ---------------- Key levels ----------------
def identify_key_levels_advanced(candles, lookback=200):
    if len(candles) < lookback:
        lookback = len(candles)
    recent_candles = candles[-lookback:]
    swing_highs = []
    swing_lows = []
    for period in [3, 5, 8, 13, 21]:
        for i in range(period, len(recent_candles) - period):
            candle = recent_candles[i]
            if all(candle['high'] >= recent_candles[j]['high'] for j in range(i-period, i+period+1) if j != i):
                swing_highs.append({'price': candle['high'], 'strength': period, 'volume': candle['volume'], 'timestamp': candle['ts']})
            if all(candle['low'] <= recent_candles[j]['low'] for j in range(i-period, i+period+1) if j != i):
                swing_lows.append({'price': candle['low'], 'strength': period, 'volume': candle['volume'], 'timestamp': candle['ts']})
    def consolidate_levels(levels, tolerance=0.003):
        if not levels:
            return []
        consolidated = []
        sorted_levels = sorted(levels, key=lambda x: x['price'])
        i = 0
        while i < len(sorted_levels):
            group = [sorted_levels[i]]
            base_price = sorted_levels[i]['price']
            j = i + 1
            while j < len(sorted_levels):
                if abs(sorted_levels[j]['price'] - base_price) / base_price < tolerance:
                    group.append(sorted_levels[j])
                    j += 1
                else:
                    break
            total_weight = sum(level['strength'] * level['volume'] for level in group)
            weighted_price = sum(level['price'] * level['strength'] * level['volume'] for level in group) / total_weight if total_weight > 0 else base_price
            consolidated.append({
                'price': weighted_price,
                'strength': sum(level['strength'] for level in group),
                'volume': sum(level['volume'] for level in group),
                'touch_count': len(group),
                'latest_touch': max(level['timestamp'] for level in group)
            })
            i = j
        consolidated.sort(key=lambda x: x['touch_count'] * x['strength'] * x['volume'], reverse=True)
        return consolidated[:10]
    support_levels = consolidate_levels(swing_lows)
    resistance_levels = consolidate_levels(swing_highs)
    return support_levels, resistance_levels

# ---------------- Multi TF analysis ----------------
def analyze_multi_timeframe_720(candles_4h, candles_1h):
    try:
        analysis = {'trends': {}, 'momentum': {}, 'key_levels': {}, 'patterns': {}, 'volume': {}, 'strength': 0, 'signals': []}
        for tf_name, candles in [("4H", candles_4h), ("1H", candles_1h)]:
            if len(candles) < 100:
                continue
            closes = [c['close'] for c in candles]
            trend_periods = [20, 50, 100, 200]
            trends = {}
            for period in trend_periods:
                if len(closes) >= period:
                    early_avg = sum(closes[-period:-period//2]) / (period//2)
                    recent_avg = sum(closes[-period//2:]) / (period//2)
                    if recent_avg > early_avg * 1.02:
                        trends[f'{period}'] = 'BULLISH'
                    elif recent_avg < early_avg * 0.98:
                        trends[f'{period}'] = 'BEARISH'
                    else:
                        trends[f'{period}'] = 'NEUTRAL'
            analysis['trends'][tf_name] = trends
            momentum_scores = []
            for i in range(1, min(21, len(candles))):
                prev_close = candles[-i-1]['close']
                curr_close = candles[-i]['close']
                momentum_scores.append((curr_close - prev_close) / prev_close * 100)
            analysis['momentum'][tf_name] = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0
            support_levels, resistance_levels = identify_key_levels_advanced(candles)
            analysis['key_levels'][tf_name] = {'support': support_levels, 'resistance': resistance_levels}
            patterns = detect_all_candlestick_patterns(candles)
            analysis['patterns'][tf_name] = patterns
            volumes = [c['volume'] for c in candles[-50:]] if len(candles) >= 1 else []
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            recent_volume = volumes[-1] if volumes else 0
            volume_trend = []
            for i in range(1, min(11, len(volumes))):
                if volumes[-i-1] == 0:
                    continue
                volume_trend.append(volumes[-i] / volumes[-i-1])
            analysis['volume'][tf_name] = {'current_vs_avg': (recent_volume / avg_volume) if avg_volume else 1,
                                           'trend': sum(volume_trend) / len(volume_trend) if volume_trend else 1,
                                           'spike': recent_volume > avg_volume * 2 if avg_volume else False}
        # strength scoring
        strength = 0
        for tf in analysis['trends']:
            for period in analysis['trends'][tf]:
                if analysis['trends'][tf][period] == 'BULLISH':
                    strength += 1
                elif analysis['trends'][tf][period] == 'BEARISH':
                    strength -= 1
        for tf in analysis['momentum']:
            if analysis['momentum'][tf] > 1:
                strength += 2
            elif analysis['momentum'][tf] < -1:
                strength -= 2
        bullish_patterns = ['Hammer', 'Bullish_Engulfing', 'Morning_Star', 'Bullish_Harami', 'Piercing_Line']
        bearish_patterns = ['Hanging_Man', 'Bearish_Engulfing', 'Evening_Star', 'Bearish_Harami', 'Dark_Cloud_Cover']
        for tf in analysis['patterns']:
            for pattern in analysis['patterns'][tf]:
                if any(bp in pattern for bp in bullish_patterns):
                    strength += 3
                elif any(bp in pattern for bp in bearish_patterns):
                    strength -= 3
        analysis['strength'] = strength
        current_price = candles_1h[-1]['close'] if candles_1h else 0
        all_support = []
        all_resistance = []
        for tf in analysis['key_levels']:
            all_support.extend([s['price'] for s in analysis['key_levels'][tf]['support']])
            all_resistance.extend([r['price'] for r in analysis['key_levels'][tf]['resistance']])
        nearest_support = max([s for s in all_support if s < current_price], default=None) if all_support else None
        nearest_resistance = min([r for r in all_resistance if r > current_price], default=None) if all_resistance else None
        if strength >= 8 and nearest_support and nearest_resistance:
            entry = current_price
            sl = nearest_support * 0.995
            tp = nearest_resistance * 0.995
            if (tp - entry) > 0 and (entry - sl) > 0:
                rr = (tp - entry) / (entry - sl)
                if rr >= MIN_RR_RATIO:
                    analysis['signals'].append({'side': 'BUY', 'entry': entry, 'sl': sl, 'tp': tp, 'rr': rr, 'confidence': min(95, 70 + strength)})
        elif strength <= -8 and nearest_support and nearest_resistance:
            entry = current_price
            sl = nearest_resistance * 1.005
            tp = nearest_support * 1.005
            if (entry - tp) > 0 and (sl - entry) > 0:
                rr = (entry - tp) / (sl - entry)
                if rr >= MIN_RR_RATIO:
                    analysis['signals'].append({'side': 'SELL', 'entry': entry, 'sl': sl, 'tp': tp, 'rr': rr, 'confidence': min(95, 70 + abs(strength))})
        return analysis
    except Exception as e:
        print(f"Multi-timeframe analysis error: {e}")
        traceback.print_exc()
        return {'signals': [], 'strength': 0, 'trends': {}, 'patterns': {}}

# ---------------- AI integration (safe) ----------------
async def get_comprehensive_ai_analysis(symbol, analysis_data):
    if not client:
        return None
    context = {
        "symbol": symbol,
        "timeframe_analysis": "4H + 1H with 720 candles each",
        "multi_timeframe_trends": analysis_data.get('trends', {}),
        "momentum_analysis": analysis_data.get('momentum', {}),
        "key_levels": {tf: {'support_count': len(levels.get('support', [])),
                            'resistance_count': len(levels.get('resistance', [])),
                            'strongest_support': levels.get('support', [{}])[0].get('price') if levels.get('support') else None,
                            'strongest_resistance': levels.get('resistance', [{}])[0].get('price') if levels.get('resistance') else None}
                       for tf, levels in analysis_data.get('key_levels', {}).items()},
        "candlestick_patterns": analysis_data.get('patterns', {}),
        "volume_analysis": analysis_data.get('volume', {}),
        "overall_strength_score": analysis_data.get('strength', 0),
        "proposed_signals": analysis_data.get('signals', [])
    }
    system_prompt = ("You are an elite crypto trading expert with 15+ years of experience. "
                     "Analyze the market data and give your opinion: CONFIRM/REJECT - CONFIDENCE:XX% - KEY_REASONS short.")
    user_prompt = f"Market summary for {symbol}:\n{json.dumps(context, indent=2)}\nRespond: CONFIRM/REJECT - CONFIDENCE:XX% - KEY_REASONS: short"
    try:
        loop = asyncio.get_running_loop()
        def call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=200,
                temperature=0.2
            )
        resp = await loop.run_in_executor(None, call)
        if hasattr(resp, "choices") and len(resp.choices) > 0:
            return getattr(resp.choices[0].message, "content", None) or getattr(resp.choices[0], "text", None)
        if isinstance(resp, dict) and resp.get("choices"):
            ch = resp["choices"][0]
            return (ch.get("message") or {}).get("content") or ch.get("text")
        return str(resp)
    except Exception as e:
        print(f"AI analysis error: {e}")
        return None

def parse_ai_response(text):
    if not text:
        return {'decision': None, 'confidence': None}
    up = text.upper()
    decision = None
    if 'CONFIRM' in up:
        decision = 'CONFIRM'
    elif 'REJECT' in up:
        decision = 'REJECT'
    import re
    m = re.search(r'(\d{1,3})\s*%', text)
    confidence = int(m.group(1)) if m else None
    return {'decision': decision, 'confidence': confidence}

# ---------------- Redis helpers ----------------
def save_signal_to_redis(symbol, signal_data):
    if not redis_client:
        return False
    try:
        signal_id = f"signal:{symbol}:{int(datetime.utcnow().timestamp())}"
        now_iso = datetime.utcnow().isoformat()
        store = dict(signal_data)
        if 'signals' in store:
            store['signals'] = json.dumps(store['signals'])
        if 'progress_alerts_sent' in store:
            pas = store.get('progress_alerts_sent') or []
            if isinstance(pas, (list, tuple)):
                store['progress_alerts_sent'] = ",".join(map(str, pas))
            else:
                store['progress_alerts_sent'] = str(pas)
        store['signal_id'] = signal_id
        store['timestamp'] = now_iso
        store['status'] = 'ACTIVE'
        mapping = {k: (v if isinstance(v, str) else str(v)) for k, v in store.items()}
        redis_client.hset(signal_id, mapping=mapping)
        redis_client.expire(signal_id, 7 * 24 * 3600)  # 7 days for signals
        redis_client.sadd("active_signals", signal_id)
        print(f"✅ Signal saved to Redis: {signal_id}")
        return signal_id
    except Exception as e:
        print(f"❌ Redis save error: {e}")
        return False

def get_active_signals():
    if not redis_client:
        return []
    try:
        signal_ids = redis_client.smembers("active_signals") or set()
        signals = []
        for sid in signal_ids:
            data = redis_client.hgetall(sid)
            if not data:
                continue
            if data.get('status') != 'ACTIVE':
                continue
            if 'signals' in data:
                try:
                    data['signals'] = json.loads(data['signals'])
                except Exception:
                    pass
            pas = data.get('progress_alerts_sent', '')
            if pas:
                data['progress_alerts_sent'] = [x for x in pas.split(',') if x != '']
            else:
                data['progress_alerts_sent'] = []
            signals.append(data)
        return signals
    except Exception as e:
        print(f"❌ Redis get error: {e}")
        return []

def update_signal_status(signal_id, status, reason=None):
    if not redis_client:
        return False
    try:
        redis_client.hset(signal_id, 'status', status)
        if reason:
            redis_client.hset(signal_id, 'exit_reason', reason)
        redis_client.hset(signal_id, 'exit_time', datetime.utcnow().isoformat())
        if status in ['HIT_TP', 'HIT_SL', 'TREND_REVERSED']:
            redis_client.srem("active_signals", signal_id)
        return True
    except Exception as e:
        print(f"❌ Redis update error: {e}")
        return False

# ---------------- Telegram utilities ----------------
async def send_telegram_message(session, text, parse_mode="Markdown"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured; message:", text[:200])
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": parse_mode}
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print("Telegram send error:", e)
        return False

async def send_signal_alert(session, symbol, signal_data, ai_analysis, analysis_summary):
    try:
        signal = signal_data['signals'][0] if isinstance(signal_data.get('signals'), list) and signal_data['signals'] else signal_data.get('signals')
        if not signal:
            print("No signal to alert for", symbol)
            return False
        side_emoji = "🟢" if signal['side'] == 'BUY' else "🔴"
        patterns_4h = analysis_summary.get('patterns', {}).get('4H', [])
        patterns_1h = analysis_summary.get('patterns', {}).get('1H', [])
        message = f"""{side_emoji} CRYPTO SIGNAL - {symbol} {side_emoji}

📊 720 Candles Analysis Complete
🕐 Timeframes: 4H + 1H (720 candles each)

TRADE DETAILS:
• Side: {signal['side']}
• Entry: {fmt_price(signal['entry'])}
• Stop Loss: {fmt_price(signal['sl'])}
• Take Profit: {fmt_price(signal['tp'])}
• Risk/Reward: 1:{signal.get('rr', 0)}
• Confidence: {signal.get('confidence', 0)}%

TECHNICAL:
• Strength Score: {analysis_summary.get('strength', 0)}
• 4H Patterns: {', '.join(patterns_4h[:3]) if patterns_4h else 'None'}
• 1H Patterns: {', '.join(patterns_1h[:3]) if patterns_1h else 'None'}

AI CONFIRMATION:
{ai_analysis if ai_analysis else 'AI not available'}

Signal ID: {signal_data.get('signal_id', 'N/A')}
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print("Signal alert error:", e)
        return False

# ---------------- Monitoring functions ----------------
async def send_signal_update(session, signal, current_price, update_type):
    try:
        symbol = signal.get('symbol')
        side = signal.get('side')
        entry = float(signal.get('entry', 0))
        if update_type == "STOP_LOSS_HIT":
            emoji = "🛑"; title = "STOP LOSS HIT"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        elif update_type == "TARGET_HIT":
            emoji = "🎯"; title = "TARGET ACHIEVED"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        else:
            emoji = "🔄"; title = "TREND REVERSAL DETECTED"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        message = f"""{emoji} {title} - {symbol} {emoji}

SIGNAL CLOSED:
• Side: {side}
• Entry: {fmt_price(entry)}
• Exit: {fmt_price(current_price)}
• P&L: {pnl:+.2f}%

Signal ID: {signal.get('signal_id', 'N/A')}
Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print("Signal update error:", e)
        return False

async def send_progress_alert(session, signal, current_price, progress_level):
    try:
        symbol = signal.get('symbol'); side = signal.get('side')
        entry = float(signal.get('entry', 0)); tp = float(signal.get('tp', 0))
        progress_percent = int(progress_level * 100)
        pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        message = f"""🚀 TARGET PROGRESS - {symbol}

{progress_percent}% TARGET REACHED
• Side: {side}
• Entry: {fmt_price(entry)}
• Current: {fmt_price(current_price)}
• Target: {fmt_price(tp)}
• Current P&L: {pnl:+.2f}%

Signal ID: {signal.get('signal_id', 'N/A')}
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print("Progress alert error:", e)
        return False

async def check_trend_reversal(session, signal, candles_1h):
    try:
        if len(candles_1h) < 20:
            return False
        symbol = signal.get('symbol'); side = signal.get('side'); signal_id = signal.get('signal_id')
        recent = candles_1h[-20:]; closes = [c['close'] for c in recent]
        early_avg = sum(closes[:10]) / 10; recent_avg = sum(closes[-10:]) / 10
        price_change = (recent_avg - early_avg) / early_avg
        reversal = False
        if side == 'BUY' and price_change < -TREND_REVERSAL_THRESHOLD:
            reversal = True
        elif side == 'SELL' and price_change > TREND_REVERSAL_THRESHOLD:
            reversal = True
        if reversal:
            current_price = closes[-1]
            await send_signal_update(session, signal, current_price, "TREND_REVERSED")
            update_signal_status(signal_id, 'TREND_REVERSED', f'Trend reversal detected: {price_change:.2%}')
            return True
        return False
    except Exception as e:
        print("Trend reversal check error:", e)
        return False

async def monitor_active_signals(session):
    active_signals = get_active_signals()
    if not active_signals:
        return
    print(f"🔍 Monitoring {len(active_signals)} active signals...")
    for signal in active_signals:
        try:
            symbol = signal.get('symbol'); signal_id = signal.get('signal_id')
            side = signal.get('side'); entry = float(signal.get('entry', 0)); sl = float(signal.get('sl', 0)); tp = float(signal.get('tp', 0))
            current = await fetch_json(session, TICKER_PRICE_URL.format(symbol=symbol))
            if not current:
                continue
            current_price = float(current['price'])
            if side == 'BUY':
                if current_price <= sl:
                    await send_signal_update(session, signal, current_price, "STOP_LOSS_HIT")
                    update_signal_status(signal_id, 'HIT_SL', 'Stop loss triggered'); continue
                elif current_price >= tp:
                    await send_signal_update(session, signal, current_price, "TARGET_HIT")
                    update_signal_status(signal_id, 'HIT_TP', 'Target achieved'); continue
                progress = (current_price - entry) / (tp - entry) if (tp - entry) != 0 else 0
            else:
                if current_price >= sl:
                    await send_signal_update(session, signal, current_price, "STOP_LOSS_HIT")
                    update_signal_status(signal_id, 'HIT_SL', 'Stop loss triggered'); continue
                elif current_price <= tp:
                    await send_signal_update(session, signal, current_price, "TARGET_HIT")
                    update_signal_status(signal_id, 'HIT_TP', 'Target achieved'); continue
                progress = (entry - current_price) / (entry - tp) if (entry - tp) != 0 else 0
            pas = signal.get('progress_alerts_sent', [])
            if isinstance(pas, str):
                pas = [x for x in pas.split(',') if x != '']
            if not isinstance(pas, list):
                pas = list(pas)
            for level in TARGET_PROGRESS_LEVELS:
                if progress >= level and str(level) not in pas:
                    await send_progress_alert(session, signal, current_price, level)
                    pas.append(str(level))
                    if redis_client:
                        redis_client.hset(signal_id, 'progress_alerts_sent', ','.join(pas))
            candles_1h_raw = await fetch_json(session, CANDLE_1H_URL.format(symbol=symbol))
            if candles_1h_raw:
                candles_1h = normalize_klines(candles_1h_raw)
                await check_trend_reversal(session, signal, candles_1h)
        except Exception as e:
            print(f"Signal monitoring error for {signal.get('symbol', 'unknown')}: {e}")
            traceback.print_exc()
            continue

# ---------------- Main analysis for a symbol ----------------
async def analyze_symbol_720_candles(session, symbol):
    try:
        print(f"🔍 Analyzing {symbol} with 720 candles...")
        c4h_raw = await fetch_candles_with_cache(session, symbol, "4h")
        c1h_raw = await fetch_candles_with_cache(session, symbol, "1h")
        if not c4h_raw or not c1h_raw:
            print(f"❌ Failed to fetch candle data for {symbol}")
            return
        candles_4h = normalize_klines(c4h_raw)
        candles_1h = normalize_klines(c1h_raw)
        if len(candles_4h) < MIN_CANDLES_REQUIRED or len(candles_1h) < MIN_CANDLES_REQUIRED:
            print(f"❌ Insufficient candle data for {symbol}")
            return
        print(f"📊 {symbol}: 4H={len(candles_4h)} candles, 1H={len(candles_1h)} candles")
        analysis_result = analyze_multi_timeframe_720(candles_4h, candles_1h)
        if not analysis_result['signals']:
            print(f"📈 {symbol}: No signals generated (Strength: {analysis_result.get('strength', 0)})")
            return
        ai_analysis = await get_comprehensive_ai_analysis(symbol, analysis_result)
        parsed_ai = parse_ai_response(ai_analysis) if ai_analysis else {'decision': None, 'confidence': None}
        ai_decision = parsed_ai['decision']; ai_conf = parsed_ai['confidence'] or 0
        signal_obj = analysis_result['signals'][0]
        orig_conf = float(signal_obj.get('confidence', 0))
        # compute ai boost
        ai_boost = 0
        if ai_decision == 'CONFIRM':
            ai_boost = int(ai_conf / 10)
        elif ai_decision == 'REJECT':
            ai_boost = -int(max(ai_conf, 50) / 10)
        final_conf = min(95, max(0, int(orig_conf + ai_boost)))
        print(f"AI decision: {ai_decision} ({ai_conf}%) -> ai_boost {ai_boost} => final_conf {final_conf}% (orig {orig_conf}%)")
        if final_conf < SIGNAL_CONF_THRESHOLD:
            print(f"📉 {symbol}: Final confidence {final_conf}% below threshold {SIGNAL_CONF_THRESHOLD}% — skipping")
            return
        # prepare signal dict for redis/telegram
        signal_data = {
            'symbol': symbol,
            'side': signal_obj['side'],
            'entry': str(signal_obj['entry']),
            'sl': str(signal_obj['sl']),
            'tp': str(signal_obj['tp']),
            'rr': str(signal_obj['rr']),
            'confidence': str(final_conf),
            'ai_analysis': ai_analysis or 'Not available',
            'strength_score': str(analysis_result.get('strength', 0)),
            'patterns_4h': ','.join(analysis_result.get('patterns', {}).get('4H', [])),
            'patterns_1h': ','.join(analysis_result.get('patterns', {}).get('1H', [])),
            'signals': analysis_result['signals'],
            'progress_alerts_sent': []
        }
        saved_id = save_signal_to_redis(symbol, signal_data)
        if saved_id:
            signal_data['signal_id'] = saved_id
            await send_signal_alert(session, symbol, signal_data, ai_analysis, analysis_result)
            print(f"🚀 {symbol}: Signal sent and saved for tracking!")
        else:
            print(f"❌ {symbol}: Failed to save signal to Redis")
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        traceback.print_exc()

# ---------------- Main loop ----------------
async def main():
    print("🚀 Advanced 720 Candles Crypto Trading Bot Started!")
    print(f"📊 Analyzing {len(SYMBOLS)} symbols every {POLL_INTERVAL} seconds")
    print(f"🤖 AI Model: {OPENAI_MODEL}")
    print(f"📈 Min Confidence: {SIGNAL_CONF_THRESHOLD}%")
    print(f"⚡ Redis: {'Connected' if redis_client else 'Not available'}")
    async with aiohttp.ClientSession() as session:
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            await send_telegram_message(session, f"🟢 Bot started • Symbols: {len(SYMBOLS)} • Candles TTL: 10 days")
    while True:
        try:
            print("\n" + "="*60)
            print(f"🕐 Starting analysis cycle - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print("="*60)
            async with aiohttp.ClientSession() as session:
                # monitor existing signals first
                await monitor_active_signals(session)
                # analyze symbols concurrently (HTTP semaphore will limit concurrency)
                tasks = [analyze_symbol_720_candles(session, symbol) for symbol in SYMBOLS]
                await asyncio.gather(*tasks, return_exceptions=True)
            print(f"\n✅ Analysis cycle completed. Next scan in {POLL_INTERVAL} seconds...")
            await asyncio.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
            break
        except Exception as e:
            print(f"❌ Main loop error: {e}")
            traceback.print_exc()
            await asyncio.sleep(60)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        sys.exit(1)
