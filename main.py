#!/usr/bin/env python3
# Advanced Crypto Trading Bot - Complete Signal Management System (Patched)
# Features: 720 candles analysis, All patterns, Redis tracking, Trade monitoring
# Patched: Redis serialization, concurrency control (semaphore), safer fetch timeouts,
# returns signal_id from save, progress alerts handling, and other robustness fixes.

import os
import asyncio
import aiohttp
import traceback
import numpy as np
import json
import redis
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
import hashlib

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]

POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 85.0))

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Use 720 candles as requested
CANDLE_4H_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=4h&limit=720"
CANDLE_1H_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"

MIN_CANDLES_REQUIRED = 720
MIN_RR_RATIO = 2.0
TREND_REVERSAL_THRESHOLD = 0.02  # 2% price change for trend reversal alert
TARGET_PROGRESS_LEVELS = [0.25, 0.50, 0.75, 0.90]  # 25%, 50%, 75%, 90% target hit alerts

# Concurrency control
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 6))
HTTP_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Initialize Redis
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

# ---------------- Redis Helper Functions ----------------
def save_signal_to_redis(symbol, signal_data):
    """Save signal to Redis for tracking (serializes complex fields).
    Returns the generated signal_id on success, otherwise False.
    """
    if not redis_client:
        return False

    try:
        signal_id = f"signal:{symbol}:{int(datetime.utcnow().timestamp())}"
        now_iso = datetime.utcnow().isoformat()
        # shallow copy to avoid mutating caller data
        store = dict(signal_data)

        # Serialize potentially complex fields
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

        # Ensure mapping values are strings
        mapping = {k: (v if isinstance(v, str) else str(v)) for k, v in store.items()}

        redis_client.hset(signal_id, mapping=mapping)
        redis_client.expire(signal_id, 604800)  # 7 days expiry

        # Add to active signals set
        redis_client.sadd("active_signals", signal_id)

        print(f"✅ Signal saved to Redis: {signal_id}")
        return signal_id
    except Exception as e:
        print(f"❌ Redis save error: {e}")
        return False


def get_active_signals():
    """Get all active signals from Redis (deserializes complex fields)."""
    if not redis_client:
        return []

    try:
        signal_ids = redis_client.smembers("active_signals") or set()
        signals = []

        for signal_id in signal_ids:
            signal_data = redis_client.hgetall(signal_id)
            if not signal_data:
                continue
            if signal_data.get('status') != 'ACTIVE':
                continue

            # Deserialize fields we previously serialized
            if 'signals' in signal_data:
                try:
                    signal_data['signals'] = json.loads(signal_data['signals'])
                except Exception:
                    pass

            # progress_alerts_sent stored as CSV string
            pas = signal_data.get('progress_alerts_sent', '')
            if pas:
                signal_data['progress_alerts_sent'] = [x for x in pas.split(',') if x != '']
            else:
                signal_data['progress_alerts_sent'] = []

            signals.append(signal_data)

        return signals
    except Exception as e:
        print(f"❌ Redis get error: {e}")
        return []


def update_signal_status(signal_id, status, reason=None):
    """Update signal status in Redis"""
    if not redis_client:
        return False

    try:
        redis_client.hset(signal_id, 'status', status)
        if reason:
            redis_client.hset(signal_id, 'exit_reason', reason)
        redis_client.hset(signal_id, 'exit_time', datetime.utcnow().isoformat())

        # Remove from active signals if closed
        if status in ['HIT_TP', 'HIT_SL', 'TREND_REVERSED']:
            redis_client.srem("active_signals", signal_id)

        return True
    except Exception as e:
        print(f"❌ Redis update error: {e}")
        return False

# ---------------- Utils ----------------
def fmt_price(p):
    try:
        p = float(p)
    except Exception:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"


def normalize_klines(raw_klines):
    """Convert raw klines to standardized format with pattern analysis data"""
    out = []
    for row in raw_klines or []:
        try:
            if len(row) >= 6:
                ts = int(row[0])
                o, h, l, c, v = float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5])

                body_size = abs(c - o)
                total_range = h - l if h - l != 0 else 0.0
                upper_wick = h - max(o, c)
                lower_wick = min(o, c) - l

                out.append({
                    "open": o, "high": h, "low": l, "close": c,
                    "volume": v, "ts": ts,
                    "body_size": body_size,
                    "total_range": total_range,
                    "upper_wick": upper_wick,
                    "lower_wick": lower_wick,
                    "is_bullish": c > o,
                    "body_ratio": (body_size / total_range) if total_range > 0 else 0,
                    "upper_wick_ratio": (upper_wick / total_range) if total_range > 0 else 0,
                    "lower_wick_ratio": (lower_wick / total_range) if total_range > 0 else 0
                })
        except Exception:
            continue
    return out

# ---------------- Advanced Pattern Recognition ----------------
# (Keep your original detection functions; they were OK. We'll include them unchanged.)

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


def detect_single_candle_patterns(candle):
    patterns = []
    c = candle
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
    elif abs(c1['high'] - c2['high']) / c1['high'] < 0.002:
        patterns.append("Tweezer_Top")
    elif abs(c1['low'] - c2['low']) / c1['low'] < 0.002:
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

# ---------------- Support/Resistance Detection ----------------
def identify_key_levels_advanced(candles, lookback=200):
    if len(candles) < lookback:
        lookback = len(candles)
    recent_candles = candles[-lookback:]
    swing_highs = []
    swing_lows = []
    for period in [3, 5, 8, 13, 21]:
        for i in range(period, len(recent_candles) - period):
            candle = recent_candles[i]
            if all(candle['high'] >= recent_candles[j]['high'] 
                  for j in range(i-period, i+period+1) if j != i):
                swing_highs.append({
                    'price': candle['high'],
                    'strength': period,
                    'volume': candle['volume'],
                    'timestamp': candle['ts']
                })
            if all(candle['low'] <= recent_candles[j]['low'] 
                  for j in range(i-period, i+period+1) if j != i):
                swing_lows.append({
                    'price': candle['low'],
                    'strength': period,
                    'volume': candle['volume'],
                    'timestamp': candle['ts']
                })
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
            weighted_price = sum(level['price'] * level['strength'] * level['volume'] 
                               for level in group) / total_weight if total_weight > 0 else base_price
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

# ---------------- Multi-Timeframe Analysis ----------------
def analyze_multi_timeframe_720(candles_4h, candles_1h):
    try:
        analysis = {
            'trends': {},
            'momentum': {},
            'key_levels': {},
            'patterns': {},
            'volume': {},
            'strength': 0,
            'signals': []
        }
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
            avg_momentum = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 0
            analysis['momentum'][tf_name] = avg_momentum
            support_levels, resistance_levels = identify_key_levels_advanced(candles)
            analysis['key_levels'][tf_name] = {
                'support': support_levels,
                'resistance': resistance_levels
            }
            patterns = detect_all_candlestick_patterns(candles)
            analysis['patterns'][tf_name'] = patterns
            volumes = [c['volume'] for c in candles[-50:]]
            avg_volume = sum(volumes) / len(volumes) if volumes else 1
            recent_volume = volumes[-1] if volumes else 0
            volume_trend = []
            for i in range(1, min(11, len(volumes))):
                if volumes[-i-1] == 0:
                    continue
                volume_trend.append(volumes[-i] / volumes[-i-1])
            analysis['volume'][tf_name] = {
                'current_vs_avg': (recent_volume / avg_volume) if avg_volume else 1,
                'trend': sum(volume_trend) / len(volume_trend) if volume_trend else 1,
                'spike': recent_volume > avg_volume * 2 if avg_volume else False
            }
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
                    analysis['signals'].append({
                        'side': 'BUY',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confidence': min(95, 70 + strength)
                    })
        elif strength <= -8 and nearest_support and nearest_resistance:
            entry = current_price
            sl = nearest_resistance * 1.005
            tp = nearest_support * 1.005
            if (entry - tp) > 0 and (sl - entry) > 0:
                rr = (entry - tp) / (sl - entry)
                if rr >= MIN_RR_RATIO:
                    analysis['signals'].append({
                        'side': 'SELL',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'rr': rr,
                        'confidence': min(95, 70 + abs(strength))
                    })
        return analysis
    except Exception as e:
        print(f"Multi-timeframe analysis error: {e}")
        traceback.print_exc()
        return {'signals': [], 'strength': 0, 'trends': {}, 'patterns': {}}

# ---------------- AI Analysis ----------------
async def get_comprehensive_ai_analysis(symbol, analysis_data):
    if not client:
        return None
    context = {
        "symbol": symbol,
        "timeframe_analysis": "4H + 1H with 720 candles each",
        "multi_timeframe_trends": analysis_data.get('trends', {}),
        "momentum_analysis": analysis_data.get('momentum', {}),
        "key_levels": {
            tf: {
                'support_count': len(levels.get('support', [])),
                'resistance_count': len(levels.get('resistance', [])),
                'strongest_support': levels.get('support', [{}])[0].get('price') if levels.get('support') else None,
                'strongest_resistance': levels.get('resistance', [{}])[0].get('price') if levels.get('resistance') else None
            } for tf, levels in analysis_data.get('key_levels', {}).items()
        },
        "candlestick_patterns": analysis_data.get('patterns', {}),
        "volume_analysis": analysis_data.get('volume', {}),
        "overall_strength_score": analysis_data.get('strength', 0),
        "proposed_signals": analysis_data.get('signals', [])
    }
    system_prompt = (
        "You are an elite crypto trading expert with 15+ years of experience. "
        "Analyze the comprehensive market data provided and give your expert opinion on the trade signals. "
        "Consider multi-timeframe alignment, pattern confluence, volume confirmation, and key level interactions. "
        "Be highly selective - only confirm trades with exceptional probability of success. "
        "Respond with: CONFIRM/REJECT - CONFIDENCE:XX% - KEY_REASONS: brief analysis"
    )
    user_prompt = f"""
COMPREHENSIVE MARKET ANALYSIS FOR {symbol}:

{json.dumps(context, indent=2)}

Based on this deep analysis using 720 candles on both 4H and 1H timeframes, 
provide your expert assessment of the trade signals.

Focus on:
1. Multi-timeframe trend alignment
2. Pattern confluence and strength  
3. Volume confirmation
4. Key level interaction
5. Overall probability of success

Response format: CONFIRM/REJECT - CONFIDENCE:XX% - KEY_REASONS: your analysis
"""
    try:
        loop = asyncio.get_running_loop()
        def call():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=250,
                temperature=0.1
            )
        resp = await loop.run_in_executor(None, call)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"AI analysis error: {e}")
        return None

# ---------------- Fetch Functions ----------------
async def fetch_json(session, url, retries=3, timeout_seconds=60):
    """Fetch JSON data with retries and concurrency control."""
    for i in range(retries):
        try:
            async with HTTP_SEMAPHORE:
                async with session.get(url, timeout=timeout_seconds) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        print(f"HTTP {resp.status} for {url}")
        except Exception as e:
            print(f"Fetch error (attempt {i+1}) for {url}: {e}")
            if i < retries - 1:
                await asyncio.sleep(2)
    return None

# ---------------- Telegram Functions ----------------
async def send_telegram_message(session, text, parse_mode="Markdown"):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"Telegram not configured. Message: {text}")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": parse_mode
    }
    try:
        async with session.post(url, json=payload, timeout=10) as resp:
            return resp.status == 200
    except Exception as e:
        print(f"Telegram send error: {e}")
        return False


async def send_signal_alert(session, symbol, signal_data, ai_analysis, analysis_summary):
    try:
        signal = signal_data['signals'][0] if isinstance(signal_data.get('signals'), list) and len(signal_data['signals'])>0 else signal_data.get('signals')
        side_emoji = "🟢" if signal['side'] == 'BUY' else "🔴"
        patterns_4h = analysis_summary.get('patterns', {}).get('4H', [])
        patterns_1h = analysis_summary.get('patterns', {}).get('1H', [])
        message = f"""
{side_emoji} **CRYPTO SIGNAL - {symbol}** {side_emoji}

📊 **720 Candles Analysis Complete**
🕐 **Timeframes:** 4H + 1H (720 candles each)

**TRADE DETAILS:**
• **Side:** {signal['side']}
• **Entry:** ${fmt_price(signal['entry'])}
• **Stop Loss:** ${fmt_price(signal['sl'])}
• **Take Profit:** ${fmt_price(signal['tp'])}
• **Risk/Reward:** {signal['rr']:.2f}
• **Confidence:** {signal['confidence']:.1f}%

**TECHNICAL ANALYSIS:**
• **Strength Score:** {analysis_summary.get('strength', 0)}/20
• **4H Patterns:** {', '.join(patterns_4h[:3]) if patterns_4h else 'None'}
• **1H Patterns:** {', '.join(patterns_1h[:3]) if patterns_1h else 'None'}

**AI CONFIRMATION:**
{ai_analysis if ai_analysis else 'AI analysis not available'}

**MULTI-TIMEFRAME TRENDS:**
• **4H Trend:** {analysis_summary.get('trends', {}).get('4H', {}).get('50', 'N/A')}
• **1H Trend:** {analysis_summary.get('trends', {}).get('1H', {}).get('50', 'N/A')}

⚠️ **Risk Management Required**
🎯 **Target Progress Alerts:** 25%, 50%, 75%, 90%

*Signal ID: {signal_data.get('signal_id', 'N/A')}*
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print(f"Signal alert error: {e}")
        return False

# ---------------- Signal Monitoring Functions ----------------
async def send_signal_update(session, signal, current_price, update_type):
    try:
        symbol = signal.get('symbol')
        side = signal.get('side')
        entry = float(signal.get('entry', 0))
        if update_type == "STOP_LOSS_HIT":
            emoji = "🛑"
            title = "STOP LOSS HIT"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        elif update_type == "TARGET_HIT":
            emoji = "🎯"
            title = "TARGET ACHIEVED"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        elif update_type == "TREND_REVERSED":
            emoji = "🔄"
            title = "TREND REVERSAL DETECTED"
            pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        message = f"""
{emoji} **{title} - {symbol}** {emoji}

**SIGNAL CLOSED:**
• **Side:** {side}
• **Entry:** ${fmt_price(entry)}
• **Exit:** ${fmt_price(current_price)}
• **P&L:** {pnl:+.2f}%

**Signal ID:** {signal.get('signal_id', 'N/A')}
**Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print(f"Signal update error: {e}")
        return False


async def send_progress_alert(session, signal, current_price, progress_level):
    try:
        symbol = signal.get('symbol')
        side = signal.get('side')
        entry = float(signal.get('entry', 0))
        tp = float(signal.get('tp', 0))
        progress_percent = int(progress_level * 100)
        pnl = ((current_price - entry) / entry * 100) if side == 'BUY' else ((entry - current_price) / entry * 100)
        message = f"""
🚀 **TARGET PROGRESS - {symbol}** 🚀

**{progress_percent}% TARGET REACHED**

• **Side:** {side}
• **Entry:** ${fmt_price(entry)}
• **Current:** ${fmt_price(current_price)}
• **Target:** ${fmt_price(tp)}
• **Current P&L:** {pnl:+.2f}%

**Progress:** {progress_percent}% ✅

**Signal ID:** {signal.get('signal_id', 'N/A')}
"""
        return await send_telegram_message(session, message)
    except Exception as e:
        print(f"Progress alert error: {e}")
        return False


async def check_trend_reversal(session, signal, candles_1h):
    try:
        if len(candles_1h) < 20:
            return False
        symbol = signal.get('symbol')
        side = signal.get('side')
        signal_id = signal.get('signal_id')
        recent_candles = candles_1h[-20:]
        closes = [c['close'] for c in recent_candles]
        early_avg = sum(closes[:10]) / 10
        recent_avg = sum(closes[-10:]) / 10
        price_change = (recent_avg - early_avg) / early_avg
        reversal_detected = False
        if side == 'BUY' and price_change < -TREND_REVERSAL_THRESHOLD:
            reversal_detected = True
        elif side == 'SELL' and price_change > TREND_REVERSAL_THRESHOLD:
            reversal_detected = True
        if reversal_detected:
            current_price = closes[-1]
            await send_signal_update(session, signal, current_price, "TREND_REVERSED")
            update_signal_status(signal_id, 'TREND_REVERSED', f'Trend reversal detected: {price_change:.2%}')
            return True
        return False
    except Exception as e:
        print(f"Trend reversal check error: {e}")
        return False

# ---------------- Signal Monitoring ----------------
async def monitor_active_signals(session):
    active_signals = get_active_signals()
    if not active_signals:
        return
    print(f"🔍 Monitoring {len(active_signals)} active signals...")
    for signal in active_signals:
        try:
            symbol = signal.get('symbol')
            signal_id = signal.get('signal_id')
            side = signal.get('side')
            entry = float(signal.get('entry', 0))
            sl = float(signal.get('sl', 0))
            tp = float(signal.get('tp', 0))
            current_data = await fetch_json(session, f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}")
            if not current_data:
                continue
            current_price = float(current_data['price'])
            if side == 'BUY':
                if current_price <= sl:
                    await send_signal_update(session, signal, current_price, "STOP_LOSS_HIT")
                    update_signal_status(signal_id, 'HIT_SL', 'Stop loss triggered')
                    continue
                elif current_price >= tp:
                    await send_signal_update(session, signal, current_price, "TARGET_HIT")
                    update_signal_status(signal_id, 'HIT_TP', 'Target achieved')
                    continue
                progress = (current_price - entry) / (tp - entry) if (tp - entry) != 0 else 0
            else:
                if current_price >= sl:
                    await send_signal_update(session, signal, current_price, "STOP_LOSS_HIT")
                    update_signal_status(signal_id, 'HIT_SL', 'Stop loss triggered')
                    continue
                elif current_price <= tp:
                    await send_signal_update(session, signal, current_price, "TARGET_HIT")
                    update_signal_status(signal_id, 'HIT_TP', 'Target achieved')
                    continue
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

# ---------------- Main Analysis Function ----------------
async def analyze_symbol_720_candles(session, symbol):
    try:
        print(f"🔍 Analyzing {symbol} with 720 candles...")
        candles_4h_raw = await fetch_json(session, CANDLE_4H_URL.format(symbol=symbol))
        candles_1h_raw = await fetch_json(session, CANDLE_1H_URL.format(symbol=symbol))
        if not candles_4h_raw or not candles_1h_raw:
            print(f"❌ Failed to fetch candle data for {symbol}")
            return
        candles_4h = normalize_klines(candles_4h_raw)
        candles_1h = normalize_klines(candles_1h_raw)
        if len(candles_4h) < MIN_CANDLES_REQUIRED or len(candles_1h) < MIN_CANDLES_REQUIRED:
            print(f"❌ Insufficient candle data for {symbol}")
            return
        print(f"📊 {symbol}: 4H={len(candles_4h)} candles, 1H={len(candles_1h)} candles")
        analysis_result = analyze_multi_timeframe_720(candles_4h, candles_1h)
        if not analysis_result['signals']:
            print(f"📈 {symbol}: No signals generated (Strength: {analysis_result.get('strength', 0)})")
            return
        ai_analysis = await get_comprehensive_ai_analysis(symbol, analysis_result)
        if ai_analysis:
            if "REJECT" in ai_analysis.upper():
                print(f"🤖 {symbol}: AI rejected signal - {ai_analysis}")
                return
            elif "CONFIRM" in ai_analysis.upper():
                print(f"✅ {symbol}: AI confirmed signal - {ai_analysis}")
            else:
                print(f"⚠️ {symbol}: Unclear AI response - {ai_analysis}")
        signal_obj = analysis_result['signals'][0]
        signal_data = {
            'symbol': symbol,
            'side': signal_obj['side'],
            'entry': str(signal_obj['entry']),
            'sl': str(signal_obj['sl']),
            'tp': str(signal_obj['tp']),
            'rr': str(signal_obj['rr']),
            'confidence': str(signal_obj['confidence']),
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

# ---------------- Main Bot Loop ----------------
async def main():
    print("🚀 Advanced 720 Candles Crypto Trading Bot Started!")
    print(f"📊 Analyzing {len(SYMBOLS)} symbols every {POLL_INTERVAL} seconds")
    print(f"🤖 AI Model: {OPENAI_MODEL}")
    print(f"📈 Min Confidence: {SIGNAL_CONF_THRESHOLD}%")
    print(f"⚡ Redis: {'Connected' if redis_client else 'Not available'}")
    async with aiohttp.ClientSession() as session:
        await send_telegram_message(session,
            f"🟢 **Bot Started Successfully**\n\n"
            f"📊 **Config:**\n"
            f"• Symbols: {len(SYMBOLS)}\n"
            f"• Candles: 720 per timeframe\n"
            f"• Timeframes: 4H + 1H\n"
            f"• Scan Interval: {POLL_INTERVAL}s\n"
            f"• AI Model: {OPENAI_MODEL}\n"
            f"• Redis: {'✅' if redis_client else '❌'}\n\n"
            f"Ready for signal generation! 🚀"
        )
    while True:
        try:
            print(f"\n{'='*50}")
            print(f"🕐 Starting analysis cycle - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            print(f"{'='*50}")
            async with aiohttp.ClientSession() as session:
                await monitor_active_signals(session)
                tasks = [analyze_symbol_720_candles(session, symbol) for symbol in SYMBOLS]
                # limit concurrency at HTTP level via semaphore; we still gather tasks
                await asyncio.gather(*tasks, return_exceptions=True)
            print(f"✅ Analysis cycle completed. Next scan in {POLL_INTERVAL} seconds...")
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
    except KeyboardInterrupt:
        print("\n👋 Bot shutdown completed")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
