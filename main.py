#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v3.0 (with local logic + multi-TF + AI merge)
import os
import re
import asyncio
import aiohttp
import traceback
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple
import json

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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")   # now mini default
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

# Globals
price_history: Dict[str, List[Dict]] = {}
last_signals: Dict[str, datetime] = {}
performance_tracking: List[Dict] = []

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
CANDLE_URL_1H = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"

# ---------------- Utilities ----------------
def fmt_price(p: Optional[float]) -> str:
    if p is None: return "N/A"
    try: v = float(p)
    except: return str(p)
    return f"{v:.6f}" if abs(v) < 1 else f"{v:.2f}"

# ---------------- Local Trade Logic ----------------
def ema(values, period):
    if len(values) < period: return []
    k = 2 / (period + 1)
    ema_arr = []
    prev = sum(values[:period]) / period
    ema_arr.extend([None] * (period - 1))
    ema_arr.append(prev)
    for v in values[period:]:
        prev = v * k + prev * (1 - k)
        ema_arr.append(prev)
    return ema_arr

def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    pts = closes[-lookback:] + highs[-lookback:] + lows[-lookback:]
    levels = []
    for p in pts:
        found = False
        for lv in levels:
            if abs((lv["price"] - p) / p) < binsize:
                lv["count"] += 1
                lv["price"] = (lv["price"] * (lv["count"] - 1) + p) / lv["count"]
                found = True
                break
        if not found:
            levels.append({"price": p, "count": 1})
    levels.sort(key=lambda x: -x["count"])
    return [lv["price"] for lv in levels[:5]]

def analyze_trade_logic(candles, volumes, rr_min=1.5):
    closes = [c[3] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    if len(closes) < 30:
        return {"side": "none", "confidence": 0, "reason": "not enough data"}
    ema_s = ema(closes, 9)
    ema_l = ema(closes, 21)
    price = closes[-1]
    es, el = ema_s[-1], ema_l[-1]
    levels = horizontal_levels(closes, highs, lows)
    sup = max([lv for lv in levels if lv < price], default=None)
    res = min([lv for lv in levels if lv > price], default=None)
    conf = 50
    reasons = []
    if price < es and price < el:
        conf += 10; reasons.append("below EMAs (bear)")
    elif price > es and price > el:
        conf += 10; reasons.append("above EMAs (bull)")
    else:
        conf -= 5; reasons.append("EMAs mixed")
    if price < es and res:
        entry, stop = price, res * 1.003
        tgt1 = sup if sup else price - (stop - entry) * 1.5
        rr = (entry - tgt1) / (stop - entry) if stop > entry else 0
        if rr >= rr_min:
            conf += 10; side = "SELL"
            return {"side": side, "entry": entry, "sl": stop, "tp": tgt1, "confidence": conf, "reason": "; ".join(reasons)}
    if price > es and res:
        entry, stop = price, sup * 0.997 if sup else price * 0.99
        tgt1 = res
        rr = (tgt1 - entry) / (entry - stop) if entry > stop else 0
        if rr >= rr_min:
            conf += 10; side = "BUY"
            return {"side": side, "entry": entry, "sl": stop, "tp": tgt1, "confidence": conf, "reason": "; ".join(reasons)}
    return {"side": "none", "confidence": conf, "reason": "; ".join(reasons)}

def multi_tf_confirmation(candles_30m, candles_1h, volumes_30m):
    sig30 = analyze_trade_logic(candles_30m, volumes_30m)
    sig1h = analyze_trade_logic(candles_1h, volumes_30m)
    if sig30["side"] == "none" or sig1h["side"] == "none":
        return {"side": "none", "confidence": 0, "reason": "no alignment"}
    if sig30["side"] == sig1h["side"]:
        sig30["confidence"] = min(100, sig30["confidence"] + 15)
        sig30["reason"] += f"; aligned {sig1h['side']} on 1h"
        return sig30
    return {"side": "none", "confidence": 0, "reason": "TF conflict"}

# ---------------- Fetch helpers ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json error:", e)
        return None

# ---------------- AI helper ----------------
async def ask_openai_for_signals(market_summary_text: str):
    if not client: return None
    prompt_system = "You are a professional crypto trader. Only output signals that meet criteria."
    prompt_user = f"""{market_summary_text}
Rules:
- Conf ≥ {int(SIGNAL_CONF_THRESHOLD)}%
- Format: SYMBOL - BUY/SELL - ENTRY: x - SL: y - TP: z - REASON: ... - CONF: n%"""
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":prompt_system},{"role":"user","content":prompt_user}],
                max_tokens=500,temperature=0.0)
        resp = await loop.run_in_executor(None, call_model)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("AI call failed:", e)
        return None

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID,"text": text}, timeout=30)
    except: pass

# ---------------- Main loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        print(f"🤖 Bot started with {len(SYMBOLS)} symbols | Poll={POLL_INTERVAL}s")
        iteration = 0
        while True:
            iteration += 1
            print(f"\n--- Iteration {iteration} --- {datetime.now().strftime('%H:%M:%S')}")
            for symbol in SYMBOLS:
                try:
                    c30 = await fetch_json(session, CANDLE_URL.format(symbol=symbol))
                    c1h = await fetch_json(session, CANDLE_URL_1H.format(symbol=symbol))
                    if not c30 or not c1h: continue
                    candles30 = [[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c30]
                    candles1h = [[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c1h]
                    vols = [float(x[5]) for x in c30]
                    local_signal = multi_tf_confirmation(candles30, candles1h, vols)
                    ai_resp = await ask_openai_for_signals(json.dumps({"symbol":symbol,"last_price":candles30[-1][3]},indent=2))
                    # Merge logic
                    if local_signal["side"] != "none":
                        conf = local_signal["confidence"]
                        if ai_resp and "NO_SIGNAL" not in ai_resp:
                            conf += 10
                            local_signal["reason"] += "; AI agrees"
                        print(f"⚡ {symbol}: {local_signal['side']} @ {fmt_price(local_signal['entry'])} | Conf {conf}% | {local_signal['reason']}")
                        if conf >= SIGNAL_CONF_THRESHOLD:
                            msg = f"{symbol} {local_signal['side']} | Entry {fmt_price(local_signal['entry'])} | SL {fmt_price(local_signal['sl'])} | TP {fmt_price(local_signal['tp'])} | Conf {conf}%"
                            await send_text(session, msg)
                    else:
                        print(f"{symbol}: no signal ({local_signal['reason']})")
                except Exception as e:
                    print("Error for", symbol, e)
            await asyncio.sleep(POLL_INTERVAL)

# ---------------- Entry ----------------
if __name__ == "__main__":
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("Stopped")
