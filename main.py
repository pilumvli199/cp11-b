import os
import asyncio
import aiohttp
import time
import traceback
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

# Load environment variables
load_dotenv()

# Config
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT,SOLUSDT").split(",")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "1800"))  # default 30m
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

client = OpenAI()

# Binance endpoint for 1h timeframe, 720 candles
CANDLE_URL_1H = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"

# ============ Utility Functions ============
def ema(values, period):
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    ema_values = [sum(values[:period]) / period]
    for price in values[period:]:
        ema_values.append(price * k + ema_values[-1] * (1 - k))
    return ema_values

async def fetch_candles(session, symbol, url):
    try:
        async with session.get(url.format(symbol=symbol)) as resp:
            data = await resp.json()
            candles = [
                {
                    "time": int(x[0]),
                    "open": float(x[1]),
                    "high": float(x[2]),
                    "low": float(x[3]),
                    "close": float(x[4]),
                    "volume": float(x[5]),
                }
                for x in data
            ]
            return candles
    except Exception as e:
        print(f"Error fetching candles for {symbol}: {e}")
        return []

def horizontal_levels(candles, lookback=50, sensitivity=0.02):
    closes = [c["close"] for c in candles[-lookback:]]
    highs = [c["high"] for c in candles[-lookback:]]
    lows = [c["low"] for c in candles[-lookback:]]
    levels = []
    for price in closes + highs + lows:
        if any(abs(price - l) / l < sensitivity for l in levels):
            continue
        levels.append(price)
    return levels

async def analyze_trade_logic(symbol, candles):
    if len(candles) < 60:
        return {"side": "none", "confidence": 0, "reason": "insufficient data"}
    closes = [c["close"] for c in candles]
    ema9 = ema(closes, 9)
    ema20 = ema(closes, 20)
    if not ema9 or not ema20:
        return {"side": "none", "confidence": 0, "reason": "not enough data for EMA"}
    last_price = closes[-1]
    if ema9[-1] > ema20[-1] and last_price > ema9[-1]:
        return {"side": "long", "confidence": 70, "reason": "EMA9 above EMA20 and price above EMA9"}
    elif ema9[-1] < ema20[-1] and last_price < ema9[-1]:
        return {"side": "short", "confidence": 70, "reason": "EMA9 below EMA20 and price below EMA9"}
    return {"side": "none", "confidence": 40, "reason": "No clear trend"}

def plot_signal_chart(symbol, candles, ema9, ema20, levels, signal):
    times = [date2num(datetime.fromtimestamp(c["time"] / 1000)) for c in candles]
    closes = [c["close"] for c in candles]
    plt.figure(figsize=(10, 5))
    plt.plot_date(times, closes, "-", label="Close")
    if ema9:
        plt.plot_date(times[-len(ema9):], ema9, "-", label="EMA9")
    if ema20:
        plt.plot_date(times[-len(ema20):], ema20, "-", label="EMA20")
    for lvl in levels:
        plt.axhline(y=lvl, linestyle="--", alpha=0.5)
    plt.title(f"{symbol} Signal: {signal['side']} ({signal['confidence']}%)")
    plt.legend()
    tmpfile = NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmpfile.name)
    plt.close()
    return tmpfile.name

async def analyze_with_openai(symbol, analysis, chart_path):
    try:
        with open(chart_path, "rb") as img:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a crypto trading assistant."},
                    {"role": "user", "content": f"Analyze {symbol} with data: {analysis}"},
                ],
                modalities=["text", "image"],
                temperature=0.3,
                max_tokens=300,
            )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI analysis error: {e}"

async def process_symbol(session, symbol):
    candles = await fetch_candles(session, symbol, CANDLE_URL_1H)
    signal = await analyze_trade_logic(symbol, candles)
    closes = [c["close"] for c in candles]
    ema9 = ema(closes, 9)
    ema20 = ema(closes, 20)
    levels = horizontal_levels(candles)
    chart_path = plot_signal_chart(symbol, candles, ema9, ema20, levels, signal)
    analysis = {"signal": signal, "levels": levels}
    gpt_analysis = await analyze_with_openai(symbol, analysis, chart_path)
    print(f"\nSymbol: {symbol}\nSignal: {signal}\nGPT Analysis: {gpt_analysis}\n")

async def main():
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [process_symbol(session, s) for s in SYMBOLS]
                await asyncio.gather(*tasks)
        except Exception as e:
            print("Error in main loop:", e)
            traceback.print_exc()
        await asyncio.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    asyncio.run(main())
