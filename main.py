import os
import redis
import time
import traceback
import asyncio
import aiohttp
from datetime import datetime, timedelta
from openai import OpenAI

# -------------------------------
# Redis Connection Setup
# -------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "redis-14246.crce206.ap-south-1-1.ec2.redns.redis-cloud.com")
REDIS_PORT = int(os.getenv("REDIS_PORT", 14246))
REDIS_USER = os.getenv("REDIS_USER", "default")
REDIS_PASS = os.getenv("REDIS_PASS", "aar1DQpfMxPl7XT3SHFO8xEu6A7A0ik1")

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        decode_responses=True,
        username=REDIS_USER,
        password=REDIS_PASS,
    )
    r.ping()
    print("✅ Redis connected successfully")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    r = None

# -------------------------------
# Main Bot Logic
# -------------------------------

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
MIN_CONFIDENCE = 70.0

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "AAVEUSDT", "TRXUSDT", "DOGEUSDT", "BNBUSDT",
    "ADAUSDT", "LTCUSDT", "LINKUSDT"
]

async def fetch_candles(symbol, timeframe="1h", limit=720):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.json()

async def analyze_symbol(symbol):
    try:
        print(f"🔍 Analyzing {symbol} with {720} candles...")
        candles_1h = await fetch_candles(symbol, "1h", 720)
        candles_4h = await fetch_candles(symbol, "4h", 720)

        # Simplified Signal Logic (EMA check placeholder)
        signal = {"side": "short", "confidence": 70, "reason": "EMA9 below EMA20 and price below EMA9"}

        print(f"📈 {symbol}: {signal}")

        # Store last signal in Redis
        if r:
            r.set(f"last_signal_{symbol}", str(signal))

        return signal
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {e}")
        traceback.print_exc()

async def main_loop():
    print(f"🚀 Advanced 720 Candles Crypto Trading Bot Started!")
    print(f"📊 Analyzing {len(SYMBOLS)} symbols every 1800 seconds")
    print(f"🤖 AI Model: {MODEL}")
    print(f"📈 Min Confidence: {MIN_CONFIDENCE}%")
    print(f"⚡ Redis: {'Available' if r else 'Not available'}")

    while True:
        print("=" * 50)
        print(f"🕐 Starting analysis cycle - {datetime.utcnow()} UTC")
        print("=" * 50)

        tasks = [analyze_symbol(symbol) for symbol in SYMBOLS]
        await asyncio.gather(*tasks)

        print("✅ Analysis cycle completed. Next scan in 1800 seconds...")
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(main_loop())
