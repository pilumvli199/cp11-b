#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v2.0 (fixed & flow updated)
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
import base64

load_dotenv()

# ---------------- CONFIG ----------------
SYMBOLS = os.getenv("SYMBOLS", "BTCUSDT,ETHUSDT").split(",")
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # change to gpt-4o-mini if available
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

# Analysis windows
RSI_PERIOD = int(os.getenv("RSI_PERIOD", 14))
MACD_FAST = int(os.getenv("MACD_FAST", 12))
MACD_SLOW = int(os.getenv("MACD_SLOW", 26))
MACD_SIGNAL = int(os.getenv("MACD_SIGNAL", 9))
MA_SHORT = int(os.getenv("MA_SHORT", 7))
MA_MEDIUM = int(os.getenv("MA_MEDIUM", 21))
MA_LONG = int(os.getenv("MA_LONG", 50))
BB_PERIOD = int(os.getenv("BB_PERIOD", 20))
BB_STD = float(os.getenv("BB_STD", 2))
VOLUME_MULTIPLIER = float(os.getenv("VOLUME_MULTIPLIER", 1.8))
MIN_CANDLES_FOR_ANALYSIS = int(os.getenv("MIN_CANDLES_FOR_ANALYSIS", 30))
LOOKBACK_PERIOD = int(os.getenv("LOOKBACK_PERIOD", 100))
FIBONACCI_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

# Globals
price_history: Dict[str, List[Dict]] = {}
signal_history: List[Dict] = []
market_sentiment: Dict[str, str] = {}
last_signals: Dict[str, datetime] = {}
performance_tracking: List[Dict] = []

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
ORDER_BOOK_URL = "https://api.binance.com/api/v3/depth?symbol={symbol}&limit=20"

# ---------------- Utility formatters ----------------
def fmt_price(p: Optional[float]) -> str:
    """Format numeric price: 6 decimals for sub-1 prices, 2 decimals otherwise."""
    if p is None:
        return "N/A"
    try:
        v = float(p)
    except Exception:
        return str(p)
    return f"{v:.6f}" if abs(v) < 1 else f"{v:.2f}"

def fmt_decimal(val, small_prec=6, large_prec=2):
    if val is None:
        return "N/A"
    try:
        v = float(val)
    except Exception:
        return str(val)
    return f"{v:.{small_prec}f}" if abs(v) < 1 else f"{v:.{large_prec}f}"

# ---------------- Indicators ----------------
def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(prices: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < MACD_SLOW:
        return None, None, None
    def ema(data, span):
        alpha = 2 / (span + 1)
        ema_values = [data[0]]
        for price in data[1:]:
            ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
        return ema_values
    ema_fast = ema(prices, MACD_FAST)
    ema_slow = ema(prices, MACD_SLOW)
    macd_line = ema_fast[-1] - ema_slow[-1]
    macd_values = [fast - slow for fast, slow in zip(ema_fast[MACD_SLOW-1:], ema_slow)]
    if len(macd_values) >= MACD_SIGNAL:
        signal_line = ema(macd_values, MACD_SIGNAL)[-1]
        histogram = macd_line - signal_line
        return round(macd_line, 6), round(signal_line, 6), round(histogram, 6)
    return round(macd_line, 6), None, None

def calculate_bollinger_bands(prices: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if len(prices) < BB_PERIOD:
        return None, None, None
    recent_prices = prices[-BB_PERIOD:]
    sma = np.mean(recent_prices)
    std = np.std(recent_prices)
    upper_band = sma + (BB_STD * std)
    lower_band = sma - (BB_STD * std)
    return round(upper_band, 6), round(sma, 6), round(lower_band, 6)

def calculate_fibonacci_retracements(highs: List[float], lows: List[float]) -> Dict[str, float]:
    if not highs or not lows:
        return {}
    recent_high = max(highs[-50:])
    recent_low = min(lows[-50:])
    diff = recent_high - recent_low
    fib_levels = {}
    for level in FIBONACCI_LEVELS:
        fib_levels[f"fib_{level}"] = recent_low + (diff * level)
    return fib_levels

def detect_divergence(prices: List[float], rsi_values: List[float]) -> str:
    if len(prices) < 10 or len(rsi_values) < 10:
        return "none"
    recent_prices = prices[-10:]
    recent_rsi = rsi_values[-10:]
    price_trend = "up" if recent_prices[-1] > recent_prices[0] else "down"
    rsi_trend = "up" if recent_rsi[-1] > recent_rsi[0] else "down"
    if price_trend == "down" and rsi_trend == "up":
        return "bullish_divergence"
    elif price_trend == "up" and rsi_trend == "down":
        return "bearish_divergence"
    return "none"

def calculate_market_structure(candles: List[List[float]]) -> Dict[str, any]:
    if len(candles) < 10:
        return {}
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    peaks = []
    troughs = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1] and highs[i] > highs[i-2] and highs[i] > highs[i+2]:
            peaks.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1] and lows[i] < lows[i-2] and lows[i] < lows[i+2]:
            troughs.append((i, lows[i]))
    structure = {"trend": "sideways", "strength": 0}
    if len(peaks) >= 2 and len(troughs) >= 2:
        if peaks[-1][1] > peaks[-2][1] and troughs[-1][1] > troughs[-2][1]:
            structure["trend"] = "uptrend"; structure["strength"] = 2
        elif peaks[-1][1] < peaks[-2][1] and troughs[-1][1] < troughs[-2][1]:
            structure["trend"] = "downtrend"; structure["strength"] = 2
    return structure

def enhanced_volume_analysis(volumes: List[float], prices: List[float]) -> Dict[str, any]:
    if len(volumes) < 20:
        return {}
    recent_volumes = volumes[-20:]
    avg_volume = np.mean(recent_volumes[:-1])
    current_volume = volumes[-1]
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    price_change = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 and prices[-2] != 0 else 0
    analysis = {
        "volume_spike": volume_ratio > VOLUME_MULTIPLIER,
        "volume_ratio": round(volume_ratio, 2),
        "price_volume_confirmation": False
    }
    if abs(price_change) > 0.01:
        if (price_change > 0 and volume_ratio > 1.2) or (price_change < 0 and volume_ratio > 1.2):
            analysis["price_volume_confirmation"] = True
    return analysis

def detect_advanced_patterns(candles: List[List[float]]) -> Dict[str, bool]:
    if len(candles) < 5:
        return {}
    patterns = {}
    recent = candles[-5:]
    for i, candle in enumerate(recent):
        open_price, high, low, close = candle
        body = abs(close - open_price)
        upper_wick = high - max(open_price, close)
        lower_wick = min(open_price, close) - low
        range_size = high - low
        if range_size == 0:
            continue
        if i == len(recent) - 1:
            if (upper_wick > 2 * body and lower_wick < body * 0.1 and close < open_price and i > 0 and recent[i-1][3] > recent[i-1][0]):
                patterns['shooting_star'] = True
            if (lower_wick > 2 * body and upper_wick < body * 0.1 and i > 0 and recent[i-1][3] < recent[i-1][0]):
                patterns['hammer'] = True
            if (upper_wick > 2 * body and lower_wick < body * 0.1 and close > open_price and i > 0 and recent[i-1][3] < recent[i-1][0]):
                patterns['inverted_hammer'] = True
    if len(recent) >= 3:
        if (recent[-3][3] < recent[-3][0] and abs(recent[-2][3] - recent[-2][0]) < (recent[-3][1] - recent[-3][2]) * 0.3 and recent[-1][3] > recent[-1][0] and recent[-1][3] > (recent[-3][0] + recent[-3][3]) / 2):
            patterns['morning_star'] = True
        if (recent[-3][3] > recent[-3][0] and abs(recent[-2][3] - recent[-2][0]) < (recent[-3][1] - recent[-3][2]) * 0.3 and recent[-1][3] < recent[-1][0] and recent[-1][3] < (recent[-3][0] + recent[-3][3]) / 2):
            patterns['evening_star'] = True
    return patterns

# ---------------- Fetching & parsing ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=20) as r:
            if r.status != 200:
                txt = await r.text() if r is not None else "<no body>"
                print(f"fetch_json {url} returned {r.status}: {txt[:200]}")
                return None
            return await r.json()
    except Exception as e:
        print("fetch_json exception for", url, e)
        return None

async def fetch_enhanced_data(session, symbol):
    ticker_task = fetch_json(session, TICKER_URL.format(symbol=symbol))
    candle_task = fetch_json(session, CANDLE_URL.format(symbol=symbol))
    orderbook_task = fetch_json(session, ORDER_BOOK_URL.format(symbol=symbol))
    ticker, candles, orderbook = await asyncio.gather(ticker_task, candle_task, orderbook_task)
    out = {}
    if ticker:
        try:
            out["price"] = float(ticker.get("lastPrice", 0))
            out["volume"] = float(ticker.get("volume", 0))
            out["price_change_24h"] = float(ticker.get("priceChangePercent", 0))
            out["high_24h"] = float(ticker.get("highPrice", 0))
            out["low_24h"] = float(ticker.get("lowPrice", 0))
            out["quote_volume"] = float(ticker.get("quoteVolume", 0))
        except Exception as e:
            print(f"Error processing ticker for {symbol}: {e}")
            out["price"] = None
    if isinstance(candles, list) and len(candles) >= MIN_CANDLES_FOR_ANALYSIS:
        try:
            parsed_candles = []
            times = []
            volumes = []
            for x in candles:
                # Binance kline: [openTime, open, high, low, close, volume, ...]
                open_price = float(x[1])
                high = float(x[2])
                low = float(x[3])
                close = float(x[4])
                volume = float(x[5])
                timestamp = int(x[0]) // 1000
                parsed_candles.append([open_price, high, low, close])
                times.append(timestamp)
                volumes.append(volume)
            out["candles"] = parsed_candles
            out["times"] = times
            out["volumes"] = volumes
            closes = [c[3] for c in parsed_candles]
            highs = [c[1] for c in parsed_candles]
            lows = [c[2] for c in parsed_candles]
            out["rsi"] = calculate_rsi(closes, RSI_PERIOD)
            out["ma_short"] = sum(closes[-MA_SHORT:]) / MA_SHORT if len(closes) >= MA_SHORT else None
            out["ma_medium"] = sum(closes[-MA_MEDIUM:]) / MA_MEDIUM if len(closes) >= MA_MEDIUM else None
            out["ma_long"] = sum(closes[-MA_LONG:]) / MA_LONG if len(closes) >= MA_LONG else None
            macd, signal, histogram = calculate_macd(closes)
            out["macd"] = macd
            out["macd_signal"] = signal
            out["macd_histogram"] = histogram
            bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
            out["bb_upper"] = bb_upper
            out["bb_middle"] = bb_middle
            out["bb_lower"] = bb_lower
            out["market_structure"] = calculate_market_structure(parsed_candles)
            out["volume_analysis"] = enhanced_volume_analysis(volumes, closes)
            out["patterns"] = detect_advanced_patterns(parsed_candles)
            out["fibonacci"] = calculate_fibonacci_retracements(highs, lows)
            if len(closes) >= 20:
                rsi_values = []
                for i in range(RSI_PERIOD, len(closes)):
                    rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
                    if rsi:
                        rsi_values.append(rsi)
                if len(rsi_values) >= 10:
                    out["divergence"] = detect_divergence(closes[-len(rsi_values):], rsi_values)
        except Exception as e:
            print(f"Enhanced candle processing error for {symbol}: {e}")
            traceback.print_exc()
    if orderbook:
        try:
            bids = [(float(x[0]), float(x[1])) for x in orderbook.get("bids", [])]
            asks = [(float(x[0]), float(x[1])) for x in orderbook.get("asks", [])]
            if bids and asks:
                out["bid"] = bids[0][0]
                out["ask"] = asks[0][0]
                out["spread"] = asks[0][0] - bids[0][0]
                out["spread_pct"] = (out["spread"] / bids[0][0]) * 100
                total_bid_volume = sum(x[1] for x in bids[:10])
                total_ask_volume = sum(x[1] for x in asks[:10])
                out["order_imbalance"] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume) if (total_bid_volume + total_ask_volume) != 0 else 0
                sig_bid_mean = np.mean([b[1] for b in bids]) if bids else 0
                sig_ask_mean = np.mean([a[1] for a in asks]) if asks else 0
                significant_bids = [x for x in bids if x[1] > sig_bid_mean * 1.5]
                significant_asks = [x for x in asks if x[1] > sig_ask_mean * 1.5]
                out["ob_support"] = significant_bids[0][0] if significant_bids else None
                out["ob_resistance"] = significant_asks[0][0] if significant_asks else None
        except Exception as e:
            print(f"Order book processing error for {symbol}: {e}")
    if symbol not in price_history:
        price_history[symbol] = []
    if out.get("price") is not None:
        price_history[symbol].append({"price": out["price"], "timestamp": datetime.now(), "volume": out.get("volume", 0), "rsi": out.get("rsi")})
        if len(price_history[symbol]) > 200:
            price_history[symbol] = price_history[symbol][-200:]
    return out

# ---------------- Charting ----------------
def enhanced_levels(candles, lookback=LOOKBACK_PERIOD):
    if not candles or len(candles) < 10:
        return (None, None, None, None, None)
    arr = candles[-min(len(candles), lookback):]
    highs = [c[1] for c in arr]
    lows = [c[2] for c in arr]
    closes = [c[3] for c in arr]
    recent_weight = 1.5
    older_weight = 1.0
    weighted_highs = []
    weighted_lows = []
    for i, (high, low) in enumerate(zip(highs, lows)):
        weight = recent_weight if i >= len(highs) * 0.7 else older_weight
        weighted_highs.extend([high] * int(weight * 10))
        weighted_lows.extend([low] * int(weight * 10))
    highs_sorted = sorted(weighted_highs, reverse=True)
    lows_sorted = sorted(weighted_lows)
    primary_resistance = np.mean(highs_sorted[:30]) if len(highs_sorted) >= 30 else None
    primary_support = np.mean(lows_sorted[:30]) if len(lows_sorted) >= 30 else None
    secondary_resistance = np.mean(highs_sorted[30:60]) if len(highs_sorted) >= 60 else None
    secondary_support = np.mean(lows_sorted[30:60]) if len(lows_sorted) >= 60 else None
    current_price = closes[-1]
    mid_level = (primary_resistance + primary_support) / 2 if primary_resistance and primary_support else current_price
    return primary_support, primary_resistance, secondary_support, secondary_resistance, mid_level

def enhanced_plot_chart(times, candles, symbol, market_data):
    if not times or not candles or len(times) != len(candles) or len(candles) < 10:
        raise ValueError("Insufficient data for enhanced plotting")
    dates = [datetime.utcfromtimestamp(int(t)) for t in times]
    closes = [c[3] for c in candles]
    highs = [c[1] for c in candles]
    lows = [c[2] for c in candles]
    x = date2num(dates)
    fig = plt.figure(figsize=(14, 9), dpi=120)
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1])
    ax_rsi = fig.add_subplot(gs[2])
    ax_macd = fig.add_subplot(gs[3])
    width = 0.6 * (x[1] - x[0]) if len(x) > 1 else 0.4
    for xi, candle in zip(x, candles):
        o, h, l, c = candle
        color = "#00aa00" if c >= o else "#aa0000"
        edge_color = "#006600" if c >= o else "#660000"
        ax_price.vlines(xi, l, h, color=edge_color, linewidth=1.2, alpha=0.8)
        rect_height = abs(c - o) if abs(c - o) > 0.0001 else 0.0001
        rect = plt.Rectangle((xi - width/2, min(o, c)), width, rect_height,
                           facecolor=color, edgecolor=edge_color, alpha=0.9, linewidth=0.8)
        ax_price.add_patch(rect)
    sup1, res1, sup2, res2, mid = enhanced_levels(candles, lookback=LOOKBACK_PERIOD)
    if res1:
        ax_price.axhline(res1, linestyle="--", alpha=0.8, linewidth=1.5, label=f"Primary Resistance: {fmt_price(res1)}")
    if sup1:
        ax_price.axhline(sup1, linestyle="--", alpha=0.8, linewidth=1.5, label=f"Primary Support: {fmt_price(sup1)}")
    if res2:
        ax_price.axhline(res2, linestyle=":", alpha=0.6, linewidth=1.2, label=f"Secondary Resistance: {fmt_price(res2)}")
    if sup2:
        ax_price.axhline(sup2, linestyle=":", alpha=0.6, linewidth=1.2, label=f"Secondary Support: {fmt_price(sup2)}")
    bb_upper = market_data.get("bb_upper")
    bb_middle = market_data.get("bb_middle")
    bb_lower = market_data.get("bb_lower")
    if bb_upper and bb_middle and bb_lower:
        bb_upper_line = [bb_upper] * len(x)
        bb_middle_line = [bb_middle] * len(x)
        bb_lower_line = [bb_lower] * len(x)
        ax_price.plot(x, bb_upper_line, linewidth=1, alpha=0.6, label="BB Upper")
        ax_price.plot(x, bb_middle_line, linewidth=1, alpha=0.6, label="BB Middle")
        ax_price.plot(x, bb_lower_line, linewidth=1, alpha=0.6, label="BB Lower")
        ax_price.fill_between(x, bb_upper_line, bb_lower_line, alpha=0.05)
    if len(closes) >= MA_LONG:
        ma_short_values, ma_medium_values, ma_long_values = [], [], []
        for i in range(len(closes)):
            if i >= MA_SHORT - 1:
                ma_short_values.append(sum(closes[i-MA_SHORT+1:i+1]) / MA_SHORT)
            else:
                ma_short_values.append(None)
            if i >= MA_MEDIUM - 1:
                ma_medium_values.append(sum(closes[i-MA_MEDIUM+1:i+1]) / MA_MEDIUM)
            else:
                ma_medium_values.append(None)
            if i >= MA_LONG - 1:
                ma_long_values.append(sum(closes[i-MA_LONG+1:i+1]) / MA_LONG)
            else:
                ma_long_values.append(None)
        valid_short = [(x[i], ma) for i, ma in enumerate(ma_short_values) if ma is not None]
        valid_medium = [(x[i], ma) for i, ma in enumerate(ma_medium_values) if ma is not None]
        valid_long = [(x[i], ma) for i, ma in enumerate(ma_long_values) if ma is not None]
        if valid_short:
            x_short, y_short = zip(*valid_short)
            ax_price.plot(x_short, y_short, linewidth=1.5, alpha=0.9, label=f"MA{MA_SHORT}")
        if valid_medium:
            x_medium, y_medium = zip(*valid_medium)
            ax_price.plot(x_medium, y_medium, linewidth=1.5, alpha=0.9, label=f"MA{MA_MEDIUM}")
        if valid_long:
            x_long, y_long = zip(*valid_long)
            ax_price.plot(x_long, y_long, linewidth=1.5, alpha=0.9, label=f"MA{MA_LONG}")
    volumes = market_data.get("volumes", [])
    if volumes and len(volumes) == len(x):
        volume_colors = []
        for i in range(len(candles)):
            if i == 0:
                volume_colors.append("gray")
            elif candles[i][3] >= candles[i-1][3]:
                volume_colors.append("green")
            else:
                volume_colors.append("red")
        ax_volume.bar(x, volumes, width=width, color=volume_colors, alpha=0.7)
        ax_volume.set_ylabel("Volume", fontsize=9)
    if market_data.get("candles") and len(market_data["candles"]) >= RSI_PERIOD + 5:
        rsi_values = []
        for i in range(RSI_PERIOD, len(closes)):
            rsi = calculate_rsi(closes[:i+1], RSI_PERIOD)
            if rsi:
                rsi_values.append(rsi)
        if rsi_values:
            rsi_x = x[-len(rsi_values):]
            ax_rsi.plot(rsi_x, rsi_values, linewidth=1.5)
            ax_rsi.axhline(70, linestyle="--", alpha=0.6)
            ax_rsi.axhline(30, linestyle="--", alpha=0.6)
            ax_rsi.set_ylim(0, 100)
            ax_rsi.set_ylabel("RSI", fontsize=9)
    macd = market_data.get("macd")
    macd_signal = market_data.get("macd_signal")
    macd_histogram = market_data.get("macd_histogram")
    if macd is not None and macd_signal is not None:
        ax_macd.axhline(macd, linewidth=1.2, label=f"MACD: {fmt_decimal(macd)}")
        ax_macd.axhline(macd_signal, linewidth=1.2, label=f"Signal: {fmt_decimal(macd_signal)}")
        if macd_histogram:
            ax_macd.bar([x[-1]], [macd_histogram], width=width*5, alpha=0.7)
        ax_macd.set_ylabel("MACD", fontsize=9)
    current_price = closes[-1]
    rsi_current = market_data.get("rsi", "N/A")
    price_change_24h = market_data.get("price_change_24h", 0)
    volume_ratio = market_data.get("volume_analysis", {}).get("volume_ratio", 0)
    trend = market_data.get("market_structure", {}).get("trend", "sideways")
    title = f"{symbol} | Price: ${fmt_price(current_price)} | 24h: {price_change_24h:+.2f}% | RSI: {rsi_current} | Vol: {volume_ratio}x | Trend: {trend.upper()} | TF:30m"
    ax_price.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax_price.legend(loc="upper left", fontsize="small", framealpha=0.9)
    ax_price.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp.name, bbox_inches="tight", dpi=120, facecolor='white')
    plt.close(fig)
    return tmp.name

# ---------------- OpenAI Analysis ----------------
async def ask_openai_for_signals(market_summary_text: str, chart_path: Optional[str] = None):
    """
    Sends the numeric/structured market summary to OpenAI and requests ENTRY/SL/TP suggestions.
    chart_path is provided as reference (filename) so model can reason about it; actual image is NOT uploaded here.
    """
    if not client:
        print("No OpenAI client configured.")
        return None
    prompt_system = "You are a professional crypto trader. Provide high-probability trade setups given the provided 30m timeframe data. Follow strict formatting."
    prompt_user = f"""DATA:
{market_summary_text}

INSTRUCTIONS (strict):
- Only give signals with CONF >= {int(SIGNAL_CONF_THRESHOLD)}%
- Provide exact ENTRY, SL, TP numeric prices.
- Minimum R:R 1.5:1
- Output format (one per line):
SYMBOL - ACTION - ENTRY: <price> - SL: <price> - TP: <price> - REASON: <<=40 words> - CONF: <{int(SIGNAL_CONF_THRESHOLD)}-100>%
If no valid signal, reply "NO_SIGNAL"."""
    try:
        loop = asyncio.get_running_loop()
        def call_model():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ],
                max_tokens=800,
                temperature=0.0
            )
        resp = await loop.run_in_executor(None, call_model)
        try:
            content = resp.choices[0].message.content
            return content.strip()
        except Exception:
            return str(resp)
    except Exception as e:
        print("OpenAI call failed:", e)
        traceback.print_exc()
        return None

# ---------------- Signal parser and validator ----------------
def enhanced_parse(text):
    out = {}
    if not text:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == "NO_SIGNAL":
            continue
        if not any(k in line.upper() for k in ("BUY", "SELL")):
            continue
        parts = [p.strip() for p in line.split(" - ")]
        if len(parts) < 3:
            continue
        symbol = parts[0].upper().replace(" ", "")
        action = parts[1].upper()
        if action not in ["BUY", "SELL"]:
            continue
        entry = sl = tp = None
        reason = ""
        conf = None
        remainder = " - ".join(parts[2:])
        m_entry = re.search(r'ENTRY\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_sl = re.search(r'\bSL\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_tp = re.search(r'\bTP\b\s*[:=]\s*([0-9\.]+)', remainder, flags=re.I)
        m_conf = re.search(r'CONF(?:IDENCE)?\s*[:=]?\s*(\d{2,3})', remainder, flags=re.I)
        m_reason = re.search(r'REASON\s*[:=]\s*(.+?)(?:\s*-\s*CONF|$)', remainder, flags=re.I)
        if m_entry:
            entry = float(m_entry.group(1))
        if m_sl:
            sl = float(m_sl.group(1))
        if m_tp:
            tp = float(m_tp.group(1))
        if m_conf:
            conf = int(m_conf.group(1))
        if m_reason:
            reason = m_reason.group(1).strip()
        if not all([entry, sl, tp, conf]):
            print(f"Incomplete signal for {symbol}: entry={entry}, sl={sl}, tp={tp}, conf={conf}")
            continue
        if conf < SIGNAL_CONF_THRESHOLD:
            print(f"Low confidence signal for {symbol}: {conf}% < {SIGNAL_CONF_THRESHOLD}%")
            continue
        if action == "BUY":
            risk = entry - sl
            reward = tp - entry
        else:
            risk = sl - entry
            reward = entry - tp
        if risk <= 0 or reward <= 0:
            print(f"Invalid risk/reward for {symbol}: risk={risk}, reward={reward}")
            continue
        risk_reward_ratio = reward / risk
        if risk_reward_ratio < 1.5:
            print(f"Poor risk:reward ratio for {symbol}: {risk_reward_ratio:.2f}")
            continue
        if symbol in last_signals:
            time_since_last = datetime.now() - last_signals[symbol]
            if time_since_last < timedelta(hours=2):
                print(f"Recent signal exists for {symbol}, skipping")
                continue
        out[symbol] = {
            "action": action,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "reason": reason,
            "confidence": conf,
            "risk_reward": round(risk_reward_ratio, 2),
            "timestamp": datetime.now()
        }
        last_signals[symbol] = datetime.now()
    return out

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Skipping send_text.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown", "disable_web_page_preview": True}, timeout=30) as r:
            if r.status != 200:
                txt = await r.text()
                print(f"Telegram send_text failed {r.status}: {txt}")
    except Exception as e:
        print("send_text error:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured. Skipping send_photo.")
        try:
            os.remove(path)
        except:
            pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("parse_mode", "Markdown")
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with session.post(url, data=data, timeout=90) as r:
                if r.status != 200:
                    text = await r.text()
                    print(f"Telegram send_photo failed {r.status}: {text}")
    except Exception as e:
        print("send_photo error:", e)
    finally:
        try:
            os.remove(path)
        except:
            pass

# ---------------- Performance ----------------
def track_signal_performance():
    if not performance_tracking:
        return
    total_signals = len(performance_tracking)
    profitable = sum(1 for p in performance_tracking if p.get("profit", 0) > 0)
    if total_signals > 0:
        win_rate = (profitable / total_signals) * 100
        avg_profit = np.mean([p.get("profit", 0) for p in performance_tracking]) if performance_tracking else 0
        print(f"Performance: {total_signals} signals, {win_rate:.1f}% win rate, avg profit: {avg_profit:.2f}%")

# ---------------- Main loop ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        startup_msg = f"🤖 ENHANCED Crypto Trading Bot v2.0\n• Symbols: {len(SYMBOLS)} • TF:30m • Poll: {POLL_INTERVAL}s • Conf≥{SIGNAL_CONF_THRESHOLD}%"
        await send_text(session, startup_msg)
        iteration = 0
        while True:
            try:
                iteration += 1
                start_time = datetime.now()
                print(f"\nITERATION {iteration} @ {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
                # Fetch data
                fetch_tasks = [fetch_enhanced_data(session, symbol) for symbol in SYMBOLS]
                results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                market = {}
                fetch_errors = 0
                for symbol, result in zip(SYMBOLS, results):
                    if isinstance(result, Exception):
                        print(f"❌ Fetch error {symbol}: {result}")
                        fetch_errors += 1
                        continue
                    if result and result.get("price") is not None:
                        market[symbol] = result
                        print(f"✅ {symbol}: ${fmt_price(result.get('price'))}")
                    else:
                        print(f"⚠️  No price data for {symbol}")
                if not market:
                    print("❌ No market data available - sleeping...")
                    await asyncio.sleep(min(120, POLL_INTERVAL))
                    continue
                # For each symbol: create chart, create market summary, ask OpenAI for signals
                aggregate_summary_texts = []
                signals_from_ai_texts = []
                for symbol, data in market.items():
                    # Build structured summary text for model
                    summary = {
                        "symbol": symbol,
                        "price": data.get("price"),
                        "rsi": data.get("rsi"),
                        "macd": data.get("macd"),
                        "macd_signal": data.get("macd_signal"),
                        "macd_histogram": data.get("macd_histogram"),
                        "bb_upper": data.get("bb_upper"),
                        "bb_middle": data.get("bb_middle"),
                        "bb_lower": data.get("bb_lower"),
                        "volume_ratio": data.get("volume_analysis", {}).get("volume_ratio"),
                        "volume_spike": data.get("volume_analysis", {}).get("volume_spike"),
                        "order_imbalance": data.get("order_imbalance"),
                        "market_structure": data.get("market_structure"),
                        "patterns": list(data.get("patterns", {}).keys()),
                        "ob_support": data.get("ob_support"),
                        "ob_resistance": data.get("ob_resistance"),
                        "last_100_candles": len(data.get("candles", []))
                    }
                    summary_text = json.dumps(summary, default=str, indent=2)
                    aggregate_summary_texts.append(f"### {symbol}\n{summary_text}\n")
                    # Create chart PNG for last 100 candles (if present)
                    chart_path = None
                    if data.get("candles") and data.get("times"):
                        try:
                            chart_path = enhanced_plot_chart(data["times"], data["candles"], symbol, data)
                        except Exception as e:
                            print(f"Chart error for {symbol}: {e}")
                    # Prompt OpenAI per-symbol (small prompt) to get entry/sl/tp
                    per_symbol_prompt = f"{summary_text}\nChartPath: {chart_path if chart_path else 'N/A'}"
                    ai_response = await ask_openai_for_signals(per_symbol_prompt, chart_path=chart_path)
                    signals_from_ai_texts.append((symbol, ai_response, chart_path))
                # Combine AI outputs, parse and send to telegram
                combined_ai_text = "\n\n".join(f"{s}\n{txt}" for s, txt, _ in signals_from_ai_texts if txt)
                parsed_signals = {}
                for symbol, txt, chart_path in signals_from_ai_texts:
                    parsed = enhanced_parse(txt)
                    # parsed is dict with keys of symbols (likely same symbol)
                    for k, v in parsed.items():
                        parsed_signals[k] = {"signal": v, "chart": chart_path}
                # Send signals to telegram
                if parsed_signals:
                    for sym, payload in parsed_signals.items():
                        sig = payload["signal"]
                        chart = payload.get("chart")
                        action = sig["action"]
                        entry = sig["entry"]
                        sl = sig["sl"]
                        tp = sig["tp"]
                        confidence = sig["confidence"]
                        rr = sig["risk_reward"]
                        reason = sig.get("reason", "")
                        potential_profit = ((tp - entry) / entry) * 100 if action == "BUY" else ((entry - tp) / entry) * 100
                        risk_pct = ((entry - sl) / entry) * 100 if action == "BUY" else ((sl - entry) / entry) * 100
                        emoji = "🟢" if action == "BUY" else "🔴"
                        caption = f"""{emoji} *SIGNAL ALERT* {emoji}

*{sym}* → *{action}*
Entry: `{fmt_price(entry)}`
SL: `{fmt_price(sl)}`
TP: `{fmt_price(tp)}`

• Confidence: *{confidence}%*
• R:R: *1:{rr}*
• Risk: {risk_pct:.1f}% • Potential: +{potential_profit:.1f}%

_Reason:_ {reason}

{datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
"""
                        if chart:
                            await send_photo(session, caption, chart)
                        else:
                            await send_text(session, caption)
                        performance_tracking.append({
                            "symbol": sym,
                            "action": action,
                            "entry": entry,
                            "sl": sl,
                            "tp": tp,
                            "confidence": confidence,
                            "risk_reward": rr,
                            "timestamp": datetime.now(),
                            "reason": reason
                        })
                else:
                    print("No high-confidence signals this iteration.")
                # Periodic status & perf logging
                if iteration % 12 == 0:
                    track_signal_performance()
                    status_msg = f"📊 Status - Iteration {iteration}\nScanned: {len(market)}/{len(SYMBOLS)} pairs\nTotal Signals Logged: {len(performance_tracking)}"
                    await send_text(session, status_msg)
                iteration_time = (datetime.now() - start_time).total_seconds()
                print(f"Iteration {iteration} completed in {iteration_time:.1f}s. Sleeping {POLL_INTERVAL}s...")
                await asyncio.sleep(POLL_INTERVAL)
            except asyncio.CancelledError:
                print("Shutdown requested")
                await send_text(session, "🤖 Bot shutting down gracefully...")
                break
            except Exception as e:
                print("MAIN LOOP ERROR:", e)
                traceback.print_exc()
                try:
                    await send_text(session, f"⚠️ Bot Error: {str(e)[:200]}")
                except:
                    pass
                await asyncio.sleep(min(120, POLL_INTERVAL))

# ---------------- Entry ----------------
if __name__ == "__main__":
    print("Starting ENHANCED Crypto Trading Bot v2.0")
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("Stopped by user")
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
