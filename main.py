#!/usr/bin/env python3
# main.py - Price Action Master Bot v5.1 (Stable)
# Features: EMA confluence, Pattern detection, Trendlines, Auto R/R optimization (1:2 default), Clean Telegram alerts

import os, re, asyncio, aiohttp, traceback, numpy as np, json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

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
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))
TARGET_RR = float(os.getenv("TARGET_RR", 2.0))  # min risk:reward target
MAX_TP_MULT = float(os.getenv("MAX_TP_MULT", 1.25))  # max TP cap multiplier

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"
MIN_CANDLES_REQUIRED = 720

# ---------------- Utility Functions ----------------
def fmt_price(p):
    try:
        p = float(p)
    except:
        return str(p)
    return f"{p:.6f}" if abs(p) < 1 else f"{p:.2f}"

def ema(values, period):
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [None] * (period - 1)
    sma = sum(values[:period]) / period
    ema_vals.append(sma)
    prev = sma
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        ema_vals.append(prev)
    return ema_vals

def normalize_klines(raw_klines):
    out = []
    for row in raw_klines or []:
        if len(row) >= 6:
            out.append({
                "open": float(row[1]), "high": float(row[2]),
                "low": float(row[3]), "close": float(row[4]),
                "volume": float(row[5]), "ts": int(row[0])
            })
    return out

def calculate_emas(closes):
    return {
        'ema_9': ema(closes, 9)[-1] if len(closes) >= 9 else None,
        'ema_20': ema(closes, 20)[-1] if len(closes) >= 20 else None,
        'ema_9_series': ema(closes, 9),
        'ema_20_series': ema(closes, 20)
    }

# ---------------- Simple Candlestick Patterns ----------------
def detect_candlestick_patterns(candles):
    patterns = []
    if len(candles) < 3: return patterns
    for i in range(2, len(candles)):
        c0, c1, c2 = candles[i-2], candles[i-1], candles[i]
        o2, h2, l2, cl2 = c2['open'], c2['high'], c2['low'], c2['close']
        body = abs(cl2 - o2)
        range2 = h2 - l2
        # Bullish
        if body > 0 and (cl2 > o2) and ((h2 - cl2) < body) and ((o2 - l2) > body * 2):
            patterns.append({"type": "Hammer", "sentiment": "bullish", "strength": 8})
        # Bearish
        if body > 0 and (cl2 < o2) and ((cl2 - l2) < body) and ((h2 - o2) > body * 2):
            patterns.append({"type": "Shooting Star", "sentiment": "bearish", "strength": 8})
        # Doji
        if body < range2 * 0.1:
            patterns.append({"type": "Doji", "sentiment": "neutral", "strength": 5})
    return patterns

# ---------------- Trade Analysis Core ----------------
def analyze_trade_logic(raw_candles, rr_min=TARGET_RR):
    try:
        candles = normalize_klines(raw_candles)
        if len(candles) < MIN_CANDLES_REQUIRED:
            return {"side":"none","confidence":0,"reason":"insufficient candles"}

        closes = [c['close'] for c in candles]
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        current_price = closes[-1]

        emas = calculate_emas(closes)
        ema9, ema20 = emas['ema_9'], emas['ema_20']
        if not ema9 or not ema20:
            return {"side":"none","confidence":0,"reason":"no EMA data"}

        # base signal scoring
        score = 0
        if current_price > ema9: score += 2
        if ema9 > ema20: score += 3
        if current_price > ema20: score += 1

        # candle pattern impact
        pats = detect_candlestick_patterns(candles[-60:])
        for p in pats:
            if p['sentiment'] == 'bullish': score += p['strength'] * 0.5
            elif p['sentiment'] == 'bearish': score -= p['strength'] * 0.5

        base_conf = min(95, 60 + abs(score)*2.5)

        # --- BUY logic (with smart TP adjust) ---
        if score >= 4.5:
            sl = current_price * 0.985
            risk = current_price - sl
            tp_target = current_price + risk * rr_min
            if tp_target > current_price * MAX_TP_MULT:
                tp_target = current_price * MAX_TP_MULT
            rr = (tp_target - current_price) / (risk) if risk>0 else 0

            if rr >= rr_min:
                return {
                    "side":"BUY","entry":current_price,"sl":sl,"tp":tp_target,
                    "confidence":int(base_conf),"score":round(score,2),
                    "rr":round(rr,2),"reason":"EMA alignment + bullish patterns"
                }
            else:
                return {
                    "side":"none","candidate_entry":current_price,"candidate_sl":sl,
                    "candidate_tp":tp_target,"candidate_rr":round(rr,2),
                    "confidence":int(base_conf),
                    "reason":f"BUY rejected: R/R {rr:.2f} < {rr_min}"
                }

        # --- SELL logic (mirror) ---
        elif score <= -4.5:
            sl = current_price * 1.015
            risk = sl - current_price
            tp_target = current_price - risk * rr_min
            if tp_target < current_price * (2 - MAX_TP_MULT):
                tp_target = current_price * (2 - MAX_TP_MULT)
            rr = (current_price - tp_target) / (risk) if risk>0 else 0

            if rr >= rr_min:
                return {
                    "side":"SELL","entry":current_price,"sl":sl,"tp":tp_target,
                    "confidence":int(base_conf),"score":round(score,2),
                    "rr":round(rr,2),"reason":"EMA bearish + negative candle patterns"
                }
            else:
                return {
                    "side":"none","candidate_entry":current_price,"candidate_sl":sl,
                    "candidate_tp":tp_target,"candidate_rr":round(rr,2),
                    "confidence":int(base_conf),
                    "reason":f"SELL rejected: R/R {rr:.2f} < {rr_min}"
                }

        else:
            return {
                "side":"none","confidence":int(base_conf),
                "reason":"no clear directional setup","score":round(score,2)
            }
    except Exception as e:
        traceback.print_exc()
        return {"side":"none","confidence":0,"reason":str(e)}

# ---------------- Chart Generation ----------------
def plot_signal_chart(symbol, raw_candles, signal):
    candles = normalize_klines(raw_candles)
    if len(candles)<50:
        tmp = NamedTemporaryFile(delete=False,suffix=".png")
        plt.figure(figsize=(5,2)); plt.text(0.5,0.5,"No data",ha='center'); plt.axis('off'); plt.savefig(tmp.name); plt.close(); return tmp.name

    dates = [datetime.utcfromtimestamp(c['ts']/1000) for c in candles]
    closes = [c['close'] for c in candles]; opens = [c['open'] for c in candles]
    highs = [c['high'] for c in candles]; lows = [c['low'] for c in candles]
    x = date2num(dates)

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,9),gridspec_kw={'height_ratios':[4,1]})
    fig.patch.set_facecolor('#0a0e27'); ax1.set_facecolor('#0f1419'); ax2.set_facecolor('#0f1419')
    for xi,o,h,l,c in zip(x,opens,highs,lows,closes):
        color='#26a69a' if c>=o else '#ef5350'
        ax1.plot([xi,xi],[l,h],color=color,linewidth=0.6)
        ax1.plot([xi,xi],[o,c],color=color,linewidth=2)

    ax1.plot(x, ema(closes,9), color='#42a5f5', label='EMA9')
    ax1.plot(x, ema(closes,20), color='#ffa726', label='EMA20')

    if signal.get('entry'):
        ax1.axhline(signal['entry'],color='#ffeb3b',linestyle='--',label='Entry')
    if signal.get('sl'):
        ax1.axhline(signal['sl'],color='#ff5252',linestyle='--',label='SL')
    if signal.get('tp'):
        ax1.axhline(signal['tp'],color='#69f0ae',linestyle='--',label='TP')

    ax1.legend(loc='upper left',fontsize=9,facecolor='#1a1f2e')
    ax1.set_title(f"{symbol} - Price Action Analysis",color='white',fontsize=15)
    ax1.tick_params(colors='white')
    ax2.bar(x,[c['volume'] for c in candles],color='#42a5f5',alpha=0.6)
    ax2.tick_params(colors='white'); plt.tight_layout()
    tmp = NamedTemporaryFile(delete=False,suffix=".png")
    plt.savefig(tmp.name,facecolor='#0a0e27',dpi=150,bbox_inches='tight'); plt.close(fig)
    return tmp.name

# ---------------- Telegram ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"})

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(caption); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path,"rb") as f:
        data=aiohttp.FormData()
        data.add_field("chat_id",TELEGRAM_CHAT_ID)
        data.add_field("caption",caption,content_type="text/html")
        data.add_field("photo",f,filename=os.path.basename(path),content_type="image/png")
        await session.post(url,data=data)
    os.unlink(path)

# ---------------- Fetch ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url,timeout=30) as r:
            if r.status!=200: return None
            return await r.json()
    except: return None

# ---------------- MAIN LOOP ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session,"🚀 <b>Price Action Master Bot v5.1 Started!</b>\nSymbols: "+", ".join(SYMBOLS))
        while True:
            print(f"\n===== {datetime.utcnow()} UTC Scan =====")
            summary=[]
            for sym in SYMBOLS:
                data=await fetch_json(session,CANDLE_URL.format(symbol=sym))
                if not data: continue
                sig=analyze_trade_logic(data)
                conf=sig.get('confidence',0)
                rr=sig.get('rr') or sig.get('candidate_rr') or 0
                if sig.get('side')!="none" and conf>=SIGNAL_CONF_THRESHOLD:
                    msg=(f"🎯 <b>{sym} {sig['side']} SIGNAL</b>\n"
                         f"📊 Conf: <b>{conf}%</b> | R/R: <b>1:{rr}</b>\n"
                         f"💰 Entry: <code>{fmt_price(sig['entry'])}</code>\n"
                         f"🛑 SL: <code>{fmt_price(sig['sl'])}</code>\n"
                         f"🎯 TP: <code>{fmt_price(sig['tp'])}</code>\n"
                         f"📈 Score: {sig.get('score',0)}\n💡 {sig.get('reason','')}")
                    chart=plot_signal_chart(sym,data,sig)
                    await send_photo(session,msg,chart)
                    print(f"✅ {sym} signal sent.")
                else:
                    summary.append(f"{sym}: {conf}% | R/R {rr}")
            await send_text(session,"📊 Scan complete:\n"+"\n".join(summary))
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try:
        asyncio.run(enhanced_loop())
    except KeyboardInterrupt:
        print("🛑 Bot stopped manually")
    except Exception as e:
        traceback.print_exc()
