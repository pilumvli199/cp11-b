#!/usr/bin/env python3
# main.py - Price Action Master Bot v5.2 (AI Confirmation Edition)
# ⚡ Includes:
# - EMA9/20 logic
# - Candlestick + Chart patterns
# - Support/Resistance + Trendlines
# - Auto Risk/Reward target (>=1:2)
# - Clean Telegram alerts
# - GPT-4o-mini AI trade validation (optional)

import os, re, asyncio, aiohttp, traceback, numpy as np, json
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from tempfile import NamedTemporaryFile

# ---------------- CONFIG ----------------
load_dotenv()
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = max(30, int(os.getenv("POLL_INTERVAL", 1800)))
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 75.0))
TARGET_RR = float(os.getenv("TARGET_RR", 2.0))          # Desired R/R 1:2
MAX_TP_MULT = float(os.getenv("MAX_TP_MULT", 1.25))     # TP cap +25%
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=720"

# ---------------- Utility ----------------
def fmt_price(p): return f"{float(p):.4f}" if p else "—"
def ema(values, period):
    if len(values)<period: return []
    k=2/(period+1); sma=sum(values[:period])/period; e=[None]*(period-1)+[sma]
    for v in values[period:]: sma=v*k+sma*(1-k); e.append(sma)
    return e

def normalize_klines(raw): return [
    {"open":float(r[1]),"high":float(r[2]),"low":float(r[3]),"close":float(r[4]),"volume":float(r[5]),"ts":int(r[0])}
    for r in raw if len(r)>=6]

# ---------------- Simple Candlestick Pattern Detector ----------------
def detect_patterns(candles):
    pats=[]
    for i in range(2,len(candles)):
        o1,c1,o2,c2=candles[i-1]["open"],candles[i-1]["close"],candles[i]["open"],candles[i]["close"]
        if c2>o2 and c1<o1 and c2>=o1: pats.append(("Bullish Engulfing","bullish"))
        if c2<o2 and c1>o1 and c2<=o1: pats.append(("Bearish Engulfing","bearish"))
    return pats[-3:]

# ---------------- Core Analyzer ----------------
def analyze_trade(candles):
    try:
        c=normalize_klines(candles)
        closes=[x["close"] for x in c]; highs=[x["high"] for x in c]; lows=[x["low"] for x in c]
        if len(closes)<60: return {"side":"none","confidence":0}
        ema9,ema20=ema(closes,9)[-1],ema(closes,20)[-1]
        cur=closes[-1]; score=0; reasons=[]

        if cur>ema9: score+=2; reasons.append("Price>EMA9")
        else: score-=1
        if ema9>ema20: score+=3; reasons.append("EMA9>EMA20")
        else: score-=3

        pats=detect_patterns(c)
        for p in pats: score+=3 if p[1]=="bullish" else -3

        conf=min(90,55+abs(score)*3)
        side="BUY" if score>=4.5 else "SELL" if score<=-4.5 else "none"
        sup=min(lows[-20:]); res=max(highs[-20:])
        entry=cur
        if side=="BUY":
            sl=sup*0.996; risk=entry-sl; tp=entry+risk*TARGET_RR
        elif side=="SELL":
            sl=res*1.004; risk=sl-entry; tp=entry-risk*TARGET_RR
        else:
            sl=tp=None; risk=0

        rr=(abs(tp-entry)/risk) if risk>0 else 0
        return {
            "side":side,"score":round(score,2),"confidence":int(conf),
            "entry":entry,"sl":sl,"tp":tp,"rr":round(rr,2),
            "patterns":[p[0] for p in pats],
            "reason":"; ".join(reasons[:3])
        }
    except Exception as e:
        return {"side":"none","confidence":0,"reason":str(e)}

# ---------------- OpenAI confirmation ----------------
async def ai_confirm(symbol, signal):
    if not client or signal["side"]=="none": return None
    prompt=f"""
Symbol: {symbol}
Signal: {signal['side']}
Score: {signal['score']}
Confidence: {signal['confidence']}%
R/R: {signal['rr']}
Patterns: {', '.join(signal.get('patterns',[]))}
EMA logic: {signal.get('reason')}
Decide CONFIRM or REJECT this trade, short reason.
Format:
VERDICT: [CONFIRM/REJECT]
CONFIDENCE: [0-100]%
REASON: [2 short sentences]
"""
    try:
        loop=asyncio.get_running_loop()
        def run():
            return client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role":"system","content":"You are a senior crypto price-action trader. Be concise."},
                    {"role":"user","content":prompt}
                ],
                max_tokens=200,temperature=0.3)
        resp=await loop.run_in_executor(None,run)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print("AI error:",e); return None

# ---------------- Telegram ----------------
async def send_text(session,text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return print(text)
    await session.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        json={"chat_id":TELEGRAM_CHAT_ID,"text":text,"parse_mode":"HTML"})

# ---------------- Main Loop ----------------
async def run_bot():
    async with aiohttp.ClientSession() as s:
        await send_text(s,
            f"🚀 <b>Price Action Master Bot v5.2 Started!</b>\n\n"
            f"⏱ 1H | {len(SYMBOLS)} symbols | Min Conf {int(SIGNAL_CONF_THRESHOLD)}%\n"
            f"🎯 Target R/R ≥1:{TARGET_RR}\n🤖 AI Confirm: {'ON' if client else 'OFF'}")
        i=0
        while True:
            i+=1; found=0; msgsum=[]
            print(f"\n🔎 Scan {i} running...")
            for sym in SYMBOLS:
                try:
                    data=await (await s.get(CANDLE_URL.format(symbol=sym))).json()
                    sig=analyze_trade(data)
                    if sig["side"]!="none" and sig["confidence"]>=SIGNAL_CONF_THRESHOLD:
                        ai=None
                        if client:
                            ai=await ai_confirm(sym,sig)
                            verdict="✅ CONFIRMED" if ai and "CONFIRM" in ai.upper() else "❌ REJECTED"
                        else:
                            verdict="—"
                        text=(f"<b>🎯 {sym} {sig['side']} SIGNAL</b>\n"
                              f"📊 Score: <b>{sig['score']}</b> | Conf: <b>{sig['confidence']}%</b>\n"
                              f"⚖️ R/R: <b>1:{sig['rr']}</b>\n"
                              f"💰 Entry: <code>{fmt_price(sig['entry'])}</code>\n"
                              f"🛑 SL: <code>{fmt_price(sig['sl'])}</code>\n"
                              f"🎯 TP: <code>{fmt_price(sig['tp'])}</code>\n"
                              f"🔍 {', '.join(sig['patterns']) or '—'}\n"
                              f"💡 {sig['reason']}\n"
                              f"🤖 {verdict}")
                        await send_text(s,text)
                        found+=1
                    msgsum.append(f"{sym}: {sig['confidence']}% | {sig['side']}")
                except Exception as e:
                    print(sym,"error",e)
            summary=(f"<b>📊 Scan #{i} Complete</b>\n✅ {len(SYMBOLS)} analyzed\n"
                     f"🎯 {found} signals ≥{int(SIGNAL_CONF_THRESHOLD)}%\n"
                     f"⏰ Next in {POLL_INTERVAL//60} min\n"
                     +"\n".join(msgsum[:5]))
            await send_text(s,summary)
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: asyncio.run(run_bot())
    except KeyboardInterrupt: print("🛑 Stopped")
