#!/usr/bin/env python3
# main.py - Enhanced Crypto Trading Bot v4.0 (with charts + startup msg)
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
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 80.0))

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
CANDLE_URL = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=30m&limit=100"
CANDLE_URL_1H = "https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1h&limit=100"

# ---------------- Utils ----------------
def fmt_price(p): return f"{p:.6f}" if abs(p)<1 else f"{p:.2f}"

# EMA + Logic
def ema(values, period):
    if len(values)<period: return []
    k=2/(period+1); prev=sum(values[:period])/period
    arr=[None]*(period-1)+[prev]
    for v in values[period:]:
        prev=v*k+prev*(1-k); arr.append(prev)
    return arr

def horizontal_levels(closes, highs, lows, lookback=50, binsize=0.002):
    pts=closes[-lookback:]+highs[-lookback:]+lows[-lookback:]; lvls=[]
    for p in pts:
        found=False
        for lv in lvls:
            if abs((lv["price"]-p)/p)<binsize:
                lv["count"]+=1
                lv["price"]=(lv["price"]*(lv["count"]-1)+p)/lv["count"]
                found=True; break
        if not found: lvls.append({"price":p,"count":1})
    lvls.sort(key=lambda x:-x["count"])
    return [lv["price"] for lv in lvls[:5]]

def analyze_trade_logic(candles, rr_min=1.5):
    closes=[c[3] for c in candles]; highs=[c[1] for c in candles]; lows=[c[2] for c in candles]
    if len(closes)<30: return {"side":"none","confidence":0,"reason":"not enough data"}
    es,el=ema(closes,9)[-1],ema(closes,21)[-1]; price=closes[-1]; lvls=horizontal_levels(closes,highs,lows)
    sup=max([lv for lv in lvls if lv<price],default=None); res=min([lv for lv in lvls if lv>price],default=None)
    conf=50; reasons=[]
    if price<es and price<el: conf+=10; reasons.append("below EMAs")
    elif price>es and price>el: conf+=10; reasons.append("above EMAs")
    else: conf-=5; reasons.append("EMAs mixed")
    if price<es and res:
        entry,stop=price,res*1.003; tgt=sup if sup else price-(stop-entry)*1.5
        rr=(entry-tgt)/(stop-entry) if stop>entry else 0
        if rr>=rr_min: return {"side":"SELL","entry":entry,"sl":stop,"tp":tgt,"confidence":conf+10,"reason":"; ".join(reasons)}
    if price>es and res:
        entry,stop=price,(sup*0.997 if sup else price*0.99); tgt=res
        rr=(tgt-entry)/(entry-stop) if entry>stop else 0
        if rr>=rr_min: return {"side":"BUY","entry":entry,"sl":stop,"tp":tgt,"confidence":conf+10,"reason":"; ".join(reasons)}
    return {"side":"none","confidence":conf,"reason":"; ".join(reasons)}

def multi_tf_confirmation(c30,c1h):
    s30=analyze_trade_logic(c30); s1h=analyze_trade_logic(c1h)
    if s30["side"]=="none" or s1h["side"]=="none": return {"side":"none","confidence":0,"reason":"no align"}
    if s30["side"]==s1h["side"]: s30["confidence"]=min(100,s30["confidence"]+15); s30["reason"]+="; aligned 1h"; return s30
    return {"side":"none","confidence":0,"reason":"TF conflict"}

# ---------------- Chart ----------------
def plot_signal_chart(symbol, candles, signal):
    dates=[datetime.utcfromtimestamp(int(x[0])/1000) for x in candles]
    closes=[float(x[4]) for x in candles]; highs=[float(x[2]) for x in candles]; lows=[float(x[3]) for x in candles]
    x=date2num(dates); fig,ax=plt.subplots(figsize=(12,6))
    for xi,c in zip(x,candles):
        o,h,l,cl=float(c[1]),float(c[2]),float(c[3]),float(c[4])
        color="g" if cl>=o else "r"; ax.plot([xi,xi],[l,h],color=color); ax.plot(xi,cl,"o",color=color)
    ax.axhline(signal["entry"],color="blue",ls="--",label=f"Entry {fmt_price(signal['entry'])}")
    ax.axhline(signal["sl"],color="red",ls="--",label=f"SL {fmt_price(signal['sl'])}")
    ax.axhline(signal["tp"],color="green",ls="--",label=f"TP {fmt_price(signal['tp'])}")
    ax.legend(); ax.set_title(f"{symbol} Signal {signal['side']} | Conf {signal['confidence']}%")
    tmp=NamedTemporaryFile(delete=False,suffix=".png"); fig.savefig(tmp.name); plt.close(fig); return tmp.name

# ---------------- Fetch ----------------
async def fetch_json(session,url):
    try:
        async with session.get(url,timeout=20) as r:
            if r.status!=200: return None
            return await r.json()
    except: return None

# ---------------- AI ----------------
async def ask_openai_for_signals(summary):
    if not client: return None
    sys="You are pro crypto trader. Only output signals." 
    usr=f"{summary}\nRules: Conf≥{int(SIGNAL_CONF_THRESHOLD)}% Format: SYMBOL - BUY/SELL - ENTRY:x - SL:y - TP:z - REASON:.. - CONF:n%"
    try:
        loop=asyncio.get_running_loop()
        def call(): return client.chat.completions.create(model=OPENAI_MODEL,
            messages=[{"role":"system","content":sys},{"role":"user","content":usr}],
            max_tokens=400,temperature=0.0)
        resp=await loop.run_in_executor(None,call)
        return resp.choices[0].message.content.strip()
    except: return None

# ---------------- Telegram ----------------
async def send_text(session,text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(text); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    await session.post(url,json={"chat_id":TELEGRAM_CHAT_ID,"text":text})

async def send_photo(session,caption,path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: print(caption); return
    url=f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(path,"rb") as f:
        data=aiohttp.FormData(); data.add_field("chat_id",TELEGRAM_CHAT_ID); data.add_field("caption",caption); data.add_field("photo",f)
        await session.post(url,data=data)

# ---------------- Main ----------------
async def enhanced_loop():
    async with aiohttp.ClientSession() as session:
        startup=f"🤖 Bot started successfully! • {len(SYMBOLS)} symbols • Poll {POLL_INTERVAL}s"
        print(startup); await send_text(session,startup)
        it=0
        while True:
            it+=1; print(f"\nITER {it} @ {datetime.now().strftime('%H:%M:%S')}")
            for sym in SYMBOLS:
                try:
                    c30=await fetch_json(session,CANDLE_URL.format(symbol=sym))
                    c1h=await fetch_json(session,CANDLE_URL_1H.format(symbol=sym))
                    if not c30 or not c1h: continue
                    can30=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c30]
                    can1h=[[float(x[1]),float(x[2]),float(x[3]),float(x[4])] for x in c1h]
                    local=multi_tf_confirmation(can30,can1h)
                    ai=await ask_openai_for_signals(json.dumps({"symbol":sym,"last_price":can30[-1][3]},indent=2))
                    if local["side"]!="none":
                        conf=local["confidence"]+(10 if ai and "NO_SIGNAL" not in ai else 0)
                        msg=f"{sym} {local['side']} | Entry {fmt_price(local['entry'])} | SL {fmt_price(local['sl'])} | TP {fmt_price(local['tp'])} | Conf {conf}%\nReason: {local['reason']}"
                        chart=plot_signal_chart(sym,c30,local)
                        await send_photo(session,msg,chart)
                        print("⚡",msg)
                    else:
                        print(f"{sym}: no signal")
                except Exception as e: print("Error",sym,e)
            await asyncio.sleep(POLL_INTERVAL)

if __name__=="__main__":
    try: asyncio.run(enhanced_loop())
    except KeyboardInterrupt: print("Stopped by user")
