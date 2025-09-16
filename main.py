# main_futures.py
# Binance Futures Bot (30m, 100 candles, Entry/SL/Targets, OI + Funding Data)

import os
import asyncio
import aiohttp
import traceback
from datetime import datetime
from dotenv import load_dotenv
from tempfile import NamedTemporaryFile
import numpy as np
from matplotlib.dates import date2num
import matplotlib.pyplot as plt

load_dotenv()

# --- Config ---
SYMBOLS = [
    "BTCUSDT","ETHUSDT","SOLUSDT","XRPUSDT","AAVEUSDT",
    "TRXUSDT","DOGEUSDT","BNBUSDT","ADAUSDT","LTCUSDT","LINKUSDT"
]
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 1800))
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SIGNAL_CONF_THRESHOLD = float(os.getenv("SIGNAL_CONF_THRESHOLD", 60.0))

# Binance Futures base URL
BASE_FAPI = "https://fapi.binance.com"

KLINE_URL = BASE_FAPI + "/fapi/v1/klines?symbol={symbol}&interval=30m&limit=100"
TICKER_URL = BASE_FAPI + "/fapi/v1/ticker/24hr?symbol={symbol}"
OI_URL = BASE_FAPI + "/fapi/v1/openInterest?symbol={symbol}"
FUNDING_URL = BASE_FAPI + "/fapi/v1/fundingRate?symbol={symbol}&limit=1"

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        await session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"})
    except Exception as e:
        print("send_text error:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("photo", f, filename="chart.png", content_type="image/png")
            await session.post(url, data=data)
    except Exception as e:
        print("send_photo error:", e)
    finally:
        try: os.remove(path)
        except: pass

# ---------------- Fetching ----------------
async def fetch_json(session, url):
    try:
        async with session.get(url, timeout=15) as r:
            if r.status != 200:
                return None
            return await r.json()
    except:
        return None

async def fetch_data(session, symbol):
    candles = await fetch_json(session, KLINE_URL.format(symbol=symbol))
    ticker = await fetch_json(session, TICKER_URL.format(symbol=symbol))
    oi = await fetch_json(session, OI_URL.format(symbol=symbol))
    funding = await fetch_json(session, FUNDING_URL.format(symbol=symbol))
    
    out = {}
    if isinstance(candles, list):
        out["candles"] = [[float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])] for x in candles]
        out["times"] = [int(x[0])//1000 for x in candles]
    if ticker:
        out["price"] = float(ticker.get("lastPrice",0))
        out["volume"] = float(ticker.get("volume",0))
    if oi:
        out["openInterest"] = float(oi.get("openInterest",0))
    if funding and isinstance(funding, list) and funding:
        out["fundingRate"] = float(funding[0].get("fundingRate",0))
    return out

# ---------------- PA Detector ----------------
def compute_levels(candles, lookback=50):
    if not candles: return (None,None,None)
    arr = candles[-lookback:]
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3,len(arr))
    res = sum(highs[:k])/k if highs else None
    sup = sum(lows[:k])/k if lows else None
    mid = (res+sup)/2 if res and sup else None
    return sup,res,mid

def detect_signal(sym, data):
    candles = data.get("candles")
    if not candles or len(candles)<5: return None
    closes = [c[3] for c in candles]
    last = candles[-1]; prev = candles[-2]
    sup,res,mid = compute_levels(candles)
    entry = last[3]; sl=None; targets=[]
    bias="NEUTRAL"; reason=[]; conf=50

    # breakout/breakdown
    if res and entry>res: bias="BUY"; reason.append("Breakout"); conf+=15
    if sup and entry<sup: bias="SELL"; reason.append("Breakdown"); conf+=15

    # engulfing
    if last[3]>last[0] and prev[3]<prev[0]: reason.append("Bullish Engulfing"); bias="BUY"; conf+=10
    if last[3]<last[0] and prev[3]>prev[0]: reason.append("Bearish Engulfing"); bias="SELL"; conf+=10

    # SL
    if bias=="BUY": sl=min([c[2] for c in candles[-6:]])*0.997
    if bias=="SELL": sl=max([c[1] for c in candles[-6:]])*1.003

    if sl:
        risk=abs(entry-sl)
        if bias=="BUY": targets=[entry+risk*r for r in (1,2,3)]
        if bias=="SELL": targets=[entry-risk*r for r in (1,2,3)]

    # add futures metrics
    oi = data.get("openInterest")
    funding = data.get("fundingRate")
    if oi: reason.append(f"OI={oi}")
    if funding: reason.append(f"Funding={funding}")

    return {"bias":bias,"entry":entry,"sl":sl,"targets":targets,"reason":"; ".join(reason),"conf":conf}

# ---------------- Chart ----------------
def plot_chart(times,candles,sym,trade):
    if not candles: return None
    dates=[datetime.utcfromtimestamp(t) for t in times]
    o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c_=[c[3] for c in candles]
    x=date2num(dates)
    fig,ax=plt.subplots(figsize=(9,5),dpi=100)

    width=0.6*(x[1]-x[0]) if len(x)>1 else 0.4
    for xi,oi,hi,li,ci in zip(x,o,h,l,c_):
        col="white" if ci>=oi else "black"
        ax.vlines(xi,li,hi,color="black",linewidth=0.7)
        ax.add_patch(plt.Rectangle((xi-width/2,min(oi,ci)),width,max(0.0001,abs(ci-oi)),facecolor=col,edgecolor="black"))

    entry=trade.get("entry"); sl=trade.get("sl"); targets=trade.get("targets",[])
    if entry: ax.axhline(entry,color="blue",label=f"Entry {entry}")
    if sl: ax.axhline(sl,color="red",linestyle="--",label=f"SL {sl}")
    for i,t in enumerate(targets): ax.axhline(t,color="green",linestyle=":",label=f"T{i+1} {t}")

    ax.set_title(sym+" Futures 30m")
    ax.legend(loc="upper left",fontsize="small")
    tmp=NamedTemporaryFile(delete=False,suffix=".png")
    fig.savefig(tmp.name,bbox_inches="tight"); plt.close(fig)
    return tmp.name

# ---------------- Loop ----------------
async def loop():
    async with aiohttp.ClientSession() as session:
        await send_text(session,"Futures Bot online 🚀")
        while True:
            try:
                tasks=[fetch_data(session,s) for s in SYMBOLS]
                res=await asyncio.gather(*tasks)
                market={s:r for s,r in zip(SYMBOLS,res)}
                for s,d in market.items():
                    trade=detect_signal(s,d)
                    if trade and trade["bias"] in ("BUY","SELL") and trade["conf"]>=SIGNAL_CONF_THRESHOLD:
                        caption=(f"🚨 {s} → {trade['bias']}\n"
                                 f"Entry: {trade['entry']}\nSL: {trade['sl']}\n"
                                 f"Targets: {trade['targets']}\n"
                                 f"Reason: {trade['reason']}\nConf: {trade['conf']}%")
                        chart=plot_chart(d.get("times"),d.get("candles"),s,trade)
                        await send_text(session,caption)
                        if chart: await send_photo(session,caption,chart)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("main loop error:",e)
                traceback.print_exc()
                await asyncio.sleep(60)

if __name__=="__main__":
    try: asyncio.run(loop())
    except KeyboardInterrupt: print("Stopped")
