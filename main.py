import asyncio
import aiohttp
import redis.asyncio as redis
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Tuple
from telegram import Bot
from telegram.error import TelegramError
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# ======================== CONFIGURATION ========================
import os

DERIBIT_API_URL = "https://www.deribit.com/api/v2/public"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_CHAT_ID")

# Redis config (Railway addon) - Use environment variable
REDIS_URL = os.getenv("REDIS_URL", None)  # Railway provides this automatically

# Trading params
ASSETS = ["BTC", "ETH"]
TIMEFRAME = "30"  # 30 minutes
ATR_PERIOD = 6
ATR_MULTIPLIER = 2.0
VOLUME_SPIKE_THRESHOLD = 1.5
MAX_PAIN_PROXIMITY_PERCENT = 3.0  # ±3%
CALL_PUT_RATIO_LONG = 1.2
CALL_PUT_RATIO_SHORT = 0.8
OI_INCREASE_30MIN = 10  # 10% increase in 30min
OI_INCREASE_2HR = 15    # 15% increase in 2hr
ATM_RANGE_PERCENT = 5.0  # ATM ±5%

# Analysis interval
ANALYSIS_INTERVAL = 30 * 60  # 30 minutes

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ======================== REDIS MEMORY MANAGER ========================
class RedisMemory:
    def __init__(self, redis_url: Optional[str]):
        self.redis_url = redis_url
        self.client = None
        self.enabled = redis_url is not None
        
    async def connect(self):
        """Connect to Redis"""
        if not self.enabled:
            logger.warning("⚠️ Redis URL not provided - running without historical memory")
            logger.warning("   OI changes will not be tracked. Set REDIS_URL environment variable to enable.")
            return
            
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.error(f"❌ Redis connection failed: {e}")
            logger.warning("⚠️ Continuing without Redis - OI changes will not be tracked")
            self.enabled = False
            self.client = None
    
    async def close(self):
        """Close Redis connection"""
        if self.client:
            await self.client.close()
            logger.info("👋 Redis connection closed")
    
    async def store_oi_data(self, asset: str, strike: float, oi_data: Dict, ttl: int = 86400):
        """Store OI data with 24hr TTL"""
        if not self.enabled or not self.client:
            return
        try:
            key = f"oi:{asset}:{strike}:{int(datetime.now().timestamp())}"
            await self.client.setex(key, ttl, json.dumps(oi_data))
        except Exception as e:
            logger.debug(f"Redis store error: {e}")
    
    async def get_oi_history(self, asset: str, strike: float, hours: int = 2) -> List[Dict]:
        """Get OI history for last N hours"""
        if not self.enabled or not self.client:
            return []
            
        try:
            pattern = f"oi:{asset}:{strike}:*"
            keys = []
            
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            # Filter by timestamp
            cutoff = datetime.now().timestamp() - (hours * 3600)
            valid_keys = [k for k in keys if int(k.split(":")[-1]) >= cutoff]
            
            # Get data
            history = []
            for key in valid_keys:
                data = await self.client.get(key)
                if data:
                    history.append(json.loads(data))
            
            return sorted(history, key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            return []
    
    async def store_price_data(self, asset: str, price: float, volume: float):
        """Store price and volume data"""
        if not self.enabled or not self.client:
            return
        try:
            key = f"price:{asset}:{int(datetime.now().timestamp())}"
            data = {'price': price, 'volume': volume, 'timestamp': datetime.now().timestamp()}
            await self.client.setex(key, 86400, json.dumps(data))
        except Exception as e:
            logger.debug(f"Redis store error: {e}")
    
    async def get_price_history(self, asset: str, hours: int = 2) -> List[Dict]:
        """Get price history for ATR calculation"""
        if not self.enabled or not self.client:
            return []
            
        try:
            pattern = f"price:{asset}:*"
            keys = []
            
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            cutoff = datetime.now().timestamp() - (hours * 3600)
            valid_keys = [k for k in keys if int(k.split(":")[-1]) >= cutoff]
            
            history = []
            for key in valid_keys:
                data = await self.client.get(key)
                if data:
                    history.append(json.loads(data))
            
            return sorted(history, key=lambda x: x.get('timestamp', 0))
        except Exception as e:
            logger.debug(f"Redis get error: {e}")
            return []

# ======================== DERIBIT API CLIENT ========================
class DeribitOptionsAPI:
    def __init__(self):
        self.session = None
        self.base_url = DERIBIT_API_URL
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_current_price(self, asset: str) -> float:
        """Get current spot price"""
        try:
            url = f"{self.base_url}/get_index_price"
            params = {"index_name": f"{asset.lower()}_usd"}
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                if data.get("result"):
                    return data["result"]["index_price"]
                return 0.0
        except Exception as e:
            logger.error(f"❌ Error fetching {asset} price: {e}")
            return 0.0
    
    async def get_active_expiries(self, asset: str) -> List[int]:
        """Get all active option expiries"""
        try:
            url = f"{self.base_url}/get_instruments"
            params = {
                "currency": asset,
                "kind": "option",
                "expired": "false"  # String, not boolean!
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("result"):
                    expiries = set()
                    for instrument in data["result"]:
                        expiries.add(instrument["expiration_timestamp"])
                    
                    return sorted(list(expiries))
                return []
        except Exception as e:
            logger.error(f"❌ Error fetching expiries: {e}")
            return []
    
    async def get_current_expiry(self, asset: str) -> Optional[int]:
        """Get current/nearest expiry timestamp"""
        expiries = await self.get_active_expiries(asset)
        if expiries:
            now = int(datetime.now().timestamp() * 1000)
            # Find nearest expiry that's still in future
            future_expiries = [exp for exp in expiries if exp > now]
            if future_expiries:
                return min(future_expiries)
        return None
    
    async def get_option_chain(self, asset: str, expiry: int) -> Dict[str, List[Dict]]:
        """Get complete option chain for an expiry"""
        try:
            url = f"{self.base_url}/get_book_summary_by_currency"
            params = {
                "currency": asset,
                "kind": "option"
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if not data.get("result"):
                    return {"calls": [], "puts": []}
                
                calls = []
                puts = []
                
                for instrument in data["result"]:
                    # Filter by expiry
                    if instrument.get("expiration_timestamp") != expiry:
                        continue
                    
                    strike = instrument.get("strike")
                    option_type = instrument.get("option_type")
                    
                    option_data = {
                        "strike": strike,
                        "open_interest": instrument.get("open_interest", 0),
                        "volume": instrument.get("volume", 0),
                        "mark_price": instrument.get("mark_price", 0),
                        "instrument_name": instrument.get("instrument_name", "")
                    }
                    
                    if option_type == "call":
                        calls.append(option_data)
                    else:
                        puts.append(option_data)
                
                # Sort by strike
                calls.sort(key=lambda x: x["strike"])
                puts.sort(key=lambda x: x["strike"])
                
                return {"calls": calls, "puts": puts}
                
        except Exception as e:
            logger.error(f"❌ Error fetching option chain: {e}")
            return {"calls": [], "puts": []}
    
    async def get_ohlcv(self, asset: str, count: int = 20) -> pd.DataFrame:
        """Get OHLCV data for ATR calculation"""
        try:
            instrument = f"{asset}-PERPETUAL"
            url = f"{self.base_url}/get_tradingview_chart_data"
            
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = end_time - (30 * 60 * 1000 * count)
            
            params = {
                "instrument_name": instrument,
                "resolution": TIMEFRAME,
                "start_timestamp": start_time,
                "end_timestamp": end_time
            }
            
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                if data.get("result"):
                    result = data["result"]
                    df = pd.DataFrame({
                        'open': result['open'],
                        'high': result['high'],
                        'low': result['low'],
                        'close': result['close'],
                        'volume': result['volume']
                    })
                    return df
                
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ Error fetching OHLCV: {e}")
            return pd.DataFrame()

# ======================== OPTIONS ANALYZER ========================
class OptionsAnalyzer:
    def __init__(self, memory: RedisMemory, api: DeribitOptionsAPI):
        self.memory = memory
        self.api = api
    
    def calculate_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> float:
        """Calculate Average True Range"""
        if len(df) < period:
            return 0.0
        
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def filter_atm_strikes(self, option_chain: Dict, current_price: float) -> Dict[str, List[Dict]]:
        """Filter strikes to ATM ±5%"""
        lower_bound = current_price * (1 - ATM_RANGE_PERCENT / 100)
        upper_bound = current_price * (1 + ATM_RANGE_PERCENT / 100)
        
        filtered_calls = [
            opt for opt in option_chain["calls"]
            if lower_bound <= opt["strike"] <= upper_bound
        ]
        
        filtered_puts = [
            opt for opt in option_chain["puts"]
            if lower_bound <= opt["strike"] <= upper_bound
        ]
        
        return {"calls": filtered_calls, "puts": filtered_puts}
    
    def calculate_max_pain(self, option_chain: Dict, current_price: float) -> float:
        """Calculate Max Pain strike"""
        strikes = set()
        
        for call in option_chain["calls"]:
            strikes.add(call["strike"])
        for put in option_chain["puts"]:
            strikes.add(put["strike"])
        
        if not strikes:
            return current_price
        
        # Calculate pain for each strike
        pain_levels = {}
        
        for strike in strikes:
            total_pain = 0
            
            # Call pain (ITM calls lose for sellers)
            for call in option_chain["calls"]:
                if current_price > call["strike"]:
                    total_pain += (current_price - call["strike"]) * call["open_interest"]
            
            # Put pain (ITM puts lose for sellers)
            for put in option_chain["puts"]:
                if current_price < put["strike"]:
                    total_pain += (put["strike"] - current_price) * put["open_interest"]
            
            pain_levels[strike] = total_pain
        
        # Return strike with minimum pain
        max_pain_strike = min(pain_levels, key=pain_levels.get)
        return max_pain_strike
    
    def get_strongest_strikes(self, option_chain: Dict) -> Dict:
        """Find strikes with highest OI"""
        if not option_chain["calls"] or not option_chain["puts"]:
            return {}
        
        strongest_call = max(option_chain["calls"], key=lambda x: x["open_interest"])
        strongest_put = max(option_chain["puts"], key=lambda x: x["open_interest"])
        
        return {
            "call": strongest_call,
            "put": strongest_put
        }
    
    def calculate_call_put_ratio(self, option_chain: Dict) -> float:
        """Calculate Call/Put OI ratio"""
        total_call_oi = sum(opt["open_interest"] for opt in option_chain["calls"])
        total_put_oi = sum(opt["open_interest"] for opt in option_chain["puts"])
        
        if total_put_oi == 0:
            return 999.9  # Infinity case
        
        return total_call_oi / total_put_oi
    
    async def get_oi_changes(self, asset: str, current_oi: Dict) -> Dict:
        """Calculate OI changes over 30min and 2hr"""
        changes = {}
        
        for strike_data in current_oi["calls"] + current_oi["puts"]:
            strike = strike_data["strike"]
            current_value = strike_data["open_interest"]
            
            # Get history
            history_2hr = await self.memory.get_oi_history(asset, strike, hours=2)
            history_30min = await self.memory.get_oi_history(asset, strike, hours=0.5)
            
            change_30min = 0
            change_2hr = 0
            
            if history_30min:
                old_value = history_30min[0].get("open_interest", current_value)
                if old_value > 0:
                    change_30min = ((current_value - old_value) / old_value) * 100
            
            if history_2hr:
                old_value = history_2hr[0].get("open_interest", current_value)
                if old_value > 0:
                    change_2hr = ((current_value - old_value) / old_value) * 100
            
            changes[strike] = {
                "30min": change_30min,
                "2hr": change_2hr
            }
        
        return changes
    
    def check_volume_spike(self, option_chain: Dict, asset: str) -> bool:
        """Check if volume is significantly high"""
        total_volume = sum(
            opt["volume"] for opt in option_chain["calls"] + option_chain["puts"]
        )
        
        # Compare with average (simplified - can enhance with Redis history)
        # For now, check if any individual option has high volume
        avg_volume = total_volume / (len(option_chain["calls"]) + len(option_chain["puts"]))
        
        for opt in option_chain["calls"] + option_chain["puts"]:
            if opt["volume"] > avg_volume * VOLUME_SPIKE_THRESHOLD:
                return True
        
        return False
    
    async def analyze_asset(self, asset: str) -> Optional[Dict]:
        """Complete analysis for an asset"""
        logger.info(f"🔍 Analyzing {asset}...")
        
        # Get current price
        current_price = await self.api.get_current_price(asset)
        if current_price == 0:
            logger.warning(f"⚠️ Could not fetch {asset} price")
            return None
        
        logger.info(f"   💰 Current Price: ${current_price:,.2f}")
        
        # Get current expiry
        expiry = await self.api.get_current_expiry(asset)
        if not expiry:
            logger.warning(f"⚠️ No active expiry for {asset}")
            return None
        
        expiry_date = datetime.fromtimestamp(expiry / 1000)
        logger.info(f"   📅 Current Expiry: {expiry_date.strftime('%Y-%m-%d %H:%M')}")
        
        # Small delay for rate limiting
        await asyncio.sleep(1)
        
        # Get option chain
        option_chain = await self.api.get_option_chain(asset, expiry)
        if not option_chain["calls"] or not option_chain["puts"]:
            logger.warning(f"⚠️ Empty option chain for {asset}")
            return None
        
        logger.info(f"   📊 Fetched {len(option_chain['calls'])} Calls, {len(option_chain['puts'])} Puts")
        
        # Filter to ATM ±5%
        filtered_chain = self.filter_atm_strikes(option_chain, current_price)
        logger.info(f"   ✂️ Filtered to ATM±5%: {len(filtered_chain['calls'])} Calls, {len(filtered_chain['puts'])} Puts")
        
        if not filtered_chain["calls"] or not filtered_chain["puts"]:
            logger.warning(f"⚠️ No strikes in ATM range")
            return None
        
        # Calculate metrics
        max_pain = self.calculate_max_pain(filtered_chain, current_price)
        strongest = self.get_strongest_strikes(filtered_chain)
        cp_ratio = self.calculate_call_put_ratio(filtered_chain)
        volume_spike = self.check_volume_spike(filtered_chain, asset)
        
        logger.info(f"   🎯 Max Pain: ${max_pain:,.0f}")
        logger.info(f"   📈 Strongest Call: ${strongest['call']['strike']:,.0f} (OI: {strongest['call']['open_interest']:,.0f})")
        logger.info(f"   📉 Strongest Put: ${strongest['put']['strike']:,.0f} (OI: {strongest['put']['open_interest']:,.0f})")
        logger.info(f"   📊 C/P Ratio: {cp_ratio:.2f}")
        logger.info(f"   🔊 Volume Spike: {'YES' if volume_spike else 'NO'}")
        
        # Get OI changes
        oi_changes = await self.get_oi_changes(asset, filtered_chain)
        
        # Store current OI data in Redis
        for opt in filtered_chain["calls"] + filtered_chain["puts"]:
            oi_data = {
                "open_interest": opt["open_interest"],
                "volume": opt["volume"],
                "timestamp": datetime.now().timestamp()
            }
            await self.memory.store_oi_data(asset, opt["strike"], oi_data)
        
        # Store price data
        ohlcv = await self.api.get_ohlcv(asset, count=20)
        await asyncio.sleep(1)
        
        if not ohlcv.empty:
            await self.memory.store_price_data(asset, current_price, ohlcv['volume'].iloc[-1])
        
        # Calculate ATR
        atr = self.calculate_atr(ohlcv) if not ohlcv.empty else 0
        stop_loss_distance = atr * ATR_MULTIPLIER
        
        logger.info(f"   📏 ATR: ${atr:.2f} | Stop Distance: ${stop_loss_distance:.2f}")
        
        return {
            "asset": asset,
            "current_price": current_price,
            "expiry": expiry,
            "expiry_date": expiry_date,
            "option_chain": filtered_chain,
            "max_pain": max_pain,
            "strongest": strongest,
            "cp_ratio": cp_ratio,
            "volume_spike": volume_spike,
            "oi_changes": oi_changes,
            "atr": atr,
            "stop_loss_distance": stop_loss_distance
        }

# ======================== SIGNAL GENERATOR ========================
class SignalGenerator:
    def generate_signal(self, analysis: Dict) -> Optional[Dict]:
        """Generate trading signal based on analysis"""
        current_price = analysis["current_price"]
        max_pain = analysis["max_pain"]
        strongest = analysis["strongest"]
        cp_ratio = analysis["cp_ratio"]
        volume_spike = analysis["volume_spike"]
        oi_changes = analysis["oi_changes"]
        
        # Calculate distance from Max Pain
        distance_from_max_pain = abs((current_price - max_pain) / max_pain) * 100
        
        # Check if near Max Pain (±3%)
        near_max_pain = distance_from_max_pain <= MAX_PAIN_PROXIMITY_PERCENT
        
        # Get OI changes for strongest strikes
        call_strike = strongest["call"]["strike"]
        put_strike = strongest["put"]["strike"]
        
        call_oi_30min = oi_changes.get(call_strike, {}).get("30min", 0)
        call_oi_2hr = oi_changes.get(call_strike, {}).get("2hr", 0)
        put_oi_30min = oi_changes.get(put_strike, {}).get("30min", 0)
        put_oi_2hr = oi_changes.get(put_strike, {}).get("2hr", 0)
        
        # If no historical data (Redis disabled), use relaxed OI requirements
        has_history = any([call_oi_30min, call_oi_2hr, put_oi_30min, put_oi_2hr])
        
        signal = None
        
        # 🟢 LONG SIGNAL (Buy Call)
        if has_history:
            # With history - full criteria
            if (
                (current_price < max_pain or current_price <= strongest["put"]["strike"]) and
                near_max_pain and
                call_oi_30min >= OI_INCREASE_30MIN and
                call_oi_2hr >= OI_INCREASE_2HR and
                volume_spike and
                cp_ratio > CALL_PUT_RATIO_LONG
            ):
                signal = {
                    "type": "LONG",
                    "action": "BUY CALL",
                    "reason": "Price below Max Pain + Call OI increasing + Volume spike + Bullish C/P ratio",
                    "scenario": "Strong Bullish (OI↑ Vol↑ Price→MP)"
                }
        else:
            # Without history - simplified criteria
            if (
                (current_price < max_pain or current_price <= strongest["put"]["strike"]) and
                near_max_pain and
                volume_spike and
                cp_ratio > CALL_PUT_RATIO_LONG and
                strongest["call"]["open_interest"] > strongest["put"]["open_interest"] * 1.2
            ):
                signal = {
                    "type": "LONG",
                    "action": "BUY CALL",
                    "reason": "Price below Max Pain + High Call OI + Volume spike + Bullish C/P ratio",
                    "scenario": "Bullish (Price→MP)"
                }
        
        # 🔴 SHORT SIGNAL (Buy Put)
        if not signal:
            if has_history:
                if (
                    (current_price > max_pain or current_price >= strongest["call"]["strike"]) and
                    near_max_pain and
                    put_oi_30min >= OI_INCREASE_30MIN and
                    put_oi_2hr >= OI_INCREASE_2HR and
                    volume_spike and
                    cp_ratio < CALL_PUT_RATIO_SHORT
                ):
                    signal = {
                        "type": "SHORT",
                        "action": "BUY PUT",
                        "reason": "Price above Max Pain + Put OI increasing + Volume spike + Bearish C/P ratio",
                        "scenario": "Strong Bearish (OI↑ Vol↑ Price→MP)"
                    }
            else:
                if (
                    (current_price > max_pain or current_price >= strongest["call"]["strike"]) and
                    near_max_pain and
                    volume_spike and
                    cp_ratio < CALL_PUT_RATIO_SHORT and
                    strongest["put"]["open_interest"] > strongest["call"]["open_interest"] * 1.2
                ):
                    signal = {
                        "type": "SHORT",
                        "action": "BUY PUT",
                        "reason": "Price above Max Pain + High Put OI + Volume spike + Bearish C/P ratio",
                        "scenario": "Bearish (Price→MP)"
                    }
        
        # 🟢 BREAKOUT LONG (Ignore Max Pain)
        if not signal:
            if has_history:
                if (
                    current_price >= strongest["call"]["strike"] and
                    call_oi_30min >= 20 and  # Stronger increase
                    volume_spike and
                    cp_ratio > 1.5  # Very bullish
                ):
                    signal = {
                        "type": "LONG",
                        "action": "BUY CALL",
                        "reason": "Breakout above resistance + Strong Call OI increase + High volume",
                        "scenario": "Bullish Breakout"
                    }
            else:
                if (
                    current_price >= strongest["call"]["strike"] and
                    volume_spike and
                    cp_ratio > 1.5
                ):
                    signal = {
                        "type": "LONG",
                        "action": "BUY CALL",
                        "reason": "Breakout above resistance + High volume + Very bullish ratio",
                        "scenario": "Bullish Breakout"
                    }
        
        # 🔴 BREAKDOWN SHORT
        if not signal:
            if has_history:
                if (
                    current_price <= strongest["put"]["strike"] and
                    put_oi_30min >= 20 and
                    volume_spike and
                    cp_ratio < 0.6  # Very bearish
                ):
                    signal = {
                        "type": "SHORT",
                        "action": "BUY PUT",
                        "reason": "Breakdown below support + Strong Put OI increase + High volume",
                        "scenario": "Bearish Breakdown"
                    }
            else:
                if (
                    current_price <= strongest["put"]["strike"] and
                    volume_spike and
                    cp_ratio < 0.6
                ):
                    signal = {
                        "type": "SHORT",
                        "action": "BUY PUT",
                        "reason": "Breakdown below support + High volume + Very bearish ratio",
                        "scenario": "Bearish Breakdown"
                    }
        
        if signal:
            # Add trade details
            entry_price = current_price
            
            if signal["type"] == "LONG":
                strike = strongest["call"]["strike"]
                target = strike * 1.05  # 5% above strike
                stop = entry_price - analysis["stop_loss_distance"]
            else:
                strike = strongest["put"]["strike"]
                target = strike * 0.95  # 5% below strike
                stop = entry_price + analysis["stop_loss_distance"]
            
            signal.update({
                "entry": entry_price,
                "strike": strike,
                "target": target,
                "stop": stop,
                "risk_reward": abs((target - entry_price) / (entry_price - stop)) if stop != entry_price else 0
            })
        
        return signal

# ======================== CHART GENERATOR ========================
class ChartGenerator:
    @staticmethod
    def create_option_heatmap(analysis: Dict, signal: Optional[Dict] = None) -> str:
        """Create option chain heatmap"""
        asset = analysis["asset"]
        option_chain = analysis["option_chain"]
        current_price = analysis["current_price"]
        max_pain = analysis["max_pain"]
        strongest = analysis["strongest"]
        
        # Prepare data
        call_strikes = [opt["strike"] for opt in option_chain["calls"]]
        call_oi = [opt["open_interest"] for opt in option_chain["calls"]]
        call_volume = [opt["volume"] for opt in option_chain["calls"]]
        
        put_strikes = [opt["strike"] for opt in option_chain["puts"]]
        put_oi = [opt["open_interest"] for opt in option_chain["puts"]]
        put_volume = [opt["volume"] for opt in option_chain["puts"]]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), facecolor='white')
        fig.suptitle(f'{asset} Option Chain Analysis - {analysis["expiry_date"].strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        
        # === TOP CHART: Open Interest ===
        ax1.bar(call_strikes, call_oi, width=50, color='#26a69a', alpha=0.8, label='Call OI')
        ax1.bar(put_strikes, [-oi for oi in put_oi], width=50, color='#ef5350', alpha=0.8, label='Put OI')
        
        # Mark current price
        ax1.axvline(current_price, color='blue', linewidth=2, linestyle='-', label=f'Current: ${current_price:,.0f}')
        
        # Mark Max Pain
        ax1.axvline(max_pain, color='purple', linewidth=2, linestyle='--', label=f'Max Pain: ${max_pain:,.0f}')
        
        # Mark strongest strikes
        ax1.axvline(strongest["call"]["strike"], color='green', linewidth=1.5, linestyle=':', 
                   label=f'Strong Call: ${strongest["call"]["strike"]:,.0f}')
        ax1.axvline(strongest["put"]["strike"], color='red', linewidth=1.5, linestyle=':', 
                   label=f'Strong Put: ${strongest["put"]["strike"]:,.0f}')
        
        ax1.set_xlabel('Strike Price', fontsize=12)
        ax1.set_ylabel('Open Interest', fontsize=12)
        ax1.set_title('Open Interest Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linewidth=0.8)
        
        # === BOTTOM CHART: Volume ===
        ax2.bar(call_strikes, call_volume, width=50, color='#26a69a', alpha=0.6, label='Call Volume')
        ax2.bar(put_strikes, [-vol for vol in put_volume], width=50, color='#ef5350', alpha=0.6, label='Put Volume')
        
        ax2.axvline(current_price, color='blue', linewidth=2, linestyle='-')
        ax2.axvline(max_pain, color='purple', linewidth=2, linestyle='--')
        
        ax2.set_xlabel('Strike Price', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.set_title('Trading Volume Distribution', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='black', linewidth=0.8)
        
        # Add signal annotation if present
        if signal:
            signal_text = f"🎯 SIGNAL: {signal['type']}\n{signal['action']}\nEntry: ${signal['entry']:,.0f}"
            ax1.text(0.02, 0.98, signal_text, transform=ax1.transAxes,
                    fontsize=12, fontweight='bold',
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='yellow' if signal['type'] == 'LONG' else 'orange', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        filename = f"/home/claude/chart_{asset}_{int(datetime.now().timestamp())}.png"
        plt.savefig(filename, dpi=150, facecolor='white')
        plt.close()
        
        logger.info(f"📊 Chart saved: {filename}")
        return filename

# ======================== TELEGRAM ALERTER ========================
class TelegramAlerter:
    def __init__(self, token: str, chat_id: str):
        self.bot = Bot(token=token)
        self.chat_id = chat_id
    
    async def send_alert(self, analysis: Dict, signal: Dict, chart_path: str):
        """Send complete alert with chart"""
        try:
            asset = analysis["asset"]
            current_price = analysis["current_price"]
            max_pain = analysis["max_pain"]
            strongest = analysis["strongest"]
            cp_ratio = analysis["cp_ratio"]
            volume_spike = analysis["volume_spike"]
            
            # Get OI changes for display
            call_strike = strongest["call"]["strike"]
            put_strike = strongest["put"]["strike"]
            oi_changes = analysis["oi_changes"]
            
            call_change_30min = oi_changes.get(call_strike, {}).get("30min", 0)
            call_change_2hr = oi_changes.get(call_strike, {}).get("2hr", 0)
            put_change_30min = oi_changes.get(put_strike, {}).get("30min", 0)
            put_change_2hr = oi_changes.get(put_strike, {}).get("2hr", 0)
            
            # Emoji for changes
            call_emoji = "📈" if call_change_30min > 0 else "📉" if call_change_30min < 0 else "➡️"
            put_emoji = "📈" if put_change_30min > 0 else "📉" if put_change_30min < 0 else "➡️"
            
            # Build message
            msg = f"""
{'🟢' if signal['type'] == 'LONG' else '🔴'} **{signal['type']} SIGNAL - {asset}**

**📊 Market Data:**
💰 Current Price: ${current_price:,.2f}
🎯 Max Pain: ${max_pain:,.2f} ({((current_price - max_pain) / max_pain * 100):+.2f}%)

**📈 Strongest Call Strike:** ${call_strike:,.0f}
   OI: {strongest['call']['open_interest']:,.0f} contracts
   Change: {call_emoji} 30min: {call_change_30min:+.1f}% | 2hr: {call_change_2hr:+.1f}%

**📉 Strongest Put Strike:** ${put_strike:,.0f}
   OI: {strongest['put']['open_interest']:,.0f} contracts
   Change: {put_emoji} 30min: {put_change_30min:+.1f}% | 2hr: {put_change_2hr:+.1f}%

**📊 Metrics:**
C/P Ratio: {cp_ratio:.2f} {'(Bullish)' if cp_ratio > 1.2 else '(Bearish)' if cp_ratio < 0.8 else '(Neutral)'}
Volume Spike: {'✅ YES' if volume_spike else '❌ NO'}

**💰 Trade Setup:**
Action: {signal['action']}
Entry: ${signal['entry']:,.2f}
Strike: ${signal['strike']:,.0f}
Target: ${signal['target']:,.2f} ({((signal['target'] - signal['entry']) / signal['entry'] * 100):+.1f}%)
Stop: ${signal['stop']:,.2f} ({((signal['stop'] - signal['entry']) / signal['entry'] * 100):+.1f}%)
Risk/Reward: {signal['risk_reward']:.2f}

**📋 Scenario:** {signal['scenario']}
**🎯 Reason:** {signal['reason']}

**⏰ Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**📅 Expiry:** {analysis['expiry_date'].strftime('%Y-%m-%d %H:%M')}
"""
            
            # Send chart with caption
            with open(chart_path, 'rb') as photo:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo,
                    caption=msg,
                    parse_mode='Markdown'
                )
            
            logger.info(f"✅ Alert sent for {asset}")
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")

# ======================== MAIN BOT ========================
class OptionsBot:
    def __init__(self):
        self.memory = RedisMemory(REDIS_URL)
        self.api = DeribitOptionsAPI()
        self.analyzer = OptionsAnalyzer(self.memory, self.api)
        self.signal_gen = SignalGenerator()
        self.chart_gen = ChartGenerator()
        self.alerter = TelegramAlerter(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.last_alerts = defaultdict(lambda: None)
    
    async def analyze_and_alert(self):
        """Run analysis and send alerts"""
        logger.info("\n" + "="*60)
        logger.info("🔍 STARTING OPTIONS ANALYSIS")
        logger.info("="*60 + "\n")
        
        for asset in ASSETS:
            try:
                # Analyze
                analysis = await self.analyzer.analyze_asset(asset)
                
                if not analysis:
                    logger.warning(f"⚠️ Analysis failed for {asset}\n")
                    continue
                
                # Generate signal
                signal = self.signal_gen.generate_signal(analysis)
                
                if signal:
                    logger.info(f"🎯 SIGNAL FOUND: {signal['type']} - {signal['action']}")
                    
                    # Check cooldown (1 hour)
                    last_alert = self.last_alerts[asset]
                    current_time = datetime.now()
                    
                    if last_alert is None or (current_time - last_alert).seconds > 3600:
                        # Generate chart
                        chart_path = self.chart_gen.create_option_heatmap(analysis, signal)
                        
                        # Send alert
                        await self.alerter.send_alert(analysis, signal, chart_path)
                        
                        self.last_alerts[asset] = current_time
                        logger.info(f"✉️ Alert sent successfully!\n")
                    else:
                        logger.info(f"⏸️ Signal found but cooldown active\n")
                else:
                    logger.info(f"ℹ️ No trading signals detected for {asset}\n")
                
                # Delay between assets
                await asyncio.sleep(2)
                
            except Exception as e:
                logger.error(f"❌ Error processing {asset}: {e}\n")
        
        logger.info("="*60)
        logger.info("✅ ANALYSIS CYCLE COMPLETE")
        logger.info("="*60 + "\n")
    
    async def run(self):
        """Main bot loop"""
        # Connect
        await self.memory.connect()
        await self.api.create_session()
        
        # Startup banner
        logger.info("\n" + "="*60)
        logger.info("🚀 OPTIONS TRADING BOT STARTED!")
        logger.info("="*60)
        logger.info(f"📊 Assets: {', '.join(ASSETS)}")
        logger.info(f"⏱️  Analysis Interval: {ANALYSIS_INTERVAL // 60} minutes")
        logger.info(f"💾 Redis Memory: {'✅ ENABLED' if self.memory.enabled else '⚠️ DISABLED (no historical OI tracking)'}")
        logger.info(f"🎯 Max Pain Range: ±{MAX_PAIN_PROXIMITY_PERCENT}%")
        logger.info(f"📈 ATM Strike Range: ±{ATM_RANGE_PERCENT}%")
        logger.info(f"🛑 Stop Loss: {ATR_MULTIPLIER}x ATR ({ATR_PERIOD} candles)")
        logger.info(f"📊 C/P Ratio: Long>{CALL_PUT_RATIO_LONG}, Short<{CALL_PUT_RATIO_SHORT}")
        logger.info(f"🔊 Volume Threshold: {VOLUME_SPIKE_THRESHOLD}x avg")
        logger.info("="*60 + "\n")
        
        try:
            while True:
                try:
                    await self.analyze_and_alert()
                    
                    next_run = datetime.now() + timedelta(seconds=ANALYSIS_INTERVAL)
                    logger.info(f"⏰ Next analysis at {next_run.strftime('%H:%M:%S')}\n")
                    
                    await asyncio.sleep(ANALYSIS_INTERVAL)
                    
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    logger.info("⏳ Retrying in 1 minute...")
                    await asyncio.sleep(60)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Bot stopped by user")
        
        finally:
            await self.api.close_session()
            await self.memory.close()
            logger.info("👋 Connections closed. Goodbye!")

# ======================== ENTRY POINT ========================
if __name__ == "__main__":
    bot = OptionsBot()
    asyncio.run(bot.run())
