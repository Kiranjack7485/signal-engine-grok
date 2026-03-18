import os
import time
import datetime
import pytz
import pandas as pd
import schedule
import requests
from dotenv import load_dotenv
from pybit.unified_trading import HTTP
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator

load_dotenv()

# ────────────────────────────────────────────────
#  CONFIG
# ────────────────────────────────────────────────
BYBIT_API_KEY    = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "SUIUSDT", "TONUSDT",
    "LINKUSDT", "DOTUSDT", "BCHUSDT", "LTCUSDT", "NEARUSDT"
]

TIMEFRAME = "15"                    # ← Changed to 15-min for strong trends
SCAN_INTERVAL_SEC = 10
MIN_SCORE_THRESHOLD = 7.5           # ← Rolled back to original strict level

DEFAULT_LEVERAGE = 8
PARTIAL_PCT = 50

ALLOWED_SESSIONS_UTC = [(13, 17), (3, 10)]

# ────────────────────────────────────────────────
session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

def is_trading_time():
    now_utc = datetime.datetime.now(pytz.UTC)
    hour = now_utc.hour
    for start_h, end_h in ALLOWED_SESSIONS_UTC:
        if start_h <= hour < end_h:
            return True
    return False

def send_telegram(message: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("[DRY] Telegram would send:\n", message[:300] + "...")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=10)
        print("✅ Strong Long-Term Trend Notification Sent!")
    except Exception as e:
        print(f"Telegram error: {e}")

def get_klines(symbol, interval="15", limit=200):
    try:
        resp = session.get_kline(category="linear", symbol=symbol, interval=interval, limit=limit)
        if resp.get("retCode") != 0:
            return None
        data = resp["result"]["list"]
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume","turnover"])
        df = df.astype(float).sort_values("timestamp")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except:
        return None

def calculate_indicators(df):
    close = df["close"]
    df["ema21"] = EMAIndicator(close, window=21).ema_indicator()
    df["ema55"] = EMAIndicator(close, window=55).ema_indicator()
    df["rsi"]   = RSIIndicator(close, window=14).rsi()
    
    stoch = StochasticOscillator(high=df["high"], low=df["low"], close=close, 
                                 window=14, smooth_window=3, fillna=True)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    return df

def score_and_analyze(df, symbol):
    if len(df) < 60:   # Need more data for strong trend confirmation
        return 0, "NEUTRAL", {}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    score = 0
    details = []
    trend = "NEUTRAL"

    # Rule 1: Strong Trend + Momentum (slower EMAs)
    if last["close"] > last["ema21"] > last["ema55"]:
        trend = "LONG"
        score += 2
        details.append("✅ Strong EMA Bullish Stack")
        if last["rsi"] > 52 and prev["rsi"] <= 52:
            score += 1
            details.append("✅ RSI Bull Cross")
        if last["stoch_k"] > last["stoch_d"] and prev["stoch_k"] <= prev["stoch_d"] and last["stoch_k"] < 40:
            score += 1
            details.append("✅ Stoch Bull Cross")
    elif last["close"] < last["ema21"] < last["ema55"]:
        trend = "SHORT"
        score += 2
        details.append("✅ Strong EMA Bearish Stack")
        if last["rsi"] < 48 and prev["rsi"] >= 48:
            score += 1
            details.append("✅ RSI Bear Cross")
        if last["stoch_k"] < last["stoch_d"] and prev["stoch_k"] >= prev["stoch_d"] and last["stoch_k"] > 60:
            score += 1
            details.append("✅ Stoch Bear Cross")

    if trend == "NEUTRAL":
        return 0, "NEUTRAL", {}

    # === NEW: Sustained Trend Filter (must hold for ~1 hour) ===
    sustained_count = 0
    for i in range(1, 6):  # last 5 candles
        candle = df.iloc[-i]
        if trend == "LONG" and candle["close"] > candle["ema21"]:
            sustained_count += 1
        elif trend == "SHORT" and candle["close"] < candle["ema21"]:
            sustained_count += 1
    if sustained_count >= 4:
        score += 2
        details.append("✅ Strong Sustained Trend (5 bars)")

    # Rule 2: Key Level (longer lookback)
    recent_low = df["low"].iloc[-30:].min()
    recent_high = df["high"].iloc[-30:].max()
    if trend == "LONG" and abs(last["close"] - recent_low) / last["close"] < 0.005:
        score += 2
        details.append("✅ At Major Swing Low")
    elif trend == "SHORT" and abs(last["close"] - recent_high) / last["close"] < 0.005:
        score += 2
        details.append("✅ At Major Swing High")

    # Rule 3: Volume Surge
    avg_vol = df["volume"].iloc[-40:-1].mean()
    vol_ratio = last["volume"] / avg_vol if avg_vol > 0 else 0
    if vol_ratio > 2.0:
        score += 2
        details.append(f"✅ Strong Volume Surge {vol_ratio:.1f}x")

    score = min(10, score)
    analysis = {
        "trend": trend,
        "score": round(score, 1),
        "rsi": round(last["rsi"], 1),
        "stoch_k": round(last["stoch_k"], 1),
        "vol_ratio": round(vol_ratio, 1),
        "details": " | ".join(details),
        "df": df
    }
    return score, trend, analysis

def scan_all():
    if not is_trading_time():
        return

    now = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"\n🔍 [{now}] STRONG TREND SCAN (15-min) - {len(SYMBOLS)} pairs...")

    best_score = 0
    best_setup = None
    best_symbol = ""

    print(f"{'Pair':<8} {'Trend':<6} {'Score':<6} {'RSI':<6} {'StochK':<7} {'Vol(x)':<7} Details")
    print("-" * 95)

    for sym in SYMBOLS:
        df = get_klines(sym)
        if df is None:
            print(f"{sym:<8} Error")
            continue

        df = calculate_indicators(df)
        score, trend, info = score_and_analyze(df, sym)

        print(f"{sym:<8} {trend:<6} {info.get('score',0):<6} {info.get('rsi',0):<6} "
              f"{info.get('stoch_k',0):<7} {info.get('vol_ratio',0):<7} {info.get('details','No setup')[:55]}")

        if score > best_score:
            best_score = score
            best_symbol = sym
            best_setup = {"symbol": sym.replace("USDT",""), **info}

    print("-" * 95)
    print(f"Best this cycle → {best_symbol} | Score: {best_score}/10\n")

    if best_setup and best_score >= MIN_SCORE_THRESHOLD:
        d = best_setup
        last_close = d["df"].iloc[-1]["close"]
        entry = round(last_close, 2)
        
        atr = (d["df"]["high"] - d["df"]["low"]).iloc[-20:].mean()
        sl = round(entry - atr*1.2 if d["trend"]=="LONG" else entry + atr*1.2, 2)
        tp = round(entry + atr*2.4 if d["trend"]=="LONG" else entry - atr*2.4, 2)
        partial = round((entry + tp)/2, 2)

        msg = (
            f"**🟢 STRONG LONG-TERM TREND SETUP DETECTED!**\n\n"
            f"**{d['symbol']}USDT** | **{d['trend']}** | **{d['score']}/10**\n\n"
            f"Entry ≈ **{entry}**\n"
            f"Take Profit: **{tp}**\n"
            f"Stop Loss: **{sl}**\n"
            f"Leverage: **{DEFAULT_LEVERAGE}x** (max)\n"
            f"Holding Time: **30–90 minutes**\n"
            f"Partial ({PARTIAL_PCT}%): **{partial}**\n\n"
            f"**Reasons:** {d['details']}"
        )
        send_telegram(msg)

# ────────────────────────────────────────────────
schedule.every(SCAN_INTERVAL_SEC).seconds.do(scan_all)

print("✅ STRONG LONG-TERM TREND SCANNER STARTED")
print("Only signals with sustained trend (≥7.5 score) will be sent.\n")

while True:
    schedule.run_pending()
    time.sleep(1)