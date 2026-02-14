import asyncio
import numpy as np
from binance import AsyncClient
import talib
import logging
import requests
from datetime import datetime
import pytz  # pip install pytz if not already
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────

# ── CONFIG ────────────────────────────────────────────────
API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([API_KEY, API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("Missing environment variables for config!")

INTERVAL_SECONDS = 300
MIN_VOLUME_USDT_TEST = 100000  # Keep low for testing
TOP_N_TEST = 10  # For Phase 1 testing

# For Phase 2+: We'll switch to this fixed high-legacy list
LEGACY_TOP_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']  # High-volume majors

KLINE_INTERVAL = '5m'
KLINE_LIMIT = 50

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload)
        r.raise_for_status()
        logger.info("Telegram sent")
    except Exception as e:
        logger.error(f"Telegram fail: {e}")

async def analyze_coin(client, candidate):
    symbol = candidate['symbol']
    try:
        klines = await client.futures_klines(symbol=symbol, interval=KLINE_INTERVAL, limit=KLINE_LIMIT)
        if len(klines) < KLINE_LIMIT:
            return None

        closes = np.array([float(k[4]) for k in klines], dtype=float)
        volumes = np.array([float(k[5]) for k in klines], dtype=float)

        ema9 = talib.EMA(closes, timeperiod=9)[-1]
        ema21 = talib.EMA(closes, timeperiod=21)[-1]
        last_price = closes[-1]
        last_vol = volumes[-1]
        avg_vol = np.mean(volumes[-11:-1]) if len(volumes) > 10 else 0

        if last_price > ema9 and last_price > ema21:
            direction = 'long'
        elif last_price < ema9 and last_price < ema21:
            direction = 'short'
        else:
            return None

        score = 4

        vol_multiplier = last_vol / avg_vol if avg_vol > 0 else 0
        if vol_multiplier > 1.5:
            score += 6
        elif vol_multiplier > 1.2:
            score += 3

        score = min(score, 10)

        return {
            'symbol': symbol,
            'score': score,
            'direction': direction,
            'price': last_price,
            'change_24h_pct': candidate['change_24h_pct'],
            'volume_usdt': candidate['volume_usdt'],
        }

    except Exception as e:
        logger.error(f"Analysis fail {symbol}: {e}")
        return None

async def get_top_and_analyze(client):
    try:
        tickers = await client.futures_ticker()
        candidates = []
        for t in tickers:
            symbol = t['symbol']
            if not symbol.endswith('USDT'):
                continue
            quote_vol = float(t.get('quoteVolume', '0') or 0)
            if quote_vol < MIN_VOLUME_USDT_TEST:
                continue
            candidates.append({
                'symbol': symbol,
                'change_24h_pct': float(t.get('priceChangePercent', 0)),
                'volume_usdt': quote_vol,
            })

        if not candidates:
            logger.warning("No candidates")
            return None

        # Phase 1: sort by change % desc
        candidates.sort(key=lambda x: x['change_24h_pct'], reverse=True)
        top_n = candidates[:TOP_N_TEST]
        logger.info(f"Top {TOP_N_TEST}: {[c['symbol'] for c in top_n]}")

        # In Phase 2: replace above with filtering to LEGACY_TOP_SYMBOLS only
        # e.g.: top_n = [c for c in candidates if c['symbol'] in LEGACY_TOP_SYMBOLS]
        # then sort those by volume or change

        analysis_tasks = [analyze_coin(client, c) for c in top_n]
        results = await asyncio.gather(*analysis_tasks)
        valid = [r for r in results if r]

        if not valid:
            logger.info("No passes")
            return None

        best = max(valid, key=lambda x: (x['score'], x['change_24h_pct']))

        trend = "Bullish 🚀" if best['direction'] == 'long' else "Bearish 📉"
        entry = best['price']
        tp = f"TP1: {entry * 1.015:.4f} (+1.5%), TP2: {entry * 1.04:.4f} (+4%)" if best['direction'] == 'long' else f"TP1: {entry * 0.985:.4f} (-1.5%), TP2: {entry * 0.96:.4f} (-4%)"
        sl = f"{entry * 0.988:.4f} (-1.2%)" if best['direction'] == 'long' else f"{entry * 1.012:.4f} (+1.2%)"
        leverage = "5-10x"

        ist_now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')

        message = f"""
*🔥 Phase 1 Test Signal @ {ist_now}*

**Coin**: {best['symbol']}
**Rating**: {best['score']}/10
**Price**: ${best['price']:.4f}
**Trend**: {trend} ({best['change_24h_pct']:+.2f}% 24h)
**TP**: {tp}
**SL**: {sl}
**Leverage**: {leverage}

**Reason**: EMA + volume test | Vol ${best['volume_usdt']/1e6:.1f}M

Test only!
"""
        send_telegram_message(message)
        logger.info(f"Sent for {best['symbol']}")

    except Exception as e:
        logger.error(f"Error: {e}")

async def main():
    client = await AsyncClient.create(API_KEY, API_SECRET)

    try:
        while True:
            logger.info("Scan start...")
            await get_top_and_analyze(client)
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())