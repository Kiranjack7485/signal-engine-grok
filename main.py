import asyncio
import numpy as np
from binance import AsyncClient
import talib
import logging
import requests
from datetime import datetime
import pytz  # pip install pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────
API_KEY = "KLWk0Hz7niXyKMyZdiwNEG8QI4X5yEu2pMotjtI6AMfC4HatblQ7XDyAxghLwbzy"
API_SECRET = "ZtHmzyPpl8T3R3twCEdt1IdIFOTTBEbZNNhPfRO0zb8XGp4Rba76LVQoSqv42CMt"
TELEGRAM_BOT_TOKEN = "8304346679:AAE-xCzuZpuq0-2gCjmVboLvR-rvm6m6mlA"
TELEGRAM_CHAT_ID = "921422015"

INTERVAL_SECONDS = 60  # Every 1 minute
MIN_VOLUME_USDT_TEST = 100000
FIXED_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'LTCUSDT', 'LINKUSDT', 'DOTUSDT']

KLINE_INTERVALS = ['3m', '5m', '15m', '1h']  # Multi-timeframe check
KLINE_LIMIT = 50
MIN_SCORE_THRESHOLD = 7  # Only send strong signals (e.g., EMA + good volume)
MAX_RETRIES = 5

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
        # Multi-timeframe EMA alignment check
        aligned_count = 0
        last_price = None
        last_vol = None
        avg_vol = None
        direction = None

        for interval in KLINE_INTERVALS:
            klines = await client.futures_klines(symbol=symbol, interval=interval, limit=KLINE_LIMIT)
            if len(klines) < KLINE_LIMIT:
                continue

            closes = np.array([float(k[4]) for k in klines], dtype=float)
            volumes = np.array([float(k[5]) for k in klines], dtype=float)

            ema9 = talib.EMA(closes, timeperiod=9)[-1]
            ema21 = talib.EMA(closes, timeperiod=21)[-1]

            if np.isnan(ema9) or np.isnan(ema21):
                continue

            current_price = closes[-1]
            current_vol = volumes[-1]
            current_avg_vol = np.mean(volumes[-11:-1]) if len(volumes) > 10 else 0

            if current_price > ema9 and current_price > ema21:
                current_direction = 'long'
            elif current_price < ema9 and current_price < ema21:
                current_direction = 'short'
            else:
                continue  # No alignment in this timeframe

            if direction is None:
                direction = current_direction
            elif direction != current_direction:
                return None  # Conflicting directions across timeframes

            aligned_count += 1
            # Use the 5m data for final price/volume (as primary)
            if interval == '5m':
                last_price = current_price
                last_vol = current_vol
                avg_vol = current_avg_vol

        # Require alignment in at least 3 timeframes for strong confirmation
        if aligned_count < 3 or last_price is None:
            return None

        # Volume check (same as before)
        score = 4  # Base for EMA alignment
        vol_multiplier = last_vol / avg_vol if avg_vol > 0 else 0
        if vol_multiplier > 1.5:
            score += 6
        elif vol_multiplier > 1.2:
            score += 3

        score = min(score, 10)

        # Only return if strong (score >= threshold)
        if score < MIN_SCORE_THRESHOLD:
            return None

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
        tickers = await client.futures_ticker(symbols=FIXED_COINS)
        candidates = []
        for t in tickers:
            if 'symbol' not in t:  # Safety check for API response
                continue
            symbol = t['symbol']
            if symbol not in FIXED_COINS:  # Double-check only fixed
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
            logger.warning("No fixed coins met volume threshold")
            return None

        # Sort by 24h change % desc
        candidates.sort(key=lambda x: x['change_24h_pct'], reverse=True)
        logger.info(f"Fixed coins sorted: {[c['symbol'] for c in candidates]}")

        analysis_tasks = [analyze_coin(client, c) for c in candidates]
        results = await asyncio.gather(*analysis_tasks)
        valid = [r for r in results if r]

        if not valid:
            logger.info("No strong signals")
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

**Reason**: Multi-TF EMA alignment + volume | Vol ${best['volume_usdt']/1e6:.1f}M

Test only!
"""
        send_telegram_message(message)
        logger.info(f"Sent strong signal for {best['symbol']}")

    except Exception as e:
        logger.error(f"Error: {e}")

async def main():
    client = None
    try:
        while True:
            retry_count = 0
            while retry_count < MAX_RETRIES:
                try:
                    if client is None:
                        client = await AsyncClient.create(API_KEY, API_SECRET)
                    logger.info("Scan start...")
                    await get_top_and_analyze(client)
                    break
                except Exception as e:
                    logger.error(f"Scan failed (retry {retry_count+1}/{MAX_RETRIES}): {e}")
                    retry_count += 1
                    if client:
                        await client.close_connection()
                        client = None
                    await asyncio.sleep(10)
            if retry_count == MAX_RETRIES:
                logger.error("Max retries reached - skipping this scan")
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        if client:
            await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())