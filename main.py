import asyncio
import numpy as np
from binance import AsyncClient
import talib
import logging
import requests
from datetime import datetime
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── CONFIG ────────────────────────────────────────────────

API_KEY = "KLWk0Hz7niXyKMyZdiwNEG8QI4X5yEu2pMotjtI6AMfC4HatblQ7XDyAxghLwbzy"
API_SECRET = "ZtHmzyPpl8T3R3twCEdt1IdIFOTTBEbZNNhPfRO0zb8XGp4Rba76LVQoSqv42CMt"
TELEGRAM_BOT_TOKEN = "8304346679:AAE-xCzuZpuq0-2gCjmVboLvR-rvm6m6mlA"
TELEGRAM_CHAT_ID = "921422015"

INTERVAL_SECONDS = 60
FIXED_COINS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT']

KLINE_INTERVALS = ['3m', '5m', '15m', '1h']
PRIMARY_INTERVAL = '5m'
KLINE_LIMIT = 60
MIN_SCORE_THRESHOLD = 8.0
MIN_ALIGNED_TF = 3

# Indian high-liquidity window (IST)
TRADING_START_IST = 18   # 18:00 IST
TRADING_END_IST   = 1    # 01:00 IST next day

# Risk / Reward settings
ACCOUNT_SIZE = 10000.0              # Simulated USDT balance
RISK_PER_TRADE_PCT = 0.01           # 1% risk per trade
SL_PCT = 0.012                      # 1.2% stop-loss distance
MIN_RISK_REWARD_RATIO = 2.0         # Minimum 1:2 RR enforced on TP1
TP2_PCT = 0.04                      # ~4% second target (more aggressive)

last_signals = {}  # symbol → dict

def is_high_liquidity_time():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    hour = now_ist.hour
    # 18:00 to 01:00 (crossing midnight)
    if TRADING_START_IST <= hour or hour < TRADING_END_IST:
        return True
    return False

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Telegram sent")
    except Exception as e:
        logger.error(f"Telegram fail: {e}")

async def get_klines(client, symbol, interval):
    try:
        klines = await client.futures_klines(symbol=symbol, interval=interval, limit=KLINE_LIMIT)
        if len(klines) < KLINE_LIMIT - 5:
            return None
        return klines
    except Exception as e:
        logger.error(f"Klines fail {symbol} {interval}: {e}")
        return None

def detect_candlestick_patterns(open_, high, low, close):
    patterns = {
        'bullish_engulfing': talib.CDLENGULFING(open_, high, low, close)[-1] > 0,
        'hammer': talib.CDLHAMMER(open_, high, low, close)[-1] > 0,
        'inverted_hammer': talib.CDLINVERTEDHAMMER(open_, high, low, close)[-1] > 0,
        'doji': talib.CDLDOJI(open_, high, low, close)[-1] != 0,
        'shooting_star': talib.CDLSHOOTINGSTAR(open_, high, low, close)[-1] > 0,
        'hanging_man': talib.CDLHANGINGMAN(open_, high, low, close)[-1] > 0,
        'bearish_engulfing': talib.CDLENGULFING(open_, high, low, close)[-1] < 0,
    }
    bull_count = sum(v for k, v in patterns.items() if v and any(word in k for word in ['bullish', 'hammer', 'inverted_hammer', 'doji']))
    bear_count = sum(v for k, v in patterns.items() if v and any(word in k for word in ['bearish', 'shooting_star', 'hanging_man']))
    return bull_count, bear_count

def check_rsi_divergence(closes, rsi):
    if len(closes) < 10:
        return 0
    price_low1 = closes[-5]
    price_low2 = min(closes[-10:-5])
    rsi_low1 = rsi[-5]
    rsi_low2 = min(rsi[-10:-5])
    bull_div = (price_low2 < price_low1) and (rsi_low2 < rsi_low1) and rsi_low1 < 40

    price_high1 = closes[-5]
    price_high2 = max(closes[-10:-5])
    rsi_high1 = rsi[-5]
    rsi_high2 = max(rsi[-10:-5])
    bear_div = (price_high2 > price_high1) and (rsi_high2 > rsi_high1) and rsi_high1 > 60

    if bull_div: return 3
    if bear_div: return -3
    return 0

def check_bollinger_squeeze(upper, middle, lower):
    if len(upper) < 2:
        return 0
    width = (upper[-1] - lower[-1]) / middle[-1]
    prev_width = (upper[-2] - lower[-2]) / middle[-2]
    if width < 0.015 and prev_width > width * 1.2:  # squeeze → potential breakout
        return 2
    return 0

async def analyze_coin(client, candidate):
    symbol = candidate['symbol']
    try:
        aligned_count = 0
        direction = None
        primary_data = {}

        for interval in KLINE_INTERVALS:
            klines = await get_klines(client, symbol, interval)
            if not klines:
                continue

            closes = np.array([float(k[4]) for k in klines])
            ema9 = talib.EMA(closes, 9)[-1]
            ema21 = talib.EMA(closes, 21)[-1]

            if np.isnan(ema9) or np.isnan(ema21):
                continue

            price = closes[-1]
            if price > ema9 and price > ema21:
                tf_dir = 'long'
            elif price < ema9 and price < ema21:
                tf_dir = 'short'
            else:
                continue

            if direction is None:
                direction = tf_dir
            elif direction != tf_dir:
                return None

            aligned_count += 1

            if interval == PRIMARY_INTERVAL:
                primary_data = {
                    'price': price,
                    'closes': closes,
                    'opens': np.array([float(k[1]) for k in klines]),
                    'highs': np.array([float(k[2]) for k in klines]),
                    'lows': np.array([float(k[3]) for k in klines]),
                    'volumes': np.array([float(k[5]) for k in klines]),
                }

        if aligned_count < MIN_ALIGNED_TF or not primary_data:
            return None

        p = primary_data
        price = p['price']
        closes = p['closes']
        volumes = p['volumes']

        last_vol = volumes[-1]
        avg_vol = np.mean(volumes[-15:-1]) if len(volumes) > 15 else 0
        vol_mult = last_vol / avg_vol if avg_vol > 0 else 0

        rsi = talib.RSI(closes, 14)
        rsi_val = rsi[-1] if not np.isnan(rsi[-1]) else 50

        upper, middle, lower = talib.BBANDS(closes, timeperiod=20, nbdevup=2, nbdevdn=2)
        bb_bonus = check_bollinger_squeeze(upper, middle, lower)

        bull_pat, bear_pat = detect_candlestick_patterns(
            p['opens'][-1:], p['highs'][-1:], p['lows'][-1:], p['closes'][-1:]
        )

        div_bonus = check_rsi_divergence(closes, rsi)

        # Score
        score = 5.0
        score += min(vol_mult - 1, 3) * 2
        score += bb_bonus
        score += div_bonus if direction == 'long' else -div_bonus
        score += bull_pat * 1.5 if direction == 'long' else bear_pat * 1.5

        if direction == 'long' and rsi_val > 45:
            score -= 2
        if direction == 'short' and rsi_val < 55:
            score -= 2

        score = round(max(min(score, 10), 0), 1)

        if score < MIN_SCORE_THRESHOLD:
            return None

        # ── Risk / Reward & Position sizing ─────────────────────────────
        if direction == 'long':
            sl_price = price * (1 - SL_PCT)
            risk_per_unit = price - sl_price
            reward_needed = risk_per_unit * MIN_RISK_REWARD_RATIO
            tp1_price = price + reward_needed
            tp1_pct = (tp1_price - price) / price * 100
            tp2_price = price * (1 + TP2_PCT)
        else:
            sl_price = price * (1 + SL_PCT)
            risk_per_unit = sl_price - price
            reward_needed = risk_per_unit * MIN_RISK_REWARD_RATIO
            tp1_price = price - reward_needed
            tp1_pct = (price - tp1_price) / price * 100
            tp2_price = price * (1 - TP2_PCT)

        risk_amount = ACCOUNT_SIZE * RISK_PER_TRADE_PCT
        position_size_usdt = risk_amount / (risk_per_unit / price)

        lev_suggest = min(8, max(3, int(score / 2)))

        return {
            'symbol': symbol,
            'score': score,
            'direction': direction,
            'price': price,
            'rsi_5m': round(rsi_val, 1),
            'vol_mult': round(vol_mult, 2),
            'position_usdt': round(position_size_usdt, 2),
            'leverage': lev_suggest,
            'sl_price': round(sl_price, 4),
            'tp1_price': round(tp1_price, 4),
            'tp1_pct': round(tp1_pct, 2),
            'tp2_price': round(tp2_price, 4),
            'change_24h_pct': candidate['change_24h_pct'],
            'volume_usdt': candidate['volume_usdt'],
        }

    except Exception as e:
        logger.error(f"Analysis fail {symbol}: {e}")
        return None

async def get_top_and_analyze(client):
    if not is_high_liquidity_time():
        logger.info("Outside high-liquidity window (18:00–01:00 IST) — monitoring only")
        await check_previous_signals(client, set())
        return

    try:
        tickers = await client.futures_ticker(symbols=FIXED_COINS)
        candidates = []
        for t in tickers:
            symbol = t.get('symbol')
            if symbol not in FIXED_COINS:
                continue
            quote_vol = float(t.get('quoteVolume', 0) or 0)
            if quote_vol < 100000:
                continue
            candidates.append({
                'symbol': symbol,
                'change_24h_pct': float(t.get('priceChangePercent', 0)),
                'volume_usdt': quote_vol,
            })

        candidates.sort(key=lambda x: x['change_24h_pct'], reverse=True)
        logger.info(f"Phase 2 scan: {', '.join(c['symbol'] for c in candidates)}")

        tasks = [analyze_coin(client, c) for c in candidates]
        results = await asyncio.gather(*tasks)
        valid_results = {r['symbol']: r for r in results if r}

        if not valid_results:
            logger.info("No Phase 2 qualified signals this cycle")
            await check_previous_signals(client, valid_results.keys())
            return

        best = max(valid_results.values(), key=lambda x: x['score'])

        ist_now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
        trend = "Bullish 🚀 LONG" if best['direction'] == 'long' else "Bearish 📉 SHORT"
        entry = best['price']

        reason = f"Aligned ≥{MIN_ALIGNED_TF} TF + Vol {best['vol_mult']}x + RSI {best['rsi_5m']} + Score {best['score']}"

        msg = f"""
**🔥 Phase 2 QUALIFIED Signal @ {ist_now}**  
**(Enforced min 1:2 RR on TP1)**

**Coin**: {best['symbol']}
**Direction**: {trend}
**Entry**: ${entry:.4f}

**TP1**: ${best['tp1_price']:.4f}  **(+{best['tp1_pct']}% – min 1:2 RR)**
**TP2**: ${best['tp2_price']:.4f}  (+{TP2_PCT*100:.1f}%)

**SL**:  ${best['sl_price']:.4f}   (-{SL_PCT*100:.1f}%)

**Risk**: 1% (~${ACCOUNT_SIZE * RISK_PER_TRADE_PCT:.0f} USDT)
**Position size**: ~${best['position_usdt']:.0f} USDT
**Suggested leverage**: {best['leverage']}x

**Reason**: {reason}

**Test/paper trade only – never risk real capital yet.**
"""

        send_telegram_message(msg)
        logger.info(f"Phase 2 signal sent: {best['symbol']} {best['direction']} @ {best['score']}")

        last_signals[best['symbol']] = {
            'direction': best['direction'],
            'score': best['score'],
            'entry': entry,
            'time': datetime.utcnow(),
            'position_usdt': best['position_usdt'],
        }

        await check_previous_signals(client, valid_results.keys())

    except Exception as e:
        logger.error(f"Phase 2 error: {e}")

async def check_previous_signals(client, current_symbols):
    to_remove = []
    for symbol, data in list(last_signals.items()):
        if symbol not in current_symbols:
            fake = {'symbol': symbol, 'change_24h_pct': 0, 'volume_usdt': 0}
            res = await analyze_coin(client, fake)
            if res:
                if res['direction'] == data['direction'] and res['score'] >= MIN_SCORE_THRESHOLD - 2:
                    send_telegram_message(f"📈 {symbol} still holding ({res['score']}/10) – confirmed, can hold / add lightly.")
                else:
                    send_telegram_message(f"⚠️ {symbol} weakening ({res['score'] if res else 'N/A'}/10) – consider reducing or exit.")
            else:
                send_telegram_message(f"❌ {symbol} invalid now – exit recommended.")
                to_remove.append(symbol)

    for s in to_remove:
        last_signals.pop(s, None)

async def main():
    client = None
    try:
        while True:
            retry_count = 0
            while retry_count < 5:
                try:
                    if client is None:
                        client = await AsyncClient.create(API_KEY, API_SECRET)
                    await get_top_and_analyze(client)
                    break
                except Exception as e:
                    logger.error(f"Retry {retry_count+1}/5: {e}")
                    retry_count += 1
                    if client:
                        await client.close_connection()
                        client = None
                    await asyncio.sleep(10)
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        if client:
            await client.close_connection()

if __name__ == "__main__":
    asyncio.run(main())