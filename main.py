import asyncio
import numpy as np
import ccxt.async_support as ccxt
import talib
import logging
import requests
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ── Load credentials from .env ──────────────────────────────────────────────
load_dotenv()

API_KEY            = os.getenv("BYBIT_API_KEY")
API_SECRET         = os.getenv("BYBIT_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")

required_vars = {
    "BYBIT_API_KEY":     API_KEY,
    "BYBIT_API_SECRET":  API_SECRET,
    "TELEGRAM_BOT_TOKEN":  TELEGRAM_BOT_TOKEN,
    "TELEGRAM_CHAT_ID":    TELEGRAM_CHAT_ID,
}

missing = [name for name, value in required_vars.items() if not value]

if missing:
    logger.error("Missing required .env variables: " + ", ".join(missing))
    exit(1)

logger.info("Credentials loaded from .env successfully")

# ── CONFIG ──────────────────────────────────────────────────────────────────
INTERVAL_SECONDS        = 60
FIXED_COINS             = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT']

KLINE_INTERVALS         = ['3m', '5m', '15m', '1h']
PRIMARY_INTERVAL        = '5m'
KLINE_LIMIT             = 60

MIN_SCORE_THRESHOLD     = 7.0
MIN_ALIGNED_TF          = 3                 # Fixed: defined here before use

# Two trading windows (IST, weekdays only)
WINDOW_1_START = (18, 0)    # 6:00 PM
WINDOW_1_END   = (23, 30)   # 11:30 PM

WINDOW_2_START = (9, 15)    # 9:15 AM
WINDOW_2_END   = (15, 30)   # 3:30 PM

# Risk settings
ACCOUNT_SIZE            = 10000.0
RISK_PER_TRADE_PCT      = 0.005
SL_PCT                  = 0.008
MIN_RISK_REWARD_RATIO   = 2.5
TP2_PCT                 = 0.035

MIN_SETUP_SCORE         = 7.0
MIN_CONFIRMED_SCORE     = 6.5

# ── GLOBAL STATE ────────────────────────────────────────────────────────────
last_signals    = {}    # Active confirmed trades
pending_setups  = {}    # Pending setups waiting for confirmation

def is_trading_window():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    
    if now_ist.weekday() > 4:  # No weekends
        return False, "Weekend – no trading"
    
    hour, minute = now_ist.hour, now_ist.minute
    current_min = hour * 60 + minute
    
    # Window 1: 18:00 – 23:30
    w1_start = WINDOW_1_START[0] * 60 + WINDOW_1_START[1]
    w1_end   = WINDOW_1_END[0]   * 60 + WINDOW_1_END[1]
    if w1_start <= current_min <= w1_end:
        return True, "Evening window (6:00 PM – 11:30 PM)"
    
    # Window 2: 9:15 – 15:30
    w2_start = WINDOW_2_START[0] * 60 + WINDOW_2_START[1]
    w2_end   = WINDOW_2_END[0]   * 60 + WINDOW_2_END[1]
    if w2_start <= current_min <= w2_end:
        return True, "Indian stock hours (9:15 AM – 3:30 PM)"
    
    return False, "Outside trading windows"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
        logger.info("Telegram sent")
    except Exception as e:
        logger.error(f"Telegram fail: {e}")

async def get_klines(exchange, symbol, interval):
    try:
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe=interval, limit=KLINE_LIMIT)
        if len(ohlcv) < KLINE_LIMIT - 5:
            return None
        # Convert to Binance-like format: [timestamp, open, high, low, close, volume]
        return [[ohlcv[i][0], ohlcv[i][1], ohlcv[i][2], ohlcv[i][3], ohlcv[i][4], ohlcv[i][5]] for i in range(len(ohlcv))]
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
    bull_count = sum(v for k, v in patterns.items() if v and any(w in k for w in ['bullish','hammer','inverted_hammer','doji']))
    bear_count = sum(v for k, v in patterns.items() if v and any(w in k for w in ['bearish','shooting_star','hanging_man']))
    return bull_count, bear_count

def check_rsi_divergence(closes, rsi):
    if len(closes) < 10: return 0
    p_low1 = closes[-5]; p_low2 = min(closes[-10:-5])
    r_low1 = rsi[-5];    r_low2 = min(rsi[-10:-5])
    bull_div = (p_low2 < p_low1) and (r_low2 < r_low1) and r_low1 < 40

    p_high1 = closes[-5]; p_high2 = max(closes[-10:-5])
    r_high1 = rsi[-5];    r_high2 = max(rsi[-10:-5])
    bear_div = (p_high2 > p_high1) and (r_high2 > r_high1) and r_high1 > 60

    if bull_div: return 3
    if bear_div: return -3
    return 0

def check_bollinger_squeeze(upper, middle, lower):
    if len(upper) < 2: return 0
    width = (upper[-1] - lower[-1]) / middle[-1]
    prev_width = (upper[-2] - lower[-2]) / middle[-2]
    if width < 0.015 and prev_width > width * 1.2:
        return 2
    return 0

def check_macd_crossover(macdfast, macdsignal):
    if len(macdfast) < 2: return 0
    if macdfast[-2] < macdsignal[-2] and macdfast[-1] > macdsignal[-1]:
        return 1
    if macdfast[-2] > macdsignal[-2] and macdfast[-1] < macdsignal[-1]:
        return -1
    return 0

async def analyze_coin(exchange, candidate, is_confirmation=False):
    symbol = candidate['symbol']
    try:
        aligned_count = 0
        direction = None
        primary_data = {}

        for interval in KLINE_INTERVALS:
            klines = await get_klines(exchange, symbol, interval)
            if not klines: continue

            closes = np.array([float(k[4]) for k in klines])
            ema9  = talib.EMA(closes, 9)[-1]
            ema21 = talib.EMA(closes, 21)[-1]

            if np.isnan(ema9) or np.isnan(ema21): continue

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

        if rsi_val < 20 or rsi_val > 80:
            return None

        upper, middle, lower = talib.BBANDS(closes, 20, 2, 2)
        bb_bonus = check_bollinger_squeeze(upper, middle, lower)

        bull_pat, bear_pat = detect_candlestick_patterns(
            p['opens'][-1:], p['highs'][-1:], p['lows'][-1:], p['closes'][-1:]
        )

        div_bonus = check_rsi_divergence(closes, rsi)

        macd, macdsignal, _ = talib.MACD(closes, 12, 26, 9)
        macd_cross = check_macd_crossover(macd, macdsignal)

        # Scoring
        score = 4.0 + (aligned_count - MIN_ALIGNED_TF) * 1.5
        score += min(vol_mult - 1.5, 4) * 3.0 if vol_mult > 1.5 else 0
        score += bb_bonus * 1.5
        score += div_bonus if direction == 'long' else -div_bonus
        score += bull_pat * 1.2 if direction == 'long' else bear_pat * 1.2

        if direction == 'long' and rsi_val > 50:   score -= 1.0
        if direction == 'short' and rsi_val < 50: score -= 1.0

        score = round(max(min(score, 10), 0), 1)

        if is_confirmation:
            if score < MIN_CONFIRMED_SCORE:
                return None
        else:
            if score < MIN_SETUP_SCORE:
                return None

        # Risk / Reward
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

        lev_suggest = min(6, max(3, int(score / 2)))

        result = {
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
            'macd_cross': macd_cross,
            'change_24h_pct': candidate['change_24h_pct'],
            'volume_usdt': candidate['volume_usdt'],
        }

        return result

    except Exception as e:
        logger.error(f"Analysis fail {symbol}: {e}")
        return None

async def get_top_and_analyze(exchange):
    active, window_name = is_trading_window()
    if not active:
        logger.info(f"Outside trading windows — idle ({window_name})")
        await check_pending_confirmations(exchange, window_name)
        return

    logger.info(f"Trading window active: {window_name}")

    try:
        tickers = await exchange.fetch_tickers(FIXED_COINS)
        candidates = []
        for sym in FIXED_COINS:
            t = tickers.get(sym)
            if not t: continue
            quote_vol = float(t.get('quoteVolume', 0) or 0)
            if quote_vol < 100000: continue
            candidates.append({
                'symbol': sym,
                'change_24h_pct': float(t.get('percentage', 0)),
                'volume_usdt': quote_vol,
            })

        candidates.sort(key=lambda x: x['change_24h_pct'], reverse=True)
        logger.info(f"Scan: {', '.join(c['symbol'] for c in candidates)}")

        tasks = [analyze_coin(exchange, c, is_confirmation=False) for c in candidates]
        results = await asyncio.gather(*tasks)
        setups = {r['symbol']: r for r in results if r}

        if not setups:
            logger.info("No setups this cycle")
            await check_pending_confirmations(exchange, window_name)
            return

        for symbol, setup in setups.items():
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
            trend = "Bullish LONG" if setup['direction'] == 'long' else "Bearish SHORT"

            msg = f"""
**⚠ SETUP ALERT @ {ist_now}** ({window_name})

**Coin**: {symbol}
**Direction**: {trend}
**Price**: ${setup['price']:.4f}
**Score**: {setup['score']}/10
**RSI**: {setup['rsi_5m']}
**Vol mult**: {setup['vol_mult']}x

Waiting for double-confirmation...
"""
            send_telegram_message(msg)
            logger.info(f"Setup alert sent: {symbol} {setup['direction']} @ {setup['score']}")

            pending_setups[symbol] = {
                'setup_time': datetime.now(pytz.utc),
                'direction': setup['direction'],
                'score': setup['score'],
                'price': setup['price'],
                'macd_cross': setup['macd_cross'],
            }

        await check_pending_confirmations(exchange, window_name)

    except Exception as e:
        logger.error(f"Scan error: {e}")

async def check_pending_confirmations(exchange, window_name):
    now = datetime.now(pytz.utc)
    to_remove = []
    for symbol, pending in list(pending_setups.items()):
        if now - pending['setup_time'] > timedelta(minutes=3):
            send_telegram_message(f"⏳ {symbol} setup expired – no confirmation within 3 min.")
            to_remove.append(symbol)
            continue

        fake = {'symbol': symbol, 'change_24h_pct': 0, 'volume_usdt': 0}
        conf = await analyze_coin(exchange, fake, is_confirmation=True)

        if not conf:
            continue

        confirmed = False
        reason = []

        if conf['direction'] != pending['direction']:
            reason.append("Direction flipped")
        else:
            if (pending['direction'] == 'long' and conf['macd_cross'] == 1) or \
               (pending['direction'] == 'short' and conf['macd_cross'] == -1):
                reason.append("MACD crossover confirmed")
                confirmed = True
            else:
                reason.append("No MACD confirmation")

            if conf['vol_mult'] >= 1.2:
                reason.append("Volume strong")
            else:
                reason.append("Volume dropped")

            if 25 < conf['rsi_5m'] < 75:
                reason.append("RSI in safe zone")
            else:
                reason.append("RSI extreme")

            btc_data = await analyze_coin(exchange, {'symbol': 'BTC/USDT:USDT', 'change_24h_pct': 0, 'volume_usdt': 0}, is_confirmation=True)
            if btc_data and btc_data['direction'] == pending['direction']:
                reason.append("BTC aligned")
            else:
                reason.append("BTC not aligned")
                confirmed = False

        if confirmed:
            ist_now = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%H:%M:%S IST')
            trend = "Bullish LONG" if conf['direction'] == 'long' else "Bearish 📉 SHORT"

            msg = f"""
**✅ DOUBLE-CONFIRMED ENTRY @ {ist_now}** ({window_name})

**Coin**: {symbol}
**Direction**: {trend}
**Entry**: ${conf['price']:.4f}

**TP1**: ${conf['tp1_price']:.4f} (+{conf['tp1_pct']}% – 1:{MIN_RISK_REWARD_RATIO} RR)
**TP2**: ${conf['tp2_price']:.4f} (+{TP2_PCT*100:.1f}%)

**SL**: ${conf['sl_price']:.4f} (-{SL_PCT*100:.1f}%)

**Risk**: 0.5% (~${ACCOUNT_SIZE * RISK_PER_TRADE_PCT:.0f} USDT)
**Position**: ~${conf['position_usdt']:.0f} USDT
**Leverage**: {conf['leverage']}x

**Confirmation reasons**: {', '.join(reason)}
"""
            send_telegram_message(msg)
            logger.info(f"Confirmed entry: {symbol} {conf['direction']}")

            last_signals[symbol] = conf
            to_remove.append(symbol)

    for s in to_remove:
        pending_setups.pop(s, None)

async def main():
    exchange = None
    try:
        while True:
            retry_count = 0
            while retry_count < 5:
                try:
                    if exchange is None:
                        exchange = ccxt.bybit({
                            'apiKey': API_KEY,
                            'secret': API_SECRET,
                            'enableRateLimit': True,
                            'options': {'defaultType': 'future'},
                        })
                        logger.info("Connected to Bybit API")
                    logger.info("Starting scan cycle...")
                    await get_top_and_analyze(exchange)
                    break
                except Exception as e:
                    logger.error(f"Retry {retry_count+1}/5: {e}")
                    retry_count += 1
                    if exchange:
                        await exchange.close()
                        exchange = None
                    await asyncio.sleep(10)
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        if exchange:
            await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())