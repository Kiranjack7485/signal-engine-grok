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
INTERVAL_SECONDS        = 10                    # Every 10 seconds

MIN_VOLUME_USDT         = 3000000               # Minimum 24h volume
MIN_CHANGE_PCT_24H      = 1.5                   # Minimum absolute 24h % change
MIN_CHANGE_PCT_1H       = 0.6                   # Minimum absolute 1h % change

KLINE_INTERVALS         = ['3m', '5m', '15m', '1h']
PRIMARY_INTERVAL        = '5m'
KLINE_LIMIT             = 60

MIN_SCORE_THRESHOLD     = 6.5
MIN_ALIGNED_TF          = 3

# Two trading windows (IST, weekdays only)
WINDOW_1_START = (18, 0)    # 6:00 PM
WINDOW_1_END   = (23, 30)   # 11:30 PM

WINDOW_2_START = (9, 15)    # 9:15 AM
WINDOW_2_END   = (15, 30)   # 3:30 PM

# Risk settings for signal only
SL_PCT                  = 0.008
MIN_RISK_REWARD_RATIO   = 2.5
TP2_PCT                 = 0.035

MIN_SETUP_SCORE         = 6.5
MIN_CONFIRMED_SCORE     = 6.0
MAX_HOLD_MIN            = 15
MIN_HOLD_MIN            = 3

# ── GLOBAL STATE ────────────────────────────────────────────────────────────
last_signals    = {}    # To avoid duplicate signals

def is_trading_window():
    now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
    if now_ist.weekday() > 4:
        return False, "Weekend – no trading"
    
    hour, minute = now_ist.hour, now_ist.minute
    current_min = hour * 60 + minute
    
    w1_start = WINDOW_1_START[0] * 60 + WINDOW_1_START[1]
    w1_end   = WINDOW_1_END[0]   * 60 + WINDOW_1_END[1]
    if w1_start <= current_min <= w1_end:
        return True, "Evening window (6:00 PM – 11:30 PM)"
    
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

async def analyze_coin(exchange, candidate):
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

        # Your original 5 golden rules scoring (untouched)
        score = 4.0 + (aligned_count - MIN_ALIGNED_TF) * 1.5
        score += min(vol_mult - 1.5, 4) * 3.0 if vol_mult > 1.5 else 0
        score += bb_bonus * 1.5
        score += div_bonus if direction == 'long' else -div_bonus
        score += bull_pat * 1.2 if direction == 'long' else bear_pat * 1.2

        if direction == 'long' and rsi_val > 50:   score -= 1.0
        if direction == 'short' and rsi_val < 50: score -= 1.0

        score = round(max(min(score, 10), 0), 1)

        if score < MIN_SETUP_SCORE:
            return None

        # Risk / Reward calculation
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

        # Suggested hold time based on volatility
        atr = talib.ATR(p['highs'], p['lows'], closes, timeperiod=14)[-1]
        volatility = atr / price * 100
        max_hold = min(30, max(5, int(100 / volatility)))

        result = {
            'symbol': symbol,
            'score': score,
            'direction': direction,
            'price': price,
            'rsi_5m': round(rsi_val, 1),
            'vol_mult': round(vol_mult, 2),
            'sl_price': round(sl_price, 4),
            'tp1_price': round(tp1_price, 4),
            'tp1_pct': round(tp1_pct, 2),
            'tp2_price': round(tp2_price, 4),
            'max_hold_min': max_hold,
            'min_hold_min': min(3, max_hold),
        }

        return result

    except Exception as e:
        logger.error(f"Analysis fail {symbol}: {e}")
        return None

async def get_top_and_analyze(exchange):
    active, window_name = is_trading_window()
    if not active:
        logger.info(f"Outside trading windows — idle ({window_name})")
        return

    logger.info(f"Trading window active: {window_name}")

    try:
        # Fetch ALL tickers
        tickers = await exchange.fetch_tickers()
        candidates = []

        for sym, t in tickers.items():
            # Strict Bybit USDT perpetual futures filter
            if not (sym.endswith('USDT') and ':' not in sym and 'PERP' not in sym and len(sym) > 5 and 'SPOT' not in sym.upper()):
                continue

            quote_vol = float(t.get('quoteVolume', 0) or 0)
            change_24h = abs(float(t.get('percentage', 0) or 0))
            change_1h = abs(float(t.get('change1h', 0) or 0))  # fallback

            if quote_vol < MIN_VOLUME_USDT:
                continue
            if change_24h < MIN_CHANGE_PCT_24H and change_1h < MIN_CHANGE_PCT_1H:
                continue

            candidates.append({
                'symbol': sym,
                'change_24h_pct': change_24h,
                'volume_usdt': quote_vol,
            })

        if not candidates:
            logger.info("No trending USDT perpetual futures found this cycle")
            return

        # Take top 10 strongest
        candidates.sort(key=lambda x: x['change_24h_pct'], reverse=True)
        top_candidates = candidates[:10]

        logger.info(f"Analyzing top 10 trending coins: {[c['symbol'] for c in top_candidates]}")

        tasks = [analyze_coin(exchange, c) for c in top_candidates]
        results = await asyncio.gather(*tasks)
        setups = [r for r in results if r is not None]

        if not setups:
            logger.info("No strong setups found after 5-rule analysis")
            return

        setups.sort(key=lambda x: x['score'], reverse=True)

        for setup in setups[:3]:  # Max 3 signals per cycle
            direction_emoji = "🟢 LONG" if setup['direction'] == 'long' else "🔴 SHORT"
            msg = f"""
**🚀 NEW SIGNAL – {direction_emoji}**

**Coin**: {setup['symbol']}
**Current Price**: ${setup['price']:.4f}
**Best Entry**: ${setup['price']:.4f}
**Take Profit 1**: ${setup['tp1_price']:.4f} (+{setup['tp1_pct']:.2f}%)
**Take Profit 2**: ${setup['tp2_price']:.4f} (+{TP2_PCT*100:.1f}%)
**Stop Loss**: ${setup['sl_price']:.4f} (-{SL_PCT*100:.1f}%)
**Rating / Score**: {setup['score']:.1f}/10
**Suggested Hold**: {setup['min_hold_min']} – {setup['max_hold_min']} min
**Volume Surge**: {setup['vol_mult']:.2f}x
**RSI (5m)**: {setup['rsi_5m']}

High conviction setup – act fast!
"""
            send_telegram_message(msg)
            logger.info(f"Signal sent for {setup['symbol']} – Score: {setup['score']}")

    except Exception as e:
        logger.error(f"Scan error: {e}")

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
                        logger.info("Successfully logged into Bybit API")
                    logger.info("Starting scan cycle...")
                    await get_top_and_analyze(exchange)
                    break
                except Exception as e:
                    logger.error(f"Retry {retry_count+1}/5: {e}")
                    retry_count += 1
                    if exchange:
                        await exchange.close()
                        exchange = None
                    await asyncio.sleep(5)
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        if exchange:
            await exchange.close()

if __name__ == "__main__":
    asyncio.run(main())