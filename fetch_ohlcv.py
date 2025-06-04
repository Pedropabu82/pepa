import ccxt.async_support as ccxt
import pandas as pd
import asyncio
import os

async def fetch_candles(symbol, timeframe, limit=300):
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'asyncio_loop': asyncio.get_event_loop()
    })
    exchange.options['defaultType'] = 'future'
    try:
        since = int((pd.Timestamp.now() - pd.Timedelta(days=30)).timestamp() * 1000)
        data = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error fetching candles for {symbol} {timeframe}: {e}")
        return None
    finally:
        await exchange.close()

async def fetch_and_save_all():
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    timeframes = ["5m", "15m", "30m", "1h", "4h"]
    os.makedirs("data", exist_ok=True)
    for symbol in symbols:
        for tf in timeframes:
            df = await fetch_candles(symbol, tf, limit=1000)
            if df is not None and not df.empty:
                df.to_csv(f"data/{symbol}_{tf}.csv", index=False)
                print(f"Saved {len(df)} candles for {symbol} {tf}")
            await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(fetch_and_save_all())
