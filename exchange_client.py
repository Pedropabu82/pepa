import ccxt.async_support as ccxt
import asyncio
import logging

logger = logging.getLogger(__name__)

class ExchangeClient:
    def __init__(self, config):
        self.name = config["name"]
        self.symbols = config["symbols"]
        self.leverage = config.get("leverage", 10)
        self.config = config
        self.exchange = self._init_exchange(config)

    def _init_exchange(self, config):
        ex_class = {
            "binance": ccxt.binance,
            "phemex": ccxt.phemex
        }.get(config["name"])

        if not ex_class:
            raise ValueError(f"Exchange '{config['name']}' não suportada.")

        exchange = ex_class({
            "apiKey": config["api_key"],
            "secret": config["api_secret"],
            "enableRateLimit": True,
            "asyncio_loop": asyncio.get_event_loop(),
            "options": {"defaultType": "future"}
        })

        if config.get("testnet"):
            exchange.set_sandbox_mode(True)

        logger.info(f"{config['name'].capitalize()}Client initialized ({'testnet' if config.get('testnet') else 'mainnet'})")
        return exchange

    async def fetch_candles(self, symbol, timeframe, limit=300):
        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            import pandas as pd
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.error(f"Erro ao buscar candles para {symbol} {timeframe}: {e}")
            return None

    async def get_balance(self):
        try:
            balance = await self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Erro ao buscar saldo: {e}")
            return {"USDT": {"free": 0}}

    async def close(self):
        try:
            await self.exchange.close()
        except Exception as e:
            logger.error(f"Erro ao fechar exchange: {e}")
