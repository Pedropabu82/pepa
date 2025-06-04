import asyncio
import json
import logging
from api_client import BinanceClient
from live_strategy import LiveMAStrategy
from backtest_engine import simulate_trades

# Configurar logging para console e arquivo
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

async def main():
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting bot...")
        with open('config.json', 'r') as f:
            cfg = json.load(f)
        logger.debug(f"Config loaded: {cfg}")
        client = BinanceClient(cfg)
        strategy = LiveMAStrategy(client, cfg)
        logger.info("Initializing strategy...")
        await strategy.initialize()

        mode = cfg.get('mode', 'live')
        if mode == 'train':
            logger.info("Running in training mode...")
            await strategy.train_mode(cfg.get('train_days', 30))
        elif mode == 'backtest':
            logger.info("Running in backtest mode...")
            metrics, equity = simulate_trades()
            logger.info(f"Backtest results: {metrics}")
        else:
            logger.info("Running in live mode...")
            await strategy.run()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        if 'client' in locals():
            await client.close()

if __name__ == "__main__":
    asyncio.run(main())
