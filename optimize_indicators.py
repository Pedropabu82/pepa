import logging
import pandas as pd
from itertools import product
from backtest_engine import simulate_trades

logger = logging.getLogger(__name__)

class IndicatorOptimizer:
    def __init__(self, strategy, backtest_days=30):
        self.strategy = strategy
        self.backtest_days = backtest_days
        self.last_optimization = None

    def should_optimize(self):
        if self.last_optimization is None:
            return True
        win_rate = self.strategy.get_win_rate('BTCUSDT') or 0.5
        return win_rate < 0.5 or (pd.Timestamp.now() - self.last_optimization).days >= 7

    def optimize(self):
        if not self.should_optimize():
            logger.info("No optimization needed.")
            return
        param_grid = {
            'ema_short': range(8, 16),
            'ema_long': range(20, 30),
            'rsi_period': range(10, 20)
        }
        best_sharpe = -float('inf')
        best_params = {}
        original_config = self.strategy.config['indicators']['BTCUSDT'].copy()
        for params in product(*param_grid.values()):
            config = original_config.copy()
            config.update(dict(zip(param_grid.keys(), params)))
            self.strategy.config['indicators']['BTCUSDT'] = config
            metrics, _ = simulate_trades(self.strategy, days=self.backtest_days)
            if metrics['Sharpe Ratio'] > best_sharpe:
                best_sharpe = metrics['Sharpe Ratio']
                best_params = config.copy()
        self.strategy.config['indicators']['BTCUSDT'] = best_params
        self.last_optimization = pd.Timestamp.now()
        logger.info(f"Optimized parameters: {best_params}, Sharpe: {best_sharpe}")
        with open('config.json', 'w') as f:
            json.dump(self.strategy.config, f, indent=4)
