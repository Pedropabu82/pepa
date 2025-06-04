import pandas as pd
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Simples motor de backtest baseado nos logs + sinais simulados

def simulate_trades(trade_log_path='trade_log.csv', initial_balance=1000):
    df = pd.read_csv(trade_log_path)
    df = df[df['type'] == 'ENTRY']
    df = df.sort_values('timestamp')

    balance = initial_balance
    equity_curve = []
    wins = 0
    losses = 0

    for _, row in df.iterrows():
        result = row['result'].lower()
        pnl_pct = row['pnl_pct']
        profit = balance * pnl_pct
        balance += profit
        equity_curve.append(balance)

        if result == 'win':
            wins += 1
        else:
            losses += 1

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades else 0
    max_drawdown = max_drawdown_calc(equity_curve)
    sharpe_ratio = calc_sharpe(equity_curve)

    metrics = {
        'Total Trades': total_trades,
        'Win Rate': round(win_rate * 100, 2),
        'Final Balance': round(balance, 2),
        'Max Drawdown': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2)
    }

    return metrics, equity_curve

def max_drawdown_calc(equity):
    peak = equity[0]
    drawdowns = []
    for value in equity:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        drawdowns.append(drawdown)
    return max(drawdowns) * 100

def calc_sharpe(equity_curve, risk_free_rate=0):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate
    if excess_returns.std() == 0:
        return 0
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

if __name__ == '__main__':
    results, equity = simulate_trades()
    print("\n--- Backtest Results ---")
    for k, v in results.items():
        print(f"{k}: {v}")
