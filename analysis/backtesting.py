"""Backtesting module for testing trading strategies on historical data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable

class BacktestResult:
    def __init__(self, strategy_name: str, ticker: str, start_date: str, end_date: str):
        self.strategy_name = strategy_name
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.trades: List[Dict] = []
        self.metrics: Dict = {}

    def add_trade(self, entry_date: str, entry_price: float, exit_date: str, exit_price: float,
                  position_type: str, quantity: float) -> None:
        """Add a trade to the backtest results."""
        trade = {
            'entry_date': entry_date,
            'entry_price': entry_price,
            'exit_date': exit_date,
            'exit_price': exit_price,
            'position_type': position_type,
            'quantity': quantity,
            'pnl': (exit_price - entry_price) * quantity if position_type == 'long' \
                  else (entry_price - exit_price) * quantity
        }
        self.trades.append(trade)

    def calculate_metrics(self) -> None:
        """Calculate performance metrics for the strategy."""
        if not self.trades:
            return

        # Calculate basic metrics
        pnls = [trade['pnl'] for trade in self.trades]
        self.metrics['total_trades'] = len(self.trades)
        self.metrics['winning_trades'] = len([pnl for pnl in pnls if pnl > 0])
        self.metrics['losing_trades'] = len([pnl for pnl in pnls if pnl < 0])
        self.metrics['total_pnl'] = sum(pnls)
        self.metrics['win_rate'] = self.metrics['winning_trades'] / self.metrics['total_trades']
        
        # Calculate advanced metrics
        returns = [trade['pnl'] / (trade['entry_price'] * trade['quantity']) for trade in self.trades]
        self.metrics['avg_return'] = np.mean(returns)
        self.metrics['std_return'] = np.std(returns)
        self.metrics['sharpe_ratio'] = self.metrics['avg_return'] / self.metrics['std_return'] \
                                      if self.metrics['std_return'] != 0 else 0
        self.metrics['max_drawdown'] = self._calculate_max_drawdown()

    def _calculate_max_drawdown(self) -> float:
        """Calculate the maximum drawdown from peak equity."""
        if not self.trades:
            return 0.0

        equity_curve = [0]  # Start with zero
        for trade in self.trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])

        # Convert to numpy array for calculations
        equity_curve = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_drawdown = np.max(drawdown)

        return max_drawdown

def backtest_strategy(data: pd.DataFrame, strategy_func: Callable, 
                      strategy_params: Dict, initial_capital: float = 100000.0) -> BacktestResult:
    """Run a backtest for a given strategy on historical data.
    
    Args:
        data: DataFrame containing OHLCV data
        strategy_func: Function that implements the trading strategy
        strategy_params: Dictionary of parameters for the strategy
        initial_capital: Initial capital for the backtest
    
    Returns:
        BacktestResult: Object containing backtest results and metrics
    """
    result = BacktestResult(
        strategy_name=strategy_func.__name__,
        ticker=strategy_params.get('ticker', 'Unknown'),
        start_date=str(data.index[0].date()),
        end_date=str(data.index[-1].date())
    )
    
    # Run strategy
    signals = strategy_func(data, **strategy_params)
    position = 0
    entry_price = 0
    entry_date = None
    available_capital = initial_capital

    for date, row in data.iterrows():
        signal = signals.loc[date]
        
        # Handle position entry
        if position == 0 and signal != 0:
            entry_price = row['Close']
            entry_date = date
            position = signal  # 1 for long, -1 for short
            quantity = (available_capital * 0.98) / entry_price  # Use 98% of available capital
        
        # Handle position exit
        elif position != 0 and ((position == 1 and signal == -1) or 
                               (position == -1 and signal == 1) or
                               date == data.index[-1]):  # Force exit on last day
            exit_price = row['Close']
            position_type = 'long' if position == 1 else 'short'
            
            result.add_trade(
                entry_date=str(entry_date.date()),
                entry_price=entry_price,
                exit_date=str(date.date()),
                exit_price=exit_price,
                position_type=position_type,
                quantity=quantity
            )
            
            # Update available capital
            trade_pnl = (exit_price - entry_price) * quantity if position == 1 \
                       else (entry_price - exit_price) * quantity
            available_capital += trade_pnl
            position = 0
    
    # Calculate final metrics
    result.calculate_metrics()
    return result

def compare_strategies(results: List[BacktestResult]) -> pd.DataFrame:
    """Compare multiple backtest results.
    
    Args:
        results: List of BacktestResult objects to compare
    
    Returns:
        DataFrame: Comparison of strategy performance metrics
    """
    comparison_data = []
    for result in results:
        result_dict = {
            'Strategy': result.strategy_name,
            'Ticker': result.ticker,
            'Start Date': result.start_date,
            'End Date': result.end_date,
            **result.metrics
        }
        comparison_data.append(result_dict)
    
    return pd.DataFrame(comparison_data)