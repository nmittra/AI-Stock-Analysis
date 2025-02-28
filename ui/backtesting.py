"""UI components for backtesting and strategy comparison."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, List

from analysis.backtesting import backtest_strategy, compare_strategies, BacktestResult
from analysis.strategies import detect_elliott_wave, can_slim_screen

def render_backtesting_ui():
    """Render the backtesting interface in the Streamlit app."""
    st.header("Strategy Backtesting")
    
    # Strategy selection
    strategy_options = {
        "Elliott Wave": detect_elliott_wave,
        "CAN SLIM": can_slim_screen
    }
    selected_strategy = st.selectbox(
        "Select Trading Strategy",
        list(strategy_options.keys())
    )
    
    # Strategy parameters
    st.subheader("Strategy Parameters")
    params = {}
    if selected_strategy == "CAN SLIM":
        params['quarterly_earnings_growth'] = st.number_input(
            "Minimum Quarterly Earnings Growth (%)",
            min_value=0, value=25
        )
        params['annual_earnings_growth'] = st.number_input(
            "Minimum Annual Earnings Growth (%)",
            min_value=0, value=25
        )
    
    # Backtesting parameters
    st.subheader("Backtesting Parameters")
    initial_capital = st.number_input(
        "Initial Capital ($)",
        min_value=1000,
        value=100000
    )
    
    # Run backtest button
    if st.button("Run Backtest"):
        if "stock_data" in st.session_state:
            results = []
            for ticker, data in st.session_state["stock_data"].items():
                params['ticker'] = ticker
                try:
                    result = backtest_strategy(
                        data=data,
                        strategy_func=strategy_options[selected_strategy],
                        strategy_params=params,
                        initial_capital=initial_capital
                    )
                    results.append(result)
                except Exception as e:
                    st.error(f"Error backtesting {ticker}: {str(e)}")
            
            if results:
                display_backtest_results(results)
        else:
            st.warning("Please fetch stock data first.")

def display_backtest_results(results: List[BacktestResult]):
    """Display the results of backtesting in an organized format.
    
    Args:
        results: List of BacktestResult objects to display
    """
    # Create comparison DataFrame
    comparison_df = compare_strategies(results)
    
    # Display summary metrics
    st.subheader("Strategy Performance Comparison")
    st.dataframe(comparison_df)
    
    # Create performance visualization
    fig = go.Figure()
    for result in results:
        equity_curve = [0]
        dates = [result.start_date]
        for trade in result.trades:
            equity_curve.append(equity_curve[-1] + trade['pnl'])
            dates.append(trade['exit_date'])
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=equity_curve,
            name=f"{result.ticker} - {result.strategy_name}",
            mode='lines'
        ))
    
    fig.update_layout(
        title="Equity Curves",
        xaxis_title="Date",
        yaxis_title="Cumulative P&L ($)",
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Export functionality
    if st.button("Export Results"):
        # Prepare detailed results for export
        export_data = {
            'comparison': comparison_df,
            'trades': pd.concat([
                pd.DataFrame(result.trades)
                for result in results
            ], keys=[result.ticker for result in results])
        }
        
        # Save to session state for download
        st.session_state['export_data'] = export_data
        
        # Create download buttons
        st.download_button(
            label="Download Comparison (CSV)",
            data=comparison_df.to_csv(index=False),
            file_name=f"strategy_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        st.download_button(
            label="Download Detailed Trades (CSV)",
            data=export_data['trades'].to_csv(),
            file_name=f"detailed_trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )