import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Optional
import warnings

def create_stock_chart(data: pd.DataFrame, symbol: str, params: Optional[Dict] = None) -> go.Figure:
    """Create an interactive stock chart with technical indicators."""
    # Ensure data is properly formatted
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame but got {type(data)}")
    
    # Check if required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Create base candlestick chart with improved hover label styling
    fig = go.Figure(data=[go.Candlestick(
        x=data.index.tolist(),
        open=data['Open'].tolist(),
        high=data['High'].tolist(),
        low=data['Low'].tolist(),
        close=data['Close'].tolist(),
        name="Candlestick",
        hoverlabel=dict(
            bgcolor="white",
            font=dict(size=14, color="black", family="Arial, sans-serif")
        )
    )])
    
    # Add technical indicators with improved styling
    data_with_indicators = add_technical_indicators(data, params)
    
    # Handle dictionary format
    if isinstance(params, dict):
        # Add SMA if enabled
        if params.get('sma', {}).get('use', False):
            period = params['sma'].get('period', 20)
            if f'SMA_{period}' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators[f'SMA_{period}'].tolist(),
                    mode='lines',
                    name=f'SMA ({period})',
                    line=dict(width=2),
                    hoverlabel=dict(
                        bgcolor="white",
                        font=dict(size=14, color="black", family="Arial, sans-serif")
                    ),
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>SMA ({period})</b>: %{{y:.2f}}<br><i>Simple Moving Average over {period} days</i><extra></extra>"
                ))
        
        # Add EMA if enabled
        if params.get('ema', {}).get('use', False):
            period = params['ema'].get('period', 20)
            if f'EMA_{period}' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators[f'EMA_{period}'].tolist(),
                    mode='lines',
                    name=f'EMA ({period})',
                    line=dict(width=2),
                    hoverlabel=dict(
                        bgcolor="white",
                        font=dict(size=14, color="black", family="Arial, sans-serif")
                    ),
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>EMA ({period})</b>: %{{y:.2f}}<br><i>Exponential Moving Average over {period} days</i><extra></extra>"
                ))
        
        # Add Bollinger Bands if enabled
        if params.get('bollinger', {}).get('use', False):
            if 'BB_Upper' in data_with_indicators.columns and 'BB_Lower' in data_with_indicators.columns:
                period = params['bollinger'].get('period', 20)
                std_dev = params['bollinger'].get('std', 2)
                
                # Upper band
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Upper'].tolist(),
                    mode='lines',
                    name=f'BB Upper ({std_dev}σ)',
                    line=dict(width=2),
                    hoverlabel=dict(
                        bgcolor="white",
                        font=dict(size=14, color="black", family="Arial, sans-serif")
                    ),
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Upper</b>: %{{y:.2f}}<br><i>Upper Bollinger Band ({std_dev} standard deviations)</i><extra></extra>"
                ))
                
                # Lower band
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Lower'].tolist(),
                    mode='lines',
                    name=f'BB Lower ({std_dev}σ)',
                    line=dict(width=2),
                    hoverlabel=dict(
                        bgcolor="white",
                        font=dict(size=14, color="black", family="Arial, sans-serif")
                    ),
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Lower</b>: %{{y:.2f}}<br><i>Lower Bollinger Band ({std_dev} standard deviations)</i><extra></extra>"
                ))
                
                # Middle band
                if 'BB_Middle' in data_with_indicators.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index.tolist(),
                        y=data_with_indicators['BB_Middle'].tolist(),
                        mode='lines',
                        name=f'BB Middle (SMA {period})',
                        line=dict(dash='dash', width=2),
                        hoverlabel=dict(
                            bgcolor="white",
                            font=dict(size=14, color="black", family="Arial, sans-serif")
                        ),
                        hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Middle</b>: %{{y:.2f}}<br><i>Middle Bollinger Band (SMA {period})</i><extra></extra>"
                    ))
        
        # Add RSI if enabled
        if params.get('rsi', {}).get('use', False) and 'RSI' in data_with_indicators.columns:
            period = params['rsi'].get('period', 14)
            fig.add_trace(go.Scatter(
                x=data.index.tolist(),
                y=data_with_indicators['RSI'].tolist(),
                mode='lines',
                name=f'RSI ({period})',
                line=dict(width=2),
                yaxis="y2",
                hoverlabel=dict(
                    bgcolor="white",
                    font=dict(size=14, color="black", family="Arial, sans-serif")
                ),
                hovertemplate=f"<b>Date</b>: %{{x}}<br><b>RSI ({period})</b>: %{{y:.2f}}<extra></extra>"
            ))
            
            # Add RSI reference lines with improved visibility
            fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
                         line=dict(color="red", width=1.5, dash="dash"), yref="y2")
            fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
                         line=dict(color="red", width=1.5, dash="dash"), yref="y2")
            
            # Using high-contrast colors for RSI axis text
            fig.update_layout(
                yaxis2=dict(
                    title=dict(
                        text="RSI",
                        font=dict(size=14, color="#00FF00", family="Arial, sans-serif")  # Bright green
                    ),
                    range=[0, 100],
                    side="right",
                    overlaying="y",
                    tickfont=dict(size=12, color="#00FF00", family="Arial, sans-serif")  # Bright green
                )
            )
    
    # Update layout with high-contrast colors that should be visible in any mode
    fig.update_layout(
        title=dict(
            text=f"{symbol} Stock Price",
            font=dict(size=24, color="#FF9900", family="Arial, sans-serif")  # Orange
        ),
        xaxis=dict(
            title=dict(
                text="Date",
                font=dict(size=14, color="#FF9900", family="Arial, sans-serif")  # Orange
            ),
            tickfont=dict(size=12, color="#FF9900", family="Arial, sans-serif"),  # Orange
            gridcolor="rgba(255, 255, 255, 0.3)"  # More visible grid lines
        ),
        yaxis=dict(
            title=dict(
                text="Price",
                font=dict(size=14, color="#FF9900", family="Arial, sans-serif")  # Orange
            ),
            tickfont=dict(size=12, color="#FF9900", family="Arial, sans-serif"),  # Orange
            gridcolor="rgba(255, 255, 255, 0.3)"  # More visible grid lines
        ),
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        height=450,
        # Removing template to have more control over colors
        # template='plotly_dark',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0, 0, 0, 0.8)",  # More opaque background
            bordercolor="#FF9900",  # Orange border
            borderwidth=1,
            font=dict(size=12, color="#FF9900", family="Arial, sans-serif")  # Orange
        ),
        plot_bgcolor="#111111",  # Very dark background
        paper_bgcolor="#222222"  # Dark gray paper background
    )
    
    # Add a more visible border
    fig.update_layout(
        shapes=[
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="#FF9900", width=2)  # Orange border
            )
        ]
    )
    
    return fig