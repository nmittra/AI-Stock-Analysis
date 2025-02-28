import plotly.graph_objects as go
import pandas as pd

def add_technical_indicators(data, indicators):
    """Add technical indicators to the data.
    
    Args:
        data: DataFrame containing stock data
        indicators: Either a list of indicator names or a dictionary with indicator settings
    
    Returns:
        DataFrame with added technical indicators
    """
    result = data.copy()
    
    # Handle dictionary format (from app.py)
    if isinstance(indicators, dict):
        # Add SMA if enabled
        if indicators.get('sma', {}).get('use', False):
            period = indicators['sma'].get('period', 20)
            result[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # Add EMA if enabled
        if indicators.get('ema', {}).get('use', False):
            period = indicators['ema'].get('period', 20)
            result[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
        
        # Add Bollinger Bands if enabled
        if indicators.get('bollinger', {}).get('use', False):
            period = indicators['bollinger'].get('period', 20)
            std_dev = indicators['bollinger'].get('std', 2)
            sma = data['Close'].rolling(window=period).mean()
            std = data['Close'].rolling(window=period).std()
            result['BB_Upper'] = sma + std_dev * std
            result['BB_Lower'] = sma - std_dev * std
            result['BB_Middle'] = sma
        
        # Add RSI if enabled
        if indicators.get('rsi', {}).get('use', False):
            period = indicators['rsi'].get('period', 14)
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            rs = avg_gain / avg_loss
            result['RSI'] = 100 - (100 / (1 + rs))
    
    # Handle list format (original implementation)
    else:
        for indicator in indicators:
            if indicator == "20-Day SMA":
                result['SMA_20'] = data['Close'].rolling(window=20).mean()
            elif indicator == "20-Day EMA":
                result['EMA_20'] = data['Close'].ewm(span=20).mean()
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                result['BB_Upper'] = sma + 2 * std
                result['BB_Lower'] = sma - 2 * std
            elif indicator == "VWAP" and 'Volume' in data.columns:
                result['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
            elif indicator == "RSI":
                delta = data['Close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / avg_loss
                result['RSI'] = 100 - (100 / (1 + rs))
    
    return result

def create_stock_chart(data, ticker, indicators):
    """Create an interactive stock chart with technical indicators."""
    # Ensure data is properly formatted
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"Expected DataFrame but got {type(data)}")
    
    # Check if required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Required column '{col}' not found in data")
    
    # Create base candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=data.index.tolist(),
        open=data['Open'].tolist(),
        high=data['High'].tolist(),
        low=data['Low'].tolist(),
        close=data['Close'].tolist(),
        name="Candlestick",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )])
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data, indicators)
    
    # Handle dictionary format (from app.py)
    if isinstance(indicators, dict):
        # Add SMA if enabled
        if indicators.get('sma', {}).get('use', False):
            period = indicators['sma'].get('period', 20)
            if f'SMA_{period}' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators[f'SMA_{period}'].tolist(),
                    mode='lines',
                    name=f'SMA ({period})',
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>SMA ({period})</b>: %{{y:.2f}}<br><i>Simple Moving Average over {period} days</i><extra></extra>"
                ))
        
        # Add EMA if enabled
        if indicators.get('ema', {}).get('use', False):
            period = indicators['ema'].get('period', 20)
            if f'EMA_{period}' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators[f'EMA_{period}'].tolist(),
                    mode='lines',
                    name=f'EMA ({period})',
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>EMA ({period})</b>: %{{y:.2f}}<br><i>Exponential Moving Average over {period} days</i><extra></extra>"
                ))
        
        # Add Bollinger Bands if enabled
        if indicators.get('bollinger', {}).get('use', False):
            if 'BB_Upper' in data_with_indicators.columns and 'BB_Lower' in data_with_indicators.columns:
                period = indicators['bollinger'].get('period', 20)
                std_dev = indicators['bollinger'].get('std', 2)
                
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Upper'].tolist(),
                    mode='lines',
                    name=f'BB Upper ({std_dev}σ)',
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Upper</b>: %{{y:.2f}}<br><i>Upper Bollinger Band ({std_dev} standard deviations)</i><extra></extra>"
                ))
                
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Lower'].tolist(),
                    mode='lines',
                    name=f'BB Lower ({std_dev}σ)',
                    hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Lower</b>: %{{y:.2f}}<br><i>Lower Bollinger Band ({std_dev} standard deviations)</i><extra></extra>"
                ))
                
                if 'BB_Middle' in data_with_indicators.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index.tolist(),
                        y=data_with_indicators['BB_Middle'].tolist(),
                        mode='lines',
                        name=f'BB Middle (SMA {period})',
                        line=dict(dash='dash'),
                        hovertemplate=f"<b>Date</b>: %{{x}}<br><b>BB Middle</b>: %{{y:.2f}}<br><i>Middle Bollinger Band (SMA {period})</i><extra></extra>"
                    ))
        
        # Add RSI if enabled
        if indicators.get('rsi', {}).get('use', False) and 'RSI' in data_with_indicators.columns:
            period = indicators['rsi'].get('period', 14)
            # Add RSI as a subplot
            fig.add_trace(go.Scatter(
                x=data.index.tolist(),
                y=data_with_indicators['RSI'].tolist(),
                mode='lines',
                name=f'RSI ({period})',
                yaxis="y2",
                hovertemplate=f"<b>Date</b>: %{{x}}<br><b>RSI ({period})</b>: %{{y:.2f}}<extra></extra>"
            ))
            
            # Add RSI reference lines (30 and 70)
            fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
                          line=dict(color="red", width=1, dash="dash"), yref="y2")
            fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
                          line=dict(color="red", width=1, dash="dash"), yref="y2")
            
            # Update layout for RSI subplot
            fig.update_layout(
                yaxis2=dict(
                    title="RSI",
                    range=[0, 100],
                    side="right",
                    overlaying="y"
                )
            )
    
    # Handle list format (original implementation)
    else:
        for indicator in indicators:
            if indicator == "20-Day SMA":
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['SMA_20'].tolist(),
                    mode='lines',
                    name='SMA (20)',
                    hovertemplate="<b>Date</b>: %{x}<br><b>SMA (20)</b>: %{y:.2f}<br><i>Simple Moving Average over 20 days</i><extra></extra>"
                ))
            elif indicator == "20-Day EMA":
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['EMA_20'].tolist(),
                    mode='lines',
                    name='EMA (20)',
                    hovertemplate="<b>Date</b>: %{x}<br><b>EMA (20)</b>: %{y:.2f}<br><i>Exponential Moving Average over 20 days</i><extra></extra>"
                ))
            elif indicator == "20-Day Bollinger Bands":
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Upper'].tolist(),
                    mode='lines',
                    name='BB Upper',
                    hovertemplate="<b>Date</b>: %{x}<br><b>BB Upper</b>: %{y:.2f}<br><i>Upper Bollinger Band (2 standard deviations)</i><extra></extra>"
                ))
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['BB_Lower'].tolist(),
                    mode='lines',
                    name='BB Lower',
                    hovertemplate="<b>Date</b>: %{x}<br><b>BB Lower</b>: %{y:.2f}<br><i>Lower Bollinger Band (2 standard deviations)</i><extra></extra>"
                ))
            elif indicator == "VWAP" and 'VWAP' in data_with_indicators.columns:
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['VWAP'].tolist(),
                    mode='lines',
                    name='VWAP',
                    hovertemplate="<b>Date</b>: %{x}<br><b>VWAP</b>: %{y:.2f}<br><i>Volume Weighted Average Price</i><extra></extra>"
                ))
            elif indicator == "RSI" and 'RSI' in data_with_indicators.columns:
                # Add RSI as a subplot
                fig.add_trace(go.Scatter(
                    x=data.index.tolist(),
                    y=data_with_indicators['RSI'].tolist(),
                    mode='lines',
                    name='RSI (14)',
                    yaxis="y2",
                    hovertemplate="<b>Date</b>: %{x}<br><b>RSI</b>: %{y:.2f}<extra></extra>"
                ))
                
                # Add RSI reference lines (30 and 70)
                fig.add_shape(type="line", x0=data.index[0], y0=30, x1=data.index[-1], y1=30,
                              line=dict(color="red", width=1, dash="dash"), yref="y2")
                fig.add_shape(type="line", x0=data.index[0], y0=70, x1=data.index[-1], y1=70,
                              line=dict(color="red", width=1, dash="dash"), yref="y2")
                
                # Update layout for RSI subplot
                fig.update_layout(
                    yaxis2=dict(
                        title="RSI",
                        range=[0, 100],
                        side="right",
                        overlaying="y"
                    )
                )
    
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        margin=dict(l=50, r=50, t=50, b=50),
        height=450,  # Reduced to 75% of original 600px height
        template='plotly_white',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        # Add border and semi-transparent background
        plot_bgcolor="rgba(0, 0, 0, 0.1)",  # 10% opacity black background
        paper_bgcolor="white",
        shapes=[
            # Add border around the chart
            dict(
                type="rect",
                xref="paper", yref="paper",
                x0=0, y0=0, x1=1, y1=1,
                line=dict(color="rgba(0, 0, 0, 0.3)", width=2)
            )
        ]
    )
    
    return fig

def create_summary_chart(summary_data):
    """Create a summary chart showing distribution of recommendations."""
    recommendation_counts = summary_data['Recommendation'].value_counts()
    fig = go.Figure(data=[go.Bar(
        x=recommendation_counts.index,
        y=recommendation_counts.values
    )])
    
    fig.update_layout(
        title="Distribution of Recommendations",
        xaxis_title="Recommendation",
        yaxis_title="Count"
    )
    
    return fig