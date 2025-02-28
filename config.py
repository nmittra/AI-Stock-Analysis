"""Configuration settings for the AI Stock Analysis application."""

# API Configuration
API_ENDPOINTS = {
    "deepseek": "https://api.deepseek.com",
    "gemini": "gemini-2.0-flash"
}

# Technical Indicators Configuration
INDICATORS = {
    "SMA": {
        "name": "20-Day SMA",
        "window": 20
    },
    "EMA": {
        "name": "20-Day EMA",
        "window": 20
    },
    "BOLLINGER_BANDS": {
        "name": "20-Day Bollinger Bands",
        "window": 20,
        "num_std": 2
    },
    "VWAP": {
        "name": "VWAP"
    }
}

# UI Configuration
UI_CONFIG = {
    "page_title": "AI-Powered Technical Stock Analysis Dashboard",
    "layout": "wide",
    "default_tickers": "AAPL,MSFT,GOOG",
    "date_range": {
        "default_days": 365
    }
}

# Chart Configuration
CHART_CONFIG = {
    "candlestick": {
        "name": "Candlestick"
    },
    "layout": {
        "rangeslider_visible": False
    }
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "recommendations": [
        "Strong Buy",
        "Buy",
        "Weak Buy",
        "Hold",
        "Weak Sell",
        "Sell",
        "Strong Sell"
    ],
    "prompt_template": (
        "Act as a financial analyst specializing in technical analysis of stocks. "
        "Analyze the stock chart for {ticker} based on its candlestick chart and technical indicators. "
        "Provide a detailed justification and recommendation ({recommendations}). "
        "Return your output as a JSON object with 'action' and 'justification' keys."
    )
}