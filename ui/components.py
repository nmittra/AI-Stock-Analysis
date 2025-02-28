import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import tempfile
import os

from api.clients import initialize_api_clients, fetch_stock_data, analyze_with_ai
from analysis.analysis import create_stock_chart, create_summary_chart

def setup_page() -> None:
    """Initialize the Streamlit page configuration.
    
    Sets up the page layout, title, and sidebar configuration for the dashboard.
    This function should be called at the start of the application.
    """
    st.set_page_config(layout="wide")
    st.title("AI-Powered Technical Stock Analysis Dashboard")
    st.sidebar.header("Configuration")

def get_user_inputs() -> tuple[str, str, list[str], datetime, datetime, list[str]]:
    """Get user inputs from the sidebar.
    
    Returns:
        tuple: A tuple containing:
            - str: Selected API choice (DeepSeek or Gemini)
            - str: Selected data provider (Alpha Vantage or Tiingo)
            - list[str]: List of stock tickers to analyze
            - datetime: Start date for analysis
            - datetime: End date for analysis
            - list[str]: Selected technical indicators
    """
    # API selection
    api_choice = st.sidebar.radio("Choose AI API", ["DeepSeek", "Gemini", "Falcon-7B", "LLama-2-13B"], index=3)
    
    # Data provider selection
    data_provider = st.sidebar.radio("Choose Data Provider", ["Alpha Vantage", "Tiingo", "FinnHub"], index=2)
    
    # Input for multiple stock tickers
    tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
    
    # Date range selection
    end_date_default = datetime.now().date()
    start_date_default = end_date_default - timedelta(days=365)
    start_date = st.sidebar.date_input("Start Date", value=start_date_default)
    end_date = st.sidebar.date_input("End Date", value=end_date_default)
    
    # Technical indicators selection
    indicators = st.sidebar.multiselect(
        "Select Indicators:",
        ["20-Day SMA", "20-Day EMA", "EMAs", "20-Day Bollinger Bands", "RSI", "MACD", "Stochastic", "Volume Analysis", "VWAP"],
        default=["20-Day SMA", "RSI", "MACD"]
    )
    
    return api_choice, data_provider, tickers, start_date, end_date, indicators

def display_date_info(current_date: datetime, start_date: datetime, end_date: datetime) -> None:
    """Display current date and selected date range in the Streamlit interface.
    
    Args:
        current_date: The current system date
        start_date: Selected start date for analysis
        end_date: Selected end date for analysis
    """
    st.write(f"Current date: {current_date}")
    st.write(f"Selected date range: {start_date} to {end_date}")

def process_stock_data(ticker: str, data: pd.DataFrame, api_choice: str, indicators: list[str]) -> tuple[go.Figure | None, dict | None]:
    """Process and display stock data for a single ticker.
    
    Args:
        ticker: Stock symbol to process
        data: DataFrame containing the stock data
        api_choice: Selected AI API (DeepSeek or Gemini)
        indicators: List of technical indicators to display
    
    Returns:
        tuple: A tuple containing:
            - Figure: Plotly figure object if successful, None if failed
            - dict: AI analysis results if successful, None if failed
    """
    try:
        with st.spinner(f'Creating chart for {ticker}...'):
            # Create chart and get AI analysis
            fig = create_stock_chart(data, ticker, indicators)
        
        # For Gemini API, we need to save the chart as an image
        if api_choice == "Gemini":
            with st.spinner(f'Preparing image for AI analysis of {ticker}...'):
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.write_image(tmpfile.name)
                    tmpfile_path = tmpfile.name
                with open(tmpfile_path, "rb") as f:
                    image_bytes = f.read()
                os.remove(tmpfile_path)
                image_data = {"data": image_bytes, "mime_type": "image/png"}
        else:
            image_data = None
        
        # Get AI analysis
        with st.spinner(f'Generating AI analysis for {ticker}...'):
            analysis_result = analyze_with_ai(
                client=st.session_state.get("ai_client"),
                model_type=api_choice,
                analysis_prompt=get_analysis_prompt(ticker),
                image_data=image_data
            )
        
        return fig, analysis_result
    except Exception as e:
        error_message = str(e)
        if "rate limit" in error_message.lower():
            st.error(f"🚫 API rate limit exceeded for {ticker}. Please wait a moment and try again.")
        elif "network" in error_message.lower():
            st.error(f"📡 Network error while processing {ticker}. Please check your internet connection.")
        elif "timeout" in error_message.lower():
            st.error(f"⏱️ Request timed out for {ticker}. The server might be busy, please try again.")
        else:
            st.error(f"❌ Error processing {ticker}: {error_message}")
        return None, None

def get_analysis_prompt(ticker):
    """Generate the analysis prompt for AI models."""
    return (
        f"Act as a financial analyst specializing in technical analysis of stocks. "
        f"Analyze the stock chart for {ticker} based on its candlestick chart and technical indicators. "
        f"Provide a detailed justification and recommendation (Strong Buy, Buy, Weak Buy, Hold, Weak Sell, Sell, Strong Sell). "
        f"Return your output as a JSON object with 'action' and 'justification' keys."
    )

def display_instructions():
    """Display usage instructions in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.write("Instructions:")
    st.sidebar.write("1. Enter stock tickers separated by commas")
    st.sidebar.write("2. Select a date range")
    st.sidebar.write("3. Choose technical indicators")
    st.sidebar.write("4. Select an AI model for analysis")
    st.sidebar.write("5. Click 'Fetch Data' to update the analysis")