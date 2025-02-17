import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
import tempfile
import os
import json
import base64
from datetime import datetime, timedelta

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")


# Configure the API key with better user experience
def initialize_api_key():
    # Try to get key from multiple sources
    api_key = None

    # Method 1: Check Streamlit secrets
    try:
        api_key = st.secrets.get("DEEPSEEK_API_KEY")
    except (KeyError, AttributeError):
        pass

    # Method 2: Environment variable
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    # Method 3: Session state (for persistence)
    if not api_key and "DEEPSEEK_API_KEY" in st.session_state:
        api_key = st.session_state.DEEPSEEK_API_KEY

    return api_key


# Get API key
api_key = initialize_api_key()

# API key input widget if needed
if not api_key:
    st.warning("⚠️ Deepseek API key not found in secrets or environment variables.")
    api_key_input = st.text_input(
        "Enter your Deepseek API key:",
        type="password",
        help="Your API key will be stored in this session only and not saved permanently."
    )

    if api_key_input:
        # Save entered key and use it
        st.session_state.DEEPSEEK_API_KEY = api_key_input
        api_key = api_key_input
        st.success("API key saved for this session. You can now proceed with analysis.")
        st.experimental_rerun()  # Force a rerun to refresh with the new API key

# Verify we have an API key before proceeding
api_key_available = bool(api_key)
if not api_key_available:
    st.error("Please provide a valid Deepseek API key to continue.")
    st.stop()  # Stop execution if no API key

# Sidebar UI
st.sidebar.header("Configuration")

# Input for multiple stock tickers (comma-separated)
tickers_input = st.sidebar.text_input("Enter Stock Tickers (comma-separated):", "AAPL,MSFT,GOOG")
# Parse tickers by stripping extra whitespace and splitting on commas
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Set the date range: start date = one year before today, end date = today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Technical indicators selection (applied to every ticker)
st.sidebar.subheader("Technical Indicators")
indicators = st.sidebar.multiselect(
    "Select Indicators:",
    ["20-Day SMA", "20-Day EMA", "20-Day Bollinger Bands", "VWAP"],
    default=["20-Day SMA"]
)


# Function to call Deepseek R1 API
def call_deepseek_r1(prompt, image_base64):
    api_url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Prepare the messages with image
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                }
            ]
        }
    ]

    payload = {
        "model": "deepseek-vision-r1",
        "messages": messages,
        "max_tokens": 1000,
        "response_format": {"type": "json_object"}
    }

    try:
        with st.spinner("Analyzing chart with Deepseek R1..."):
            response = requests.post(api_url, json=payload, headers=headers)

            # Detailed error handling
            if response.status_code != 200:
                error_message = f"API Error (Status: {response.status_code})"
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        error_message += f": {error_json['error'].get('message', '')}"
                except:
                    error_message += f": {response.text[:200]}"

                if response.status_code == 401:
                    # Reset API key on authentication failure
                    st.session_state.pop("DEEPSEEK_API_KEY", None)
                    error_message = "Invalid API key. Please re-enter your Deepseek API key."
                    st.error(error_message)
                    st.experimental_rerun()

                return {"action": "Error", "justification": error_message}

            result = response.json()
            response_content = result["choices"][0]["message"]["content"]
            return json.loads(response_content)
    except requests.exceptions.RequestException as e:
        return {"action": "Error", "justification": f"API request error: {str(e)}"}
    except json.JSONDecodeError as e:
        return {"action": "Error",
                "justification": f"JSON parsing error: {str(e)}. Raw response: {response.text[:500]}"}
    except Exception as e:
        return {"action": "Error", "justification": f"General error: {str(e)}"}


# Button to fetch data for all tickers
if st.sidebar.button("Fetch Data") and tickers:
    stock_data = {}
    with st.spinner("Fetching stock data..."):
        for ticker in tickers:
            # Download data for each ticker using yfinance
            data = yf.download(ticker, start=start_date, end=end_date)
            if not data.empty:
                stock_data[ticker] = data
            else:
                st.warning(f"No data found for {ticker}.")
    if stock_data:
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.error("No valid stock data found for any of the provided tickers.")

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    # Define a function to build chart, call the Deepseek API and return structured result
    def analyze_ticker(ticker, data):
        # Build candlestick chart for the given ticker's data
        fig = go.Figure(data=[
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Candlestick"
            )
        ])

        # Add selected technical indicators
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=20).mean()
                fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))

        for ind in indicators:
            add_indicator(ind)
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Save chart as temporary PNG file and convert to base64
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name

        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        os.remove(tmpfile_path)

        # Updated prompt asking for a detailed justification of technical analysis and a recommendation.
        analysis_prompt = (
            f"Act as a financial analyst specializing in technical analysis of stocks, ETFs, and cryptocurrencies. Your task is to analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
            f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
            f"Then, based solely on the chart, provide a recommendation from the following options: "
            f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
            f"Return your output as a JSON object with two keys: 'action' and 'justification'."
        )

        # Call the Deepseek API with text and image input
        result = call_deepseek_r1(analysis_prompt, image_base64)

        return fig, result


    # Create tabs: first tab for overall summary, subsequent tabs per ticker
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # List to store overall results
    overall_results = []

    # Process each ticker and populate results
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        # Analyze ticker: get chart figure and structured output result
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
        # In each ticker-specific tab, display the chart and detailed justification
        with tabs[i + 1]:
            st.subheader(f"Analysis for {ticker}")
            st.plotly_chart(fig)
            st.write("**Detailed Justification:**")
            st.write(result.get("justification", "No justification provided."))

    # In the Overall Summary tab, display a table of all results
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")