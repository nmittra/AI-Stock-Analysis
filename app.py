print("Hello Papa!")

# Libraries
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from openai import OpenAI
import google.generativeai as genai
import tempfile
import os
from alpha_vantage.timeseries import TimeSeries

# Configure API keys - IMPORTANT: Use Streamlit secrets for security
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
ALPHA_VANTAGE_API_KEY = st.secrets["ALPHA_VANTAGE_API_KEY"]

# Initialize API clients
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
genai.configure(api_key=GOOGLE_API_KEY)
gen_model = genai.GenerativeModel('gemini-2.0-flash')  # or other model

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("AI-Powered Technical Stock Analysis Dashboard")
st.sidebar.header("Configuration")

# API selection
api_choice = st.sidebar.radio("Choose API", ["DeepSeek", "Gemini"])

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

# Button to fetch data for all tickers
if st.sidebar.button("Fetch Data"):
    stock_data = {}
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    for ticker in tickers:
        try:
            st.write(f"Attempting to fetch data for {ticker}...")
            data, meta_data = ts.get_daily(symbol=ticker, outputsize='full')
            st.write(f"Meta data: {meta_data}")  # Add this line
            data = data.loc[start_date:end_date]
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            st.write(f"Raw data for {ticker}:", data)
            if not data.empty:
                stock_data[ticker] = data
                st.success(f"Data fetched successfully for {ticker}")
            else:
                st.warning(f"No data found for {ticker}.")
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            st.write(f"Error type: {type(e).__name__}")
            st.write(f"Error details: {e}")

    if stock_data:
        st.session_state["stock_data"] = stock_data
        st.success("Stock data loaded successfully for: " + ", ".join(stock_data.keys()))
    else:
        st.error("No data was fetched for any ticker. Please check your inputs and try again.")

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    # Define a function to build chart, call the selected API and return structured result
    def analyze_ticker(ticker, data, api_choice):
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

        if api_choice == "DeepSeek":
            # DeepSeek API call
            analysis_prompt = (
                f"Act as a financial analyst specializing in technical analysis of stocks, ETFs, and cryptocurrencies. "
                f"Your expertise includes asset trends, momentum, volatility, and volume assessments. "
                f"You employ various strategies such as screening high-quality stocks, evaluating trend, idiosyncratic, and risk-adjusted momentum, and identifying top sectors. "
                f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
                f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
                f"Then, based solely on the chart, provide a recommendation from the following options: "
                f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
                f"Return your output as a JSON object with two keys: 'action' and 'justification'."
            )
            response = deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {"role": "system", "content": analysis_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                stream=False
            )
            result_text = response.choices[0].message.content
        else:  # Gemini API
            # Save chart as temporary PNG file and read image bytes
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                fig.write_image(tmpfile.name)
                tmpfile_path = tmpfile.name
            with open(tmpfile_path, "rb") as f:
                image_bytes = f.read()
            os.remove(tmpfile_path)

            image_part = {
                "data": image_bytes,
                "mime_type": "image/png"
            }

            analysis_prompt = (
                f"You are a Stock Trader specializing in Technical Analysis at a top financial institution. "
                f"Analyze the stock chart for {ticker} based on its candlestick chart and the displayed technical indicators. "
                f"Provide a detailed justification of your analysis, explaining what patterns, signals, and trends you observe. "
                f"Then, based solely on the chart, provide a recommendation from the following options: "
                f"'Strong Buy', 'Buy', 'Weak Buy', 'Hold', 'Weak Sell', 'Sell', or 'Strong Sell'. "
                f"Return your output as a JSON object with two keys: 'action' and 'justification'."
            )

            contents = [
                {"role": "user", "parts": [analysis_prompt]},
                {"role": "user", "parts": [image_part]}
            ]

            response = gen_model.generate_content(contents=contents)
            result_text = response.text

        try:
            # Attempt to parse JSON from the response text
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1  # +1 to include the closing brace
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the response")

        except json.JSONDecodeError as e:
            result = {"action": "Error", "justification": f"JSON Parsing error: {e}. Raw response text: {result_text}"}
        except ValueError as ve:
            result = {"action": "Error", "justification": f"Value Error: {ve}. Raw response text: {result_text}"}
        except Exception as e:
            result = {"action": "Error", "justification": f"General Error: {e}. Raw response text: {result_text}"}

        return fig, result

    # Create tabs: first tab for overall summary, subsequent tabs per ticker
    tab_names = ["Overall Summary"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # List to store overall results
    overall_results = []

    # Process each ticker and populate results
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        try:
            # Analyze ticker: get chart figure and structured output result
            fig, result = analyze_ticker(ticker, data, api_choice)
            overall_results.append({"Stock": ticker, "Recommendation": result.get("action", "N/A")})
            # In each ticker-specific tab, display the chart and detailed justification
            with tabs[i + 1]:
                st.subheader(f"Analysis for {ticker}")
                st.plotly_chart(fig, use_container_width=True)
                st.write("**Detailed Justification:**")
                st.write(result.get("justification", "No justification provided."))
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")

    # In the Overall Summary tab, display a table of all results
    with tabs[0]:
        st.subheader("Overall Structured Recommendations")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Please fetch stock data using the sidebar.")

# Add this at the end of your script to display API keys (for debugging only, remove in production)
st.sidebar.write("DeepSeek API Key:", DEEPSEEK_API_KEY[:5] + "..." if DEEPSEEK_API_KEY else "Not set")
st.sidebar.write("Google API Key:", GOOGLE_API_KEY[:5] + "..." if GOOGLE_API_KEY else "Not set")
st.sidebar.write("Alpha Vantage API Key:", ALPHA_VANTAGE_API_KEY[:5] + "..." if ALPHA_VANTAGE_API_KEY else "Not set")