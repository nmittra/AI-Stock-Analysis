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
end_date_default = datetime.now().date()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Start Date", value=start_date_default)
end_date = st.sidebar.date_input("End Date", value=end_date_default)

# Display current date and selected date range
current_date = datetime.now().date()
st.write(f"Current date: {current_date}")
st.write(f"Selected date range: {start_date} to {end_date}")

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
            data = data.rename(columns={
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. volume': 'Volume'
            })
            st.write(f"Raw API response for {ticker}:")
            st.write(data)
            st.write(meta_data)
            st.write(f"Raw data for {ticker}:", data)
            st.write(f"Data range: {data.index.min()} to {data.index.max()}")

            # Filter data based on selected date range
            data = data.loc[start_date:end_date]

            if not data.empty:
                stock_data[ticker] = data
                st.success(f"Data fetched successfully for {ticker}")
            else:
                st.warning(f"No data found for {ticker} in the selected date range.")
        except ValueError as ve:
            st.error(f"ValueError for {ticker}: {ve}")
        except KeyError as ke:
            st.error(f"KeyError for {ticker}: {ke}")
        except Exception as e:
            st.error(f"Unexpected error for {ticker}: {e}")

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

    # Display overall summary in the first tab
    # Display overall summary in the first tab
    with tabs[0]:
        # Create a DataFrame from overall_results
        summary_df = pd.DataFrame(overall_results)

        # Calculate and display the distribution of recommendations
        recommendation_counts = summary_df['Recommendation'].value_counts()
        fig_summary = go.Figure(data=[go.Bar(x=recommendation_counts.index, y=recommendation_counts.values)])
        fig_summary.update_layout(title="Distribution of Recommendations", xaxis_title="Recommendation",
                                  yaxis_title="Count")
        st.plotly_chart(fig_summary, use_container_width=True)

        # Display any additional overall insights or summaries here
        st.write("This summary provides an overview of the recommendations for all analyzed stocks. "
                 "Please refer to individual stock tabs for detailed analysis and justifications.")

    # Add some instructions for the user
    st.sidebar.markdown("---")
    st.sidebar.write("Instructions:")
    st.sidebar.write("1. Enter stock tickers separated by commas")
    st.sidebar.write("2. Select a date range")
    st.sidebar.write("3. Choose technical indicators")
    st.sidebar.write("4. Select an AI model for analysis")
    st.sidebar.write("5. Click 'Fetch Data' to update the analysis")

