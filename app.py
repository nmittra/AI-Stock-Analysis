from datetime import datetime, timedelta
import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Import API clients
from api.clients import initialize_api_clients, fetch_stock_data
from api.tiingo_client import TiingoDataClient
from api.finnhub_client import FinnHubDataClient
from api.llm_client import OpenSourceLLMClient
from analysis.analysis import create_stock_chart, create_summary_chart

# Import LLM libraries
from openai import OpenAI
import google.generativeai as genai
import re  # Add this import at the top with other imports

# Set page configuration
st.set_page_config(
    page_title="Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'api_clients' not in st.session_state:
    st.session_state.api_clients = {}
if 'selected_stocks' not in st.session_state:
    st.session_state.selected_stocks = []
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = {}

def load_api_keys():
    """Load API keys from environment variables or Streamlit secrets"""
    api_keys = {
        'tiingo': os.environ.get('TIINGO_API_KEY') or st.secrets.get('TIINGO_API_KEY', ''),
        'finnhub': os.environ.get('FINNHUB_API_KEY') or st.secrets.get('FINNHUB_API_KEY', ''),
        'openai': os.environ.get('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY', ''),
        'google': os.environ.get('GOOGLE_API_KEY') or st.secrets.get('GOOGLE_API_KEY', ''),
        'openrouter': os.environ.get('OPENROUTER_API_KEY') or st.secrets.get('OPENROUTER_API_KEY', '')
    }
    return api_keys

def initialize_ai_clients(api_keys: Dict[str, str]) -> Dict[str, Any]:
    """Initialize AI clients with provided keys."""
    clients = {}
    
    # Initialize OpenRouter client if available
    if api_keys.get('openrouter'):
        try:
            clients['openrouter'] = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_keys['openrouter'],
                default_headers={
                    "HTTP-Referer": "https://github.com/traetechnologies",
                    "X-Title": "Stock Analysis Dashboard"
                }
            )
        except Exception as e:
            pass
    
    return clients

def initialize_app():
    """Initialize the application"""
    st.title("📊 Advanced Stock Analysis Dashboard")
    
    # Load API keys and initialize clients
    api_keys = load_api_keys()
    
    with st.sidebar:
        st.header("Configuration")
        
        # Data source configuration
        st.subheader("Data Sources")
        
        # OpenRouter configuration
        use_openrouter = st.checkbox("Use OpenRouter AI", value=True)
        
        if api_keys.get('openrouter'):
            st.session_state.openrouter_model = "cognitivecomputations/dolphin3.0-r1-mistral-24b:free"
        
        st.markdown("---")
        
        # Data source selection
        use_tiingo = st.checkbox("Use Tiingo API", value=True)
        use_finnhub = st.checkbox("Use Finnhub API", value=True)
        
        # Initialize API clients
        if st.button("Connect APIs"):
            data_clients = {}
            llm_clients = {}
            
            # Initialize OpenRouter client
            if use_openrouter and api_keys.get('openrouter'):
                try:
                    llm_clients['openrouter'] = OpenAI(
                        base_url="https://openrouter.ai/api/v1",
                        api_key=api_keys['openrouter'],
                        default_headers={
                            "HTTP-Referer": "https://github.com/traetechnologies",
                            "X-Title": "Stock Analysis Dashboard"
                        }
                    )
                    st.success("✅ OpenRouter AI connected")
                except Exception as e:
                    st.error(f"❌ OpenRouter error: {str(e)}")
            
            # Initialize data clients
            if use_tiingo and api_keys['tiingo']:
                try:
                    data_clients['tiingo'] = TiingoDataClient(api_keys['tiingo'])
                    st.success("✅ Tiingo API connected")
                except Exception as e:
                    st.error(f"❌ Tiingo API error: {str(e)}")
                    
            if use_finnhub and api_keys['finnhub']:
                try:
                    data_clients['finnhub'] = FinnHubDataClient(api_keys['finnhub'])
                    st.success("✅ Finnhub API connected")
                except Exception as e:
                    st.error(f"❌ Finnhub API error: {str(e)}")
            
            # Store all clients in session state
            st.session_state.api_clients = {
                'data': data_clients,
                'llm': llm_clients
            }
            
            # Check if we have at least one data source
            if not data_clients:
                st.warning("⚠️ No data sources connected. Please provide valid API keys.")
    
    return api_keys

def stock_selection():
    """Handle stock selection UI"""
    st.header("Stock Selection")
    
    # Stock selection input
    col1, col2 = st.columns([3, 1])
    with col1:
        stock_input = st.text_input("Enter stock symbols (comma separated):", "AAPL, MSFT, GOOGL")
    
    with col2:
        if st.button("Add Stocks"):
            new_stocks = [s.strip().upper() for s in stock_input.split(",") if s.strip()]
            # Add only unique stocks that aren't already in the list
            for stock in new_stocks:
                if stock not in st.session_state.selected_stocks:
                    st.session_state.selected_stocks.append(stock)
    
    # Display and manage selected stocks
    if st.session_state.selected_stocks:
        st.subheader("Selected Stocks")
        cols = st.columns(4)
        stocks_to_remove = []
        
        for i, stock in enumerate(st.session_state.selected_stocks):
            col_idx = i % 4
            with cols[col_idx]:
                if st.button(f"❌ {stock}", key=f"remove_{stock}"):
                    stocks_to_remove.append(stock)
        
        # Remove stocks marked for deletion
        for stock in stocks_to_remove:
            st.session_state.selected_stocks.remove(stock)
            if stock in st.session_state.stock_data:
                del st.session_state.stock_data[stock]
            if stock in st.session_state.analysis_results:
                del st.session_state.analysis_results[stock]
        
        if stocks_to_remove:
            st.rerun()

def date_selection():
    """Handle date range selection UI"""
    st.header("Time Period")
    
    col1, col2 = st.columns(2)
    
    # Default to last 3 months
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=90)
    
    with col1:
        start_date = st.date_input("Start Date", value=default_start_date)
    with col2:
        end_date = st.date_input("End Date", value=default_end_date)
    
    # Validate date range
    if start_date >= end_date:
        st.error("Error: End date must be after start date")
        return None, None
    
    return start_date, end_date

def fetch_data(start_date, end_date):
    """Fetch stock data for selected stocks"""
    if not st.session_state.api_clients.get('data'):
        st.warning("No data sources connected. Please connect APIs first.")
        return
    
    if not st.session_state.selected_stocks:
        st.info("Please select at least one stock to analyze.")
        return
    
    st.header("Fetching Data")
    progress_bar = st.progress(0)
    
    # Initialize company_profiles in session state if it doesn't exist
    if 'company_profiles' not in st.session_state:
        st.session_state.company_profiles = {}
    
    for i, symbol in enumerate(st.session_state.selected_stocks):
        progress = (i / len(st.session_state.selected_stocks))
        progress_bar.progress(progress)
        st.write(f"Fetching data for {symbol}...")
        
        try:
            # Use the fetch_stock_data function from api.clients
            data, metadata, error = fetch_stock_data(
                symbol, 
                start_date, 
                end_date, 
                st.session_state.api_clients['data']
            )
            
            if error:
                st.error(f"❌ Error fetching data for {symbol}: {error}")
            elif data is not None and not data.empty:
                st.session_state.stock_data[symbol] = data
                st.write(f"✅ Successfully fetched data for {symbol}")
                
                # Fetch and store company profile if Finnhub client is available
                if 'finnhub' in st.session_state.api_clients['data']:
                    try:
                        profile = st.session_state.api_clients['data']['finnhub'].get_company_profile(symbol)
                        if profile:
                            st.session_state.company_profiles[symbol] = profile
                            st.write(f"✅ Successfully fetched company profile for {symbol}")
                    except Exception as e:
                        st.warning(f"⚠️ Could not fetch company profile for {symbol}: {str(e)}")
            else:
                st.error(f"❌ No data returned for {symbol}")
        except Exception as e:
            st.error(f"❌ Error fetching data for {symbol}: {str(e)}")
    
    progress_bar.progress(1.0)
    st.success("Data fetching completed!")

def analyze_data():
    """Run analysis on the fetched stock data"""
    if not st.session_state.stock_data:
        st.info("No data available for analysis. Please fetch data first.")
        return
    
    st.header("Stock Analysis")
    
    # Technical analysis settings
    with st.expander("Technical Analysis Settings"):
        use_sma = st.checkbox("Simple Moving Average (SMA)", value=True)
        sma_period = st.slider("SMA Period", min_value=5, max_value=200, value=20, step=5)
        
        use_ema = st.checkbox("Exponential Moving Average (EMA)", value=True)
        ema_period = st.slider("EMA Period", min_value=5, max_value=200, value=20, step=5)
        
        use_bollinger = st.checkbox("Bollinger Bands", value=True)
        bollinger_period = st.slider("Bollinger Period", min_value=5, max_value=50, value=20, step=1)
        bollinger_std = st.slider("Standard Deviation", min_value=1, max_value=3, value=2, step=1)
        
        use_rsi = st.checkbox("Relative Strength Index (RSI)", value=True)
        rsi_period = st.slider("RSI Period", min_value=7, max_value=30, value=14, step=1)
    
    # Create analysis parameters dictionary
    analysis_params = {
        'sma': {'use': use_sma, 'period': sma_period},
        'ema': {'use': use_ema, 'period': ema_period},
        'bollinger': {'use': use_bollinger, 'period': bollinger_period, 'std': bollinger_std},
        'rsi': {'use': use_rsi, 'period': rsi_period}
    }
    
    # Run analysis for each stock
    for symbol, data in st.session_state.stock_data.items():
        st.subheader(f"Analysis for {symbol}")
        
        # Display company profile if available
        if hasattr(st.session_state, 'company_profiles') and symbol in st.session_state.company_profiles:
            profile = st.session_state.company_profiles[symbol]
            if profile:
                with st.expander("Company Information", expanded=True):
                    col1, col2 = st.columns([1, 3])
                    
                    with col1:
                        if 'logo' in profile and profile['logo']:
                            st.image(profile['logo'], width=100)
                    
                    with col2:
                        st.markdown(f"### {profile.get('name', symbol)}")
                        st.markdown(f"**Industry:** {profile.get('finnhubIndustry', 'N/A')}")
                        st.markdown(f"**Exchange:** {profile.get('exchange', 'N/A')}")
                        st.markdown(f"**Currency:** {profile.get('currency', 'N/A')}")
                        st.markdown(f"**IPO Date:** {profile.get('ipo', 'N/A')}")
                        st.markdown(f"**Market Cap:** ${profile.get('marketCapitalization', 0):,.2f}M")
                        st.markdown(f"**Shares Outstanding:** {profile.get('shareOutstanding', 0):,.2f}M")
                        
                        if 'weburl' in profile and profile['weburl']:
                            st.markdown(f"[Company Website]({profile['weburl']})")
                            
                    # Add company description if available
                    if 'description' in profile and profile['description']:
                        st.markdown("---")
                        st.markdown("**Business Description:**")
                        st.markdown(f"*{profile['description']}*")
        
        # Generate stock chart with indicators
        fig = create_stock_chart(data, symbol, analysis_params)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display news if available
        if hasattr(st.session_state, 'company_news') and symbol in st.session_state.company_news:
            news_items = st.session_state.company_news[symbol]
            if news_items:
                with st.expander("Recent News", expanded=True):
                    for item in news_items:
                        col1, col2 = st.columns([1, 4])
                        
                        with col1:
                            if 'image' in item and item['image']:
                                st.image(item['image'], width=100)
                            else:
                                st.markdown("📰")
                        
                        with col2:
                            st.markdown(f"### {item.get('headline', 'News Update')}")
                            st.markdown(f"*{item.get('summary', '')}*")
                            st.markdown(f"**Source:** {item.get('source', 'Unknown')} | **Date:** {item.get('datetime', '')}")
                            
                            if 'url' in item and item['url']:
                                st.markdown(f"[Read More]({item['url']})")
                        
                        st.markdown("---")
        
        # Store analysis results
        st.session_state.analysis_results[symbol] = {
            'data': data,
            'chart': fig,
            'params': analysis_params
        }

def generate_ai_insights():
    """Generate AI insights using connected LLM clients"""
    if not st.session_state.analysis_results:
        st.info("No analysis results available. Please run analysis first.")
        return
    
    if not st.session_state.api_clients.get('llm'):
        st.warning("No LLM clients connected. Please connect APIs with valid LLM keys.")
        return
    
    st.header("AI Market Insights")
    
    # Remove the Generate All Insights button from the top
    # We'll add it next to the individual Generate Insights button instead
    for symbol in st.session_state.selected_stocks:
        if symbol not in st.session_state.analysis_results:
            continue
                
            # Get analysis data and parameters
            stock_data = st.session_state.analysis_results[symbol]['data']
            analysis_params = st.session_state.analysis_results[symbol]['params']
            
            # Use the first available LLM provider
            available_providers = [p for p in st.session_state.api_clients['llm'].keys() 
                                 if p != 'openrouter_model']
            if not available_providers:
                st.error("No LLM providers available")
                continue
                
            llm_provider = available_providers[0]
            
            try:
                with st.spinner(f"Generating insights for {symbol}..."):
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                    # Prepare the prompt
                    prompt = f"""Analyze the stock {symbol} based on...
                    - SMA (Simple Moving Average): {analysis_params['sma']}
                    - EMA (Exponential Moving Average): {analysis_params['ema']}
                    - Bollinger Bands: {analysis_params['bollinger']}
                    - RSI (Relative Strength Index): {analysis_params['rsi']}
                    
                    Recent price data:
                    {stock_data.tail().to_string()}
                    
                    Please provide:
                    1. Technical Analysis Summary
                    2. Key Observations
                    3. Potential Trading Signals
                    4. Risk Factors
                    
                    Also, provide market sentiment and confidence at the end:
                    SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
                    CONFIDENCE: [0-100]%
                    """
                    
                    # Generate insights using the first available provider
                    client = st.session_state.api_clients['llm'][llm_provider]
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a professional stock market analyst."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    insights = response.choices[0].message.content
                    
                    # Process and store the insights
                    if 'insights' not in st.session_state.analysis_results[symbol]:
                        st.session_state.analysis_results[symbol]['insights'] = {}
                    
                    st.session_state.analysis_results[symbol]['insights'][llm_provider] = {
                        'text': insights,
                        'timestamp': current_time
                    }
                        
            except Exception as e:
                st.error(f"Error generating insights for {symbol}: {str(e)}")
    
    # Create a tab for each stock
    tabs = st.tabs(st.session_state.selected_stocks)
    
    for i, symbol in enumerate(st.session_state.selected_stocks):
        if symbol not in st.session_state.analysis_results:
            continue
            
        with tabs[i]:
            st.subheader(f"AI Insights for {symbol}")
            
            # Get analysis data 
            stock_data = st.session_state.analysis_results[symbol]['data']
            analysis_params = st.session_state.analysis_results[symbol]['params']
            
            # Select LLM provider - filter out non-provider keys
            available_providers = [p for p in st.session_state.api_clients['llm'].keys() 
                                 if p != 'openrouter_model']
            
            llm_provider = st.selectbox(
                "Select AI Provider", 
                options=available_providers,
                key=f"llm_provider_{symbol}"
            )
            
            # Display OpenRouter model selection directly in the insights section
            if llm_provider == 'openrouter':
                # Define model options
                model_options = [
                    "qwen/qwen-vl-plus:free",
                    "meta-llama/llama-3.1-8b-instruct:free",
                    "cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
                    "qwen/qwen-2-7b-instruct:free",
                    "openai/gpt-3.5-turbo",
                    "openai/gpt-4o-mini",
                    "google/gemini-2.0-flash-001"
                ]
                
                # Create a unique key for each stock's model selector
                model_key = f"or_model_select_{symbol}"
                
                # Initialize default model in session state if needed
                if model_key not in st.session_state:
                    st.session_state[model_key] = model_options[0]
                
                # Let user select model directly in the insights section
                selected_model = st.selectbox(
                    "Select OpenRouter Model",
                    options=model_options,
                    key=model_key
                )
                
                # Remove this line - Streamlit automatically updates session state
                # st.session_state[model_key] = selected_model
                
                st.info(f"Using OpenRouter model: {selected_model}")
            
            # Add Generate All Insights button next to individual Generate Insights button
            col1, col2 = st.columns(2)
            with col2:
                if st.button("Generate All Insights", key=f"generate_all_{symbol}"):
                    for stock in st.session_state.selected_stocks:
                        if stock not in st.session_state.analysis_results:
                            continue
                        
                        # Get analysis data and parameters for each stock
                        stock_data = st.session_state.analysis_results[stock]['data']
                        analysis_params = st.session_state.analysis_results[stock]['params']
                        
                        try:
                            with st.spinner("Generating insights..."):
                                # Add current_time definition at the start of the try block
                                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Prepare the prompt
                                prompt = f"""Analyze the stock {stock} based on...
                                - SMA (Simple Moving Average): {analysis_params['sma']}
                                - EMA (Exponential Moving Average): {analysis_params['ema']}
                                - Bollinger Bands: {analysis_params['bollinger']}
                                - RSI (Relative Strength Index): {analysis_params['rsi']}
                                
                                Recent price data:
                                {stock_data.tail().to_string()}
                                
                                Please provide:
                                1. Technical Analysis Summary
                                2. Key Observations
                                3. Potential Trading Signals
                                4. Risk Factors
                                
                                Also, provide market sentiment and confidence at the end:
                                SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
                                CONFIDENCE: [0-100]%
                                """
                                
                                # Generate insights based on provider
                                if llm_provider == 'openrouter':
                                    client = st.session_state.api_clients['llm'][llm_provider]
                                    # Get model from the stock-specific session state key
                                    model_key = f"or_model_select_{symbol}"
                                    model = st.session_state[model_key]
                                    
                                    response = client.chat.completions.create(
                                        model=model,
                                        messages=[
                                            {"role": "system", "content": "You are a professional stock market analyst."},
                                            {"role": "user", "content": prompt}
                                        ],
                                        temperature=0.7,
                                        max_tokens=1000
                                    )
                                    insights = response.choices[0].message.content
                                else:
                                    client = st.session_state.api_clients['llm'][llm_provider]
                                    response = client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[
                                            {"role": "system", "content": "You are a professional stock market analyst."},
                                            {"role": "user", "content": prompt}
                                        ]
                                    )
                                    insights = response.choices[0].message.content
                                
                                # Process and store the insights
                                if 'insights' not in st.session_state.analysis_results[stock]:
                                    st.session_state.analysis_results[stock]['insights'] = {}
                                
                                st.session_state.analysis_results[stock]['insights'][llm_provider] = {
                                    'text': insights,
                                    'timestamp': current_time
                                }
                        except Exception as e:
                            st.error(f"Error generating insights for {stock}: {str(e)}")
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Generate Insights", key=f"generate_{symbol}"):
                    try:
                        with st.spinner("Generating insights..."):
                            # Add current_time definition at the start of the try block
                            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # Prepare the prompt
                            prompt = f"""Analyze the stock {symbol} based on...
                            - SMA (Simple Moving Average): {analysis_params['sma']}
                            - EMA (Exponential Moving Average): {analysis_params['ema']}
                            - Bollinger Bands: {analysis_params['bollinger']}
                            - RSI (Relative Strength Index): {analysis_params['rsi']}
                            
                            Recent price data:
                            {stock_data.tail().to_string()}
                            
                            Please provide:
                            1. Technical Analysis Summary
                            2. Key Observations
                            3. Potential Trading Signals
                            4. Risk Factors
                            """
                            
                            # Add sentiment analysis request to the prompt
                            prompt += """
                            
                            Also, provide market sentiment and confidence at the end:
                            SENTIMENT: [BULLISH/BEARISH/NEUTRAL]
                            CONFIDENCE: [0-100]%
                            """
                            
                            # Generate insights based on provider
                            if llm_provider == 'openrouter':
                                client = st.session_state.api_clients['llm'][llm_provider]
                                # Get model from the stock-specific session state key
                                model_key = f"or_model_select_{symbol}"
                                model = st.session_state[model_key]
                                
                                if not model:
                                    st.error("OpenRouter model not selected")
                                    return
                                
                                # Add debug information
                                st.write(f"Selected provider: {llm_provider}")
                                st.write(f"Using model: {model}")
                                
                                response = client.chat.completions.create(
                                    model=model,
                                    messages=[
                                        {"role": "system", "content": "You are a professional stock market analyst."},
                                        {"role": "user", "content": prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000
                                )
                                insights = response.choices[0].message.content
                            else:
                                # Filter out openrouter_model from provider options
                                if llm_provider == 'openrouter_model':
                                    st.error("Invalid provider selected")
                                    return
                                    
                                client = st.session_state.api_clients['llm'][llm_provider]
                                response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are a professional stock market analyst."},
                                        {"role": "user", "content": prompt}
                                    ]
                                )
                                insights = response.choices[0].message.content
                            
                            # Extract sentiment and confidence first
                            sentiment_match = re.search(r"SENTIMENT: (BULLISH|BEARISH|NEUTRAL)", insights)
                            confidence_match = re.search(r"CONFIDENCE: (\d+)%", insights)
                            
                            # Set default values
                            sentiment = "UNKNOWN"
                            confidence = 50
                            
                            # Update values if matches found
                            if sentiment_match:
                                sentiment = sentiment_match.group(1)
                            if confidence_match:
                                try:
                                    confidence = int(confidence_match.group(1))
                                    confidence = min(max(confidence, 0), 100)  # Ensure value is between 0-100
                                except ValueError:
                                    confidence = 50
                            
                            # Format confidence display
                            confidence_display = f"<span style='color: #1f77b4; font-weight: bold;'>{confidence}%</span>"
                            
                            # Replace confidence in insights text with styled version
                            insights = re.sub(
                                r"CONFIDENCE: \d+%",
                                f"CONFIDENCE: {confidence_display}",
                                insights
                            )
                            
                            # Determine sentiment color
                            sentiment_color = {
                                "BULLISH": "#2E8B57",  # Soft sea green
                                "BEARISH": "#CD5C5C",  # Soft indian red
                                "NEUTRAL": "#DAA520",  # Soft goldenrod
                                "UNKNOWN": "#808080"   # Default soft gray
                            }.get(sentiment, "#808080")
                            
                            # Extract key points and metrics
                            key_points = re.findall(r"(?:^|\n)(?:•|\*)s*(.*?)(?=\n(?:•|\*)|\n\n|$)", insights, re.MULTILINE)
                            
                            # Extract sections from the insights
                            sections = {
                                "Technical Analysis Summary": "",
                                "Key Observations": "",
                                "Potential Trading Signals": "",
                                "Risk Factors": ""
                            }
                            
                            # Parse the insights into sections
                            current_section = None
                            lines = insights.split('\n')
                            section_content = []
                            
                            for line in lines:
                                # Check if this line is a section header
                                for section_name in sections.keys():
                                    if section_name.lower() in line.lower() or f"{section_name.split()[0].lower()}" in line.lower():
                                        # Save previous section content if any
                                        if current_section and section_content:
                                            sections[current_section] = '\n'.join(section_content)
                                        # Start new section
                                        current_section = section_name
                                        section_content = []
                                        break
                                
                                # Add line to current section if we're in one
                                if current_section and not any(s.lower() in line.lower() for s in sections.keys()):
                                    section_content.append(line)
                            
                            # Save the last section content
                            if current_section and section_content:
                                sections[current_section] = '\n'.join(section_content)
                            
                            # Highlight metrics in the insights
                            metrics_pattern = r"(\d+\.?\d*%|\$\d+\.?\d*|\d+\.?\d* points?|\d+\.?\d* MA|\d+\.?\d* RSI)"
                            highlighted_insights = re.sub(
                                metrics_pattern,
                                r'<span style="background-color: rgba(255, 255, 0, 0.2); padding: 0 2px; border-radius: 3px; font-weight: 500;">\1</span>',
                                insights
                            )
                            
                            # Apply the same highlighting to each section
                            for section in sections:
                                if sections[section]:
                                    sections[section] = re.sub(
                                        metrics_pattern,
                                        r'<span style="background-color: rgba(255, 255, 0, 0.2); padding: 0 2px; border-radius: 3px; font-weight: 500;">\1</span>',
                                        sections[section]
                                    )
                            
                            # Display insights with sections
                            st.markdown("### AI Generated Insights")
                            
                            # Create columns for sentiment, confidence, and analyst recommendation gauges
                            col1, col2, col3 = st.columns(3)
                            
                            # Create sentiment indicator
                            with col1:
                                fig_sentiment = go.Figure(go.Indicator(
                                    mode="gauge+number+delta",
                                    value=1 if sentiment == "BULLISH" else (-1 if sentiment == "BEARISH" else 0),
                                    title={'text': "Market Sentiment"},
                                    gauge={
                                        'axis': {'range': [-1, 1]},
                                        'bar': {'color': sentiment_color},
                                        'steps': [
                                            {'range': [-1, -0.3], 'color': "#CD5C5C"},
                                            {'range': [-0.3, 0.3], 'color': "#DAA520"},
                                            {'range': [0.3, 1], 'color': "#2E8B57"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "black", 'width': 3},
                                            'thickness': 0.75,
                                            'value': 1 if sentiment == "BULLISH" else (-1 if sentiment == "BEARISH" else 0)
                                        }
                                    }
                                ))
                                fig_sentiment.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                                st.plotly_chart(fig_sentiment, use_container_width=True)
                            
                            # Create confidence gauge
                            with col2:
                                fig_confidence = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=confidence,
                                    title={'text': "Confidence Level"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "#1f77b4"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "#ffcdd2"},
                                            {'range': [30, 70], 'color': "#fff9c4"},
                                            {'range': [70, 100], 'color': "#c8e6c9"}
                                        ],
                                        'threshold': {
                                            'line': {'color': "black", 'width': 3},
                                            'thickness': 0.75,
                                            'value': confidence
                                        }
                                    }
                                ))
                                fig_confidence.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                                st.plotly_chart(fig_confidence, use_container_width=True)
                                
                            # Create analyst recommendations gauge
                            with col3:
                                try:
                                    # Get analyst recommendations from Finnhub
                                    if 'data' in st.session_state.api_clients and 'finnhub' in st.session_state.api_clients['data']:
                                        finnhub_client = st.session_state.api_clients['data']['finnhub']
                                        rec_data = finnhub_client.get_recommendation_trends(symbol)
                                        
                                        if rec_data['success'] and rec_data['recommendations']:
                                            recs = rec_data['recommendations']
                                            # Calculate recommendation score (-1 to 1)
                                            total_recs = sum([recs['strongBuy'], recs['buy'], recs['hold'], recs['sell'], recs['strongSell']])
                                            if total_recs > 0:
                                                score = ((recs['strongBuy'] * 1.0 + recs['buy'] * 0.5 + 
                                                         recs['sell'] * -0.5 + recs['strongSell'] * -1.0) / total_recs)
                                                
                                                fig_rec = go.Figure(go.Indicator(
                                                    mode="gauge+number",
                                                    value=score,
                                                    title={'text': "Analyst Recommendations"},
                                                    gauge={
                                                        'axis': {'range': [-1, 1]},
                                                        'bar': {'color': "#2E8B57" if score > 0 else "#CD5C5C"},
                                                        'steps': [
                                                            {'range': [-1, -0.3], 'color': "#CD5C5C"},
                                                            {'range': [-0.3, 0.3], 'color': "#DAA520"},
                                                            {'range': [0.3, 1], 'color': "#2E8B57"}
                                                        ],
                                                        'threshold': {
                                                            'line': {'color': "black", 'width': 3},
                                                            'thickness': 0.75,
                                                            'value': score
                                                        }
                                                    }
                                                ))
                                                fig_rec.update_layout(height=200, margin=dict(l=10, r=10, t=50, b=10))
                                                st.plotly_chart(fig_rec, use_container_width=True)
                                                
                                                # Display recommendation breakdown
                                                with st.expander("Recommendation Details"):
                                                    st.write(f"Strong Buy: {recs['strongBuy']}")
                                                    st.write(f"Buy: {recs['buy']}")
                                                    st.write(f"Hold: {recs['hold']}")
                                                    st.write(f"Sell: {recs['sell']}")
                                                    st.write(f"Strong Sell: {recs['strongSell']}")
                                                    st.write(f"Period: {recs['period']}")
                                            else:
                                                st.info("No analyst recommendations available")
                                        else:
                                            st.info("No analyst recommendations available")
                                    else:
                                        st.info("Finnhub API not connected")
                                except Exception as e:
                                    st.error(f"Error fetching analyst recommendations: {str(e)}")
                            
                            # Display organized sections
                            st.markdown("#### Analysis Sections")
                            
                            # Display Technical Analysis Summary first
                            if sections["Technical Analysis Summary"]:
                                with st.expander("📊 Technical Analysis Summary", expanded=True):
                                    st.markdown(sections["Technical Analysis Summary"].strip())
                            
                            # Display Key Observations
                            if sections["Key Observations"]:
                                with st.expander("🔍 Key Observations", expanded=True):
                                    st.markdown(sections["Key Observations"].strip())
                            
                            # Display Potential Trading Signals
                            if sections["Potential Trading Signals"]:
                                with st.expander("💡 Trading Signals", expanded=True):
                                    st.markdown(sections["Potential Trading Signals"].strip())
                            
                            # Display Risk Factors
                            if sections["Risk Factors"]:
                                with st.expander("⚠️ Risk Factors", expanded=True):
                                    st.markdown(sections["Risk Factors"].strip())
                            
                            # Display key points if found
                            if key_points:
                                with st.expander("📌 Key Points Summary", expanded=True):
                                    for point in key_points:
                                        st.markdown(f"• {point.strip()}")
                            
                            # Store insights in session state
                            if 'insights' not in st.session_state.analysis_results[symbol]:
                                st.session_state.analysis_results[symbol]['insights'] = {}
                            
                            st.session_state.analysis_results[symbol]['insights'][llm_provider] = {
                                'text': insights,
                                'sentiment_color': sentiment_color,
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'timestamp': current_time
                            }
                            
                            # Initialize sentiment history tracking
                            if 'sentiment_history' not in st.session_state:
                                st.session_state.sentiment_history = {}
                            
                            if symbol not in st.session_state.sentiment_history:
                                st.session_state.sentiment_history[symbol] = []
                            
                            # Add current sentiment to history
                            st.session_state.sentiment_history[symbol].append({
                                'timestamp': current_time,
                                'sentiment': sentiment,
                                'confidence': confidence,
                                'provider': llm_provider
                            })
                            
                            # Display sentiment history
                            display_sentiment_history(symbol)
                    except Exception as e:
                        st.error(f"Error generating insights: {str(e)}")

# Define time frame mapping outside the loop to avoid recreating it
TIME_DELTA_MAP = {
    "Last 5 Minutes": timedelta(minutes=5),
    "Last 15 Minutes": timedelta(minutes=15),
    "Last 30 Minutes": timedelta(minutes=30),
    "Last Hour": timedelta(hours=1),
    "Last 4 Hours": timedelta(hours=4),
    "Last Day": timedelta(days=1),
}

# Sentiment mapping for consistent values
SENTIMENT_VALUE_MAP = {
    'BULLISH': 1,
    'NEUTRAL': 0,
    'BEARISH': -1,
    'UNKNOWN': 0
}

# Colors for different sentiment values
SENTIMENT_COLORS = {
    'BULLISH': 'green',
    'NEUTRAL': 'gray',
    'BEARISH': 'red',
    'UNKNOWN': 'lightgray'
}

def display_sentiment_history(symbol):
    """
    Display sentiment history chart for a given symbol.
    
    Args:
        symbol (str): The trading symbol to display sentiment for
    """
    if 'sentiment_history' not in st.session_state:
        return
        
    if symbol not in st.session_state.sentiment_history:
        return
        
    # Get the sentiment history for this symbol
    history = st.session_state.sentiment_history[symbol]
    
    # Check if we have enough data for a meaningful comparison
    if len(history) <= 1:
        # Display the single sentiment point without comparison
        if len(history) == 1:
            entry = history[0]
            sentiment = entry['sentiment']
            confidence = entry['confidence']
            provider = entry['provider']
            timestamp = entry['timestamp']
            
            # Determine sentiment color
            sentiment_color = SENTIMENT_COLORS.get(sentiment, 'lightgray')
            
            # Create a simple display for a single sentiment point
            st.markdown("### Current Sentiment")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sentiment", sentiment)
            with col2:
                st.metric("Confidence", f"{confidence}%")
            with col3:
                st.metric("Provider", provider)
                
            st.info(f"Only one sentiment data point available (from {timestamp}). Generate more insights to see sentiment trends.")
            return
        return
    
    with st.expander("Sentiment History", expanded=True):
        # Add time frame selection
        time_frame = st.selectbox(
            "Select Time Frame",
            list(TIME_DELTA_MAP.keys()),
            index=6,  # Default to 30 days
            key=f"timeframe_{symbol}"
        )
        
        selected_delta = TIME_DELTA_MAP[time_frame]
        filter_time = datetime.now() - selected_delta
        
        try:
            # Create DataFrame from sentiment history
            history_df = pd.DataFrame(st.session_state.sentiment_history[symbol])
            
            # Convert timestamp strings to datetime objects
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            
            # Add sentiment value column once
            history_df['sentiment_value'] = history_df['sentiment'].map(SENTIMENT_VALUE_MAP)
            
            # Filter based on selected time frame
            filtered_history = history_df[history_df['timestamp'] >= filter_time]
            
            # If we have data in the filtered timeframe, display it
            if not filtered_history.empty:
                # Calculate average sentiment for the period
                avg_sentiment = filtered_history['sentiment_value'].mean()
                st.metric(
                    "Average Sentiment", 
                    f"{avg_sentiment:.2f}", 
                    delta=None,
                    delta_color="normal"
                )
                
                # Create sentiment history chart
                fig = go.Figure()
                
                # Add a horizontal line at y=0
                fig.add_shape(
                    type="line",
                    x0=filtered_history['timestamp'].min(),
                    y0=0,
                    x1=filtered_history['timestamp'].max(),
                    y1=0,
                    line=dict(
                        color="lightgray",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Add traces for each provider
                for provider in filtered_history['provider'].unique():
                    provider_data = filtered_history[filtered_history['provider'] == provider]
                    
                    # Get colors based on sentiment
                    colors = [SENTIMENT_COLORS[sentiment] for sentiment in provider_data['sentiment']]
                    
                    fig.add_trace(go.Scatter(
                        x=provider_data['timestamp'],
                        y=provider_data['sentiment_value'],
                        name=provider,
                        mode='lines+markers',
                        line=dict(width=2),
                        marker=dict(
                            size=8,
                            color=colors
                        ),
                        hovertemplate=(
                            "Time: %{x}<br>"
                            "Sentiment: %{text}<br>"
                            "Confidence: %{customdata}%"
                        ),
                        text=provider_data['sentiment'],
                        customdata=provider_data['confidence']
                    ))
                
                fig.update_layout(
                    title=f"Sentiment History ({time_frame})",
                    yaxis=dict(
                        ticktext=["Bearish", "Neutral", "Bullish"],
                        tickvals=[-1, 0, 1],
                        range=[-1.5, 1.5]
                    ),
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info(f"No sentiment data available for {time_frame}.")
                
        except Exception as e:
            st.error(f"Error displaying sentiment history: {str(e)}")

# Usage in your main app
for symbol in st.session_state.selected_stocks:
    # Other symbol-specific display code...
    # Only display sentiment history if it exists
    if 'sentiment_history' in st.session_state and symbol in st.session_state.get('sentiment_history', {}):
        display_sentiment_history(symbol)

    # Add sentiment comparison after insights generation
    if (symbol in st.session_state.analysis_results and
        'insights' in st.session_state.analysis_results[symbol] and
        len(st.session_state.analysis_results[symbol]['insights']) > 1):
        st.markdown("### Sentiment Comparison")
        
        providers = list(st.session_state.analysis_results[symbol]['insights'].keys())
        sentiments = []
        confidences = []
                
        for provider in providers:
            insight_data = st.session_state.analysis_results[symbol]['insights'][provider]
            sentiment_match = re.search(r"SENTIMENT: (BULLISH|BEARISH|NEUTRAL)", insight_data['text'])
            sentiment = sentiment_match.group(1) if sentiment_match else "UNKNOWN"
            sentiments.append(sentiment)
            confidences.append(insight_data.get('confidence', 50))
                        
        # Create sentiment comparison dataframe
        sentiment_df = pd.DataFrame({
            'Provider': providers,
            'Sentiment': sentiments,
            'Confidence': confidences
        })
        
        # Map sentiments to numeric values for visualization
        sentiment_map = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1, 'UNKNOWN': 0}
        sentiment_df['Sentiment_Value'] = sentiment_df['Sentiment'].map(sentiment_map)
        
        # Create columns for chart and legend
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Create custom bar chart with colors
            fig = go.Figure()
            
            for idx, row in sentiment_df.iterrows():
                color = "#2E8B57" if row['Sentiment'] == "BULLISH" else \
                       "#CD5C5C" if row['Sentiment'] == "BEARISH" else \
                       "#DAA520"  # NEUTRAL or UNKNOWN
                        
                fig.add_trace(go.Bar(
                    x=[row['Provider']],
                    y=[row['Sentiment_Value']],
                    name=row['Sentiment'],
                    marker_color=color,
                    opacity=row['Confidence']/100,
                    text=f"{row['Sentiment']}<br>{row['Confidence']}% confidence",
                    hoverinfo='text'
                ))
            
            fig.update_layout(
                title="Sentiment Analysis Comparison",
                yaxis=dict(
                    ticktext=["Bearish", "Neutral", "Bullish"],
                    tickvals=[-1, 0, 1],
                    range=[-1.5, 1.5]
                ),
                showlegend=False,
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display sentiment legend
            st.markdown("### Legend")
            for sentiment in ['BULLISH', 'NEUTRAL', 'BEARISH']:
                color = "#2E8B57" if sentiment == "BULLISH" else \
                       "#CD5C5C" if sentiment == "BEARISH" else \
                       "#DAA520"
                st.markdown(f'<div style="color: {color};">● {sentiment}</div>', unsafe_allow_html=True)
            
            try:
                # Code that might raise an exception
                pass  # Replace with actual code if needed
            except Exception as e:
                st.error(f"Error generating insights: {str(e)}")

def main():
    """Main function to run the application"""
    # Initialize the app
    api_keys = initialize_app()
    
    # Create app sections
    stock_selection()
    start_date, end_date = date_selection()
    
    if start_date and end_date:
        # Fetch data button
        if st.button("Fetch Stock Data"):
            fetch_data(start_date, end_date)
        
        # Show analysis section if we have data
        if st.session_state.stock_data:
            analyze_data()
            
            # Show AI insights section if we have analysis results
            if st.session_state.analysis_results:
                generate_ai_insights()
    
    # Add footer
    st.markdown("---")
    st.markdown("### About this App")
    st.info(
        """
        This dashboard provides advanced stock analysis tools with technical indicators and AI-powered insights.
        Data is sourced from multiple providers for reliability and accuracy.
        """
    )

if __name__ == "__main__":
    main()