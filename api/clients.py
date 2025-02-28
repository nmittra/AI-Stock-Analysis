from openai import OpenAI
import google.generativeai as genai
from alpha_vantage.timeseries import TimeSeries
import streamlit as st
import pandas as pd
from datetime import datetime, date, timezone
from .error_handling import (
    APIError, ValidationError, APIKeyError, RateLimitError,
    NetworkError, DataNotFoundError, retry_with_backoff,
    validate_ticker, validate_date_range
)
from .llm_models import OpenSourceLLMClient

def initialize_api_clients() -> tuple[OpenAI | None, genai.GenerativeModel | None, OpenSourceLLMClient | None, OpenSourceLLMClient | None, TimeSeries | None]:
    """Initialize and configure all API clients with validated API keys.
    
    Returns:
        tuple: A tuple containing:
            - OpenAI: DeepSeek client for AI analysis
            - GenerativeModel: Google's Gemini model for AI analysis
            - OpenSourceLLMClient: Falcon-7B model for AI analysis
            - OpenSourceLLMClient: LLama-2-13B model for AI analysis
            - TimeSeries: Alpha Vantage client for stock data
            Returns (None, None, None, None, None) if initialization fails
    
    Raises:
        Exception: If API key validation or client initialization fails
    """
    try:
        # Get validated API keys
        from .key_management import APIKeyManager
        keys = APIKeyManager.get_validated_keys()
        
        # Initialize clients with validated keys
        deepseek_client = OpenAI(api_key=keys['deepseek'], base_url="https://api.deepseek.com")
        genai.configure(api_key=keys['google'])
        gen_model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Initialize open-source LLM models
        falcon_client = OpenSourceLLMClient('falcon-7b')
        llama_client = OpenSourceLLMClient('llama-2-13b')
        
        alpha_vantage_client = TimeSeries(key=keys['alpha_vantage'], output_format='pandas')
        
        return deepseek_client, gen_model, falcon_client, llama_client, alpha_vantage_client
    except Exception as e:
        st.error(f"Failed to initialize API clients: {str(e)}")
        return None, None, None, None, None

@st.cache_data(ttl=3600)  # Cache for 1 hour
@retry_with_backoff()
def fetch_stock_data(ticker: str, 
                   _start_date: datetime, 
                   _end_date: datetime,
                   _client) -> tuple[pd.DataFrame | None, dict | None, str | None]:
    """Fetch stock data using Alpha Vantage API with caching.
    
    Args:
        client: Alpha Vantage TimeSeries client
        ticker: Stock symbol to fetch data for
        _start_date: Start date for the data range
        _end_date: End date for the data range
    
    Returns:
        tuple: A tuple containing:
            - DataFrame: Stock price data if successful, None if failed
            - dict: Metadata about the stock if successful, None if failed
            - str: Error message if failed, None if successful
    
    Raises:
        APIKeyError: If Alpha Vantage client is not initialized or key is invalid
        RateLimitError: If API rate limit is exceeded
        NetworkError: If there are network connectivity issues
        DataNotFoundError: If no data is found for the ticker
    """
    try:
        # Validate inputs
        ticker = validate_ticker(ticker)
        start_date, end_date = validate_date_range(_start_date, _end_date)
        
        if not _client:
            raise APIKeyError("Alpha Vantage client not initialized")
        
        # Handle different types of clients (dict or direct client)
        if isinstance(_client, dict):
            # If _client is a dictionary of clients, try each one until successful
            data = None
            meta_data = None
            last_error = None
            
            for client_name, client in _client.items():
                try:
                    data, meta_data = client.get_daily(symbol=ticker, outputsize='full')
                    if data is not None and not isinstance(data, pd.DataFrame):
                        raise TypeError(f"Expected DataFrame but got {type(data)}")
                    if data is not None and not data.empty:
                        break  # Successfully got data, exit the loop
                except Exception as e:
                    last_error = e
                    continue  # Try the next client
            
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                if last_error:
                    raise last_error
                raise DataNotFoundError(f"No data found for ticker {ticker}")
        else:
            # Direct client
            try:
                data, meta_data = _client.get_daily(symbol=ticker, outputsize='full')
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    raise DataNotFoundError(f"No data found for ticker {ticker}")
            except ValueError as e:
                if "Invalid API call" in str(e):
                    raise APIKeyError("Invalid Alpha Vantage API key")
                elif "premium" in str(e).lower():
                    raise RateLimitError("API rate limit exceeded")
                else:
                    raise NetworkError(f"Failed to fetch data: {str(e)}")
        
        # Ensure data is a DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame but got {type(data)}")
            
        data = data.rename(columns={
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '4. close': 'Close',
            '5. volume': 'Volume'
        })
        
        # Ensure index is timezone-naive before filtering
        if hasattr(data.index, 'tz') and data.index.tz is not None:
            # Convert index to timezone-naive for consistent comparison
            data.index = data.index.tz_localize(None)
        
        # Filter data based on date range
        # Ensure start_date and end_date are timezone-naive for consistent comparison
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
            end_date = end_date.replace(tzinfo=None)
            
        data = data.loc[start_date:end_date]
        if data.empty:
            raise DataNotFoundError(f"No data found for {ticker} in selected date range")
        
        # Downsample data if it's too large (e.g., more than 1000 points)
        if len(data) > 1000:
            data = downsample_data(data, 1000)
            
        return data, meta_data, None
    except (APIError, ValidationError) as e:
        return None, None, str(e)
    except Exception as e:
        return None, None, f"Unexpected error: {str(e)}"

@st.cache_data(ttl=300)  # Cache for 5 minutes
@retry_with_backoff()
def analyze_with_ai(client, model_type, analysis_prompt, image_data=None):
    """Analyze stock data using AI models with caching."""
    try:
        if not client:
            raise APIKeyError(f"{model_type} client not initialized")
            
        if model_type == "DeepSeek":
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {"role": "system", "content": analysis_prompt},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if "rate limit" in str(e).lower():
                    raise RateLimitError("DeepSeek API rate limit exceeded")
                elif "invalid" in str(e).lower() and "api key" in str(e).lower():
                    raise APIKeyError("Invalid DeepSeek API key")
                else:
                    raise NetworkError(f"DeepSeek API error: {str(e)}")
        elif model_type == "Gemini":
            try:
                contents = [
                    {"role": "user", "parts": [analysis_prompt]},
                    {"role": "user", "parts": [image_data]}
                ]
                response = client.generate_content(contents=contents)
                return response.text
            except Exception as e:
                if "quota" in str(e).lower():
                    raise RateLimitError("Gemini API rate limit exceeded")
                elif "invalid" in str(e).lower() and "api key" in str(e).lower():
                    raise APIKeyError("Invalid Gemini API key")
                else:
                    raise NetworkError(f"Gemini API error: {str(e)}")
        elif model_type in ["Falcon-7B", "LLama-2-13B"]:
            try:
                return client.generate_analysis(analysis_prompt)
            except Exception as e:
                raise NetworkError(f"{model_type} error: {str(e)}")
    except (APIError, ValidationError) as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def downsample_data(data, n_points):
    """Downsample the data to n_points while preserving important features."""
    # Calculate the sampling interval
    interval = max(1, len(data) // n_points)
    
    # Create resampled dataframe
    resampled = pd.DataFrame()
    for column in data.columns:
        if column in ['Open', 'High', 'Low', 'Close', 'Volume']:
            # For OHLC data, use appropriate aggregation
            if column == 'High':
                resampled[column] = data[column].rolling(interval).max()
            elif column == 'Low':
                resampled[column] = data[column].rolling(interval).min()
            elif column in ['Open', 'Close']:
                resampled[column] = data[column].rolling(interval).mean()
            elif column == 'Volume':
                resampled[column] = data[column].rolling(interval).sum()
    
    # Remove NaN values and sample every nth row
    resampled = resampled.dropna()
    resampled = resampled.iloc[::interval]
    
    return resampled