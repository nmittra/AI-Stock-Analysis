"""Module for managing and validating API keys with environment variable fallbacks."""

import os
import streamlit as st
from openai import OpenAI
import google.generativeai as genai
from alpha_vantage.timeseries import TimeSeries
from .finnhub_client import FinnHubDataClient
from typing import Dict

class APIKeyManager:
    """Manages API keys with validation and fallback mechanisms."""

    @staticmethod
    def get_api_key(key_name, fallback_env_var):
        """Get API key from Streamlit secrets with environment variable fallback."""
        try:
            # Try to get key from Streamlit secrets first
            api_key = st.secrets.get(key_name)
            if not api_key:
                # Fallback to environment variable
                api_key = os.environ.get(fallback_env_var)
            
            if not api_key:
                raise ValueError(f"No API key found for {key_name}")
            
            return api_key
        except Exception as e:
            raise ValueError(f"Error retrieving API key for {key_name}: {str(e)}")

    @staticmethod
    def validate_deepseek_key(api_key):
        """Validate DeepSeek API key by attempting to create a client."""
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            # Attempt a simple API call to validate the key
            client.models.list()
            return True
        except Exception:
            return False

    @staticmethod
    def validate_google_key(api_key):
        """Validate Google API key by attempting to configure the client."""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-2.0-flash')
            # Attempt a simple generation to validate the key
            model.generate_content("test")
            return True
        except Exception:
            return False

    @staticmethod
    def validate_alpha_vantage_key(api_key):
        """Validate Alpha Vantage API key by attempting to create a client."""
        try:
            client = TimeSeries(key=api_key)
            # Attempt a simple API call to validate the key
            client.get_daily(symbol='AAPL', outputsize='compact')
            return True
        except Exception:
            return False

    @staticmethod
    def validate_tiingo_key(api_key: str) -> bool:
        """Validate Tiingo API key by attempting to create a client."""
        try:
            # Skip validation if no key is provided
            if not api_key:
                return True
            # Only validate if Tiingo is the selected provider
            if 'data_provider' not in st.session_state or st.session_state['data_provider'] != 'Tiingo':
                return True
            # Import here to avoid circular import
            from .tiingo_client import TiingoDataClient
            return TiingoDataClient.validate_api_key(api_key)
        except Exception:
            return False

    @staticmethod
    def validate_finnhub_key(api_key):
        """Validate FinnHub API key by attempting to create a client."""
        return FinnHubDataClient.validate_api_key(api_key)

    @classmethod
    def get_validated_keys(cls):
        """Get and validate all required API keys."""
        keys = {}
        validation_errors = []
        
        try:
            # Get and validate DeepSeek API key
            deepseek_key = cls.get_api_key("DEEPSEEK_API_KEY", "DEEPSEEK_API_KEY")
            if not cls.validate_deepseek_key(deepseek_key):
                validation_errors.append("Invalid DeepSeek API key")
            keys['deepseek'] = deepseek_key

            # Get and validate Google API key
            google_key = cls.get_api_key("GOOGLE_API_KEY", "GOOGLE_API_KEY")
            if not cls.validate_google_key(google_key):
                validation_errors.append("Invalid Google API key")
            keys['google'] = google_key

            # Get and validate Alpha Vantage API key
            alpha_vantage_key = cls.get_api_key("ALPHA_VANTAGE_API_KEY", "ALPHA_VANTAGE_API_KEY")
            if not cls.validate_alpha_vantage_key(alpha_vantage_key):
                validation_errors.append("Invalid Alpha Vantage API key")
            keys['alpha_vantage'] = alpha_vantage_key

            # Get and validate Tiingo API key
            tiingo_key = cls.get_api_key("TIINGO_API_KEY", "TIINGO_API_KEY")
            if not cls.validate_tiingo_key(tiingo_key):
                validation_errors.append("Invalid Tiingo API key")
            keys['tiingo'] = tiingo_key

            # Get and validate FinnHub API key
            finnhub_key = cls.get_api_key("FINNHUB_API_KEY", "FINNHUB_API_KEY")
            if not cls.validate_finnhub_key(finnhub_key):
                validation_errors.append("Invalid FinnHub API key")
            keys['finnhub'] = finnhub_key

            if validation_errors:
                raise ValueError(f"API key validation failed: {', '.join(validation_errors)}")

            return keys

        except Exception as e:
            raise ValueError(f"API key validation failed: {str(e)}")