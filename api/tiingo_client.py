"""Tiingo API client for fetching stock data."""
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TiingoDataClient:
    """Client for fetching data from Tiingo API."""
    
    BASE_URL = "https://api.tiingo.com/tiingo"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Authorization': f'Token {api_key}'
        })
        logger.info("Tiingo client initialized")
    
    def get_daily(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        outputsize: str = 'full'  # Added this parameter
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
        endpoint = f"{self.BASE_URL}/daily/{symbol.upper()}/prices"
        params = {
            'startDate': start_date,
            'endDate': end_date,
            'format': 'json',
            'resampleFreq': 'daily'
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame(), {'symbol': symbol, 'error': 'No data returned'}
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Rename columns to standard format
            column_mapping = {
                'adjOpen': 'Open',
                'adjHigh': 'High',
                'adjLow': 'Low',
                'adjClose': 'Close',
                'adjVolume': 'Volume'
            }
            df = df.rename(columns=column_mapping)
            
            metadata = {
                'symbol': symbol,
                'source': 'Tiingo',
                'start_date': start_date,
                'end_date': end_date,
                'rows': len(df)
            }
            
            return df, metadata
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Tiingo: {str(e)}")
            raise
