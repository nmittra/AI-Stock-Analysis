"""
Finnhub API client for fetching stock data.
"""
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, Any, Tuple, List
import logging

logger = logging.getLogger(__name__)

class FinnHubDataClient:
    """Client for fetching data from Finnhub API."""
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str):
        """
        Initialize the Finnhub client.
        
        Args:
            api_key: Finnhub API key
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': api_key
        })
        logger.info("Finnhub client initialized")
    
    def get_daily(
        self, 
        symbol: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        outputsize: str = 'full'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get daily stock data for a symbol.
        
        Note: Finnhub uses UNIX timestamps for date parameters.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            outputsize: Size of output (ignored, included for API compatibility)
            
        Returns:
            Tuple containing:
                - DataFrame with daily price data
                - Dictionary with metadata
        
        Raises:
            Exception: If API request fails
        """
        # Set default dates if not provided
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date:
            # Default to 1 year of data
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # Convert dates to UNIX timestamps
        try:
            start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            # Add a day to end_date to ensure we get the last day's data
            end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp()) + 86400
        except ValueError as e:
            logger.error(f"Date format error: {str(e)}")
            raise ValueError(f"Invalid date format. Use YYYY-MM-DD: {str(e)}")
        
        logger.info(f"Fetching Finnhub data for {symbol} from {start_date} to {end_date}")
        
        # Build the API endpoint
        endpoint = f"{self.BASE_URL}/stock/candle"
        
        # Set up parameters
        params = {
            'symbol': symbol.upper(),
            'resolution': 'D',  # Daily resolution
            'from': start_ts,
            'to': end_ts
        }
        
        try:
            # Make the request
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            
            data = response.json()
            
            # Check for error response
            if data.get('s') == 'no_data':
                logger.warning(f"No data returned from Finnhub for {symbol}")
                return pd.DataFrame(), {'symbol': symbol, 'error': 'No data returned'}
            
            # Convert response to DataFrame
            if all(k in data for k in ['c', 'h', 'l', 'o', 'v', 't']):
                df = pd.DataFrame({
                    'close': data['c'],
                    'high': data['h'],
                    'low': data['l'],
                    'open': data['o'],
                    'volume': data['v'],
                    'timestamp': data['t']
                })
                
                # Convert timestamp to datetime and set as index
                df['date'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('date', inplace=True)
                df.drop('timestamp', axis=1, inplace=True)
                
                # Sort by date
                df = df.sort_index()
                
                # Calculate daily returns
                df['daily_return'] = df['close'].pct_change() * 100
                
                # Create metadata
                metadata = {
                    'symbol': symbol,
                    'source': 'Finnhub',
                    'start_date': start_date,
                    'end_date': end_date,
                    'rows': len(df),
                    'status': data.get('s', 'unknown')
                }
                
                logger.info(f"Successfully fetched {len(df)} rows for {symbol} from Finnhub")
                
                return df, metadata
            else:
                logger.error(f"Unexpected response format from Finnhub: {data}")
                return pd.DataFrame(), {'error': 'Invalid response format', 'response': data}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Finnhub: {str(e)}")
            
            # Print actual response content for debugging
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
                
            raise Exception(f"Finnhub API request failed: {str(e)}")
        
        except Exception as e:
            logger.exception(f"Unexpected error with Finnhub API: {str(e)}")
            raise
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Any:
        """
        Make a request to the Finnhub API.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Optional query parameters
            
        Returns:
            JSON response data
        """
        if not params:
            params = {}
            
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error making request to {endpoint}: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response content: {e.response.text}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error with Finnhub API request: {str(e)}")
            raise
    
    def get_company_news(self, symbol: str, from_date: str, to_date: str) -> List[Dict]:
        """Fetch company news from Finnhub API"""
        try:
            endpoint = f"/company-news?symbol={symbol}&from={from_date}&to={to_date}"
            response = self._make_request(endpoint)
            
            # Limit to 10 most recent news items
            return response[:10] if response else []
        except Exception as e:
            print(f"Error fetching news for {symbol}: {str(e)}")
            return []
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Fetch company profile from Finnhub API"""
        try:
            endpoint = f"/stock/profile2?symbol={symbol}"
            return self._make_request(endpoint) or {}
        except Exception as e:
            print(f"Error fetching company profile for {symbol}: {str(e)}")
            return {}
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """
        Get real-time quote data.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with quote data
        """
        endpoint = f"{self.BASE_URL}/quote"
        
        params = {
            'symbol': symbol.upper()
        }
        
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {str(e)}")
            return {'error': str(e)}
            
    def get_recommendation_trends(self, symbol: str) -> Dict[str, Any]:
        """
        Get analyst recommendations for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing recommendation trends with the following structure:
            {
                'success': bool,
                'recommendations': List[Dict] or None,
                'error': str or None
            }
        """
        try:
            endpoint = f"{self.BASE_URL}/stock/recommendation"
            params = {'symbol': symbol.upper()}
            
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return {
                    'success': False,
                    'recommendations': None,
                    'error': 'No recommendation data available'
                }
            
            # Get the most recent recommendation
            latest_rec = data[0] if data else None
            
            if latest_rec:
                return {
                    'success': True,
                    'recommendations': {
                        'strongBuy': latest_rec.get('strongBuy', 0),
                        'buy': latest_rec.get('buy', 0),
                        'hold': latest_rec.get('hold', 0),
                        'sell': latest_rec.get('sell', 0),
                        'strongSell': latest_rec.get('strongSell', 0),
                        'period': latest_rec.get('period', '')
                    },
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'recommendations': None,
                    'error': 'No recent recommendations available'
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching recommendations for {symbol}: {str(e)}")
            return {
                'success': False,
                'recommendations': None,
                'error': f'API request failed: {str(e)}'
            }
        except Exception as e:
            logger.error(f"Unexpected error fetching recommendations for {symbol}: {str(e)}")
            return {
                'success': False,
                'recommendations': None,
                'error': f'Unexpected error: {str(e)}'
            }

# Testing function to verify the client works
def test_finnhub_client():
    """Test function to verify Finnhub client functionality."""
    import os
    from dotenv import load_dotenv
    
    # Load API key from environment
    load_dotenv()
    api_key = os.getenv('FINNHUB_API_KEY')
    
    if not api_key:
        print("No Finnhub API key found. Set FINNHUB_API_KEY environment variable.")
        return
    
    # Create client and fetch data
    client = FinnHubDataClient(api_key)
    
    try:
        # Test with a known symbol
        symbol = 'AAPL'
        start_date = '2023-01-01'
        end_date = '2023-01-31'
        
        data, metadata = client.get_daily(symbol, start_date, end_date)
        
        print(f"\nMetadata: {metadata}")
        print(f"\nData shape: {data.shape}")
        print(f"\nData columns: {data.columns.tolist()}")
        print(f"\nFirst 5 rows:\n{data.head()}")
        
        if not data.empty:
            print("\nTest passed! Finnhub client is working correctly.")
        else:
            print("\nWarning: No data returned but no error was raised.")
        
        # Test quotes
        quote = client.get_quote(symbol)
        print(f"\nCurrent quote: {quote}")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")

if __name__ == "__main__":
    # Set up logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the test
    test_finnhub_client()