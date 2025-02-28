"""
Debugging script to test data fetching functionality.
Run this script directly to verify that the data fetching works.
"""
import os
import pandas as pd
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from api.tiingo_client import TiingoDataClient
from api.finnhub_client import FinnHubDataClient
from api.clients import initialize_api_clients, fetch_stock_data

def debug_data_fetch():
    """Test function to debug data fetching."""
    print("=== Stock Data Fetching Debug ===\n")
    
    # Load environment variables
    load_dotenv()
    
    # Get API keys
    api_keys = {
        'tiingo': os.getenv('TIINGO_API_KEY'),
        'finnhub': os.getenv('FINNHUB_API_KEY'),
        'openai': os.getenv('OPENAI_API_KEY'),
        'google': os.getenv('GOOGLE_API_KEY')
    }
    
    print("API Keys:")
    for k, v in api_keys.items():
        if v:
            print(f"  {k}: {'*' * 8}{v[-4:] if v else 'Not found'}")
        else:
            print(f"  {k}: Not found")
    
    print("\nInitializing API clients...")
    clients = initialize_api_clients(api_keys)
    
    data_clients = clients.get('data', {})
    if not data_clients:
        print("No data clients initialized. Please check API keys.")
        return
    
    print(f"Initialized {len(data_clients)} data clients: {', '.join(data_clients.keys())}")
    
    # Test parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    end_date_str = datetime.now().strftime('%Y-%m-%d')
    start_date_str = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Convert string dates to datetime objects for fetch_stock_data
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    
    print(f"\nTesting date range: {start_date_str} to {end_date_str}")
    
    # Test each client directly first
    for client_name, client in data_clients.items():
        print(f"\n=== Testing {client_name} client directly ===")
        
        for symbol in symbols:
            print(f"\nFetching {symbol} data from {client_name}...")
            try:
                # Use string dates for direct client calls as expected by the client APIs
                data, metadata = client.get_daily(symbol, start_date_str, end_date_str)
                
                if data is not None and not data.empty:
                    print(f"  Success! Got {len(data)} rows of data.")
                    print(f"  Data columns: {data.columns.tolist()}")
                    print(f"  First row: {data.iloc[0].to_dict()}")
                else:
                    print(f"  No data returned for {symbol} from {client_name}")
            except Exception as e:
                print(f"  Error: {str(e)}")
    
    # Now test the fetch_stock_data function
    print("\n=== Testing fetch_stock_data function ===")
    
    for symbol in symbols:
        print(f"\nFetching {symbol} data via fetch_stock_data...")
        try:
            # Use datetime objects for fetch_stock_data as it expects datetime objects
            data, metadata, error = fetch_stock_data(symbol, start_date, end_date, data_clients)
            
            if error:
                print(f"  Error: {error}")
            elif data is not None and not data.empty:
                print(f"  Success! Got {len(data)} rows of data.")
                print(f"  Data columns: {data.columns.tolist()}")
                print(f"  First row: {data.iloc[0].to_dict()}")
            else:
                print(f"  No data returned for {symbol} from fetch_stock_data")
        except Exception as e:
            print(f"  Error: {str(e)}")

if __name__ == "__main__":
    debug_data_fetch()