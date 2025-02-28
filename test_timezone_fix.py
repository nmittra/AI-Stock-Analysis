from datetime import datetime, timezone
import streamlit as st
from api.error_handling import validate_date_range
from api.clients import fetch_stock_data

# Test the validate_date_range function directly
print("Testing validate_date_range function...")
try:
    # Create a timezone-naive datetime
    start_naive = datetime.now()
    # Create a timezone-aware datetime
    end_aware = datetime.now(timezone.utc)
    
    print(f"Start date (naive): {start_naive}")
    print(f"End date (aware): {end_aware}")
    
    # This would have failed before our fix
    validated_start, validated_end = validate_date_range(start_naive, end_aware)
    
    print(f"Validated start: {validated_start} (tzinfo: {validated_start.tzinfo})")
    print(f"Validated end: {validated_end} (tzinfo: {validated_end.tzinfo})")
    print("✅ Date validation successful!")
except Exception as e:
    print(f"❌ Error in validate_date_range: {e}")

# Test the fetch_stock_data function with a mock client
print("\nTesting fetch_stock_data function...")
try:
    # Create a simple mock client that just returns the dates
    class MockClient:
        def get_daily(self, symbol, outputsize):
            # Just return empty data to test the date handling
            import pandas as pd
            data = pd.DataFrame()
            metadata = {}
            return data, metadata
    
    # Test with timezone-naive and timezone-aware dates
    ticker = "AAPL"
    start_date = datetime.now()
    end_date = datetime.now(timezone.utc)
    
    print(f"Fetching data for {ticker} from {start_date} to {end_date}")
    
    # This would have failed before our fix
    result = fetch_stock_data(ticker, start_date, end_date, MockClient())
    
    print("✅ fetch_stock_data executed without timezone errors!")
except Exception as e:
    print(f"❌ Error in fetch_stock_data: {e}")