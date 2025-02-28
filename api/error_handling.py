"""Module for custom exceptions and retry logic for API calls."""

from functools import wraps
import time
import streamlit as st

class APIError(Exception):
    """Base exception class for API-related errors."""
    pass

class ValidationError(Exception):
    """Exception raised for input validation errors."""
    pass

class APIKeyError(APIError):
    """Exception raised for API key related errors."""
    pass

class RateLimitError(APIError):
    """Exception raised when API rate limits are hit."""
    pass

class NetworkError(APIError):
    """Exception raised for network connectivity issues."""
    pass

class DataNotFoundError(APIError):
    """Exception raised when requested data is not found."""
    pass

def validate_ticker(ticker):
    """Validate stock ticker symbol."""
    if not ticker or not isinstance(ticker, str):
        raise ValidationError("Ticker must be a non-empty string")
    if not ticker.isalnum():
        raise ValidationError("Ticker must contain only letters and numbers")
    return ticker.upper()

def validate_date_range(start_date, end_date):
    """Validate date range for stock data.
    
    Args:
        start_date: Start date for the data range (datetime.date or datetime.datetime)
        end_date: End date for the data range (datetime.date or datetime.datetime)
        
    Returns:
        tuple: Validated (start_date, end_date) with consistent timezone handling
        
    Raises:
        ValidationError: If dates are invalid or in wrong order
    """
    from datetime import date, datetime, timezone
    
    if not start_date or not end_date:
        raise ValidationError("Both start date and end date must be provided")
        
    # Validate date types
    valid_types = (date, datetime)
    if not isinstance(start_date, valid_types):
        raise ValidationError(f"Start date must be a date or datetime object, got {type(start_date)}")
    if not isinstance(end_date, valid_types):
        raise ValidationError(f"End date must be a date or datetime object, got {type(end_date)}")
        
    # Convert to datetime if needed
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        start_date = datetime.combine(start_date, datetime.min.time())
    if isinstance(end_date, date) and not isinstance(end_date, datetime):
        end_date = datetime.combine(end_date, datetime.max.time())
    
    # Handle timezone awareness - ALWAYS make both timezone-naive for consistent comparison
    # This is the safest approach to avoid comparison errors
    if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
        # Convert to UTC first to preserve the actual time, then remove timezone info
        start_date = start_date.astimezone(timezone.utc).replace(tzinfo=None)
    
    if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
        # Convert to UTC first to preserve the actual time, then remove timezone info
        end_date = end_date.astimezone(timezone.utc).replace(tzinfo=None)
    
    # Now both dates are timezone-naive and can be safely compared
    if start_date > end_date:
        raise ValidationError("Start date must be before end date")
        
    return start_date, end_date

def retry_with_backoff(max_retries=3, initial_delay=1, max_delay=10, exponential_base=2):
    """Decorator for implementing exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for retry in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    last_exception = e
                    st.warning(f"Rate limit reached. Retrying in {delay} seconds...")
                except NetworkError as e:
                    last_exception = e
                    st.warning(f"Network error occurred. Retrying in {delay} seconds...")
                except (APIKeyError, ValidationError, DataNotFoundError):
                    # Don't retry for these errors
                    raise
                except Exception as e:
                    last_exception = e
                    st.warning(f"Unexpected error occurred. Retrying in {delay} seconds...")

                time.sleep(delay)
                delay = min(delay * exponential_base, max_delay)

            raise last_exception or Exception("Max retries exceeded")
        return wrapper
    return decorator