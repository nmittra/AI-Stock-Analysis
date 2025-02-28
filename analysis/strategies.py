"""Advanced trading strategies including Elliott Wave and CAN SLIM screening."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def detect_elliott_wave(data: pd.DataFrame) -> Dict[str, any]:
    """Detect Elliott Wave patterns in the price data.
    
    Args:
        data: DataFrame containing OHLCV data
    
    Returns:
        dict: Dictionary containing Elliott Wave analysis results
    """
    # Initialize variables
    close_prices = data['Close'].values
    highs = data['High'].values
    lows = data['Low'].values
    
    # Find potential wave points using local extrema
    peaks = []
    troughs = []
    
    for i in range(1, len(close_prices)-1):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            peaks.append((i, highs[i]))
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            troughs.append((i, lows[i]))
    
    # Analyze wave structure
    wave_structure = analyze_wave_structure(peaks, troughs)
    
    return {
        'wave_structure': wave_structure,
        'current_wave': identify_current_wave(wave_structure),
        'next_targets': calculate_wave_targets(wave_structure, close_prices[-1])
    }

def analyze_wave_structure(peaks: List[Tuple[int, float]], troughs: List[Tuple[int, float]]) -> Dict[str, any]:
    """Analyze the wave structure based on peaks and troughs.
    
    Args:
        peaks: List of peak points (index, price)
        troughs: List of trough points (index, price)
    
    Returns:
        dict: Wave structure analysis
    """
    # Combine and sort all points
    all_points = sorted(peaks + troughs, key=lambda x: x[0])
    
    # Initialize wave counts
    impulse_waves = 0
    corrective_waves = 0
    
    # Analyze wave patterns
    waves = []
    for i in range(len(all_points)-1):
        start_point = all_points[i]
        end_point = all_points[i+1]
        
        # Calculate wave characteristics
        wave_length = end_point[0] - start_point[0]
        price_change = end_point[1] - start_point[1]
        
        # Classify wave
        if price_change > 0:
            wave_type = 'impulse'
            impulse_waves += 1
        else:
            wave_type = 'corrective'
            corrective_waves += 1
            
        waves.append({
            'type': wave_type,
            'start': start_point,
            'end': end_point,
            'length': wave_length,
            'magnitude': abs(price_change)
        })
    
    return {
        'waves': waves,
        'impulse_count': impulse_waves,
        'corrective_count': corrective_waves
    }

def identify_current_wave(wave_structure: Dict[str, any]) -> str:
    """Identify the current wave position.
    
    Args:
        wave_structure: Dictionary containing wave analysis
    
    Returns:
        str: Current wave position description
    """
    waves = wave_structure['waves']
    if not waves:
        return 'No clear wave pattern'
    
    last_wave = waves[-1]
    total_waves = len(waves)
    
    if total_waves >= 5:
        if wave_structure['impulse_count'] == 3 and wave_structure['corrective_count'] == 2:
            return 'Completing 5-wave impulse pattern'
        elif wave_structure['impulse_count'] > 3:
            return 'Possible extended wave pattern'
    
    return f'Wave {total_waves} - {last_wave["type"].capitalize()}'

def calculate_wave_targets(wave_structure: Dict[str, any], current_price: float) -> Dict[str, float]:
    """Calculate potential price targets based on wave relationships.
    
    Args:
        wave_structure: Dictionary containing wave analysis
        current_price: Current stock price
    
    Returns:
        dict: Price targets for next waves
    """
    waves = wave_structure['waves']
    if not waves:
        return {'support': current_price * 0.95, 'resistance': current_price * 1.05}
    
    # Calculate Fibonacci relationships
    wave_magnitudes = [wave['magnitude'] for wave in waves]
    avg_magnitude = np.mean(wave_magnitudes)
    
    targets = {
        'support': current_price - (avg_magnitude * 0.618),  # 61.8% retracement
        'resistance': current_price + (avg_magnitude * 1.618)  # 161.8% extension
    }
    
    return targets

def can_slim_screen(data: pd.DataFrame, fundamentals: Dict[str, any]) -> Dict[str, bool]:
    """Screen stocks based on CAN SLIM criteria.
    
    Args:
        data: DataFrame containing OHLCV data
        fundamentals: Dictionary containing fundamental data
    
    Returns:
        dict: CAN SLIM screening results
    """
    results = {
        'current_earnings': False,  # C: Current quarterly earnings
        'annual_earnings': False,   # A: Annual earnings growth
        'new_factors': False,       # N: New products, management, highs
        'supply_demand': False,     # S: Supply and demand
        'leader_laggard': False,    # L: Leader or laggard
        'institutional_support': False,  # I: Institutional support
        'market_direction': False    # M: Market direction
    }
    
    # Current quarterly earnings (C)
    if fundamentals.get('quarterly_earnings_growth', 0) >= 25:
        results['current_earnings'] = True
    
    # Annual earnings growth (A)
    if fundamentals.get('annual_earnings_growth', 0) >= 25:
        results['annual_earnings'] = True
    
    # New factors (N)
    if fundamentals.get('new_products', False) or fundamentals.get('new_management', False):
        results['new_factors'] = True
    
    # Supply and demand (S)
    volume = data['Volume'].values
    avg_volume = np.mean(volume[-50:])  # 50-day average volume
    if volume[-1] > avg_volume * 1.5:  # 50% above average
        results['supply_demand'] = True
    
    # Leader or laggard (L)
    if fundamentals.get('relative_strength_rating', 0) >= 80:
        results['leader_laggard'] = True
    
    # Institutional support (I)
    if fundamentals.get('institutional_ownership', 0) >= 30:
        results['institutional_support'] = True
    
    # Market direction (M)
    market_trend = calculate_market_trend(data)
    if market_trend == 'uptrend':
        results['market_direction'] = True
    
    return results

def calculate_market_trend(data: pd.DataFrame) -> str:
    """Calculate the overall market trend.
    
    Args:
        data: DataFrame containing OHLCV data
    
    Returns:
        str: Market trend direction
    """
    close_prices = data['Close'].values
    sma_50 = pd.Series(close_prices).rolling(window=50).mean().values
    sma_200 = pd.Series(close_prices).rolling(window=200).mean().values
    
    current_price = close_prices[-1]
    
    if current_price > sma_50[-1] and sma_50[-1] > sma_200[-1]:
        return 'uptrend'
    elif current_price < sma_50[-1] and sma_50[-1] < sma_200[-1]:
        return 'downtrend'
    else:
        return 'sideways'